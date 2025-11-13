#include "p5pf_fisheye.h"
#include "p4pf_fisheye.h"
#include "p5lp_radial.h"
#include "p5pf.h"
#include <PoseLib/robust/optim/absolute.h>
#include <PoseLib/robust/optim/lm_impl.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Cholesky>

namespace poselib {
bool refine_tz_and_f_fisheye(
    const std::vector<Eigen::Vector2d>& points2d,
    const std::vector<Eigen::Vector3d>& points3d,
    CameraPose& pose,   // 传引用，里面的 t_z 会被修改
    double& f,          // 传引用，焦距会被修改
    int max_iters = 5)
{
    // 根据你原来的过滤逻辑，f 一般在“像素数量级”
    const double f_min = 1e-3;   // 这些范围可以按你数据情况调
    const double f_max = 1e6;
    const double step_damping = 0.5;    // 步长，防止一下子跑飞
    const double tol_step = 1e-6;        // 步长收敛阈值
    const double tol_res  = 1e-8;        // 残差阈值（看情况）

    if (!std::isfinite(f) || f <= 0.0) {
        return false; // 初始就不靠谱，放弃
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        Eigen::Matrix2d JtJ = Eigen::Matrix2d::Zero();
        Eigen::Vector2d Jtg = Eigen::Vector2d::Zero();
        double res_sum = 0.0;
        int valid_cnt = 0;

        const double tz = pose.t(2);

        for (int k = 0; k < 5; ++k) {
            const Eigen::Vector2d& p2 = points2d[k];
            const Eigen::Vector3d& X  = points3d[k];

            // 旋转 + 平移
            Eigen::Vector3d RX = pose.rotate(X);
            double Xc = RX(0) + pose.t(0);
            double Yc = RX(1) + pose.t(1);
            double Zc = RX(2); // c

            // 按你原来的定义
            double a = Xc / p2[0];
            double b = Yc / p2[1];
            double c = Zc;
            double d = std::min(std::abs(a), std::abs(b));

            // 过滤掉几何太扯的点（你的原逻辑）
            if (d > 1e-2) continue;

            double r = p2.norm();      // r_i
            if (r <= 0.0) continue;

            double theta = r / f;      // r_i / f
            if (theta < 1e-4) continue;
            if (!std::isfinite(theta) || std::abs(theta) > 1e3) {
                continue; // theta 太大或非数，过滤
            }

            double tan_theta = std::tan(theta);
            double cos_theta = std::cos(theta);
            if (std::abs(cos_theta) < 1e-6) continue; // 接近 pi/2，会炸
            double sec_theta = 1.0 / cos_theta;

            if (!std::isfinite(tan_theta) || !std::isfinite(sec_theta)) {
                continue;
            }

            double ci_plus_tz = c + tz;

            // 残差 g_i = (c_i + t_z) * tan(theta_i) - r_i * d_i
            double g = ci_plus_tz * tan_theta - r * d;

            // 导数：
            // dg/dtz = tan(theta)
            double dg_dtz = tan_theta;

            // dg/df = (c + tz) * d/d f( tan(r/f) )
            //       = (c + tz) * sec^2(theta) * (-r/f^2)
            //       = -(c + tz) * sec^2(theta) * theta / f
            // 因为 theta = r/f → r/f^2 = theta/f
            double dg_df = -(ci_plus_tz) * (sec_theta * sec_theta) * (theta / f);

            if (!std::isfinite(g) || !std::isfinite(dg_dtz) || !std::isfinite(dg_df)) {
                continue;
            }

            Eigen::Vector2d Ji(dg_dtz, dg_df);

            JtJ.noalias() += Ji * Ji.transpose();   // 2x2
            Jtg.noalias() += g * Ji;                // 2x1
            res_sum       += std::abs(g);
            ++valid_cnt;
        }

        // 有效点太少就不要解这个 2D 问题了
        if (valid_cnt < 2) {
            // 视为没有好的更新，返回 false 或 true 看你需求
            return false;
        }

        // 残差几乎为 0，就算收敛了
        if (res_sum / valid_cnt < tol_res) {
            return true;
        }

        // 检查 JtJ 是否数值正常
        if (!JtJ.allFinite() || !Jtg.allFinite()) {
            return false;
        }

        // 不用 inverse —— 用 LDLT
        Eigen::LDLT<Eigen::Matrix2d> ldlt(JtJ);
        if (ldlt.info() != Eigen::Success) {
            return false; // 分解失败，说明 JtJ 病态
        }

        Eigen::Vector2d delta = -ldlt.solve(Jtg);
        if (!delta.allFinite()) {
            return false; // 解出来就已经 nan/inf 了
        }

        // 步长控制，防止一步走太大
        double dtz = step_damping * delta(0);
        double df  = step_damping * delta(1);

        // 步长很小就可以停了
        if (std::abs(dtz) + std::abs(df) < tol_step) {
            return true;
        }

        // 更新参数（注意先算新 f, 再检查范围）
        double f_new  = f  + df;
        double tz_new = pose.t(2) + dtz;

        // 对 f 做个简单范围限制，避免跑成负数或 0
        if (!std::isfinite(f_new) || f_new < f_min || f_new > f_max) {
            return false; // 认为发散了
        }

        pose.t(2) = tz_new;
        f         = f_new;
    }

    // 迭代结束，认为 OK（是否加额外检查看你）
    return std::isfinite(f) && pose.t(2) == pose.t(2);
};

int p5pf_fisheye(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
         std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, bool normalize_input) {
    if (normalize_input) {
        double focal0 = 0.0;
        for (int i = 0; i < 5; ++i) {
            focal0 += points2d[i].norm();
        }
        focal0 /= 5;

        std::vector<Eigen::Vector2d> scaled_points2d;
        scaled_points2d.reserve(5);
        for (int i = 0; i < 5; ++i) {
            scaled_points2d.push_back(points2d[i] / focal0);
        }

        int n_sols = p5pf_fisheye(scaled_points2d, points3d, output_poses, output_focals, false);

        for (int i = 0; i < n_sols; ++i) {
            (*output_focals)[i] *= focal0;
        }
        return n_sols;
    }

    std::vector<CameraPose> poses_p5pf;
    std::vector<double> focals_p5pf;
    p5pf(points2d, points3d, &poses_p5pf, &focals_p5pf);

    // bool refine_success = false;
    // for (int i = 0; i < poses_p5pf.size(); i++) {
    //     CameraPose p = poses_p5pf[i];
    //     double f = focals_p5pf[i];
    //     refine_success = refine_tz_and_f_fisheye(points2d, points3d, p, f, 5);
    //     if (refine_success){
    //         poses_p5pf[i] = p;
    //         focals_p5pf[i] = f;
    //     }
    // }

    // n-step Newton refinement
    for (int i = 0; i < poses_p5pf.size(); i++) {
        CameraPose p = poses_p5pf[i];
        double f = focals_p5pf[i];
        double t_z = p.t(2);

        Eigen::Matrix2d JtJ;
        Eigen::Vector2d Jtg;
        double res = 0;
        for (int iter = 0; iter < 5; iter++) {
            JtJ.setZero();
            Jtg.setZero();
            res = 0;
            for (int k = 0; k < 5; ++k) {

                Eigen::Vector3d RX = p.rotate(points3d[k]);
                double a = (RX(0) + p.t(0)) / points2d[k][0];
                double b = (RX(1) + p.t(1)) / points2d[k][1];
                double c = RX(2);
                double d = std::min(std::abs(a), std::abs(b));
                // if (d > 1e-2){continue;}

                double r = std::sqrt(points2d[k][0] * points2d[k][0] + points2d[k][1] * points2d[k][1]);
                double theta = r / f;
                // if (theta < 1e-4) {continue;}

                double dgdtz = std::tan(theta);
                double sec_theta = 1.0 / std::cos(theta);
                double dgdf = -(c + t_z) * sec_theta * sec_theta * theta/f;
                double g = (c + t_z) * dgdtz - r * d;
                res += std::abs(g);
                
                JtJ(0,0) += dgdtz * dgdtz;
                JtJ(0,1) += dgdtz * dgdf;
                JtJ(1,0) += dgdf * dgdtz;
                JtJ(1,1) += dgdf * dgdf;
                Jtg(0) += g * dgdtz;
                Jtg(1) += g * dgdf;
            }
            if (res < 1e-8){break;}

            // Eigen::Vector2d delta_sol = -JtJ.inverse() * Jtg;
            Eigen::LDLT<Eigen::Matrix2d> ldlt(JtJ);
            if (ldlt.info() != Eigen::Success) break;
            Eigen::Vector2d delta_sol = -ldlt.solve(Jtg);

            t_z = t_z+delta_sol(0);
            f = f+delta_sol(1);

            if ((std::abs(delta_sol(0)) + std::abs(delta_sol(1)) ) < 1e-6) {
                break;
            }

        }
        p.t(2) = t_z;
        focals_p5pf[i] = f;
        poses_p5pf[i] = p;
    }


    int nSols_p5pf = 0;
    output_poses->clear();
    output_focals->clear();
    for (int i = 0; i < poses_p5pf.size(); i++) {
        if (std::isnan(focals_p5pf[i])) {continue;}
        if (is_valid_fisheye(points2d, points3d, poses_p5pf[i], focals_p5pf[i], 1e-2)){
            output_poses->emplace_back(poses_p5pf[i]);
            output_focals->emplace_back(focals_p5pf[i]);
            nSols_p5pf++;
        }
    }
    return nSols_p5pf;
}


int p5pf_fisheye_lm(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals) {
            
    std::vector<CameraPose> poses_p5pf;
    std::vector<double> focals_p5pf;
    p5pf(points2d, points3d, &poses_p5pf, &focals_p5pf);

    // LM refine
    int nSols_LM = 0;
    for (int i = 0; i < poses_p5pf.size(); i++) {
        if (is_valid_fisheye(points2d, points3d, poses_p5pf[i], focals_p5pf[i], 1e-2)){

            CameraPose pose_initial = poses_p5pf[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p5pf[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            BundleOptions bundle_opt;
            bundle_opt.refine_focal_length = true;
            std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

            AbsolutePoseRefiner<> refiner(points2d, points3d, camera_refine_idx);
            lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

            output_poses->emplace_back(Img_initial.pose);
            output_focals->emplace_back(Img_initial.camera.params[0]);
            nSols_LM++;
        }
        else {
            continue;
        }
    }

    return nSols_LM;
}

double getDerivative(double r_i, double f){

    double theta = r_i / f;
    double sec_theta = 1.0 / std::cos(theta);
    return - sec_theta * sec_theta * theta / f;
}

int p5pf_fisheye2(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital) {
    
    std::vector<CameraPose> poses_radial;
    p5lp_radial(points2d, points3d, &poses_radial);

    Eigen::Matrix3d AtA;
    Eigen::Vector3d Atb;
    Eigen::Matrix<double, 5, 3> A;
    Eigen::Vector<double, 5> b_;
    
    output_poses->clear();
    output_focals->clear();
    for (size_t i = 0; i < poses_radial.size(); ++i) {
        CameraPose p = poses_radial[i];
        
        AtA.setZero();
        Atb.setZero();
        for (int k = 0; k < 5; ++k) {
            Eigen::Vector3d RX = p.rotate(points3d[k]);

            // (c_i + t_z) * tan(theta_i) = r_i * d_i
            // taylor expansion around f_inital
            // h(f) = h(f_inital) + h'(f_inital) * (h - h_inital)
            
            double a = (RX(0) + p.t(0)) / points2d[k][0];
            double b = (RX(1) + p.t(1)) / points2d[k][1];
            double c = RX(2);
            double d = std::min(std::abs(a), std::abs(b));
            double r = points2d[k].norm();
            double theta0 = r / f_inital;

            double dh0 = getDerivative(theta0, f_inital);
            double e = - dh0 * f_inital;
            double A_i = std::tan(theta0) + e;
            double B_i = dh0;
            double C_i = c * dh0;
            double D_i = d * r - c * e - c * std::tan(theta0);
            
            A(k,0) = A_i;
            A(k,1) = B_i;
            A(k,2) = C_i;
            b_(k)   = D_i;

        }

        AtA += A.transpose() * A;
        Atb += A.transpose() * b_;

        //solve for focal length and t_z
        // Eigen::Vector3d sol = AtA.inverse() * Atb;
        Eigen::Vector3d sol = AtA.ldlt().solve(Atb);
        double focal = sol(2);
        double t_z = sol(0);
        p.t(2) = t_z;

        // std::cout << "focal: " << focal << std::endl;
        // std::cout << "t_z: " << t_z << std::endl;
        // std::cout << "f*t_z: " << focal * t_z << std::endl;
        // std::cout << "sol(1): " << sol(1) << std::endl;
        
        if (focal < 0) {
            focal = -focal;
            Eigen::Matrix3d R = p.R();
            R.row(0) = -R.row(0);
            R.row(1) = -R.row(1);
            p.q = rotmat_to_quat(R);
            p.t(0) = -p.t(0);
            p.t(1) = -p.t(1);
        }

        output_poses->emplace_back(p);
        output_focals->emplace_back(focal);

    }

    return output_poses->size();
}

} // namespace poselib