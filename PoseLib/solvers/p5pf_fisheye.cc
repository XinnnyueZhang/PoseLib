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

double getgDerivative(double r_i, double f){

    double theta = r_i / f;
    double sec_theta = 1.0 / std::cos(theta);
    return - sec_theta * sec_theta * theta / f;
}

double gethDerivative(double r_i, double f){

    double theta = r_i / f; 
    double g = 1.0 / std::tan(theta);
    return - g * g * getgDerivative(r_i, f);
}

int p5pf_fisheye2(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital) {
    
    std::vector<CameraPose> poses_radial;
    p5lp_radial(points2d, points3d, &poses_radial);

    Eigen::Matrix2d AtA;
    Eigen::Vector2d Atb;
    Eigen::Matrix<double, 5, 2> A;
    Eigen::Vector<double, 5> b_;
    
    output_poses->clear();
    output_focals->clear();
    for (size_t i = 0; i < poses_radial.size(); ++i) {
        CameraPose p = poses_radial[i];
        
        AtA.setZero();
        Atb.setZero();
        for (int k = 0; k < 5; ++k) {
            Eigen::Vector3d RX = p.rotate(points3d[k]);

            // (c_i + t_z) = r_i * d_i / tan(theta_i)
            double a = (RX(0) + p.t(0)) / points2d[k][0];
            double b = (RX(1) + p.t(1)) / points2d[k][1];
            double c = RX(2);
            double d = std::min(std::abs(a), std::abs(b));
            double r = points2d[k].norm();
            double theta0 = r / f_inital;

            double e = d * r;
            double dh0 = gethDerivative(r, f_inital);

            A(k,0) = -1;
            A(k,1) = dh0 * e;

            b_(k) = c - e*(1/std::tan(theta0)) + e*dh0*f_inital;

        }

        AtA += A.transpose() * A;
        Atb += A.transpose() * b_;

        //solve for focal length and t_z
        // Eigen::Vector3d sol = AtA.inverse() * Atb;
        Eigen::Vector2d sol = AtA.ldlt().solve(Atb);
        double focal = sol(1);
        double t_z = sol(0);
        p.t(2) = t_z;
        
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

int p5pf_fisheye2_valid(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital) {
    
    std::vector<CameraPose> poses_p5pf2;
    std::vector<double> focals_p5pf2;
    p5pf_fisheye2(points2d, points3d, &poses_p5pf2, &focals_p5pf2, f_inital);
    
    int nSols_p5pf = 0;
    output_poses->clear();
    output_focals->clear();
    for (int i = 0; i < poses_p5pf2.size(); i++) {
        if (std::isnan(focals_p5pf2[i])) {continue;}
        if (is_valid_fisheye(points2d, points3d, poses_p5pf2[i], focals_p5pf2[i], 1e-2)){
            output_poses->emplace_back(poses_p5pf2[i]);
            output_focals->emplace_back(focals_p5pf2[i]);
            nSols_p5pf++;
        }
    }

    return nSols_p5pf;
}

int p5pf_fisheye2_lm(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital) {
    
    std::vector<CameraPose> poses_p5pf2;
    std::vector<double> focals_p5pf2;
    p5pf_fisheye2(points2d, points3d, &poses_p5pf2, &focals_p5pf2, f_inital);

    // LM refine
    int nSols_LM = 0;
    for (int i = 0; i < poses_p5pf2.size(); i++) {
        if (is_valid_fisheye(points2d, points3d, poses_p5pf2[i], focals_p5pf2[i], 1e-2)){

            CameraPose pose_initial = poses_p5pf2[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p5pf2[i], 0.0, 0.0};
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


int p5pf_fisheye2_noValid_lm(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital) {
    
    std::vector<CameraPose> poses_p5pf2;
    std::vector<double> focals_p5pf2;
    p5pf_fisheye2(points2d, points3d, &poses_p5pf2, &focals_p5pf2, f_inital);

    // LM refine
    int nSols_LM = 0;
    for (int i = 0; i < poses_p5pf2.size(); i++) {
        CameraPose pose_initial = poses_p5pf2[i];
        Camera camera_initial;
        camera_initial.model_id = 12;
        camera_initial.params = {focals_p5pf2[i], 0.0, 0.0};
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

    return nSols_LM;
}

} // namespace poselib