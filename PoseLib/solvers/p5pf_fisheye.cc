#include "p5pf_fisheye.h"
#include "p4pf_fisheye.h"

#include "p5lp_radial.h"

#include <Eigen/Dense>

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

    std::vector<CameraPose> poses_radial;
    p5lp_radial(points2d, points3d, &poses_radial);

    Eigen::Matrix2d AtA;
    Eigen::Vector2d Atb;

    std::vector<CameraPose> poses_p5pf;
    std::vector<double> focals_p5pf;

    for (size_t i = 0; i < poses_radial.size(); ++i) {
        AtA.setZero();
        Atb.setZero();

        CameraPose p = poses_radial[i];
        for (int k = 0; k < 5; ++k) {
            Eigen::Vector3d RX = p.rotate(points3d[k]);
            double a = (RX(0) + p.t(0)) / points2d[k][0];
            double b = (RX(1) + p.t(1)) / points2d[k][1];
            double c = RX(2);
            double d = (a + b)/2;

            AtA(0, 0) += d * d;
            AtA(0, 1) += -d;
            AtA(1, 0) += -d;
            AtA(1, 1) += 1;
            Atb(0) += d * c;
            Atb(1) += -c;
        }
        Eigen::Vector2d sol = AtA.inverse() * Atb;
        double focal = sol(0);
        p.t(2) = sol(1);

        // Correct sign
        if (focal < 0) {
            focal = -focal;
            Eigen::Matrix3d R = p.R();
            R.row(0) = -R.row(0);
            R.row(1) = -R.row(1);
            p.q = rotmat_to_quat(R);
            p.t(0) = -p.t(0);
            p.t(1) = -p.t(1);
        }
        poses_p5pf.emplace_back(p);
        focals_p5pf.emplace_back(focal);
    }
    // return output_poses->size();

    int nSols_p5pf = 0;
    output_poses->clear();
    output_focals->clear();
    for (int i = 0; i < poses_p5pf.size(); i++) {
        if (is_valid_fisheye(points2d, points3d, poses_p5pf[i], focals_p5pf[i], 1e-2)){
            output_poses->emplace_back(poses_p5pf[i]);
            output_focals->emplace_back(focals_p5pf[i]);
            nSols_p5pf++;
        }
    }
    return nSols_p5pf;
}

} // namespace poselib