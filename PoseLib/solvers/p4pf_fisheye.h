#ifndef P4PF_FISHEYE_H
#define P4PF_FISHEYE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/HCsolvers/homotopy_continuation.h"
#include "PoseLib/HCsolvers/HCproblems/absolute_fisheye.h"
#include "PoseLib/solvers/p3p_ding.h"

namespace poselib {
     // solve for fisheye camera pose and focal length f such that: lambda*[tan(theta) x/rd; 1] = R*X+t

     // HC solvers for benchmark
     int p4pf_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                         const Image &Img_initial, CameraPose *solutions, double *focals);

     int p4pf_fisheye_lie(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                         const Image &Img_initial, CameraPose *solutions, double *focals);

     // solvers for RANSAC
     bool is_valid_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
          const CameraPose &pose, double focal, double tol);

     int p4pfr_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                         CameraPoseVector *solutions, std::vector<double> *focals);

     int p4pfr_lm_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                         CameraPoseVector *solutions, std::vector<double> *focals);

     int p4pfr_hc_pose_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                         CameraPoseVector *solutions, std::vector<double> *focals);

     int p4pfr_hc_depth_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                         CameraPoseVector *solutions, std::vector<double> *focals);

     // sampling p3p still use 4 points (sampling fov looks converge faster than sampling focal particularly for large fov scene)
     int p3p_fisheye_lm(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                        const double image_size, CameraPoseVector *solutions, std::vector<double> *focals);

     int p3p_fisheye_hc(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                        const double image_size, CameraPoseVector *solutions, std::vector<double> *focals);

     // NEW: p5pfr
     int p5pfr_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                        CameraPoseVector *solutions, std::vector<double> *focals);

     int p5pfr_lm_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                        CameraPoseVector *solutions, std::vector<double> *focals);

     // with gt or predicted focal length as input
     int p3p_fisheye_givenf(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, double focal_initial,
                     CameraPoseVector *solutions, std::vector<double> *focals);

}

#endif