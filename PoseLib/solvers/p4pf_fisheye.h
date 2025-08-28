#ifndef P4PF_FISHEYE_H
#define P4PF_FISHEYE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/HCsolvers/homotopy_continuation.h"
#include "PoseLib/HCsolvers/HCproblems/absolute_fisheye.h"

namespace poselib {
    // solve for fisheye camera pose and focal length f such that: lambda*[tan(theta) x/rd; 1] = R*X+t

    int p4pf_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                     const Image &Img_initial, CameraPose *solutions, double *focals);

    int p4pf_fisheye_lie(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                     const Image &Img_initial, CameraPose *solutions, double *focals);

}

#endif