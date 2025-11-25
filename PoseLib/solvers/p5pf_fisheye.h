
// Copyright (c) 2024, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_P5PF_FISHEYE_H_
#define POSELIB_P5PF_FISHEYE_H_

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Solves for camera pose and focal length f and dist param k such that: lambda*diag(1/f,1/f,1)*[x;1+k*f^2*|x|^2] = R*X+t
// Re-implementation of the p5pfr solver from
//    Kukelova et al., Real-time solution to the absolute pose problem with unknown radial distortion and focal length,
//    ICCV 2013
int p5pf_fisheye(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
         std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, bool normalize_input = false);

int p5pf_fisheye_lm(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals);

int p5pf_fisheye2(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital);
    
int p5pf_fisheye2_valid(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital);

int p5pf_fisheye2_lm(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital);

int p5pf_fisheye2_noValid_lm(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
    std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, double f_inital);

} // namespace poselib

#endif
