/* C++ Port of the planar solver from Magnus Oskarsson.
   See https://github.com/hamburgerlady/fast_planar_camera_pose/blob/master/solver_planar_p4pfr_fast.m
   for the original Matlab implementation.
   Please cite 
    Oskarsson, A fast minimal solver for absolute camera pose with unknown focal length and radial distortion from four planar points, arxiv
   if you use this solver.   

   Note that this solver assumes that coordinate system is chosen such that X(3,:) = 0 !

    */
#ifndef OSKARSSON_ARXIV18_H
#define OSKARSSON_ARXIV18_H
#pragma once
// #include "pose_estimator.h"
// #include "../misc/distortion.h"
// #include "../misc/refinement.h"
#include "PoseLib/camera_pose.h"

namespace poselib {
    int p4pfr_planar(const std::vector<Eigen::Vector2d> &image_points, const std::vector<Eigen::Vector3d> &world_points, 
		std::vector<CameraPose> *output_poses, std::vector<double> *output_focals);
        
	int oskarsson_arxiv18(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d, 
		std::vector<CameraPose> *output_poses, std::vector<double> *output_focals);
};

#endif