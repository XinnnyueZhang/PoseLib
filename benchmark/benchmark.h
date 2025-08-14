#pragma once

#include "PoseLib/poselib.h"
#include "problem_generator.h"
#include <PoseLib/robust/optim/absolute.h>
#include <PoseLib/robust/optim/lm_impl.h>

#include <Eigen/Dense>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>
#include <random>

namespace poselib {

std::default_random_engine random_engine;
std::uniform_real_distribution<double> focal_perturbation_gen(-10.0, 10.0);
std::uniform_real_distribution<double> focal_gen(100.0, 1000.0);

struct BenchmarkResult {
    std::string name_;
    ProblemOptions options_;
    int instances_ = 0;
    int solutions_ = 0;
    int valid_solutions_ = 0;
    int found_gt_pose_ = 0;
    int64_t runtime_ns_ = 0;
};

// Wrappers for the Benchmarking code

struct SolverP3P {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p3p(instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p3p"; }
};

struct SolverP3P_ding {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p3p_ding(instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p3p_ding"; }
};

struct SolverP4PF {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_[i].hnormalized();
        }
        return p4pf(p2d, instance.X_point_, solutions, focals, true);
    }
    typedef UnknownFocalValidator validator;
    static std::string name() { return "p4pf"; }
};


struct SolverP35PF {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_[i].hnormalized();
        }
        return p35pf(p2d, instance.X_point_, solutions, focals, true);
    }
    typedef UnknownFocalValidator validator;
    static std::string name() { return "p3.5pf"; }
};

// NEW migrated from radialpose p4pfr, for fisheye camera resectioning
// 1D division radial distortion model fx = fy = f
struct SolverP4PFr {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals, std::vector<double> *ks) {
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_[i].hnormalized();
        }
        return p4pfr(p2d, instance.X_point_, solutions, focals, ks);
    }
    typedef UnknownFocalRadialValidator validator;
    static std::string name() { return "p4pfr"; }
};


struct SolverP5PF {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        std::vector<Eigen::Vector2d> p2d(5);
        for (int i = 0; i < 5; ++i) {
            p2d[i] = instance.x_point_[i].hnormalized();
        }
        return p5pf(p2d, instance.X_point_, solutions, focals, true);
    }
    typedef UnknownFocalValidator validator;
    static std::string name() { return "p5pf"; }
};

struct SolverGP3P {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gp3p(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "gp3p"; }
};

struct SolverGP4PS {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *scales) {
        return gp4ps(instance.p_point_, instance.x_point_, instance.X_point_, solutions, scales);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "gp4ps"; }
};

struct SolverP2P2PL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p2p2pl(instance.x_point_, instance.X_point_, instance.x_line_, instance.X_line_, instance.V_line_,
                      solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p2p2pl"; }
};

struct SolverP6LP {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p6lp(instance.l_line_point_, instance.X_line_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p6lp"; }
};
struct SolverP5LP_Radial {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p5lp_radial(instance.l_line_point_, instance.X_line_point_, solutions);
    }
    typedef RadialPoseValidator validator;
    static std::string name() { return "p5lp_radial"; }
};

struct SolverP2P1LL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p2p1ll(instance.x_point_, instance.X_point_, instance.l_line_line_, instance.X_line_line_,
                      instance.V_line_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p2p1ll"; }
};

struct SolverP1P2LL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p1p2ll(instance.x_point_, instance.X_point_, instance.l_line_line_, instance.X_line_line_,
                      instance.V_line_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p1p2ll"; }
};

struct SolverP3LL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p3ll(instance.l_line_line_, instance.X_line_line_, instance.V_line_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p3ll"; }
};

struct SolverUP2P {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return up2p(instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "up2p"; }
};

struct SolverUP1P1LL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return up1p1ll(instance.x_point_[0], instance.X_point_[0], instance.l_line_line_[0], instance.X_line_line_[0],
                       instance.V_line_line_[0], solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "up1p1ll"; }
};

struct SolverUGP2P {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return ugp2p(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "ugp2p"; }
};

struct SolverUGP3PS {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *scales) {
        return ugp3ps(instance.p_point_, instance.x_point_, instance.X_point_, solutions, scales);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "ugp3ps"; }
};

struct SolverUP1P2PL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return up1p2pl(instance.x_point_, instance.X_point_, instance.x_line_, instance.X_line_, instance.V_line_,
                       solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "up1p2pl"; }
};

struct SolverUP4PL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return up4pl(instance.x_line_, instance.X_line_, instance.V_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "up4pl"; }
};

struct SolverUGP4PL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return ugp4pl(instance.p_line_, instance.x_line_, instance.X_line_, instance.V_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "ugp4pl"; }
};

struct SolverRelUpright3pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_upright_3pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "RelUpright3pt"; }
};

struct SolverGenRelUpright4pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_upright_4pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "GenRelUpright4pt"; }
};

struct SolverRel8pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_8pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "Rel8pt"; }
};

struct SolverRelRD10pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<ProjectiveImagePair> *solutions) {
        return relpose_k2Fk1_10pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef ProjectiveImagePair Solution;
    static std::string name() { return "RelRD10pt"; }
};

struct SolverRelRD9pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<ProjectiveImagePair> *solutions) {
        return relpose_kFk_9pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef ProjectiveImagePair Solution;
    static std::string name() { return "RelSharedRD9pt"; }
};

struct SolverSharedFocalRel6pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::ImagePairVector *solutions) {
        return relpose_6pt_shared_focal(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef ImagePair Solution;
    static std::string name() { return "SharedFocalRel6pt"; }
};

struct SolverRel5pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_5pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "Rel5pt"; }
};

struct SolverGenRel5p1pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_5p1pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "GenRel5p1pt"; }
};

struct SolverGenRel6pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_6pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "GenRel6pt"; }
};

struct SolverRelUprightPlanar2pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_upright_planar_2pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "RelUprightPlanar2pt"; }
};

struct SolverRelUprightPlanar3pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_upright_planar_3pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "RelUprightPlanar3pt"; }
};

template <bool CheiralCheck = false> struct SolverHomography4pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<Eigen::Matrix3d> *solutions) {
        Eigen::Matrix3d H;
        int sols = homography_4pt(instance.x1_, instance.x2_, &H, CheiralCheck);
        solutions->clear();
        if (sols == 1) {
            solutions->push_back(H);
        }
        return sols;
    }
    typedef HomographyValidator validator;
    static std::string name() {
        if (CheiralCheck) {
            return "Homography4pt(C)";
        } else {
            return "Homography4pt";
        }
    }
};

// Below are the solvers for fisheye camera resectioning with unknown focal

// NEW p4pfr used for fisheye camera resectioning
struct SolverFisheye_P4PFr {
    // difference from p4pfr is that do not store distortion parameters and fisheye validator
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        std::vector<double> ks;
        return p4pfr(p2d, instance.X_point_, solutions, focals, &ks);
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pfr"; }
};

struct SolverFisheye_P4PFr_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // normalize input
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        CameraPoseVector solutions_p4pfr;
        std::vector<double> focals_p4pfr;
        std::vector<double> ks;
        int nSols_p4pfr = p4pfr(p2d, instance.X_point_, &solutions_p4pfr, &focals_p4pfr, &ks);

        if (nSols_p4pfr == 0) {
            return 0;
        }

        // LM refine
        int nSols_LM = 0;
        for (int i = 0; i < nSols_p4pfr; i++) {
            CameraPose pose_initial = solutions_p4pfr[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            AbsolutePoseRefiner<> refiner(p2d, instance.X_point_);

            BundleOptions bundle_opt;
            bundle_opt.step_tol = 1e-12;
            lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

            solutions->push_back(Img_initial.pose);
            focals->push_back(Img_initial.camera.params[0]);
            nSols_LM++;
        }
        return nSols_LM;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pfr_LM"; }
};


struct SolverFisheye_random_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        CameraPose pose_initial;
        set_random_pose(pose_initial, false, false);

        double focal_initial = focal_gen(random_engine); // uniformly distributed between 100 and 1000

        Camera camera_initial;
        camera_initial.model_id = 12;
        camera_initial.params = {focal_initial, 0.0, 0.0};
        Image Img_initial(pose_initial, camera_initial);

        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        

        // LM refine
        AbsolutePoseRefiner<> refiner(x_fisheye, instance.X_point_);

        BundleOptions bundle_opt;
        bundle_opt.step_tol = 1e-12;
        lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

        solutions->push_back(Img_initial.pose);
        focals->push_back(Img_initial.camera.params[0]);
        return 1;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_random_LM"; }
};

// NEW HC solvers for Fisheye camera resectioning with unknown focal
struct SolverFisheye_HC_pose_gtDebug {
    // polynomial with poses as unknowns (8 unknowns)
    // debuging with gt pose
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // CameraPose pose_initial = instance.pose_gt;
        // Camera camera_initial;
        // camera_initial.model_id = 12;
        // camera_initial.params = {(instance.focal_gt - 50), 0.0, 0.0};

        // Image Img_initial(pose_initial, camera_initial);


        Eigen::Matrix3d R_gt = instance.pose_gt.R();
        Eigen::Vector3d t_gt = instance.pose_gt.t;
        Eigen::Matrix3d R_initial = R_gt * Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t_initial = t_gt + Eigen::Vector3d::Random() * 0.1;
        double focal_perturbation = focal_perturbation_gen(random_engine)*2; // uniformly distributed between -20 and 20

        CameraPose pose_initial;
        pose_initial.q = rotmat_to_quat(R_initial);
        pose_initial.t = t_initial;

        Camera camera_initial;
        camera_initial.model_id = 12;
        camera_initial.params = {(instance.focal_gt + focal_perturbation), 0.0, 0.0};
        Image Img_initial(pose_initial, camera_initial);

        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPose solution_;
        double focal_;
        p4pf_fisheye(x_fisheye, instance.X_point_, Img_initial, &solution_, &focal_);
        solutions->push_back(solution_);
        focals->push_back(focal_);

        return 1;
    }
    // static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
    //                         std::vector<double> *focals) {
    //     p4pf_fisheye(instance.x_point_fisheye_, instance.X_point_, instance.Img_initial_, solutions, focals);
    //     return 1;
    // }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_hc_pose_gtDebug"; }
};


struct SolverFisheye_HC_pose_p4pfr {
    // polynomial with poses as unknowns (8 unknowns)

    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // use p4pfr to get the initial pose and focal length
        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPoseVector solutions_p4pfr;
        std::vector<double> focals_p4pfr;
        std::vector<double> ks;
        int nSols_p4pfr = p4pfr(x_fisheye, instance.X_point_, &solutions_p4pfr, &focals_p4pfr, &ks);

        if (nSols_p4pfr == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p4pfr; i++) {
            CameraPose pose_initial = solutions_p4pfr[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            CameraPose solution_HC;
            double focal_HC;
            int HC_success = p4pf_fisheye(x_fisheye, instance.X_point_, Img_initial, &solution_HC, &focal_HC);

            if (HC_success == 1) {
                solutions->push_back(solution_HC);
                focals->push_back(focal_HC);
                nSols_HC++;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_hc_pose_p4pfr"; }
};

struct SolverFisheye_HC_depth_gtDebug {
    // polynomial with depths as unknowns (5 unknowns)
    // Randomly perturb the initial pose and focal length

    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        Eigen::Matrix3d R_gt = instance.pose_gt.R();
        Eigen::Vector3d t_gt = instance.pose_gt.t;
        Eigen::Matrix3d R_initial = R_gt * Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t_initial = t_gt + Eigen::Vector3d::Random() * 0.1;
        double focal_perturbation = focal_perturbation_gen(random_engine)*2; // uniformly distributed between -20 and 20

        CameraPose pose_initial;
        pose_initial.q = rotmat_to_quat(R_initial);
        pose_initial.t = t_initial;
        
        Camera camera_initial;
        camera_initial.model_id = 12;
        camera_initial.params = {(instance.focal_gt + focal_perturbation), 0.0, 0.0};
        Image Img_initial(pose_initial, camera_initial);

        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        
        CameraPose solution_;
        double focal_;
        p4pf_fisheye_depth(x_fisheye, instance.X_point_, Img_initial, &solution_, &focal_);
        solutions->push_back(solution_);
        focals->push_back(focal_);
        
        return 1;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_hc_depth_gtDebug"; }
};

struct SolverFisheye_HC_depth_random {
    // polynomial with depths as unknowns (5 unknowns)

    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        
        CameraPose pose_initial;
        set_random_pose(pose_initial, false, false);

        double focal_initial = focal_gen(random_engine); // uniformly distributed between 100 and 1000

        Camera camera_initial;
        camera_initial.model_id = 12;
        camera_initial.params = {focal_initial, 0.0, 0.0};
        Image Img_initial(pose_initial, camera_initial);

        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        
        CameraPose solution_;
        double focal_;
        p4pf_fisheye_depth(x_fisheye, instance.X_point_, Img_initial, &solution_, &focal_);
        solutions->push_back(solution_);
        focals->push_back(focal_);
        
        return 1;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_hc_depth_random"; }
};

struct SolverFisheye_HC_depth_p4pfr {
    // polynomial with depths as unknowns (5 unknowns)

    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // use p4pfr to get the initial pose and focal length
        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPoseVector solutions_p4pfr;
        std::vector<double> focals_p4pfr;
        std::vector<double> ks;
        int nSols_p4pfr = p4pfr(x_fisheye, instance.X_point_, &solutions_p4pfr, &focals_p4pfr, &ks);

        if (nSols_p4pfr == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p4pfr; i++) {
            CameraPose pose_initial = solutions_p4pfr[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            CameraPose solution_HC;
            double focal_HC;
            int HC_success = p4pf_fisheye_depth(x_fisheye, instance.X_point_, Img_initial, &solution_HC, &focal_HC);

            if (HC_success == 1) {
                solutions->push_back(solution_HC);
                focals->push_back(focal_HC);
                nSols_HC++;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_hc_depth_p4pfr"; }
};


struct SolverFisheye_HC_depth_p4pfr_LM {
    // polynomial with depths as unknowns (5 unknowns)

    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // use p4pfr to get the initial pose and focal length
        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPoseVector solutions_p4pfr;
        std::vector<double> focals_p4pfr;
        std::vector<double> ks;
        int nSols_p4pfr = p4pfr(x_fisheye, instance.X_point_, &solutions_p4pfr, &focals_p4pfr, &ks);

        if (nSols_p4pfr == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p4pfr; i++) {
            CameraPose pose_initial = solutions_p4pfr[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            CameraPose solution_HC;
            double focal_HC;
            int HC_success = p4pf_fisheye_depth(x_fisheye, instance.X_point_, Img_initial, &solution_HC, &focal_HC);

            if (HC_success == 1) {

                // LM refine (for numerical stability)
                CameraPose pose_LM_initial = solution_HC;
                Camera camera_LM_initial;
                camera_LM_initial.model_id = 12;
                camera_LM_initial.params = {focal_HC, 0.0, 0.0};
                Image Img_LM_initial(pose_LM_initial, camera_LM_initial);

                AbsolutePoseRefiner<> refiner(x_fisheye, instance.X_point_);
                BundleOptions bundle_opt;
                bundle_opt.step_tol = 1e-12;
                lm_impl<decltype(refiner)>(refiner, &Img_LM_initial, bundle_opt);

                solutions->push_back(Img_LM_initial.pose);
                focals->push_back(Img_LM_initial.camera.params[0]);
                nSols_HC++;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_hc_depth_p4pfr_LM"; }
};


// NEW using p4pf as initial
struct SolverFisheye_P35PF {
    // difference from p4pfr is that do not store distortion parameters and fisheye validator
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        
        return p35pf(p2d, instance.X_point_, solutions, focals, true);
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p35pf"; }
};

struct SolverFisheye_P35PF_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // normalize input
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        CameraPoseVector solutions_p35pf;
        std::vector<double> focals_p35pf;
        int nSols_p35pf = p35pf(p2d, instance.X_point_, &solutions_p35pf, &focals_p35pf, true);

        if (nSols_p35pf == 0) {
            return 0;
        }

        // LM refine
        int nSols_LM = 0;
        for (int i = 0; i < nSols_p35pf; i++) {
            CameraPose pose_initial = solutions_p35pf[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p35pf[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            AbsolutePoseRefiner<> refiner(p2d, instance.X_point_);

            BundleOptions bundle_opt;
            bundle_opt.step_tol = 1e-12;
            lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

            solutions->push_back(Img_initial.pose);
            focals->push_back(Img_initial.camera.params[0]);
            nSols_LM++;
        }
        return nSols_LM;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p35pf_LM"; }
};


struct SolverFisheye_HC_pose_p35pf {
    // polynomial with poses as unknowns (8 unknowns)
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // use p4pfr to get the initial pose and focal length
        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPoseVector solutions_p35pf;
        std::vector<double> focals_p35pf;
        int nSols_p35pf = p35pf(x_fisheye, instance.X_point_, &solutions_p35pf, &focals_p35pf, true);

        if (nSols_p35pf == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p35pf; i++) {
            CameraPose pose_initial = solutions_p35pf[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p35pf[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            CameraPose solution_HC;
            double focal_HC;
            int HC_success = p4pf_fisheye(x_fisheye, instance.X_point_, Img_initial, &solution_HC, &focal_HC);

            if (HC_success == 1) {
                solutions->push_back(solution_HC);
                focals->push_back(focal_HC);
                nSols_HC++;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_hc_pose_p35pf"; }
};


struct SolverFisheye_HC_depth_p35pf {
    // polynomial with depths as unknowns (5 unknowns)
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // use p4pfr to get the initial pose and focal length
        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPoseVector solutions_p35pf;
        std::vector<double> focals_p35pf;
        int nSols_p35pf = p35pf(x_fisheye, instance.X_point_, &solutions_p35pf, &focals_p35pf, true);

        if (nSols_p35pf == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p35pf; i++) {
            CameraPose pose_initial = solutions_p35pf[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p35pf[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            CameraPose solution_HC;
            double focal_HC;
            int HC_success = p4pf_fisheye_depth(x_fisheye, instance.X_point_, Img_initial, &solution_HC, &focal_HC);

            if (HC_success == 1) {
                solutions->push_back(solution_HC);
                focals->push_back(focal_HC);
                nSols_HC++;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_hc_depth_p35pf"; }
};


} // namespace poselib
