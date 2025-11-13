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
static const double kPI = 3.14159265358979323846;

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

struct SolverFisheye_P5PFr {
    // difference from p5pfr is that do not store distortion parameters and fisheye validator
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        std::vector<Eigen::Vector2d> p2d(5);
        for (int i = 0; i < 5; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        std::vector<double> ks;
        return p5pfr(p2d, instance.X_point_, solutions, focals, &ks);
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p5pfr"; }
};

struct SolverFisheye_P4PFr_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // dehomogenize input
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
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p4pfr[i], focals_p4pfr[i], 1e-2)){

                CameraPose pose_initial = solutions_p4pfr[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
                Image Img_initial(pose_initial, camera_initial);

                BundleOptions bundle_opt;
                // bundle_opt.step_tol = 1e-12;
                bundle_opt.refine_focal_length = true;
                std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

                AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
                lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

                solutions->push_back(Img_initial.pose);
                focals->push_back(Img_initial.camera.params[0]);
                nSols_LM++;
            }
            else {
                continue;
            }
        }
        return nSols_LM;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pfr_LM"; }
};

struct SolverFisheye_P5PFr_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(5);
        for (int i = 0; i < 5; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        CameraPoseVector solutions_p5pfr;
        std::vector<double> focals_p5pfr;
        std::vector<double> ks;
        int nSols_p5pfr = p5pfr(p2d, instance.X_point_, &solutions_p5pfr, &focals_p5pfr, &ks);

        if (nSols_p5pfr == 0) {
            return 0;
        }

        // LM refine
        int nSols_LM = 0;
        for (int i = 0; i < nSols_p5pfr; i++) {
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p5pfr[i], focals_p5pfr[i], 1e-2)){

                CameraPose pose_initial = solutions_p5pfr[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p5pfr[i], 0.0, 0.0};
                Image Img_initial(pose_initial, camera_initial);

                BundleOptions bundle_opt;
                // bundle_opt.step_tol = 1e-12;
                bundle_opt.refine_focal_length = true;
                std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

                AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
                lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

                solutions->push_back(Img_initial.pose);
                focals->push_back(Img_initial.camera.params[0]);
                nSols_LM++;
            }
            else {
                continue;
            }
        }
        return nSols_LM;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p5pfr_LM"; }
};


struct SolverFisheye_P4PFr_HC_pose {
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
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p4pfr[i], focals_p4pfr[i], 1e-2)){

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
            else {
                continue;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pfr_hc_pose"; }
};


struct SolverFisheye_P4PFr_HC_pose_Lie {
    // polynomial with poses as unknowns (7 unknowns)

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
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p4pfr[i], focals_p4pfr[i], 1e-2)){

                CameraPose pose_initial = solutions_p4pfr[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
                Image Img_initial(pose_initial, camera_initial);

                CameraPose solution_HC;
                double focal_HC;
                int HC_success = p4pf_fisheye_lie(x_fisheye, instance.X_point_, Img_initial, &solution_HC, &focal_HC);

                if (HC_success == 1) {
                    solutions->push_back(solution_HC);
                    focals->push_back(focal_HC);
                    nSols_HC++;
                }
            }
            else {
                continue;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pfr_hc_Liepose"; }
};


struct SolverFisheye_P5PFr_HC_pose {
    // polynomial with poses as unknowns (7 unknowns)

    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // use p5pfr to get the initial pose and focal length
        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPoseVector solutions_p5pfr;
        std::vector<double> focals_p5pfr;
        std::vector<double> ks;
        int nSols_p5pfr = p5pfr(x_fisheye, instance.X_point_, &solutions_p5pfr, &focals_p5pfr, &ks);

        if (nSols_p5pfr == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p5pfr; i++) {
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p5pfr[i], focals_p5pfr[i], 1e-2)){

                CameraPose pose_initial = solutions_p5pfr[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p5pfr[i], 0.0, 0.0};
                Image Img_initial(pose_initial, camera_initial);

                CameraPose solution_HC;
                double focal_HC;
                // bad
                int HC_success = p4pf_fisheye_lie(x_fisheye, instance.X_point_, Img_initial, &solution_HC, &focal_HC);

                if (HC_success == 1) {
                    solutions->push_back(solution_HC);
                    focals->push_back(focal_HC);
                    nSols_HC++;
                }
            }
            else {
                continue;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p5pfr_hc_Liepose"; }
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

struct SolverFisheye_random_HC_depth {
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
    static std::string name() { return "fisheye_random_hc_depth"; }
};

struct SolverFisheye_P4PFr_HC_depth {
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
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p4pfr[i], focals_p4pfr[i], 1e-2)){

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
            else {
                continue;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pfr_hc_depth"; }
};


struct SolverFisheye_P4PFr_HC_depth_LM {
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

                BundleOptions bundle_opt;
                // bundle_opt.step_tol = 1e-12;
                bundle_opt.refine_focal_length = true;
                std::vector<size_t> camera_refine_idx = Img_LM_initial.camera.get_param_refinement_idx(bundle_opt);
    
                AbsolutePoseRefiner<> refiner(x_fisheye, instance.X_point_, camera_refine_idx);
                lm_impl<decltype(refiner)>(refiner, &Img_LM_initial, bundle_opt);

                solutions->push_back(Img_LM_initial.pose);
                focals->push_back(Img_LM_initial.camera.params[0]);
                nSols_HC++;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pfr_hc_depth_LM"; }
};


// NEW using p3.5pf as initial
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
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p35pf[i], focals_p35pf[i], 1e-2)){

                CameraPose pose_initial = solutions_p35pf[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p35pf[i], 0.0, 0.0};
                Image Img_initial(pose_initial, camera_initial);

                BundleOptions bundle_opt;
                // bundle_opt.step_tol = 1e-12;
                bundle_opt.refine_focal_length = true;
                std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

                AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
                lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

                solutions->push_back(Img_initial.pose);
                    focals->push_back(Img_initial.camera.params[0]);
                    nSols_LM++;
            }
            else {
                continue;
            }
        }
        return nSols_LM;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p35pf_LM"; }
};


struct SolverFisheye_P35PF_HC_pose {
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
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p35pf[i], focals_p35pf[i], 1e-2)){

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
            else {
                continue;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p35pf_hc_pose"; }
};


struct SolverFisheye_P35PF_HC_depth {
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
            if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p35pf[i], focals_p35pf[i], 1e-2)){

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
            else {
                continue;
            }

        }

        return nSols_HC;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p35pf_hc_depth"; }
};


// NEW using p4pf as initial
struct SolverFisheye_P4PF {
    // difference from p4pfr is that do not store distortion parameters and fisheye validator
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        
        return p4pf(p2d, instance.X_point_, solutions, focals, true);
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pf"; }
};

// struct SolverFisheye_P4PF_LMf {
//     static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
//                             std::vector<double> *focals) {

//         // normalize input
//         std::vector<Eigen::Vector2d> p2d(4);
//         for (int i = 0; i < 4; ++i) {
//             p2d[i] = instance.x_point_fisheye_[i].hnormalized();
//         }

//         CameraPoseVector solutions_p4pf;
//         std::vector<double> focals_p4pf;
//         int nSols_p4pf = p4pf(p2d, instance.X_point_, &solutions_p4pf, &focals_p4pf, true);

//         if (nSols_p4pf == 0) {
//             return 0;
//         }

//         // LM refine
//         int nSols_LM = 0;
//         for (int i = 0; i < nSols_p4pf; i++) {
//             CameraPose pose_initial = solutions_p4pf[i];
//             Camera camera_initial;
//             camera_initial.model_id = 12;
//             camera_initial.params = {focals_p4pf[i], 0.0, 0.0};
//             Image Img_initial(pose_initial, camera_initial);

//             AbsolutePoseRefiner<> refiner(p2d, instance.X_point_);

//             BundleOptions bundle_opt;
//             bundle_opt.step_tol = 1e-12;
//             lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

//             solutions->push_back(Img_initial.pose);
//             focals->push_back(Img_initial.camera.params[0]);
//             nSols_LM++;
//         }
//         return nSols_LM;
//     }
//     typedef UnknownFocalFisheyeValidator validator;
//     static std::string name() { return "fisheye_p4pf_LM"; }
// };


struct SolverFisheye_P4PF_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // normalize input
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        CameraPoseVector solutions_p4pf;
        std::vector<double> focals_p4pf;
        int nSols_p4pf = p4pf(p2d, instance.X_point_, &solutions_p4pf, &focals_p4pf, true);

        if (nSols_p4pf == 0) {
            return 0;
        }

        // LM refine
        int nSols_LM = 0;
        for (int i = 0; i < nSols_p4pf; i++) {
            CameraPose pose_initial = solutions_p4pf[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p4pf[i], 0.0, 0.0};
            Image Img_initial(pose_initial, camera_initial);

            BundleOptions bundle_opt;
            // bundle_opt.step_tol = 1e-12;
            bundle_opt.refine_focal_length = true;
            std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

            AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
            lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

            solutions->push_back(Img_initial.pose);
            focals->push_back(Img_initial.camera.params[0]);
            nSols_LM++;
        }
        return nSols_LM;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p4pf_LM"; }
};


struct SolverFisheye_P4PF_HC_pose {
    // polynomial with poses as unknowns (8 unknowns)
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // use p4pfr to get the initial pose and focal length
        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPoseVector solutions_p4pf;
        std::vector<double> focals_p4pf;
        int nSols_p4pf = p4pf(x_fisheye, instance.X_point_, &solutions_p4pf, &focals_p4pf, true);

        if (nSols_p4pf == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p4pf; i++) {
            CameraPose pose_initial = solutions_p4pf[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p4pf[i], 0.0, 0.0};
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
    static std::string name() { return "fisheye_p4pf_hc_pose"; }
};


struct SolverFisheye_P4PF_HC_depth {
    // polynomial with depths as unknowns (5 unknowns)
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {

        // use p4pfr to get the initial pose and focal length
        std::vector<Eigen::Vector2d> x_fisheye(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
            x_fisheye[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        CameraPoseVector solutions_p4pf;
        std::vector<double> focals_p4pf;
        int nSols_p4pf = p4pf(x_fisheye, instance.X_point_, &solutions_p4pf, &focals_p4pf, true);

        if (nSols_p4pf == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p4pf; i++) {
            CameraPose pose_initial = solutions_p4pf[i];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focals_p4pf[i], 0.0, 0.0};
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
    static std::string name() { return "fisheye_p4pf_hc_depth"; }
};


// sampling + p3p
// no refiner
struct SolverFisheye_P3P {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {

        int nSols = 0;

        // std::vector<double> fov_list = {30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
        std::vector<double> focal_list = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        for (double focal : focal_list) {
            Camera camera;
            // camera.model_id = 12;
            // camera.params = {focal, 0.0, 0.0};
            camera.model_id = 5;
            camera.params = {focal, focal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(instance.x_point_fisheye_.size());
            for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
                camera.unproject(instance.x_point_fisheye_[i].hnormalized(), &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, instance.X_point_, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {
                solutions->push_back(solutions_p3p[j]);
                focals->push_back(focal);
                nSols++;
            }
        }

        return nSols;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p3p_sampling"; }
};


struct SolverFisheye_P3P_focal_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {

        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        int nSols = 0;
        double min_reproj_error = std::numeric_limits<double>::max();
        double focal_best = 0.0;
        CameraPose pose_best;

        std::vector<double> focal_list = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        for (double focal : focal_list) {
            Camera camera;
            camera.model_id = 12;
            camera.params = {focal, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
            for (int i = 0; i < 3; i++) {
                camera.unproject(p2d[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, instance.X_point_, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {

                // check reprojection with the 4th point
                Eigen::Vector2d reprojected;
                Eigen::Vector3d x_ = solutions_p3p[j].R() * instance.X_point_[3] + solutions_p3p[j].t;
                camera.project(x_, &reprojected);
                double res = (reprojected - p2d[3]).norm();

                if (res < min_reproj_error) {
                    min_reproj_error = res;
                    focal_best = focal;
                    pose_best = solutions_p3p[j];
                }
            }
        }


        CameraPose pose_initial = pose_best;
        Camera camera_initial;
        camera_initial.model_id = 12;
        camera_initial.params = {focal_best, 0.0, 0.0};
        Image Img_initial(pose_initial, camera_initial);

        BundleOptions bundle_opt;
        // bundle_opt.step_tol = 1e-12;
        bundle_opt.refine_focal_length = true;
        std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

        AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
        lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

        solutions->push_back(Img_initial.pose);
        focals->push_back(Img_initial.camera.params[0]);
        nSols++;

        return nSols;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p3p_sampling_focal_LM"; }
};


struct SolverFisheye_P3P_fov_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {

        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        int nSols = 0;
        double min_reproj_error = std::numeric_limits<double>::max();
        double focal_best = 0.0;
        CameraPose pose_best;

        std::vector<double> fov_list = {100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 210, 220};
        for (double fov : fov_list) {
            // double focal = instance.focal_gt * std::tan(instance.camera_fov_ / 2.0 * kPI / 180.0) / std::tan(fov / 2.0 * kPI / 180.0);
            double image_size = instance.focal_gt * std::tan(instance.camera_fov_ / 2.0 * kPI / 180.0);
            double focal = image_size / (fov * M_PI / 180.0);

            Camera camera;
            camera.model_id = 12;
            camera.params = {focal, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
            for (int i = 0; i < 3; i++) {
                camera.unproject(p2d[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, instance.X_point_, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {

                // check reprojection with the 4th point

                Eigen::Vector2d reprojected;
                Eigen::Vector3d x_ = solutions_p3p[j].R() * instance.X_point_[3] + solutions_p3p[j].t;
                camera.project(x_, &reprojected);
                double res = (reprojected - p2d[3]).norm();

                if (res < min_reproj_error) {
                    min_reproj_error = res;
                    focal_best = focal;
                    pose_best = solutions_p3p[j];
                }
            }
        }


        CameraPose pose_initial = pose_best;
        Camera camera_initial;
        camera_initial.model_id = 12;
        camera_initial.params = {focal_best, 0.0, 0.0};
        Image Img_initial(pose_initial, camera_initial);

        BundleOptions bundle_opt;
        // bundle_opt.step_tol = 1e-12;
        bundle_opt.refine_focal_length = true;
        std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

        AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
        lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

        solutions->push_back(Img_initial.pose);
        focals->push_back(Img_initial.camera.params[0]);
        nSols++;

        return nSols;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p3p_sampling_fov_LM"; }
};


struct SolverFisheye_P3P_fov {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {

        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        int nSols = 0;
        double min_reproj_error = std::numeric_limits<double>::max();
        double focal_best = 0.0;
        CameraPose pose_best;

        std::vector<double> fov_list = {100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 210, 220};
        for (double fov : fov_list) {
            // double focal = instance.focal_gt * std::tan(instance.camera_fov_ / 2.0 * kPI / 180.0) / std::tan(fov / 2.0 * kPI / 180.0);
            double image_size = instance.focal_gt * std::tan(instance.camera_fov_ / 2.0 * kPI / 180.0);
            double focal = image_size / (fov * M_PI / 180.0);
            
            Camera camera;
            camera.model_id = 12;
            camera.params = {focal, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
            for (int i = 0; i < 3; i++) {
                camera.unproject(p2d[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, instance.X_point_, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {

                // check reprojection with the 4th point

                Eigen::Vector2d reprojected;
                Eigen::Vector3d x_ = solutions_p3p[j].R() * instance.X_point_[3] + solutions_p3p[j].t;
                camera.project(x_, &reprojected);
                double res = (reprojected - p2d[3]).norm();

                if (res < min_reproj_error) {
                    min_reproj_error = res;
                    focal_best = focal;
                    pose_best = solutions_p3p[j];
                }
            }
        }

        solutions->push_back(pose_best);
        focals->push_back(focal_best);
        nSols++;

        return nSols;
    }

    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p3p_sampling_fov"; }
};

struct SolverFisheye_P3P_HC {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {

        //TODO: using the reprojection error find the best and run LM

        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        int nSols = 0;
        double min_reproj_error = std::numeric_limits<double>::max();
        double focal_best = 0.0;
        CameraPose pose_best;

        std::vector<double> focal_list = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        for (double focal : focal_list) {
            Camera camera;
            camera.model_id = 12;
            camera.params = {focal, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
            for (int i = 0; i < 3; i++) {
                camera.unproject(p2d[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, instance.X_point_, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {


                Eigen::Vector2d reprojected;
                Eigen::Vector3d x_ = solutions_p3p[j].R() * instance.X_point_[3] + solutions_p3p[j].t;
                camera.project(x_, &reprojected);
                double res = (reprojected - p2d[3]).norm();

                if (res < min_reproj_error) {
                    min_reproj_error = res;
                    focal_best = focal;
                    pose_best = solutions_p3p[j];
                }
            }
        }


        CameraPose pose_initial = pose_best;
        Camera camera_initial;
        camera_initial.model_id = 12;
        camera_initial.params = {focal_best, 0.0, 0.0};
        Image Img_initial(pose_initial, camera_initial);

        CameraPose solution_HC;
        double focal_HC;
        int HC_success = p4pf_fisheye_lie(p2d, instance.X_point_, Img_initial, &solution_HC, &focal_HC);

        if (HC_success == 1) {
            solutions->push_back(solution_HC);
            focals->push_back(focal_HC);
            nSols++;
        }

        return nSols;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p3p_sampling_HC"; }
};


struct SolverFisheye_P3P_focal_all_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {

        //TODO: using the reprojection error find the best and run LM

        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        int nSols = 0;

        
        std::vector<double> focal_list = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        for (double focal : focal_list) {
            
            Camera camera;
            // camera.model_id = 12;
            // camera.params = {focal, 0.0, 0.0};

            camera.model_id = 5;
            camera.params = {focal, focal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(instance.x_point_fisheye_.size());
            for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
                camera.unproject(p2d[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, instance.X_point_, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {

                if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p3p[j], focal, 1e-2)){
                    CameraPose pose_initial = solutions_p3p[j];
                    Camera camera_initial;
                    // camera_initial.model_id = 12;
                    // camera_initial.params = {focal, 0.0, 0.0};
                    camera_initial.model_id = 5;
                    camera_initial.params = {focal, focal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    Image Img_initial(pose_initial, camera_initial);
        
                    BundleOptions bundle_opt;
                    // bundle_opt.step_tol = 1e-12;
                    bundle_opt.refine_focal_length = true;
                    std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);
        
                    AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
                    lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);
        
                    solutions->push_back(Img_initial.pose);
                    focals->push_back(Img_initial.camera.params[0]);
                    nSols++;

                } else {
                    continue;
                }
            }
        }

        return nSols;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p3p_sampling_focal_all_LM"; }
};


struct SolverFisheye_P3P_fov_all_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {

        //TODO: using the reprojection error find the best and run LM

        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(instance.x_point_fisheye_.size());
        for (int i = 0; i < instance.x_point_fisheye_.size(); ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        int nSols = 0;

        std::vector<double> fov_list = {30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
        for (double fov : fov_list) {

            double focal = instance.focal_gt * std::tan(instance.camera_fov_ / 2.0 * kPI / 180.0) / std::tan(fov / 2.0 * kPI / 180.0);
            Camera camera;
            // camera.model_id = 12;
            // camera.params = {focal, 0.0, 0.0};

            camera.model_id = 5;
            camera.params = {focal, focal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(instance.x_point_fisheye_.size());
            for (int i = 0; i < instance.x_point_fisheye_.size(); i++) {
                camera.unproject(p2d[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, instance.X_point_, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {

                if (UnknownFocalFisheyeValidator::is_valid_inner(instance, solutions_p3p[j], focal, 1e-2)){
                    CameraPose pose_initial = solutions_p3p[j];
                    Camera camera_initial;
                    // camera_initial.model_id = 12;
                    // camera_initial.params = {focal, 0.0, 0.0};
                    camera_initial.model_id = 5;
                    camera_initial.params = {focal, focal, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    Image Img_initial(pose_initial, camera_initial);
        
                    BundleOptions bundle_opt;
                    // bundle_opt.step_tol = 1e-12;
                    bundle_opt.refine_focal_length = true;
                    std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);
        
                    AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
                    lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);
        
                    solutions->push_back(Img_initial.pose);
                    focals->push_back(Img_initial.camera.params[0]);
                    nSols++;

                } else {
                    continue;
                }
            }
        }

        return nSols;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p3p_sampling_fov_all_LM"; }
};


struct SolverFisheye_P5PF_LM {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {
        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(5);
        for (int i = 0; i < 5; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }
        
        return p5pf_fisheye_lm(p2d, instance.X_point_, solutions, focals);
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p5pf_LM"; }
};


struct SolverFisheye_P5PF {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {
        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(5);
        for (int i = 0; i < 5; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        return p5pf_fisheye(p2d, instance.X_point_, solutions, focals, false);
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p5pf_Newton"; }
};


// struct SolverFisheye_P5PF_LM {
//     static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
//         std::vector<double> *focals) {
//         // dehomogenize input
//         std::vector<Eigen::Vector2d> p2d(5);
//         for (int i = 0; i < 5; ++i) {
//             p2d[i] = instance.x_point_fisheye_[i].hnormalized();
//         }

//         CameraPoseVector solutions_p5pf;
//         std::vector<double> focals_p5pf;
//         int nSols_p5pf = p5pf(p2d, instance.X_point_, &solutions_p5pf, &focals_p5pf);

//         for (int i = 0; i < nSols_p5pf; i++) {
//             CameraPose pose_initial = solutions_p5pf[i];
//             Camera camera_initial;
//             camera_initial.model_id = 12;
//             camera_initial.params = {focals_p5pf[i], 0.0, 0.0};
//             Image Img_initial(pose_initial, camera_initial);

//             BundleOptions bundle_opt;
//             bundle_opt.refine_focal_length = true;
//             std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

//             AbsolutePoseRefiner<> refiner(p2d, instance.X_point_, camera_refine_idx);
//             lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

//             solutions->push_back(Img_initial.pose);
//             focals->push_back(Img_initial.camera.params[0]);
//         }

//         return nSols_p5pf;
//     }
//     typedef UnknownFocalFisheyeValidator validator;
//     static std::string name() { return "fisheye_p5pf_lm"; }
// };

struct SolverFisheye_P5PF_original {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {
        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(5);
        for (int i = 0; i < 5; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        int nSols_p5pf = p5pf(p2d, instance.X_point_, solutions, focals);


        return nSols_p5pf;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p5pf"; }
};


struct SolverFisheye_P5PF_TaylorExpansion {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
        std::vector<double> *focals) {
        // dehomogenize input
        std::vector<Eigen::Vector2d> p2d(5);
        for (int i = 0; i < 5; ++i) {
            p2d[i] = instance.x_point_fisheye_[i].hnormalized();
        }

        double f_inital = instance.focal_gt-1;

        int nSols_p5pf = p5pf_fisheye2(p2d, instance.X_point_, solutions, focals, f_inital);

        return nSols_p5pf;
    }
    typedef UnknownFocalFisheyeValidator validator;
    static std::string name() { return "fisheye_p5pf_TaylorExpansion"; }
};

} // namespace poselib
