#include "PoseLib/solvers/p4pf_fisheye.h"
#include "PoseLib/solvers/p4pfr.h"
#include "PoseLib/solvers/p4pf_fisheye_depth.h"
#include <PoseLib/robust/optim/absolute.h>
#include <PoseLib/robust/optim/lm_impl.h>

namespace poselib {

    int p4pf_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                     const Image &Img_initial, CameraPose *solution, double *focal) {
        
        AbsoluteFisheyeHCProblem problem(x, X, Img_initial);

        const HCOptions opt;
        AbsoluteFisheyeHCProblem::sol_t HC_output;
        typedef AbsoluteFisheyeHCProblem::sol_t sol_t;
        typedef AbsoluteFisheyeHCProblem::poly_t poly_t;
        typedef AbsoluteFisheyeHCProblem::jacobian_t jacobian_t;

        HCStats stats = HC_impl<AbsoluteFisheyeHCProblem, sol_t, poly_t, jacobian_t>(problem, opt, HC_output);

        if (stats.success) {
            problem.get_solution(HC_output, solution, focal);
            return 1;
        } else {
            return 0;
        }

    }

    int p4pf_fisheye_lie(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                     const Image &Img_initial, CameraPose *solution, double *focal) {
        
        AbsoluteFisheyeHCProblemPoseLie problem(x, X, Img_initial);
        
        const HCOptions opt;
        Image HC_output;
        typedef AbsoluteFisheyeHCProblemPoseLie::dX_t dX_t;
        typedef AbsoluteFisheyeHCProblemPoseLie::poly_t poly_t;
        typedef AbsoluteFisheyeHCProblemPoseLie::jacobian_t jacobian_t;

        HCStats stats = HC_impl_update_Image<AbsoluteFisheyeHCProblemPoseLie, dX_t, poly_t, jacobian_t>(problem, opt, HC_output);

        if (stats.success) {
            solution->q = HC_output.pose.q;
            solution->t = HC_output.pose.t;
            *focal = HC_output.camera.params[0];
            return 1;
        } else {
            return 0;
        }
        
    }

    // solvers for RANSAC

    bool is_valid_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
         const CameraPose &pose, double focal, double tol) {

        if ((pose.R().transpose() * pose.R() - Eigen::Matrix3d::Identity()).norm() > tol)
            return false;
    
        if (focal < 0)
            return false;
    
        // lambda*[tan(theta) x/rd; 1] = R*X + t
        for (int i = 0; i < x.size(); ++i) {
            double rd = std::sqrt(x[i](0) * x[i](0) + x[i](1) * x[i](1));
            double theta = rd / focal;
            Eigen::Vector3d x_fisheye = Eigen::Vector3d{x[i](0) / rd * std::tan(theta), x[i](1) / rd * std::tan(theta), 1.0};
            double inner_product = (x_fisheye).normalized().dot((pose.R() * X[i] + pose.t).normalized());
            if (inner_product < 0) {
                return false;
            }
            double err = 1.0 - std::abs(inner_product);
            if (err > tol)
                return false;
        }

        return true;
    }

    int p4pfr_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                     CameraPoseVector *solutions, std::vector<double> *focals) {
        std::vector<double> ks;
        return p4pfr(x, X, solutions, focals, &ks);
    }
    
    int p4pfr_lm_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                     CameraPoseVector *solutions, std::vector<double> *focals) {

        CameraPoseVector solutions_p4pfr;
        std::vector<double> focals_p4pfr;
        std::vector<double> ks;
        int nSols_p4pfr = p4pfr(x, X, &solutions_p4pfr, &focals_p4pfr, &ks);

        if (nSols_p4pfr == 0) {
            return 0;
        }

        // LM refine
        int nSols_LM = 0;
        for (int i = 0; i < nSols_p4pfr; i++) {
            if (is_valid_fisheye(x, X, solutions_p4pfr[i], focals_p4pfr[i], 1e-2)){

                CameraPose pose_initial = solutions_p4pfr[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
                Image Img_initial(pose_initial, camera_initial);

                BundleOptions bundle_opt;
                // bundle_opt.step_tol = 1e-12;
                bundle_opt.refine_focal_length = true;
                std::vector<size_t> camera_refine_idx = Img_initial.camera.get_param_refinement_idx(bundle_opt);

                AbsolutePoseRefiner<> refiner(x, X, camera_refine_idx);
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
    
    int p4pfr_hc_pose_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                     CameraPoseVector *solutions, std::vector<double> *focals) {
                       
        CameraPoseVector solutions_p4pfr;
        std::vector<double> focals_p4pfr;
        std::vector<double> ks;
        int nSols_p4pfr = p4pfr(x, X, &solutions_p4pfr, &focals_p4pfr, &ks);

        if (nSols_p4pfr == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p4pfr; i++) {
            if (is_valid_fisheye(x, X, solutions_p4pfr[i], focals_p4pfr[i], 1e-2)){

                CameraPose pose_initial = solutions_p4pfr[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
                Image Img_initial(pose_initial, camera_initial);

                CameraPose solution_HC;
                double focal_HC;
                int HC_success = p4pf_fisheye_lie(x, X, Img_initial, &solution_HC, &focal_HC);

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
    
    int p4pfr_hc_depth_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                     CameraPoseVector *solutions, std::vector<double> *focals) {
                        
        CameraPoseVector solutions_p4pfr;
        std::vector<double> focals_p4pfr;
        std::vector<double> ks;
        int nSols_p4pfr = p4pfr(x, X, &solutions_p4pfr, &focals_p4pfr, &ks);

        if (nSols_p4pfr == 0) {
            return 0;
        }

        int nSols_HC = 0;
        for (int i = 0; i < nSols_p4pfr; i++) {
            if (is_valid_fisheye(x, X, solutions_p4pfr[i], focals_p4pfr[i], 1e-2)){

                CameraPose pose_initial = solutions_p4pfr[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p4pfr[i], 0.0, 0.0};
                Image Img_initial(pose_initial, camera_initial);

                CameraPose solution_HC;
                double focal_HC;
                int HC_success = p4pf_fisheye_depth(x, X, Img_initial, &solution_HC, &focal_HC);

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

}