#include "PoseLib/solvers/p4pf_fisheye.h"
#include "PoseLib/solvers/p4pfr.h"
#include "PoseLib/solvers/p4pf_fisheye_depth.h"
#include "PoseLib/solvers/p5pfr.h"
#include "PoseLib/solvers/p5pf.h"
#include "PoseLib/solvers/p35pf.h"
#include "PoseLib/solvers/p4pfr_planar.h"
#include <PoseLib/robust/optim/absolute.h>
#include <PoseLib/robust/optim/lm_impl.h>

// bool is_planar(const std::vector<Eigen::Vector3d> &X){
//     double pz = 0;

//     for (int i = 0; i < X.size(); i++) {
//         pz+=X[i](2);
//     }
//     pz /= X.size();
    
//     return true;
// }

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
        // HCStats stats = HC_impl_binary_adaptive<AbsoluteFisheyeHCProblem, sol_t, poly_t, jacobian_t>(problem, opt, HC_output);

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
            // double err = 1.0 - std::abs(inner_product);
            // if (err > tol)
            //     return false;
        }

        return true;
    }

    // int p4pfr_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
    //                  CameraPoseVector *solutions, std::vector<double> *focals) {
    //     std::vector<double> ks;
    //     return p4pfr(x, X, solutions, focals, &ks);
    // }

    int p4pfr_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
        CameraPoseVector *solutions, std::vector<double> *focals) {
        std::vector<double> ks;
        std::vector<CameraPose> poses_p4pfr;
        std::vector<double> focals_p4pfr;
        int nSols_p4pfr = p4pfr(x, X, &poses_p4pfr, &focals_p4pfr, &ks);
        if (nSols_p4pfr == 0) {
            return 0;
        }
        int nSols = 0;
        for (int i = 0; i < nSols_p4pfr; i++) {
            if (is_valid_fisheye(x, X, poses_p4pfr[i], focals_p4pfr[i], 1e-2)){
                solutions->push_back(poses_p4pfr[i]);
                focals->push_back(focals_p4pfr[i]);
                nSols++;
            }
        }
        return nSols;
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

    int p3p_fisheye_lm(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                     const double image_size, CameraPoseVector *solutions, std::vector<double> *focals) {
        
        // double half_size = image_size / 2.0;
        
        int nSols = 0;
        double min_reproj_error = std::numeric_limits<double>::max();
        double focal_best = 0.0;
        CameraPose pose_best;

        // TODO: check the gt fov 200 deg
        std::vector<double> fov_list = {100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220};
        for (double fov : fov_list) {
            // double focal = half_size / std::tan(fov / 2.0 * M_PI / 180.0);
            double focal = image_size / (fov * M_PI / 180.0);
            
            Camera camera;
            camera.model_id = 12;
            camera.params = {focal, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
            for (int i = 0; i < 3; i++) {
                camera.unproject(x[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, X, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {

                // check reprojection with the 4th point
                Eigen::Vector2d reprojected;
                Eigen::Vector3d x_ = solutions_p3p[j].R() * X[3] + solutions_p3p[j].t;
                camera.project(x_, &reprojected);
                double res = (reprojected - x[3]).norm();

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

        AbsolutePoseRefiner<> refiner(x, X, camera_refine_idx);
        lm_impl<decltype(refiner)>(refiner, &Img_initial, bundle_opt);

        solutions->push_back(Img_initial.pose);
        focals->push_back(Img_initial.camera.params[0]);
        nSols++;

        return nSols;
    }

    int p3p_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                     const double image_size, CameraPoseVector *solutions, std::vector<double> *focals) {
        
        // double half_size = image_size / 2.0;
        
        double min_reproj_error = std::numeric_limits<double>::max();
        double focal_best = 0.0;
        CameraPose pose_best;

        std::vector<double> fov_list = {100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220};
        for (double fov : fov_list) {
            // double focal = half_size / std::tan(fov / 2.0 * M_PI / 180.0);
            double focal = image_size / (fov * M_PI / 180.0);
            
            Camera camera;
            camera.model_id = 12;
            camera.params = {focal, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
            for (int i = 0; i < 3; i++) {
                camera.unproject(x[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, X, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {

                // check reprojection with the 4th point
                Eigen::Vector2d reprojected;
                Eigen::Vector3d x_ = solutions_p3p[j].R() * X[3] + solutions_p3p[j].t;
                camera.project(x_, &reprojected);
                double res = (reprojected - x[3]).norm();

                if (res < min_reproj_error) {
                    min_reproj_error = res;
                    focal_best = focal;
                    pose_best = solutions_p3p[j];
                }
            }
        }

        solutions->push_back(pose_best);
        focals->push_back(focal_best);

        return 1;
    }

    int p3p_fisheye_hc(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                     const double image_size, CameraPoseVector *solutions, std::vector<double> *focals) {
        
        // double half_size = image_size / 2.0;
        
        int nSols = 0;
        double min_reproj_error = std::numeric_limits<double>::max();
        double focal_best = 0.0;
        CameraPose pose_best;

        // TODO: check the gt fov 200 deg
        std::vector<double> fov_list = {100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220};
        for (double fov : fov_list) {
            // double focal = half_size / std::tan(fov / 2.0 * M_PI / 180.0);
            double focal = image_size / (fov * M_PI / 180.0);
            
            Camera camera;
            camera.model_id = 12;
            camera.params = {focal, 0.0, 0.0};

            std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
            for (int i = 0; i < 3; i++) {
                camera.unproject(x[i], &x_fisheye_normalized[i]);
            }

            CameraPoseVector solutions_p3p;
            int nSols_p3p = p3p_ding(x_fisheye_normalized, X, &solutions_p3p);

            for (int j = 0; j < nSols_p3p; j++) {

                // check reprojection with the 4th point
                Eigen::Vector2d reprojected;
                Eigen::Vector3d x_ = solutions_p3p[j].R() * X[3] + solutions_p3p[j].t;
                camera.project(x_, &reprojected);
                double res = (reprojected - x[3]).norm();

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
        int HC_success = p4pf_fisheye_lie(x, X, Img_initial, &solution_HC, &focal_HC);

        if (HC_success == 1) {
            solutions->push_back(solution_HC);
            focals->push_back(focal_HC);
            nSols++;
        }

        return nSols;
    }


    // int p5pfr_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
    //     CameraPoseVector *solutions, std::vector<double> *focals) {
    //     std::vector<double> ks;
    //     return p5pfr(x, X, solutions, focals, &ks);
    // }

    int p5pfr_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
        CameraPoseVector *solutions, std::vector<double> *focals) {
        // check the solutions are valid
        std::vector<double> ks;
        std::vector<CameraPose> poses_p5pfr;
        std::vector<double> focals_p5pfr;
        int nSols_p5pfr = p5pfr(x, X, &poses_p5pfr, &focals_p5pfr, &ks);
        if (nSols_p5pfr == 0) {
            return 0;
        }
        int nSols = 0;
        for (int i = 0; i < nSols_p5pfr; i++) {
            if (is_valid_fisheye(x, X, poses_p5pfr[i], focals_p5pfr[i], 1e-2)){
                solutions->push_back(poses_p5pfr[i]);
                focals->push_back(focals_p5pfr[i]);
                nSols++;
            }
        }
        return nSols;
    }

    int p5pfr_lm_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
            CameraPoseVector *solutions, std::vector<double> *focals) {

        CameraPoseVector solutions_p5pfr;
        std::vector<double> focals_p5pfr;
        std::vector<double> ks;
        int nSols_p5pfr = p5pfr(x, X, &solutions_p5pfr, &focals_p5pfr, &ks);

        if (nSols_p5pfr == 0) {
            return 0;
        }

        // LM refine
        int nSols_LM = 0;
        for (int i = 0; i < nSols_p5pfr; i++) {
            if (is_valid_fisheye(x, X, solutions_p5pfr[i], focals_p5pfr[i], 1e-2)){

                CameraPose pose_initial = solutions_p5pfr[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p5pfr[i], 0.0, 0.0};
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


    int p3p_fisheye_givenf(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, double focal_initial,
                     CameraPoseVector *solutions, std::vector<double> *focals) {

        Camera camera;
        camera.model_id = 12;
        camera.params = {focal_initial, 0.0, 0.0};

        std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
        for (int i = 0; i < 3; i++) {
            camera.unproject(x[i], &x_fisheye_normalized[i]);
        }

        CameraPoseVector solutions_p3p;
        int nSols_p3p = p3p_ding(x_fisheye_normalized, X, &solutions_p3p);
        
        for (int j = 0; j < nSols_p3p; j++) {
            solutions->push_back(solutions_p3p[j]);
            focals->push_back(focal_initial);
        }

        return nSols_p3p;
    }


    int p3p_fisheye_givenf_lm(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, double focal_initial,
                     CameraPoseVector *solutions, std::vector<double> *focals) {

        Camera camera;
        camera.model_id = 12;
        camera.params = {focal_initial, 0.0, 0.0};

        std::vector<Eigen::Vector3d> x_fisheye_normalized(3);
        for (int i = 0; i < 3; i++) {
            camera.unproject(x[i], &x_fisheye_normalized[i]);
        }

        CameraPoseVector solutions_p3p;
        int nSols_p3p = p3p_ding(x_fisheye_normalized, X, &solutions_p3p);

        int nSols_LM = 0;
        
        for (int j = 0; j < nSols_p3p; j++) {
            CameraPose pose_initial = solutions_p3p[j];
            Camera camera_initial;
            camera_initial.model_id = 12;
            camera_initial.params = {focal_initial, 0.0, 0.0};
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

        return nSols_LM;
    }


    int p5pf_orgin_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
        CameraPoseVector *solutions, std::vector<double> *focals) {
        std::vector<double> ks;
        std::vector<CameraPose> poses_p5pf;
        std::vector<double> focals_p5pf;
        int nSols_p5pf = p5pf(x, X, &poses_p5pf, &focals_p5pf);
        if (nSols_p5pf == 0) {
            return 0;
        }
        int nSols = 0;
        for (int i = 0; i < nSols_p5pf; i++) {
            if (is_valid_fisheye(x, X, poses_p5pf[i], focals_p5pf[i], 1e-2)){
                solutions->push_back(poses_p5pf[i]);
                focals->push_back(focals_p5pf[i]);
                nSols++;
            }
        }
        return nSols;
    }

    int p35pf_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
        CameraPoseVector *solutions, std::vector<double> *focals) {
        std::vector<double> ks;
        std::vector<CameraPose> poses_p35pf;
        std::vector<double> focals_p35pf;
        int nSols_p35pf = p35pf(x, X, &poses_p35pf, &focals_p35pf);
        if (nSols_p35pf == 0) {
            return 0;
        }
        int nSols = 0;
        for (int i = 0; i < nSols_p35pf; i++) {
            if (is_valid_fisheye(x, X, poses_p35pf[i], focals_p35pf[i], 1e-2)){
                solutions->push_back(poses_p35pf[i]);
                focals->push_back(focals_p35pf[i]);
                nSols++;
            }
        }
        return nSols;
    }

    int p35pf_lm_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
        CameraPoseVector *solutions, std::vector<double> *focals) {

        CameraPoseVector solutions_p35pf;
        std::vector<double> focals_p35pf;
        std::vector<double> ks;
        int nSols_p35pf = p35pf(x, X, &solutions_p35pf, &focals_p35pf);

        if (nSols_p35pf == 0) {
        return 0;
        }

        // LM refine
        int nSols_LM = 0;
        for (int i = 0; i < nSols_p35pf; i++) {
            if (is_valid_fisheye(x, X, solutions_p35pf[i], focals_p35pf[i], 1e-2)){

                CameraPose pose_initial = solutions_p35pf[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p35pf[i], 0.0, 0.0};
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

    // update for p4pfr planar solver
    bool isCoplanar(const std::vector<Eigen::Vector3d>& X, double eps = 1e-6)
    {
        // Assume X.size() == 4
        const Eigen::Vector3d v1 = X[1] - X[0];
        const Eigen::Vector3d v2 = X[2] - X[0];
        const Eigen::Vector3d v3 = X[3] - X[0];

        double det = v1.dot(v2.cross(v3));

        return std::abs(det) < eps;
    }

    int p4pfr_planar_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
        CameraPoseVector *solutions, std::vector<double> *focals) {
        if (!isCoplanar(X)) {
            return p4pfr_fisheye(x, X, solutions, focals);
        }
        std::vector<CameraPose> poses_p4pfr_planar;
        std::vector<double> focals_p4pfr_planar;
        int nSols_p4pfr_planar = p4pfr_planar(x, X, &poses_p4pfr_planar, &focals_p4pfr_planar);
        if (nSols_p4pfr_planar == 0) {
            return 0;
        }
        int nSols = 0;
        for (int i = 0; i < nSols_p4pfr_planar; i++) {
            if (is_valid_fisheye(x, X, poses_p4pfr_planar[i], focals_p4pfr_planar[i], 1e-2)){
                solutions->push_back(poses_p4pfr_planar[i]);
                focals->push_back(focals_p4pfr_planar[i]);
                nSols++;
            }
        }   
        return nSols;
    }
    
    int p4pfr_planar_lm_fisheye(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                     CameraPoseVector *solutions, std::vector<double> *focals) {

        if (!isCoplanar(X)) {
            return p4pfr_lm_fisheye(x, X, solutions, focals);
        }

        CameraPoseVector solutions_p4pfr_planar;
        std::vector<double> focals_p4pfr_planar;
        int nSols_p4pfr_planar = p4pfr_planar(x, X, &solutions_p4pfr_planar, &focals_p4pfr_planar);

        if (nSols_p4pfr_planar == 0) {
            return 0;
        }

        // LM refine
        int nSols_LM = 0;
        for (int i = 0; i < nSols_p4pfr_planar; i++) {
            if (is_valid_fisheye(x, X, solutions_p4pfr_planar[i], focals_p4pfr_planar[i], 1e-2)){

                CameraPose pose_initial = solutions_p4pfr_planar[i];
                Camera camera_initial;
                camera_initial.model_id = 12;
                camera_initial.params = {focals_p4pfr_planar[i], 0.0, 0.0};
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

}