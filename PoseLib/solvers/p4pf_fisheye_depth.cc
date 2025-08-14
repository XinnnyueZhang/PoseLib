#include "PoseLib/solvers/p4pf_fisheye_depth.h"

namespace poselib {

    int p4pf_fisheye_depth(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, 
                           const Image &Img_initial, CameraPose *solution, double *focal) {

        AbsoluteFisheyeHCProblemDepth problem(x, X, Img_initial);

        const HCOptions opt;
        AbsoluteFisheyeHCProblemDepth::sol_t HC_output;
        typedef AbsoluteFisheyeHCProblemDepth::sol_t sol_t;
        typedef AbsoluteFisheyeHCProblemDepth::poly_t poly_t;
        typedef AbsoluteFisheyeHCProblemDepth::jacobian_t jacobian_t;
        
        HCStats stats = HC_impl<AbsoluteFisheyeHCProblemDepth, sol_t, poly_t, jacobian_t>(problem, opt, HC_output);

        if (stats.success) {
            problem.get_solution(HC_output, solution, focal);
            return 1;
        } else {
            return 0;
        }


    }

}