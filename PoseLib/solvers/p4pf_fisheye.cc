#include "PoseLib/solvers/p4pf_fisheye.h"

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

}