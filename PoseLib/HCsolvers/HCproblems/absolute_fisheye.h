#ifndef POSELIB_HC_ABSOLUTE_FISHEYE_H_
#define POSELIB_HC_ABSOLUTE_FISHEYE_H_

#include "PoseLib/types.h"
#include "PoseLib/camera_pose.h"
#include "PoseLib/HCsolvers/HCproblems/HCproblem_base.h"
#include "PoseLib/HCsolvers/HCproblems/helper.h"

#include <iostream>

namespace poselib {

class AbsoluteFisheyeHCProblem : public HCProblemBase<8, 8, Image> {
    // TODO: this class should include what I used for define functions for specific problems, 
    // especially for the constraints evaluation and its jacobian and simulator
    // camera model is SimpleFisheyeCameraModel
    // Rotation is formulated with quaternion representation, and we need an additional constraint to make it unit
    // btw not polynomials cuz triangular function involved

    public:

        // For pose as unknowns it is 2*3 + 1 + 1 constraints with 4 + 3 + 1 parameters
        // static constexpr int num_params = 8;
        // static constexpr int num_polys = 8;
        int nParams = 8;
        int nPolys = 8;

        Image pose_initial;

        AbsoluteFisheyeHCProblem(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                 const Image &_pose_initial)
            : HCProblemBase<8, 8, Image>(), pose_initial(_pose_initial) {

            // Initialize base class member variables
            this->x = points2D;
            this->X = points3D;
            
            // initialize the x_simulated
            simulator(pose_initial);
            
        }

        sol_t get_sol_vector() {
            // Assume fx = fy
            const CameraPose pose = pose_initial.pose;
            const Camera camera = pose_initial.camera;
            sol_t sol;
            sol << pose.q, pose.t, camera.params[0]; // size 4+3+1 = 8
            return sol;
        }

        void get_solution(const sol_t &sol, CameraPose *pose, double *focal) {
            const Eigen::Vector4d q = sol.head(4);
            const Eigen::Vector3d t = sol.block<3, 1>(4, 0);
            const double f = sol(7);
            *pose = CameraPose(q, t);
            *focal = f;
        }

        void simulator(const Image &pose_initial) {
            const CameraPose pose = pose_initial.pose;
            const Camera camera = pose_initial.camera;
            const Eigen::Matrix3d R = pose.R();
            const Eigen::Vector3d t = pose.t;

            // project the 3D points to the image plane (fisheye equidistant)
            for (int i = 0; i < X.size(); ++i) {
                const Eigen::Vector3d Z = R * X[i] + t;
                Eigen::Vector2d xp;
                SimpleFisheyeCameraModel::project(camera.params, Z, &xp);
                this->x_simulated.push_back(xp);
            }
            
        }

        void compute_polys(const sol_t &sol, const std::vector<Point2D> &x_, poly_t &polys) { 
            // input is the solution vector
            
            Eigen::Vector4d q = sol.head(4);
            const Eigen::Vector3d t = sol.block<3, 1>(4, 0);
            const double f = sol(7);
            const Eigen::Matrix3d R = getRfromq(q);
            const Eigen::Vector3d r1 = R.row(0);
            const Eigen::Vector3d r2 = R.row(1);
            const Eigen::Vector3d r3 = R.row(2);

            polys.setZero();
            
            for (int i = 0; i < X.size()-1; ++i) {
                double rd = std::sqrt(x_[i][0]*x_[i][0] + x_[i][1]*x_[i][1]);
                double theta = rd / f;
                polys(i*2) = -(r3.dot(X[i]) + t(2))*std::tan(theta)*x_[i][0]/rd + r1.dot(X[i]) + t(0);
                polys(i*2+1) = -(r3.dot(X[i]) + t(2))*std::tan(theta)*x_[i][1]/rd + r2.dot(X[i]) + t(1);
            }
            double rd_3 = std::sqrt(x_[3][0]*x_[3][0] + x_[3][1]*x_[3][1]);
            polys(6) = x_[3][0] / rd_3 * (r2.dot(X[3]) + t(1)) - x_[3][1] / rd_3 * (r1.dot(X[3]) + t(0));
            polys(7) = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] - 1;

        }

        void compute_jacobian(const sol_t &sol, const std::vector<Point2D> &x_, jacobian_t &jacobian) {

            Eigen::Vector4d q = sol.head(4);
            const Eigen::Vector3d t = sol.block<3, 1>(4, 0);
            const double f = sol(7);
            const Eigen::Matrix3d R = getRfromq(q);
            const Eigen::Vector3d r3 = R.row(2);
            const Eigen::Matrix3d dqw = dRdq1(q);
            const Eigen::Matrix3d dqx = dRdq2(q);
            const Eigen::Matrix3d dqy = dRdq3(q);
            const Eigen::Matrix3d dqz = dRdq4(q);

            jacobian.setZero();

            for (int i = 0; i < X.size()-1; ++i) {
                double rd = std::sqrt(x_[i][0]*x_[i][0] + x_[i][1]*x_[i][1]);
                double ui = x_[i][0] / rd;
                double vi = x_[i][1] / rd;
                double theta = rd / f;

                // g1 Jacobian wrt qw qx qy qz
                jacobian(i*2, 0) = X[i].dot(dqw.row(0)) - ui * std::tan(theta) * X[i].dot(dqw.row(2));
                jacobian(i*2, 1) = X[i].dot(dqx.row(0)) - ui * std::tan(theta) * X[i].dot(dqx.row(2));
                jacobian(i*2, 2) = X[i].dot(dqy.row(0)) - ui * std::tan(theta) * X[i].dot(dqy.row(2));
                jacobian(i*2, 3) = X[i].dot(dqz.row(0)) - ui * std::tan(theta) * X[i].dot(dqz.row(2));

                // g1 Jacobian wrt tx ty tz
                jacobian(i*2, 4) = 1;
                jacobian(i*2, 5) = 0;
                jacobian(i*2, 6) = -ui*std::tan(theta);

                // g1 Jacobian wrt f
                jacobian(i*2, 7) = ui*(r3.dot(X[i]) + t(2)) * (sec(theta) * sec(theta) * theta/f);

                // g2 Jacobian wrt qw qx qy qz
                jacobian(i*2+1, 0) = X[i].dot(dqw.row(1)) - vi * std::tan(theta) * X[i].dot(dqw.row(2));
                jacobian(i*2+1, 1) = X[i].dot(dqx.row(1)) - vi * std::tan(theta) * X[i].dot(dqx.row(2));
                jacobian(i*2+1, 2) = X[i].dot(dqy.row(1)) - vi * std::tan(theta) * X[i].dot(dqy.row(2));
                jacobian(i*2+1, 3) = X[i].dot(dqz.row(1)) - vi * std::tan(theta) * X[i].dot(dqz.row(2));

                // g2 Jacobian wrt tx ty tz
                jacobian(i*2+1, 4) = 0;
                jacobian(i*2+1, 5) = 1;
                jacobian(i*2+1, 6) = -vi*std::tan(theta);

                // g2 Jacobian wrt f
                jacobian(i*2+1, 7) = vi*(r3.dot(X[i]) + t(2)) * (sec(theta) * sec(theta) * theta/f);
            }

            // g3 Jacobian wrt qw qx qy qz
            double rd = std::sqrt(x_[3][0]*x_[3][0] + x_[3][1]*x_[3][1]);
            double ui = x_[3][0] / rd;
            double vi = x_[3][1] / rd;

            jacobian(6, 0) = ui * X[3].dot(dqw.row(1)) - vi * X[3].dot(dqw.row(0));
            jacobian(6, 1) = ui * X[3].dot(dqx.row(1)) - vi * X[3].dot(dqx.row(0));
            jacobian(6, 2) = ui * X[3].dot(dqy.row(1)) - vi * X[3].dot(dqy.row(0));
            jacobian(6, 3) = ui * X[3].dot(dqz.row(1)) - vi * X[3].dot(dqz.row(0));

            // g3 Jacobian wrt tx ty tz
            jacobian(6, 4) = -vi;
            jacobian(6, 5) = ui;
            jacobian(6, 6) = 0;

            // g3 Jacobian wrt f
            jacobian(6, 7) = 0;

            // quaternion constraint jacobian
            jacobian(7, 0) = 2 * q(0);
            jacobian(7, 1) = 2 * q(1);
            jacobian(7, 2) = 2 * q(2);
            jacobian(7, 3) = 2 * q(3);


        }



};


class AbsoluteFisheyeHCProblemDepth : public HCProblemBase<5, 5, Image> {
    // TODO: this class should include what I used for define functions for specific problems, 
    // especially for the constraints evaluation and its jacobian and simulator
    // camera model is SimpleFisheyeCameraModel
    // constraints are formulated with depth and focal length

    public:
        // For pose as unknowns it is 4 + 1 constraints with 4 + 1 parameters
        // static constexpr int num_params = 5;
        // static constexpr int num_polys = 5;
        int nParams = 5;
        int nPolys = 5;

        Image pose_initial;

        AbsoluteFisheyeHCProblemDepth(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                 const Image &_pose_initial)
            : HCProblemBase<5, 5, Image>(), pose_initial(_pose_initial) {

            // Initialize base class member variables
            this->x = points2D;
            this->X = points3D;
            
            // initialize the x_simulated
            simulator(pose_initial);
        }

        sol_t get_sol_vector() {
            // from pose to depth
            const CameraPose pose = pose_initial.pose;
            const Camera camera = pose_initial.camera;
            const Eigen::Matrix3d R = pose.R();
            const Eigen::Vector3d t = pose.t;

            sol_t sol;
            sol.setZero();
            
            // Fill depths for each 3D point
            for (int i = 0; i < X.size(); ++i) {
                const Eigen::Vector3d Z = R * X[i] + t;
                sol(i) = Z(2);
            }
            
            // Add focal length at the end
            sol(4) = camera.params[0];

            // std::cout << "sol initial: " << sol << std::endl;

            return sol;
        }

        void get_solution(const sol_t &sol, CameraPose *pose, double *focal) {
            // from depth to pose

            *focal = sol(4);

            std::vector<Eigen::Vector3d> X_hat;
            for (int i = 0; i < X.size(); i++) {
                double di = sol(i);
                double rd = std::sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1]);
                double theta = rd / *focal;  
                Eigen::Vector2d x_hat = x[i] / rd;

                Eigen::Vector3d X_hat_i;
                X_hat_i << di * std::tan(theta) * x_hat[0], di * std::tan(theta) * x_hat[1], di;
                X_hat.push_back(X_hat_i);
            }

            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            procrustes(X_hat, X, R, t);
            *pose = CameraPose(R, t);

        }

        void simulator(const Image &pose_initial) {
            const CameraPose pose = pose_initial.pose;
            const Camera camera = pose_initial.camera;
            const Eigen::Matrix3d R = pose.R();
            const Eigen::Vector3d t = pose.t;

            // project the 3D points to the image plane (fisheye equidistant)
            for (int i = 0; i < X.size(); ++i) {
                const Eigen::Vector3d Z = R * X[i] + t;
                Eigen::Vector2d xp;
                SimpleFisheyeCameraModel::project(camera.params, Z, &xp);
                this->x_simulated.push_back(xp);

                // std::cout << "x_simulated[" << i << "]: " << xp << std::endl;
                // std::cout << "x[" << i << "]: " << x[i] << std::endl;
            }
            
        }

        void compute_polys(const sol_t &sol, const std::vector<Point2D> &x_, poly_t &polys) { 
            // input is the solution vector
            
            const double f = sol(4);

            polys.setZero();
            
            int k = 0;
            for (int i = 0; i < X.size()-1; ++i) {
                double rd_i = std::sqrt(x_[i][0]*x_[i][0] + x_[i][1]*x_[i][1]);
                double theta_i = rd_i / f;
                double tan_theta_i = std::tan(theta_i);
                Eigen::Vector2d x_hat_i = x_[i] / rd_i;
                Eigen::Vector3d p_i = X[i];
                double d_i = sol(i);

                for (int j = i+1; j < X.size(); ++j) {
                    double rd_j = std::sqrt(x_[j][0]*x_[j][0] + x_[j][1]*x_[j][1]);
                    double theta_j = rd_j / f;
                    double tan_theta_j = std::tan(theta_j);
                    Eigen::Vector2d x_hat_j = x_[j] / rd_j;
                    Eigen::Vector3d p_j = X[j];
                    Eigen::Vector3d p_ij = p_i - p_j;
                    double d_j = sol(j);
                    Eigen::Vector2d temp = x_hat_i * d_i * tan_theta_i - x_hat_j * d_j * tan_theta_j;
                    
                    polys(k) = temp.dot(temp) + (d_i - d_j)*(d_i - d_j) -p_ij.dot(p_ij);
                    k++;
                    if (k == nPolys-1) {
                        break;
                    }
                }
            }

        }

        void compute_jacobian(const sol_t &sol, const std::vector<Point2D> &x_, jacobian_t &jacobian) {

            const double f = sol(4);

            jacobian.setZero();

            int k = 0;
            for (int i = 0; i < X.size()-1; ++i) {
                double rd_i = std::sqrt(x_[i][0]*x_[i][0] + x_[i][1]*x_[i][1]);
                double theta_i = rd_i / f;
                double tan_theta_i = std::tan(theta_i);
                Eigen::Vector2d x_hat_i = x_[i] / rd_i;
                double d_i = sol(i);
                double dtan_theta_i_df = -sec(theta_i) * sec(theta_i) * theta_i / f;

                for (int j = i+1; j < X.size(); ++j) {
                    double rd_j = std::sqrt(x_[j][0]*x_[j][0] + x_[j][1]*x_[j][1]);
                    double theta_j = rd_j / f;
                    double tan_theta_j = std::tan(theta_j);
                    Eigen::Vector2d x_hat_j = x_[j] / rd_j;
                    double d_j = sol(j);
                    Eigen::Vector2d temp = x_hat_i * d_i * tan_theta_i - x_hat_j * d_j * tan_theta_j;
                    double dtan_theta_j_df = -sec(theta_j) * sec(theta_j) * theta_j / f;
                    
                    jacobian(k, i) = 2 * tan_theta_i * temp.dot(x_hat_i) + 2 * (d_i - d_j);
                    jacobian(k, j) = -2 * tan_theta_j * temp.dot(x_hat_j) - 2 * (d_i - d_j);
                    jacobian(k, 4) = 2 * temp.dot(x_hat_i * d_i * dtan_theta_i_df - x_hat_j * d_j * dtan_theta_j_df);

                    k++;
                    if (k == nPolys-1) {
                        break;
                    }

                }
            }
            
            // std::cout << "stop here" << std::endl;
            // std::abort();

        }
        
};

} // namespace poselib

#endif