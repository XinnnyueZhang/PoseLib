#ifndef POSELIB_HC_HC_PROBLEM_BASE_H_
#define POSELIB_HC_HC_PROBLEM_BASE_H_

#include "PoseLib/types.h"
#include <iostream>

namespace poselib {

template <int num_params, int num_polys, typename Model = Image>
class HCProblemBase {

public:

    typedef Eigen::Matrix<double, num_params, 1> sol_t;
    typedef Eigen::Matrix<double, num_polys, 1> poly_t;
    typedef Eigen::Matrix<double, num_polys, num_params> jacobian_t;


    std::vector<Point2D> x;
    std::vector<Point2D> x_simulated;
    std::vector<Point3D> X;

    HCProblemBase() {}
    virtual void simulator(const Model &model) = 0;

    virtual void compute_polys(const sol_t &sol, 
        const std::vector<Point2D> &x_, poly_t &polys)  = 0;

    virtual void compute_jacobian(const sol_t &sols, 
        const std::vector<Point2D> &x_,
        jacobian_t &J) = 0;
    
    void compute_PolysandJacobian(const sol_t &sol, const std::vector<Point2D> &x_, 
        poly_t &polys, jacobian_t &jacobian) {

        compute_polys(sol, x_, polys);
        compute_jacobian(sol, x_, jacobian);

    }

    void compute_HpolysandJacobian(double t, const sol_t &sol, 
        poly_t &Hpolys, jacobian_t &Hjacobian) {

        poly_t Fpolys, Gpolys;
        jacobian_t Fjacobian, Gjacobian;
        
        compute_PolysandJacobian(sol, x, Fpolys, Fjacobian);
        compute_PolysandJacobian(sol, x_simulated, Gpolys, Gjacobian);

        Hpolys = t * Fpolys + (1 - t) * Gpolys;
        Hjacobian = t * Fjacobian + (1 - t) * Gjacobian;

    }


};


template <int num_params, int num_polys, typename Model = Image>
class HCProblemBase2 {
// update Image struct instead of the Solution vector
public:

    typedef Eigen::Matrix<double, num_params, 1> dX_t;
    typedef Eigen::Matrix<double, num_polys, 1> poly_t;
    typedef Eigen::Matrix<double, num_polys, num_params> jacobian_t;


    std::vector<Point2D> x;
    std::vector<Point2D> x_simulated;
    std::vector<Point3D> X;

    HCProblemBase2() {}
    virtual void simulator(const Model &model) = 0;

    virtual void compute_polys(const Image &sol, 
        const std::vector<Point2D> &x_, poly_t &polys)  = 0;

    virtual void compute_jacobian(const Image &sol, 
        const std::vector<Point2D> &x_,
        jacobian_t &J) = 0;
    
    void compute_PolysandJacobian(const Image &sol, const std::vector<Point2D> &x_, 
        poly_t &polys, jacobian_t &jacobian) {

        compute_polys(sol, x_, polys);
        compute_jacobian(sol, x_, jacobian);

    }

    void compute_HpolysandJacobian(double t, const Image &sol, 
        poly_t &Hpolys, jacobian_t &Hjacobian) {

        poly_t Fpolys, Gpolys;
        jacobian_t Fjacobian, Gjacobian;
        
        compute_PolysandJacobian(sol, x, Fpolys, Fjacobian);
        compute_PolysandJacobian(sol, x_simulated, Gpolys, Gjacobian);

        Hpolys = t * Fpolys + (1 - t) * Gpolys;
        Hjacobian = t * Fjacobian + (1 - t) * Gjacobian;

    }

    Image step(const Image &sol, const dX_t &dx) {
        Image sol_new;
        sol_new.camera = sol.camera;

        // update quaternion with lie algebra vector
        sol_new.pose.q = quat_step_post(sol.pose.q, dx.block(0, 0, 3, 1));
        sol_new.pose.t = sol.pose.t + dx.block(3, 0, 3, 1);
        sol_new.camera.params[0] += dx(6);

        return sol_new;
    }


};

} // namespace poselib

#endif