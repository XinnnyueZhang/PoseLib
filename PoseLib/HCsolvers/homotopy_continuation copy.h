#ifndef POSELIB_HOMOTOPY_CONTINUATION_H_
#define POSELIB_HOMOTOPY_CONTINUATION_H_


#include "PoseLib/types.h"
#include "PoseLib/HCsolvers/HCproblems/helper.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace poselib {

// Notes: Problem (HC problems) is for polynomial functions and Model (Image struct) is for the parameters to be optimized
// problem.x is 2D points and problem.x_simulated is 2D points simulated by the initial solution

template<typename Problem, typename Solution, typename Poly, typename Jacobian>
bool compute_dx(double t, Problem &problem, Solution &sol, Solution &dx) {

    Poly polysF, polysG;
    Jacobian JF, JG;

    problem.compute_PolysandJacobian(sol, problem.x, polysF, JF);
    problem.compute_PolysandJacobian(sol, problem.x_simulated, polysG, JG);

    Jacobian Jx = t * JF + (1 - t) * JG;
    Poly t_grad = polysF - polysG;

    if (Jx.rows() == Jx.cols()) {
        // Use LU decomposition - fastest for square matrices
        dx = -Jx.lu().solve(t_grad);    
    } else {
        // Use QR decomposition for non-square matrices
        dx = -Jx.colPivHouseholderQr().solve(t_grad);
    }
    // dx = -(Jx.transpose() * Jx).inverse() * Jx.transpose() * t_grad;

    return true;
}


template<typename Problem, typename Solution, typename Poly, typename Jacobian>
HCStats HC_impl(Problem &problem, const HCOptions &opt, Solution &sol)
{
    HCStats stats;

    const auto clamp_step = [&](double value) {
        return std::max(opt.min_step_size, std::min(opt.max_step_size, value));
    };

    auto predictor = [&](Solution base_state, double current_t, double step,
                         Solution &out_state, double &update_norm) -> bool {
        if (opt.forth_predictor) {
            Solution k1, k2, k3, k4;
            if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t, problem, base_state, k1)) {
                return false;
            }

            Solution sol_temp1 = base_state + (step * 0.5) * k1;
            if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + step * 0.5, problem, sol_temp1, k2)) {
                return false;
            }

            Solution sol_temp2 = base_state + (step * 0.5) * k2;
            if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + step * 0.5, problem, sol_temp2, k3)) {
                return false;
            }

            Solution sol_temp3 = base_state + step * k3;
            if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + step, problem, sol_temp3, k4)) {
                return false;
            }

            Solution delta = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (step / 6.0);
            out_state = base_state + delta;
            update_norm = delta.norm();
        } else {
            Solution k1;
            if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t, problem, base_state, k1)) {
                return false;
            }
            Solution delta = k1 * step;
            out_state = base_state + delta;
            update_norm = delta.norm();
        }
        return std::isfinite(update_norm);
    };

    auto corrector = [&](double current_t, Solution &state, double &dx_norm, double &residual_norm) -> bool {
        dx_norm = 0.0;
        residual_norm = std::numeric_limits<double>::infinity();
        for (size_t iter = 0; iter < opt.newton_iter; ++iter) {
            Poly Hpolys;
            Jacobian JH;
            problem.compute_HpolysandJacobian(current_t, state, Hpolys, JH);

            if (opt.debug_output) {
                std::cout << "Hpolys: " << Hpolys << std::endl;
            }

            residual_norm = Hpolys.norm();
            if (!std::isfinite(residual_norm)) {
                return false;
            }
            if (residual_norm < opt.newton_tol) {
                dx_norm = 0.0;
                return true;
            }

            if (!checkValid(JH)) {
                return false;
            }

            Solution dx_newton;
            if (opt.adaptive_flag) {
                if (JH.rows() == JH.cols()) {
                    dx_newton = -JH.lu().solve(Hpolys);
                } else {
                    dx_newton = -JH.colPivHouseholderQr().solve(Hpolys);
                }
            } else {
                dx_newton = -(JH.transpose() * JH).inverse() * JH.transpose() * Hpolys;
            }

            dx_norm = dx_newton.norm();
            if (!std::isfinite(dx_norm)) {
                return false;
            }

            state = state + dx_newton;
            if (dx_norm < opt.newton_tol) {
                return true;
            }
        }

        return residual_norm < opt.newton_max_residual;
    };

    double t = 0.0;
    sol = problem.get_sol_vector();
    double step = clamp_step(opt.step_size);
    const double target_time = opt.target_time;

    while (stats.iterations < opt.max_iterations && t + opt.min_step_size <= target_time) {
        double trial_step = std::min(step, target_time - t);
        bool accepted_step = false;
        size_t retries = 0;

        while (retries <= opt.max_step_halvings) {
            Solution candidate;
            double predictor_norm = 0.0;
            double correction_norm = 0.0;
            double residual_norm = std::numeric_limits<double>::infinity();

            if (!predictor(sol, t, trial_step, candidate, predictor_norm) ||
                predictor_norm > opt.predictor_max_update) {
                ++stats.rejected_steps;
                ++retries;
                step = clamp_step(trial_step * opt.step_shrink);
                trial_step = std::min(step, target_time - t);
                continue;
            }

            if (!corrector(t + trial_step, candidate, correction_norm, residual_norm)) {
                ++stats.rejected_steps;
                ++stats.newton_failures;
                ++retries;
                step = clamp_step(trial_step * opt.step_shrink);
                trial_step = std::min(step, target_time - t);
                continue;
            }

            if (correction_norm > opt.newton_max_update) {
                ++stats.rejected_steps;
                ++retries;
                step = clamp_step(trial_step * opt.step_shrink);
                trial_step = std::min(step, target_time - t);
                continue;
            }

            sol = candidate;
            t += trial_step;
            ++stats.iterations;
            stats.final_residual = residual_norm;
            stats.final_time = t;
            accepted_step = true;

            if (predictor_norm < opt.predictor_small_update && correction_norm < opt.newton_tol) {
                step = clamp_step(step * opt.step_grow);
            } else if (correction_norm > opt.newton_tol) {
                step = clamp_step(step * opt.step_shrink);
            }

            stats.final_step_size = step;

            if (opt.debug_output) {
                std::cout << "t: " << t
                          << " step: " << trial_step
                          << " predictor_norm: " << predictor_norm
                          << " residual_norm: " << residual_norm << std::endl;
            }

            break;
        }

        if (!accepted_step) {
            stats.success = false;
            return stats;
        }
    }

    if (t + opt.min_step_size < target_time) {
        stats.success = false;
    }

    return stats;

};

// updating the Image struct instead of the Solution vector

template<typename Problem, typename dX, typename Poly, typename Jacobian>
bool compute_dx(double t, Problem &problem, Image &sol, dX &dx) {

    Poly polysF, polysG;
    Jacobian JF, JG;


    problem.compute_PolysandJacobian(sol, problem.x, polysF, JF);
    problem.compute_PolysandJacobian(sol, problem.x_simulated, polysG, JG);

    Jacobian Jx = t * JF + (1 - t) * JG;
    Poly t_grad = polysF - polysG;

    if (Jx.rows() == Jx.cols()) {
        // Use LU decomposition - fastest for square matrices
        dx = -Jx.lu().solve(t_grad);    
    } else {
        // Use QR decomposition for non-square matrices
        dx = -Jx.colPivHouseholderQr().solve(t_grad);
    }
    // dx = -(Jx.transpose() * Jx).inverse() * Jx.transpose() * t_grad;

    return true;
}



template<typename Problem, typename dX, typename Poly, typename Jacobian>
HCStats HC_impl_update_Image(Problem &problem, const HCOptions &opt, Image &sol)
{
    HCStats stats;

    const auto clamp_step = [&](double value) {
        return std::max(opt.min_step_size, std::min(opt.max_step_size, value));
    };

    auto predictor = [&](Image base_state, double current_t, double step,
                         Image &out_state, double &update_norm) -> bool {
        if (opt.forth_predictor) {
            dX k1, k2, k3, k4;
            if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t, problem, base_state, k1)) {
                return false;
            }

            Image sol_temp1 = problem.step(base_state, (step * 0.5) * k1);
            if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + step * 0.5, problem, sol_temp1, k2)) {
                return false;
            }

            Image sol_temp2 = problem.step(base_state, (step * 0.5) * k2);
            if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + step * 0.5, problem, sol_temp2, k3)) {
                return false;
            }

            Image sol_temp3 = problem.step(base_state, step * k3);
            if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + step, problem, sol_temp3, k4)) {
                return false;
            }

            dX delta = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (step / 6.0);
            update_norm = delta.norm();
            out_state = problem.step(base_state, delta);
        } else {
            dX k1;
            if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t, problem, base_state, k1)) {
                return false;
            }
            dX delta = k1 * step;
            update_norm = delta.norm();
            out_state = problem.step(base_state, delta);
        }

        return std::isfinite(update_norm);
    };

    auto corrector = [&](double current_t, Image &state, double &dx_norm, double &residual_norm) -> bool {
        dx_norm = 0.0;
        residual_norm = std::numeric_limits<double>::infinity();
        for (size_t iter = 0; iter < opt.newton_iter; ++iter) {
            Poly Hpolys;
            Jacobian JH;
            problem.compute_HpolysandJacobian(current_t, state, Hpolys, JH);

            if (opt.debug_output) {
                std::cout << "Hpolys: " << Hpolys << std::endl;
            }

            residual_norm = Hpolys.norm();
            if (!std::isfinite(residual_norm)) {
                return false;
            }
            if (residual_norm < opt.newton_tol) {
                dx_norm = 0.0;
                return true;
            }

            if (!checkValid(JH)) {
                return false;
            }

            dX dx_newton;
            if (opt.adaptive_flag) {
                if (JH.rows() == JH.cols()) {
                    dx_newton = -JH.lu().solve(Hpolys);
                } else {
                    dx_newton = -JH.colPivHouseholderQr().solve(Hpolys);
                }
            } else {
                dx_newton = -(JH.transpose() * JH).inverse() * JH.transpose() * Hpolys;
            }

            dx_norm = dx_newton.norm();
            if (!std::isfinite(dx_norm)) {
                return false;
            }

            state = problem.step(state, dx_newton);
            if (dx_norm < opt.newton_tol) {
                return true;
            }
        }

        return residual_norm < opt.newton_max_residual;
    };

    double t = 0.0;
    sol = problem.get_sol();
    double step = clamp_step(opt.step_size);
    const double target_time = opt.target_time;

    while (stats.iterations < opt.max_iterations && t + opt.min_step_size <= target_time) {
        double trial_step = std::min(step, target_time - t);
        bool accepted_step = false;
        size_t retries = 0;

        while (retries <= opt.max_step_halvings) {
            Image candidate;
            double predictor_norm = 0.0;
            double correction_norm = 0.0;
            double residual_norm = std::numeric_limits<double>::infinity();

            if (!predictor(sol, t, trial_step, candidate, predictor_norm) ||
                predictor_norm > opt.predictor_max_update) {
                ++stats.rejected_steps;
                ++retries;
                step = clamp_step(trial_step * opt.step_shrink);
                trial_step = std::min(step, target_time - t);
                continue;
            }

            if (!corrector(t + trial_step, candidate, correction_norm, residual_norm)) {
                ++stats.rejected_steps;
                ++stats.newton_failures;
                ++retries;
                step = clamp_step(trial_step * opt.step_shrink);
                trial_step = std::min(step, target_time - t);
                continue;
            }

            if (correction_norm > opt.newton_max_update) {
                ++stats.rejected_steps;
                ++retries;
                step = clamp_step(trial_step * opt.step_shrink);
                trial_step = std::min(step, target_time - t);
                continue;
            }

            sol = candidate;
            t += trial_step;
            ++stats.iterations;
            stats.final_residual = residual_norm;
            stats.final_time = t;
            accepted_step = true;

            if (predictor_norm < opt.predictor_small_update && correction_norm < opt.newton_tol) {
                step = clamp_step(step * opt.step_grow);
            } else if (correction_norm > opt.newton_tol) {
                step = clamp_step(step * opt.step_shrink);
            }

            stats.final_step_size = step;

            if (opt.debug_output) {
                std::cout << "t: " << t
                          << " step: " << trial_step
                          << " predictor_norm: " << predictor_norm
                          << " residual_norm: " << residual_norm << std::endl;
            }

            break;
        }

        if (!accepted_step) {
            stats.success = false;
            return stats;
        }
    }

    if (t + opt.min_step_size < target_time) {
        stats.success = false;
    }

    return stats;

};


} // namespace poselib

#endif
