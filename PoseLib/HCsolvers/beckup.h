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

// -----------------------------------------------------------------------------
// Options
// -----------------------------------------------------------------------------

struct HCOptions {
    // Homotopy is parameterized on t in [0, target_time]
    double step_size = 0.05;
    double target_time = 1.0;
    double min_step_size = 1e-3;
    double max_step_size = 0.15;

    // Heuristic growth/shrink (secondary controller)
    double step_grow = 1.5;
    double step_shrink = 0.5;
    size_t max_step_halvings = 6;

    // Predictor controls
    double predictor_max_update = std::numeric_limits<double>::infinity();
    double predictor_small_update = 1e-3;

    // Newton controls
    double newton_tol = 1e-8;
    double newton_max_update = 1.0;
    double newton_max_residual = 5e-3;
    size_t newton_iter = 5;

    // Max outer iterations (default heuristic)
    size_t max_iterations = static_cast<size_t>(target_time / step_size) + 5;

    // Predictor & solver behaviour
    bool forth_predictor = true;   // RK4 predictor
    bool adaptive_flag   = true;   // use LU/QR rather than normal equations

    // RK-based local error control in t
    double rk_tol        = 1e-4;   // target local error (state increment)
    double rk_safety     = 0.9;    // safety factor
    double rk_min_factor = 0.2;    // min multiplicative factor
    double rk_max_factor = 5.0;    // max multiplicative factor
    bool   use_rk_error_control = true;

    bool debug_output = false;
};

// -----------------------------------------------------------------------------
// compute_dx: vector Solution
// -----------------------------------------------------------------------------

template<typename Problem, typename Solution, typename Poly, typename Jacobian>
bool compute_dx(double t, Problem &problem, const Solution &sol, Solution &dx) {
    Poly polysF, polysG;
    Jacobian JF, JG;

    problem.compute_PolysandJacobian(sol, problem.x, polysF, JF);
    problem.compute_PolysandJacobian(sol, problem.x_simulated, polysG, JG);

    Jacobian Jx = t * JF + (1.0 - t) * JG;
    Poly t_grad = polysF - polysG;

    if (Jx.rows() == Jx.cols()) {
        dx = -Jx.lu().solve(t_grad);
    } else {
        dx = -Jx.colPivHouseholderQr().solve(t_grad);
    }
    return true;
}

// -----------------------------------------------------------------------------
// compute_dx: Image + dX
// -----------------------------------------------------------------------------

template<typename Problem, typename dX, typename Poly, typename Jacobian>
bool compute_dx(double t, Problem &problem, const Image &sol, dX &dx) {
    Poly polysF, polysG;
    Jacobian JF, JG;

    problem.compute_PolysandJacobian(sol, problem.x, polysF, JF);
    problem.compute_PolysandJacobian(sol, problem.x_simulated, polysG, JG);

    Jacobian Jx = t * JF + (1.0 - t) * JG;
    Poly t_grad = polysF - polysG;

    if (Jx.rows() == Jx.cols()) {
        dx = -Jx.lu().solve(t_grad);
    } else {
        dx = -Jx.colPivHouseholderQr().solve(t_grad);
    }
    return true;
}

// -----------------------------------------------------------------------------
// RK4 predictor with embedded error: vector Solution
// -----------------------------------------------------------------------------

template<typename Problem, typename Solution, typename Poly, typename Jacobian>
bool rk4_predict_with_error(
    Problem &problem,
    const Solution &base_state,
    double current_t,
    double step,
    Solution &out_state,
    double &update_norm,
    double &err_est)
{
    // one big step
    Solution k1, k2, k3, k4;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t, problem, base_state, k1)) return false;

    Solution tmp1 = base_state + (step * 0.5) * k1;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + step * 0.5, problem, tmp1, k2)) return false;

    Solution tmp2 = base_state + (step * 0.5) * k2;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + step * 0.5, problem, tmp2, k3)) return false;

    Solution tmp3 = base_state + step * k3;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + step, problem, tmp3, k4)) return false;

    Solution delta_big = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (step / 6.0);
    Solution y_big = base_state + delta_big;

    // two half-steps
    double h2 = step * 0.5;

    // first half
    Solution k1a, k2a, k3a, k4a;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t, problem, base_state, k1a)) return false;

    Solution tmp1a = base_state + (h2 * 0.5) * k1a;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + h2 * 0.5, problem, tmp1a, k2a)) return false;

    Solution tmp2a = base_state + (h2 * 0.5) * k2a;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + h2 * 0.5, problem, tmp2a, k3a)) return false;

    Solution tmp3a = base_state + h2 * k3a;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + h2, problem, tmp3a, k4a)) return false;

    Solution delta_small1 = (k1a + 2.0 * k2a + 2.0 * k3a + k4a) * (h2 / 6.0);
    Solution mid_state = base_state + delta_small1;

    // second half
    Solution k1b, k2b, k3b, k4b;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + h2, problem, mid_state, k1b)) return false;

    Solution tmp1b = mid_state + (h2 * 0.5) * k1b;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + h2 + h2 * 0.5, problem, tmp1b, k2b)) return false;

    Solution tmp2b = mid_state + (h2 * 0.5) * k2b;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + h2 + h2 * 0.5, problem, tmp2b, k3b)) return false;

    Solution tmp3b = mid_state + h2 * k3b;
    if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t + step, problem, tmp3b, k4b)) return false;

    Solution delta_small2 = (k1b + 2.0 * k2b + 2.0 * k3b + k4b) * (h2 / 6.0);
    Solution y_small = mid_state + delta_small2;

    out_state = y_small;
    update_norm = (out_state - base_state).norm();
    err_est = (y_small - y_big).norm();

    return std::isfinite(update_norm) && std::isfinite(err_est);
}

// -----------------------------------------------------------------------------
// RK4 predictor with embedded error: Image + dX
// -----------------------------------------------------------------------------

template<typename Problem, typename dX, typename Poly, typename Jacobian>
bool rk4_predict_with_error_image(
    Problem &problem,
    const Image &base_state,
    double current_t,
    double step,
    Image &out_state,
    double &update_norm,
    double &err_est)
{
    // one big step
    dX k1, k2, k3, k4;

    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t, problem, base_state, k1))
        return false;

    Image tmp1 = problem.step(base_state, (step * 0.5) * k1);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + step * 0.5, problem, tmp1, k2))
        return false;

    Image tmp2 = problem.step(base_state, (step * 0.5) * k2);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + step * 0.5, problem, tmp2, k3))
        return false;

    Image tmp3 = problem.step(base_state, step * k3);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + step, problem, tmp3, k4))
        return false;

    dX delta_big = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (step / 6.0);
    Image y_big = problem.step(base_state, delta_big);

    // two half-steps
    double h2 = step * 0.5;

    // first half
    dX k1a, k2a, k3a, k4a;
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t, problem, base_state, k1a))
        return false;

    Image tmp1a = problem.step(base_state, (h2 * 0.5) * k1a);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + h2 * 0.5, problem, tmp1a, k2a))
        return false;

    Image tmp2a = problem.step(base_state, (h2 * 0.5) * k2a);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + h2 * 0.5, problem, tmp2a, k3a))
        return false;

    Image tmp3a = problem.step(base_state, h2 * k3a);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + h2, problem, tmp3a, k4a))
        return false;

    dX delta_small1 = (k1a + 2.0 * k2a + 2.0 * k3a + k4a) * (h2 / 6.0);
    Image mid_state = problem.step(base_state, delta_small1);

    // second half
    dX k1b, k2b, k3b, k4b;
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + h2, problem, mid_state, k1b))
        return false;

    Image tmp1b = problem.step(mid_state, (h2 * 0.5) * k1b);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + h2 + h2 * 0.5, problem, tmp1b, k2b))
        return false;

    Image tmp2b = problem.step(mid_state, (h2 * 0.5) * k2b);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + h2 + h2 * 0.5, problem, tmp2b, k3b))
        return false;

    Image tmp3b = problem.step(mid_state, h2 * k3b);
    if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t + step, problem, tmp3b, k4b))
        return false;

    dX delta_small2 = (k1b + 2.0 * k2b + 2.0 * k3b + k4b) * (h2 / 6.0);
    dX delta_small_total = delta_small1 + delta_small2;
    Image y_small = problem.step(mid_state, delta_small2);

    out_state = y_small;
    update_norm = delta_small_total.norm();
    err_est = (delta_small_total - delta_big).norm();

    return std::isfinite(update_norm) && std::isfinite(err_est);
}

// -----------------------------------------------------------------------------
// HC_impl: vector Solution
// -----------------------------------------------------------------------------

template<typename Problem, typename Solution, typename Poly, typename Jacobian>
HCStats HC_impl(Problem &problem, const HCOptions &opt, Solution &sol)
{
    HCStats stats;

    auto clamp_step = [&](double value) {
        return std::max(opt.min_step_size, std::min(opt.max_step_size, value));
    };

    auto predictor = [&](const Solution &base_state, double current_t, double step,
                         Solution &out_state, double &update_norm, double &err_est) -> bool {
        if (opt.forth_predictor) {
            return rk4_predict_with_error<Problem, Solution, Poly, Jacobian>(
                problem, base_state, current_t, step, out_state, update_norm, err_est);
        } else {
            Solution k1;
            if (!compute_dx<Problem, Solution, Poly, Jacobian>(current_t, problem, base_state, k1))
                return false;
            Solution delta = k1 * step;
            out_state = base_state + delta;
            update_norm = delta.norm();
            err_est = 0.0;
            return std::isfinite(update_norm);
        }
    };

    auto corrector = [&](double current_t, Solution &state,
                         double &dx_norm, double &residual_norm) -> bool {
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
            if (!std::isfinite(residual_norm)) return false;
            if (residual_norm < opt.newton_tol) {
                dx_norm = 0.0;
                return true;
            }

            if (!checkValid(JH)) return false;

            Solution dx_newton;
            if (opt.adaptive_flag) {
                if (JH.rows() == JH.cols()) {
                    dx_newton = -JH.lu().solve(Hpolys);
                } else {
                    dx_newton = -JH.colPivHouseholderQr().solve(Hpolys);
                }
            } else {
                dx_newton = -JH.colPivHouseholderQr().solve(Hpolys);
            }

            dx_norm = dx_newton.norm();
            if (!std::isfinite(dx_norm)) return false;

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
            double err_est = 0.0;

            if (!predictor(sol, t, trial_step, candidate, predictor_norm, err_est) ||
                predictor_norm > opt.predictor_max_update) {
                ++stats.rejected_steps;
                ++retries;
                step = clamp_step(trial_step * opt.step_shrink);
                trial_step = std::min(step, target_time - t);
                continue;
            }

            // RK-based control before Newton
            if (opt.forth_predictor && opt.use_rk_error_control && err_est > 0.0) {
                double tol     = opt.rk_tol;
                double safety  = opt.rk_safety;
                double order   = 4.0;

                double factor = safety * std::pow(tol / err_est, 1.0 / (order + 1.0));
                factor = std::max(opt.rk_min_factor, std::min(opt.rk_max_factor, factor));

                if (err_est > tol) {
                    ++stats.rejected_steps;
                    ++retries;
                    step = clamp_step(trial_step * factor);
                    trial_step = std::min(step, target_time - t);
                    continue;
                } else {
                    step = clamp_step(trial_step * factor);
                }
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

            // accept step
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
                          << " residual_norm: " << residual_norm
                          << " err_est: " << err_est
                          << std::endl;
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
}

// -----------------------------------------------------------------------------
// HC_impl_update_Image: Image + dX
// -----------------------------------------------------------------------------

template<typename Problem, typename dX, typename Poly, typename Jacobian>
HCStats HC_impl_update_Image(Problem &problem, const HCOptions &opt, Image &sol)
{
    HCStats stats;

    const auto clamp_step = [&](double value) {
        return std::max(opt.min_step_size, std::min(opt.max_step_size, value));
    };

    auto predictor = [&](const Image &base_state, double current_t, double step,
                         Image &out_state, double &update_norm, double &err_est) -> bool {
        if (opt.forth_predictor) {
            return rk4_predict_with_error_image<Problem, dX, Poly, Jacobian>(
                problem, base_state, current_t, step, out_state, update_norm, err_est);
        } else {
            dX k1;
            if (!compute_dx<Problem, dX, Poly, Jacobian>(current_t, problem, base_state, k1)) {
                return false;
            }
            dX delta = k1 * step;
            update_norm = delta.norm();
            out_state = problem.step(base_state, delta);
            err_est = 0.0;
            return std::isfinite(update_norm);
        }
    };

    auto corrector = [&](double current_t, Image &state,
                         double &dx_norm, double &residual_norm) -> bool {
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
            if (!std::isfinite(residual_norm)) return false;
            if (residual_norm < opt.newton_tol) {
                dx_norm = 0.0;
                return true;
            }

            if (!checkValid(JH)) return false;

            dX dx_newton;
            if (opt.adaptive_flag) {
                if (JH.rows() == JH.cols()) {
                    dx_newton = -JH.lu().solve(Hpolys);
                } else {
                    dx_newton = -JH.colPivHouseholderQr().solve(Hpolys);
                }
            } else {
                dx_newton = -JH.colPivHouseholderQr().solve(Hpolys);
            }

            dx_norm = dx_newton.norm();
            if (!std::isfinite(dx_norm)) return false;

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
            double err_est = 0.0;

            if (!predictor(sol, t, trial_step, candidate, predictor_norm, err_est) ||
                predictor_norm > opt.predictor_max_update) {
                ++stats.rejected_steps;
                ++retries;
                step = clamp_step(trial_step * opt.step_shrink);
                trial_step = std::min(step, target_time - t);
                continue;
            }

            // RK-based control before Newton
            if (opt.forth_predictor && opt.use_rk_error_control && err_est > 0.0) {
                double tol    = opt.rk_tol;
                double safety = opt.rk_safety;
                double order  = 4.0;

                double factor = safety * std::pow(tol / err_est, 1.0 / (order + 1.0));
                factor = std::max(opt.rk_min_factor, std::min(opt.rk_max_factor, factor));

                if (err_est > tol) {
                    ++stats.rejected_steps;
                    ++retries;
                    step = clamp_step(trial_step * factor);
                    trial_step = std::min(step, target_time - t);
                    continue;
                } else {
                    step = clamp_step(trial_step * factor);
                }
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

            // accept
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
                          << " residual_norm: " << residual_norm
                          << " err_est: " << err_est
                          << std::endl;
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

#endif // POSELIB_HOMOTOPY_CONTINUATION_H_
