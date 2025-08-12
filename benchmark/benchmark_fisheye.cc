#include "benchmark.h"

#include "problem_generator.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace poselib {

template <typename Solver> BenchmarkResult benchmark(int n_problems, const ProblemOptions &options, double tol = 1e-6) {

    std::vector<AbsolutePoseProblemInstance> problem_instances;
    generate_abspose_problems(n_problems, &problem_instances, options);

    BenchmarkResult result;
    result.instances_ = n_problems;
    result.name_ = Solver::name();
    if (options.additional_name_ != "") {
        result.name_ += options.additional_name_;
    }
    result.options_ = options;
    std::cout << "Running benchmark: " << result.name_ << std::flush;

    // Run benchmark where we check solution quality
    for (const AbsolutePoseProblemInstance &instance : problem_instances) {
        CameraPoseVector solutions;

        int sols = Solver::solve(instance, &solutions);

        double pose_error = std::numeric_limits<double>::max();

        result.solutions_ += sols;
        // std::cout << "\nGt: " << instance.pose_gt.R() << "\n"<< instance.pose_gt.t << "\n";
        // std::cout << "gt valid = " << Solver::validator::is_valid(instance, instance.pose_gt, 1.0, tol) << "\n";
        for (const CameraPose &pose : solutions) {
            if (Solver::validator::is_valid(instance, pose, 1.0, tol))
                result.valid_solutions_++;
            // std::cout << "Pose: " << pose.R() << "\n" << pose.t << "\n";
            pose_error = std::min(pose_error, Solver::validator::compute_pose_error(instance, pose, 1.0));
        }
        if (pose_error < tol)
            result.found_gt_pose_++;
    }

    std::vector<long> runtimes;
    CameraPoseVector solutions;
    for (int iter = 0; iter < 10; ++iter) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (const AbsolutePoseProblemInstance &instance : problem_instances) {
            solutions.clear();
            Solver::solve(instance, &solutions);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    }

    std::sort(runtimes.begin(), runtimes.end());
    result.runtime_ns_ = runtimes[runtimes.size() / 2];
    std::cout << "\r                                                                                \r";
    return result;
}

template <typename Solver>
BenchmarkResult benchmark_w_extra(int n_problems, const ProblemOptions &options, double tol = 1e-6) {

    std::vector<AbsolutePoseProblemInstance> problem_instances;
    generate_abspose_problems(n_problems, &problem_instances, options);

    BenchmarkResult result;
    result.instances_ = n_problems;
    result.name_ = Solver::name();
    if (options.additional_name_ != "") {
        result.name_ += options.additional_name_;
    }
    result.options_ = options;
    std::cout << "Running benchmark: " << result.name_ << std::flush;

    // Run benchmark where we check solution quality
    for (const AbsolutePoseProblemInstance &instance : problem_instances) {
        CameraPoseVector solutions;
        std::vector<double> extra;

        int sols = Solver::solve(instance, &solutions, &extra);

        double pose_error = std::numeric_limits<double>::max();

        result.solutions_ += sols;
        for (size_t k = 0; k < solutions.size(); ++k) {
            if (Solver::validator::is_valid(instance, solutions[k], extra[k], tol))
                result.valid_solutions_++;
            if (options.focalError_) {
                pose_error = std::min(pose_error, Solver::validator::compute_pose_error(instance, solutions[k], extra[k]));}
            else {
                pose_error = std::min(pose_error, Solver::validator::compute_pose_error(instance, solutions[k]));
            }
        }

        if (pose_error < tol)
            result.found_gt_pose_++;
    }

    std::vector<long> runtimes;
    CameraPoseVector solutions;
    std::vector<double> extra;
    for (int iter = 0; iter < 10; ++iter) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (const AbsolutePoseProblemInstance &instance : problem_instances) {
            solutions.clear();
            extra.clear();

            Solver::solve(instance, &solutions, &extra);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    }

    std::sort(runtimes.begin(), runtimes.end());
    result.runtime_ns_ = runtimes[runtimes.size() / 2];
    std::cout << "\r                                                                                \r";
    return result;
}

// NEW for demo test p4pfr with 1D division radial distortion model (pinhole)
template <typename Solver>
BenchmarkResult benchmark_w_extra2(int n_problems, const ProblemOptions &options, double tol = 1e-6) {

    std::vector<AbsolutePoseProblemInstance> problem_instances;
    generate_abspose_problems(n_problems, &problem_instances, options);

    BenchmarkResult result;
    result.instances_ = n_problems;
    result.name_ = Solver::name();
    if (options.additional_name_ != "") {
        result.name_ += options.additional_name_;
    }
    result.options_ = options;
    std::cout << "Running benchmark: " << result.name_ << std::flush;

    // Run benchmark where we check solution quality
    for (const AbsolutePoseProblemInstance &instance : problem_instances) {
        CameraPoseVector solutions;
        std::vector<double> focals;
        std::vector<double> ks;

        int sols = Solver::solve(instance, &solutions, &focals, &ks);

        double pose_error = std::numeric_limits<double>::max();

        result.solutions_ += sols;
        for (size_t k = 0; k < solutions.size(); ++k) {
            if (Solver::validator::is_valid(instance, solutions[k], focals[k], ks[k], tol))
                result.valid_solutions_++;
            pose_error = std::min(pose_error, Solver::validator::compute_pose_error(instance, solutions[k], focals[k], ks[k]));
        }

        if (pose_error < tol)
            result.found_gt_pose_++;
    }

    std::vector<long> runtimes;
    CameraPoseVector solutions;
    std::vector<double> focals;
    std::vector<double> ks;
    for (int iter = 0; iter < 10; ++iter) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (const AbsolutePoseProblemInstance &instance : problem_instances) {
            solutions.clear();
            focals.clear();
            ks.clear();

            Solver::solve(instance, &solutions, &focals, &ks);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    }

    std::sort(runtimes.begin(), runtimes.end());
    result.runtime_ns_ = runtimes[runtimes.size() / 2];
    std::cout << "\r                                                                                \r";
    return result;
}

} // namespace poselib

void print_runtime(double runtime_ns) {
    if (runtime_ns < 1e3) {
        std::cout << runtime_ns << " ns";
    } else if (runtime_ns < 1e6) {
        std::cout << runtime_ns / 1e3 << " us";
    } else if (runtime_ns < 1e9) {
        std::cout << runtime_ns / 1e6 << " ms";
    } else {
        std::cout << runtime_ns / 1e9 << " s";
    }
}

void display_result(const std::vector<poselib::BenchmarkResult> &results) {
    // Print PoseLib version and buidling type
    std::cout << "\n" << poselib_info() << "\n\n";

    int w = 13;
    // display header
    std::cout << std::setw(6 * w) << "Solver";
    std::cout << std::setw(w) << "Solutions";
    std::cout << std::setw(w) << "Valid";
    std::cout << std::setw(w) << "GT found";
    std::cout << std::setw(w) << "Runtime"
              << "\n";
    for (int i = 0; i < w * 6; ++i)
        std::cout << "-";
    std::cout << "\n";

    int prec = 6;

    for (const poselib::BenchmarkResult &result : results) {
        double num_tests = static_cast<double>(result.instances_);
        double solutions = result.solutions_ / num_tests;
        double valid_sols = result.valid_solutions_ / static_cast<double>(result.solutions_) * 100.0;
        double gt_found = result.found_gt_pose_ / num_tests * 100.0;
        double runtime_ns = result.runtime_ns_ / num_tests;

        std::cout << std::setprecision(prec) << std::setw(6 * w) << result.name_;
        std::cout << std::setprecision(prec) << std::setw(w) << solutions;
        std::cout << std::setprecision(prec) << std::setw(w) << valid_sols;
        std::cout << std::setprecision(prec) << std::setw(w) << gt_found;
        std::cout << std::setprecision(prec) << std::setw(w - 3);
        print_runtime(runtime_ns);
        std::cout << "\n";
    }
}


void save_result(const std::vector<poselib::BenchmarkResult> &results) {

    std::string filename = "benchmark_result.txt";
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
 
    const int w_name = 30; // Wider column for the name
    const int w_data = 15; // Column width for numerical data
    const int prec = 4;    // Precision for floating-point numbers

    file << std::left // Left-align text
         << std::setw(w_name) << "Solver Name"
         << std::right // Right-align numbers
         << std::setw(w_data) << "Avg Sols"
         << std::setw(w_data) << "Valid Sols %"
         << std::setw(w_data) << "GT Found %"
         << std::setw(w_data) << "Avg Time (ns)" << '\n';
    
    file << std::string(w_name + 4 * w_data, '-') << '\n'; // A separator line

    file << std::fixed << std::setprecision(prec);

    for (const poselib::BenchmarkResult &result : results) {
        // Avoid division by zero if there are no test instances
        if (result.instances_ == 0) continue;

        double num_tests = static_cast<double>(result.instances_);
        double solutions = result.solutions_ / num_tests;

        // CRITICAL: Handle potential division by zero if no solutions were found
        double valid_sols = (result.solutions_ > 0)
                                ? (result.valid_solutions_ / static_cast<double>(result.solutions_) * 100.0)
                                : 0.0;

        double gt_found = result.found_gt_pose_ / num_tests * 100.0;
        double runtime_ns = result.runtime_ns_ / num_tests;

        // --- FIX: Write to `file` instead of `std::cout` ---
        file << std::left
             << std::setw(w_name) << result.name_
             << std::right // Switch back to right-align for numbers
             << std::setw(w_data) << solutions
             << std::setw(w_data) << valid_sols
             << std::setw(w_data) << gt_found
             << std::setw(w_data) << runtime_ns
             << '\n'; // Use '\n' for a new line
    }

    std::cout << "Benchmark results successfully saved to " << filename << std::endl;
}

int main() {

    std::vector<poselib::BenchmarkResult> results;

    poselib::ProblemOptions options;
    // options.camera_fov_ = 45; // Narrow
    options.camera_fov_ = 75; // Medium
    // options.camera_fov_ = 120; // Wide

    double tol = 1e-6;

    // P3P
    // poselib::ProblemOptions p3p_opt = options;
    // p3p_opt.n_point_point_ = 3;
    // p3p_opt.n_point_line_ = 0;
    // results.push_back(poselib::benchmark<poselib::SolverP3P>(1e5, p3p_opt, tol));
    // results.push_back(poselib::benchmark<poselib::SolverP3P_ding>(1e5, p3p_opt, tol));

    // // P35Pf
    // poselib::ProblemOptions p35pf_opt = options;
    // p35pf_opt.n_point_point_ = 4;
    // p35pf_opt.n_point_line_ = 0;
    // p35pf_opt.unknown_focal_ = true;
    // results.push_back(poselib::benchmark_w_extra<poselib::SolverP35PF>(1e4, p35pf_opt, tol));

    // NEW for demo test p4pfr
    poselib::ProblemOptions p4pfr_opt = options;
    p4pfr_opt.n_point_point_ = 4;
    p4pfr_opt.n_point_line_ = 0;
    p4pfr_opt.unknown_focal_ = true;
    p4pfr_opt.unknown_rd_ = true;
    results.push_back(poselib::benchmark_w_extra2<poselib::SolverP4PFr>(1e4, p4pfr_opt, tol*100));

    // NEW for demo test p4pfr_fisheye_LM
    poselib::ProblemOptions p4pfr_fisheye_LM_opt = options;
    p4pfr_fisheye_LM_opt.n_point_point_ = 4;
    p4pfr_fisheye_LM_opt.n_point_line_ = 0;
    p4pfr_fisheye_LM_opt.unknown_focal_ = true;
    p4pfr_fisheye_LM_opt.unknown_rd_ = true;
    results.push_back(poselib::benchmark_w_extra2<poselib::SolverP4PFr_Fisheye_LM>(1e4, p4pfr_fisheye_LM_opt, tol*100));
    
    // NEW for P4PFr Fisheye camera resectioning
    poselib::ProblemOptions p4pfr_fisheye_opt = options;
    p4pfr_fisheye_opt.n_point_point_ = 4;
    p4pfr_fisheye_opt.n_point_line_ = 0;
    p4pfr_fisheye_opt.unknown_focal_ = true;
    p4pfr_fisheye_opt.focalError_ = false;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP4PFr_Fisheye>(1e4, p4pfr_fisheye_opt, tol*1e4));

    // NEW for P4Pf Fisheye camera resectioning with unknown focal
    poselib::ProblemOptions p4pf_fisheye_opt = options;
    p4pf_fisheye_opt.n_point_point_ = 4;
    p4pf_fisheye_opt.n_point_line_ = 0;
    p4pf_fisheye_opt.unknown_focal_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP4PF_Fisheye>(1, p4pf_fisheye_opt, tol*1e2));

    // NEW for P4Pf Fisheye camera resectioning with unknown focal using Depth
    // small perturbation from gt pose as initial guess
    poselib::ProblemOptions p4pf_fisheye_depth_opt = options;
    p4pf_fisheye_depth_opt.n_point_point_ = 4;
    p4pf_fisheye_depth_opt.n_point_line_ = 0;
    p4pf_fisheye_depth_opt.unknown_focal_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP4PF_Fisheye_depth_small_perturbation>(1e4, p4pf_fisheye_depth_opt, tol*1e4));

    // random initial
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP4PF_Fisheye_depth_random_initial>(1e4, p4pf_fisheye_depth_opt, tol*1e4));

    // p4pfr as initial
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP4PF_Fisheye_depth_p4pfr_initial>(1e4, p4pf_fisheye_depth_opt, tol*1e4));

    display_result(results);
    save_result(results);

    return 0;
}
