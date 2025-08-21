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
        
        for (const CameraPose &pose : solutions) {
            if (Solver::validator::is_valid(instance, pose, 1.0, tol))
                result.valid_solutions_++;
                
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

    std::vector<long long> runtimes;
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


template <typename Solver>
BenchmarkResult benchmark_w_extra_save_result(int n_problems, const ProblemOptions &options, double tol = 1e-6) {

    int fov = options.camera_fov_;

    std::string filename = "results_fov_" + std::to_string(fov) + "/" + Solver::name() + ".txt";
    std::filesystem::create_directories("results_fov_" + std::to_string(fov));

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }

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
        int valid = 0;
        for (size_t k = 0; k < solutions.size(); ++k) {
            if (Solver::validator::is_valid(instance, solutions[k], extra[k], tol)){
                result.valid_solutions_++;
                valid++;
            }

            double RError, tError, fError;
            Solver::validator::compute_pose_error(instance, solutions[k], extra[k], RError, tError, fError);

            file << RError << " " << tError << " " << fError << " ";

            pose_error = std::min(pose_error, Solver::validator::compute_pose_error(instance, solutions[k], extra[k]));
            
        }

        file << valid << std::endl;

        if (pose_error < tol)
            result.found_gt_pose_++;
    }

    file.close();
    std::cout << "\rResults saved to " << filename << std::flush;

    std::vector<long long> runtimes;
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
    std::cout << std::setw(3 * w) << "Solver";
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


int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <fov>" << std::endl;
        return 1;
    }

    int fov = std::stoi(argv[1]);
    std::vector<poselib::BenchmarkResult> results;

    poselib::ProblemOptions options;
    // options.camera_fov_ = 45; // Narrow
    // options.camera_fov_ = 75; // Medium
    // options.camera_fov_ = 120; // Wide
    options.camera_fov_ = fov; // Wide

    double tol = 1e-6;

    // // P3P
    // poselib::ProblemOptions p3p_opt = options;
    // p3p_opt.n_point_point_ = 3;
    // p3p_opt.n_point_line_ = 0;
    // results.push_back(poselib::benchmark<poselib::SolverP3P>(1e5, p3p_opt, tol));
    // results.push_back(poselib::benchmark<poselib::SolverP3P_ding>(1e5, p3p_opt, tol));

    // // Sampling + p3p
    // poselib::ProblemOptions fisheye_3pts_opt = options;
    // fisheye_3pts_opt.n_point_point_ = 3;
    // fisheye_3pts_opt.unknown_focal_ = true;
    // results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_P3P>(1e4, fisheye_3pts_opt, tol*1e4));
    // results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_P3P_LM>(1e4, fisheye_3pts_opt, tol*1e4));

    // NEW for Fisheye camera resectioning with unknown focal
    poselib::ProblemOptions fisheye_4pts_opt = options;
    fisheye_4pts_opt.n_point_point_ = 4;
    fisheye_4pts_opt.unknown_focal_ = true;

    // // other tests
    // // random initial + LM refiner
    // results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_random_LM>(1e4, fisheye_4pts_opt, tol*1e4));

    // // gt debug + HC pose refiner
    // results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_pose_gtDebug>(1e4, fisheye_4pts_opt, tol*1e4));

    // // small perturbation from gt pose + HC depth refiner
    // results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_depth_gtDebug>(1e4, fisheye_4pts_opt, tol*1e4));

    // // random initial + HC depth refiner
    // results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_depth_random>(1e4, fisheye_4pts_opt, tol*1e4));



    // p4pfr
    // p4pfr + no refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_P4PFr>(1e4, fisheye_4pts_opt, tol*1e4));

    // p4pfr + LM refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_P4PFr_LM>(1e4, fisheye_4pts_opt, tol*1e4));

    // p4pfr as initial + HC pose refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_pose_p4pfr>(1e4, fisheye_4pts_opt, tol*1e4));
    
    // p4pfr as initial + HC depth refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_depth_p4pfr>(1e4, fisheye_4pts_opt, tol*1e4));


    // p3.5pf
    // p3.5pf + no refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_P35PF>(1e4, fisheye_4pts_opt, tol*1e4));

    // p3.5pf + LM refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_P35PF_LM>(1e4, fisheye_4pts_opt, tol*1e4));

    // p3.5pf + HC pose refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_pose_p35pf>(1e4, fisheye_4pts_opt, tol*1e4));

    // p3.5pf + HC depth refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_depth_p35pf>(1e4, fisheye_4pts_opt, tol*1e4));


    // p4pf
    // p4pf + no refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_P4PF>(1e4, fisheye_4pts_opt, tol*1e4));

    // p4pf + LM refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_P4PF_LM>(1e4, fisheye_4pts_opt, tol*1e4));

    // p4pf + HC pose refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_pose_p4pf>(1e4, fisheye_4pts_opt, tol*1e4));

    // p4pf + HC depth refiner
    results.push_back(poselib::benchmark_w_extra_save_result<poselib::SolverFisheye_HC_depth_p4pf>(1e4, fisheye_4pts_opt, tol*1e4));

    display_result(results);

    return 0;
}
