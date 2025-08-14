# Regular benchmark executable
add_executable(benchmark
    benchmark.cc
    problem_generator.cc
)

target_link_libraries(benchmark PRIVATE PoseLib::PoseLib Eigen3::Eigen)

# Fisheye benchmark executable
add_executable(benchmark_fisheye
    benchmark_fisheye.cc
    problem_generator.cc
)

target_link_libraries(benchmark_fisheye PRIVATE PoseLib::PoseLib Eigen3::Eigen)

add_executable(benchmark_debugHC
    benchmark_debugHC.cc
    problem_generator.cc
)

target_link_libraries(benchmark_debugHC PRIVATE PoseLib::PoseLib Eigen3::Eigen)

install(
  TARGETS benchmark benchmark_fisheye benchmark_debugHC
  RUNTIME DESTINATION bin
  COMPONENT "${INSTALL_BIN_DIR}"
)

# Compilation options
set_compile_flags(benchmark)
set_compile_flags(benchmark_fisheye)
set_compile_flags(benchmark_debugHC)
