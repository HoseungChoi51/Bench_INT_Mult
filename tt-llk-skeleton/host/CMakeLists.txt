cmake_minimum_required(VERSION 3.20)
project(bench_blackhole CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME is not set. Source the TT-Metal setup script first, "
                        "or `export TT_METAL_HOME=/path/to/tt-metal`.")
endif()
set(TT_METAL_HOME $ENV{TT_METAL_HOME})

# Locate the installed tt-metalium CMake config. The build_Release tree
# ships lib/cmake/tt-metalium/tt-metalium-config.cmake which exports the
# TT::Metalium imported target. Note: there is also a stub config at
# build_Release/tt-metalium-config.cmake that references a missing
# Metalium.cmake — point find_package directly at the working one.
foreach(_candidate
    "${TT_METAL_HOME}/build_Release/lib/cmake/tt-metalium"
    "${TT_METAL_HOME}/build/lib/cmake/tt-metalium"
    "${TT_METAL_HOME}/build/install/lib/cmake/tt-metalium")
    if(EXISTS "${_candidate}/tt-metalium-config.cmake")
        set(tt-metalium_DIR "${_candidate}")
        break()
    endif()
endforeach()
if(NOT DEFINED tt-metalium_DIR)
    message(FATAL_ERROR "Could not locate tt-metalium-config.cmake under "
                        "${TT_METAL_HOME}/build_Release/lib/cmake/tt-metalium "
                        "or its peers.")
endif()

# tt-metalium-config.cmake calls find_dependency() for fmt, nlohmann_json, umd,
# spdlog, tt-logger. They live alongside it under lib/cmake/<dep>/. Add the
# parent directory to CMAKE_PREFIX_PATH so each dep's config can be found.
get_filename_component(_TT_LIB_CMAKE_DIR "${tt-metalium_DIR}/.." ABSOLUTE)
list(APPEND CMAKE_PREFIX_PATH "${_TT_LIB_CMAKE_DIR}")
# share/cmake hosts nlohmann_json
list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/build_Release/share/cmake")
# Also point at the parent build_Release for any package that uses
# CONFIG-mode and lives at the build root.
list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/build_Release")

find_package(tt-metalium REQUIRED CONFIG)

add_executable(bench_blackhole main.cpp)
target_link_libraries(bench_blackhole PRIVATE TT::Metalium)

# Inject the absolute kernel directory so main.cpp can pass full paths to
# CreateKernel; this avoids depending on the runtime CWD or
# TT_METAL_KERNEL_PATH search path.
get_filename_component(KERNEL_DIR_ABS "${CMAKE_SOURCE_DIR}/../kernels" ABSOLUTE)
target_compile_definitions(bench_blackhole PRIVATE
    "KERNEL_DIR=\"${KERNEL_DIR_ABS}\"")
