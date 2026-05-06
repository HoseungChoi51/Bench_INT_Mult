cmake_minimum_required(VERSION 3.20)
project(bench_int8_tuned_mcast CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME is not set. Source the TT-Metal setup script first, "
                        "or `export TT_METAL_HOME=/path/to/tt-metal`.")
endif()
set(TT_METAL_HOME $ENV{TT_METAL_HOME})

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

get_filename_component(_TT_LIB_CMAKE_DIR "${tt-metalium_DIR}/.." ABSOLUTE)
list(APPEND CMAKE_PREFIX_PATH "${_TT_LIB_CMAKE_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/build_Release/share/cmake")
list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/build_Release")

find_package(tt-metalium REQUIRED CONFIG)

add_executable(bench_int8_tuned_mcast main.cpp)
target_link_libraries(bench_int8_tuned_mcast PRIVATE TT::Metalium)
