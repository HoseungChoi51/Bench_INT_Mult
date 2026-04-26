cmake_minimum_required(VERSION 3.20)
project(bench_blackhole CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate TT-Metal. The standard install exports a CMake config under
# $TT_METAL_HOME/build/install/cmake; we use the env var to find it.
if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME is not set. Source the TT-Metal setup script first.")
endif()
set(TT_METAL_HOME $ENV{TT_METAL_HOME})

# This list is conservative — adjust to match your installed layout. Newer
# TT-Metal versions ship a single `tt_metal::tt_metal` imported target;
# older ones ship the headers + libraries split.
list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/build/install" "${TT_METAL_HOME}")
find_package(tt-metalium QUIET)
if(NOT tt-metalium_FOUND)
    # Fallback: include + link by hand. Edit these paths to match your install.
    include_directories(
        ${TT_METAL_HOME}/tt_metal
        ${TT_METAL_HOME}/tt_metal/include
        ${TT_METAL_HOME}/tt_metal/api
        ${TT_METAL_HOME}/tt_metal/hw/inc
    )
    link_directories(${TT_METAL_HOME}/build/lib)
endif()

add_executable(bench_blackhole main.cpp)

if(tt-metalium_FOUND)
    target_link_libraries(bench_blackhole PRIVATE tt-metalium::tt-metalium)
else()
    target_link_libraries(bench_blackhole PRIVATE tt_metal device)  # adjust per your install
endif()

# Kernels live next to the host but are dispatched at runtime; we copy them
# to the build dir so the binary can find them via a known relative path.
file(GLOB KERNEL_SOURCES "${CMAKE_SOURCE_DIR}/../kernels/*.cpp")
foreach(K ${KERNEL_SOURCES})
    get_filename_component(NAME ${K} NAME)
    configure_file(${K} ${CMAKE_BINARY_DIR}/kernels/${NAME} COPYONLY)
endforeach()
