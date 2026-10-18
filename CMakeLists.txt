cmake_minimum_required(VERSION 3.16)
project(md_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default, with optimization flags that matter for MD
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Sanitizer builds for CI, e.g. -DMD_SANITIZE=address,undefined or -DMD_SANITIZE=thread.
# Applied to every target (including GoogleTest), so it must come before any target is defined
set(MD_SANITIZE "" CACHE STRING "Comma-separated sanitizers to enable (address, undefined, thread)")
if(MD_SANITIZE)
    add_compile_options(-fsanitize=${MD_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all -g)
    add_link_options(-fsanitize=${MD_SANITIZE})
endif()

# Simulation engine, shared by the command-line driver and the visualiser
add_library(md_core STATIC
    src/System.cpp
    src/Forcefield.cpp
    src/Integrator.cpp
    src/Initialisation.cpp
    src/Thermostat.cpp
    src/CellList.cpp
    src/NeighbourList.cpp
)

target_include_directories(md_core PUBLIC include)
target_compile_options(md_core PUBLIC -Wall -Wextra -Wpedantic)

# Apple clang ships without an OpenMP runtime, so point CMake at Homebrew's libomp (brew install libomp)
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(md_core PUBLIC OpenMP::OpenMP_CXX)
else()
    # Still builds, just serially: the thread pragmas are ignored, but -fopenmp-simd keeps the SIMD force kernel
    message(WARNING "OpenMP not found, building single-threaded")
    target_compile_options(md_core PUBLIC -fopenmp-simd -Wno-unknown-pragmas)
endif()

add_executable(md_sim src/main.cpp)
target_link_libraries(md_sim PRIVATE md_core)

# Real-time OpenGL visualiser, built only when GLFW is available (brew install glfw / apt install libglfw3-dev)
find_package(glfw3 3.3 CONFIG QUIET)
find_package(OpenGL QUIET)
if(glfw3_FOUND AND OpenGL_FOUND)
    add_executable(md_vis vis/main.cpp vis/Renderer.cpp)
    target_link_libraries(md_vis PRIVATE md_core glfw OpenGL::GL)
    target_compile_definitions(md_vis PRIVATE GL_SILENCE_DEPRECATION)
else()
    message(STATUS "GLFW or OpenGL not found, skipping the md_vis visualiser")
endif()

# Unit tests (GoogleTest is downloaded at configure time). Run with: ctest --test-dir build/release
option(MD_BUILD_TESTS "Build the unit tests" ON)
if(MD_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(md_tests
        tests/test_system.cpp
        tests/test_forcefield.cpp
        tests/test_neighbourlist.cpp
        tests/test_dynamics.cpp
        tests/test_determinism.cpp
    )
    target_link_libraries(md_tests PRIVATE md_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(md_tests DISCOVERY_TIMEOUT 60)
endif()
