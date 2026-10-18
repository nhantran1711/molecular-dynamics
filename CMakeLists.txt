cmake_minimum_required(VERSION 3.16)
project(md_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default, with optimization flags that matter for MD
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

add_executable(md_sim
    src/main.cpp
    src/System.cpp
    src/Forcefield.cpp
    src/Integrator.cpp
    src/Initialisation.cpp
    src/Thermostat.cpp
    src/CellList.cpp
    src/NeighbourList.cpp
)

target_include_directories(md_sim PRIVATE include)
target_compile_options(md_sim PRIVATE -Wall -Wextra -Wpedantic)

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
    target_link_libraries(md_sim PRIVATE OpenMP::OpenMP_CXX)
else()
    # Still builds, just serially: the pragmas are ignored
    message(WARNING "OpenMP not found, building single-threaded")
    target_compile_options(md_sim PRIVATE -Wno-unknown-pragmas)
endif()
