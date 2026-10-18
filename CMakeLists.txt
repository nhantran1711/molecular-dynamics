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
)

target_include_directories(md_sim PRIVATE include)
target_compile_options(md_sim PRIVATE -Wall -Wextra -Wpedantic)
