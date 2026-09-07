#pragma once

#include <array>

// AoS vs SoA
struct Particle {
    std::array<double, 3> position {0.0, 0.0, 0.0};
    std::array<double, 3> velocity {0.0, 0.0, 0.0};
    std::array<double, 3> force {0.0, 0.0, 0.0};
    
    // std::vector<double> x, y, z, vx, vy, vz, fx, fy, fz;

    double mass {1.0};
};
