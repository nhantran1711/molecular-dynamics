#pragma once

#include <array>
#include <cstddef>

// One particle as a plain record: used to build a System and to read or write single particles.
// Inside System, positions are packed per particle and everything else is one array per component.
struct Particle {
    std::array<double, 3> position {0.0, 0.0, 0.0};
    std::array<double, 3> velocity {0.0, 0.0, 0.0};
    std::array<double, 3> force {0.0, 0.0, 0.0};
    double mass {1.0};
    std::size_t id {0}; // stable identity: survives the re-sorting done on neighbour list rebuilds
};
