#pragma once

#include "Particle.hpp"
#include <vector>

class System {
public:
    System(std::vector<Particle> particles, double boxLength);

    std::vector<Particle>& getParticles() {
        return particles_;
    };

private:
    double boxLength_;
    std::vector<Particle> particles_;
}