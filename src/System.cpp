#pragma once

#include "Particle.hpp"
#include <vector>

class System {
public:
    System(std::vector<Particle> particles, double boxLength);

    // Get vector of particles
    std::vector<Particle>& getParticles() {
        return particles_;
    };
    const std::vector<Particle>& getParticles() const {
        return particles_;
    };

private:
    double boxLength_;
    std::vector<Particle> particles_;
};