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

    // Get box length
    double getBoxLength() const {
        return boxLength_;
    };


    // Using PBC, find the minimum image distance between two particles
    std::array<double, 3> minimumImageDistance(const Particle& p1, const Particle& p2) const;

    // Wrap the position of a particle back into the simulation box
    void wrapPosition(Particle& p) const;

    // Calculate the total kinetic energy of the system
    double kineticEnergy() const;
private:
    double boxLength_;
    std::vector<Particle> particles_;
};