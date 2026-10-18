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
    std::array<double, 3> minimumImageDistance(const std::array<double, 3>& ri, const std::array<double, 3>& rj) const;

    // Wrap the position of a particle back into the simulation box
    void wrapPosition(std::array<double, 3>& r) const;

    // Calculate the total kinetic energy of the system
    double kineticEnergy() const;

    // Calculate the instantaneous temperature (k_B = 1) from the kinetic energy
    double temperature() const;
private:
    double boxLength_;
    std::vector<Particle> particles_;
};