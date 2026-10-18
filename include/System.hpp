#pragma once

#include "Particle.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
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

    double getInverseBoxLength() const {
        return inverseBoxLength_;
    };


    // Using PBC, find the minimum image distance between two particles.
    // Defined in the header so it inlines into the hot pair loops in other source files
    std::array<double, 3> minimumImageDistance(const std::array<double, 3>& ri, const std::array<double, 3>& rj) const {
        std::array<double, 3> dr {};

        for (int k = 0; k < 3; ++k) {
            dr[k] = ri[k] - rj[k];

            // PBC
            dr[k] -= boxLength_ * std::round(dr[k] * inverseBoxLength_);
        }
        return dr;
    };

    // Wrap the position of a particle back into the simulation box
    void wrapPosition(std::array<double, 3>& r) const {
        for (int k = 0; k < 3; ++k) {
            r[k] -= boxLength_ * std::floor(r[k] * inverseBoxLength_);
        }
    };

    // Permute the particles so that new particle k is old particle order[k].
    // Particle indices are not stable identities: anything holding indices must be rebuilt afterwards
    void reorder(std::span<const std::size_t> order);

    // Calculate the total kinetic energy of the system
    double kineticEnergy() const;

    // Hash of every position and velocity bit pattern: equal hashes mean bitwise-identical states
    std::uint64_t stateHash() const;

    // Calculate the instantaneous temperature (k_B = 1) from the kinetic energy
    double temperature() const;
private:
    double boxLength_;
    double inverseBoxLength_;
    std::vector<Particle> particles_;
};