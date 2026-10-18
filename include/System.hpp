#pragma once

#include "Particle.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// A system of particles in a cubic box with periodic boundary conditions. The particles are stored in a structure-of-arrays layout for efficient vectorisation and cache usage. The box is defined by its length, and the positions of the particles are wrapped into the box using periodic boundary conditions. The system provides access to the positions, velocities, forces, masses, and ids of the particles, as well as methods for computing distances and energies.
class System {
public:
    // Ids are assigned in input order (0, 1, 2, ...); any ids in the input records are ignored
    System(const std::vector<Particle>& particles, double boxLength);

    std::size_t size() const {
        return mass_.size();
    };

    // Packed positions: particle i is at packedPositions()[positionStride * i + k], k = 0, 1, 2
    static constexpr std::size_t positionStride = 4;
    double* packedPositions() {
        return positions_.data();
    };
    const double* packedPositions() const {
        return positions_.data();
    };

    // Component arrays, k = 0, 1, 2 for x, y, z
    std::span<double> velocity(int k) {
        return velocity_[k];
    };
    std::span<const double> velocity(int k) const {
        return velocity_[k];
    };
    std::span<double> force(int k) {
        return force_[k];
    };
    std::span<const double> force(int k) const {
        return force_[k];
    };
    std::span<const double> mass() const {
        return mass_;
    };

    // Stable particle ids: ids()[i] is the original index of the particle now stored at index i
    std::span<const std::size_t> ids() const {
        return id_;
    };

    // Single-particle access (not for hot loops)
    std::array<double, 3> positionOf(std::size_t i) const {
        const double* r = positions_.data() + positionStride * i;
        return {r[0], r[1], r[2]};
    };
    std::array<double, 3> velocityOf(std::size_t i) const {
        return {velocity_[0][i], velocity_[1][i], velocity_[2][i]};
    };
    std::array<double, 3> forceOf(std::size_t i) const {
        return {force_[0][i], force_[1][i], force_[2][i]};
    };
    void setPosition(std::size_t i, const std::array<double, 3>& r) {
        for (int k = 0; k < 3; ++k) {
            positions_[positionStride * i + k] = r[k];
        }
    };
    void setVelocity(std::size_t i, const std::array<double, 3>& v) {
        for (int k = 0; k < 3; ++k) {
            velocity_[k][i] = v[k];
        }
    };
    Particle particle(std::size_t i) const;

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

    // Minimum image distance between particles i and j
    std::array<double, 3> minimumImageDistance(std::size_t i, std::size_t j) const {
        return minimumImageDistance(positionOf(i), positionOf(j));
    };

    // Wrap the position of a particle back into the simulation box
    void wrapPosition(std::array<double, 3>& r) const {
        for (int k = 0; k < 3; ++k) {
            r[k] = wrapCoordinate(r[k]);
        }
    };

    double wrapCoordinate(double x) const {
        return x - boxLength_ * std::floor(x * inverseBoxLength_);
    };

    // Permute the particles so that new particle k is old particle order[k].
    // Indices are not stable identities (ids are): anything holding indices must be rebuilt afterwards
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

    std::vector<double> positions_;
    std::array<std::vector<double>, 3> velocity_;
    std::array<std::vector<double>, 3> force_;
    std::vector<double> mass_;
    std::vector<std::size_t> id_;
};
