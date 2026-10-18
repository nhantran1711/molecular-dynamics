#include "System.hpp"
#include "Parallel.hpp"
#include <cmath>
#include <cstring>
#include <type_traits>
#include <utility>

// Split the particle records into one array per component
System::System(const std::vector<Particle>& particles, double boxLength)
    : boxLength_(boxLength), inverseBoxLength_(1.0 / boxLength) {
    const std::size_t n = particles.size();
    positions_.assign(positionStride * n, 0.0);
    for (int k = 0; k < 3; ++k) {
        velocity_[k].resize(n);
        force_[k].resize(n);
    }
    mass_.resize(n);
    id_.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) {
            positions_[positionStride * i + k] = particles[i].position[k];
            velocity_[k][i] = particles[i].velocity[k];
            force_[k][i] = particles[i].force[k];
        }
        mass_[i] = particles[i].mass;
        id_[i] = i;
    }
}

Particle System::particle(std::size_t i) const {
    Particle p;
    p.position = positionOf(i);
    p.velocity = velocityOf(i);
    p.force = forceOf(i);
    p.mass = mass_[i];
    p.id = id_[i];
    return p;
}

// Permute every component array into the given order
void System::reorder(std::span<const std::size_t> order) {
    auto permute = [&order](auto& values) {
        std::remove_reference_t<decltype(values)> sorted(values.size());
        for (std::size_t k = 0; k < order.size(); ++k) {
            sorted[k] = values[order[k]];
        }
        values = std::move(sorted);
    };
    std::vector<double> sortedPositions(positions_.size());
    for (std::size_t k = 0; k < order.size(); ++k) {
        for (std::size_t c = 0; c < positionStride; ++c) {
            sortedPositions[positionStride * k + c] = positions_[positionStride * order[k] + c];
        }
    }
    positions_ = std::move(sortedPositions);

    for (int k = 0; k < 3; ++k) {
        permute(velocity_[k]);
        permute(force_[k]);
    }
    permute(mass_);
    permute(id_);
}

// Calculate the kinetic energy of the system
double System::kineticEnergy() const {
    const double* vx = velocity_[0].data();
    const double* vy = velocity_[1].data();
    const double* vz = velocity_[2].data();
    const double* m = mass_.data();

    // Fixed summation order, since the thermostat feeds this back into the trajectory
    return deterministicSum(
        size(), [=](std::size_t i) { return 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]); });
}

// Temperature from equipartition: KE = (dof / 2) * T, with 3 dof removed for zero total momentum
double System::temperature() const {
    double dof = 3.0 * static_cast<double>(size()) - 3.0;
    return 2.0 * kineticEnergy() / dof;
}

// FNV-1a over the raw bytes of every position and velocity, in particle order
std::uint64_t System::stateHash() const {
    std::uint64_t hash = 14695981039346656037ull;
    auto mix = [&hash](double value) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned byte = 0; byte < 8; ++byte) {
            hash ^= (bits >> (8u * byte)) & 0xffu;
            hash *= 1099511628211ull;
        }
    };

    // Same order as the original array-of-structs layout (x, vx, y, vy, z, vz per particle), so hashes can be
    // compared across the two layouts
    for (std::size_t i = 0; i < size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            mix(positions_[positionStride * i + k]);
            mix(velocity_[k][i]);
        }
    }
    return hash;
}
