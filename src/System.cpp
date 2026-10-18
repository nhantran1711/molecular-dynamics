#include "System.hpp"
#include "Parallel.hpp"
#include <cmath>
#include <cstring>
#include <utility>

// Constructor for the System class
System::System(std::vector<Particle> particles, double boxLength): 
    boxLength_(boxLength), inverseBoxLength_(1.0 / boxLength), particles_(std::move(particles)) {}


// Permute the particles into the given order
void System::reorder(std::span<const std::size_t> order) {
    std::vector<Particle> sorted;
    sorted.reserve(particles_.size());
    for (std::size_t old : order) {
        sorted.push_back(particles_[old]);
    }
    particles_ = std::move(sorted);
}

// Calculate the kinetic energy of the system
double System::kineticEnergy() const {

    // Fixed summation order, since the thermostat feeds this back into the trajectory
    return deterministicSum(particles_.size(), [this](std::size_t i) {
        const Particle& particle = particles_[i];
        double v2 = particle.velocity[0] * particle.velocity[0] + particle.velocity[1] * particle.velocity[1] + particle.velocity[2] * particle.velocity[2];
        return 0.5 * particle.mass * v2;
    });
}

// Temperature from equipartition: KE = (dof / 2) * T, with 3 dof removed for zero total momentum
double System::temperature() const {
    double dof = 3.0 * static_cast<double>(particles_.size()) - 3.0;
    return 2.0 * kineticEnergy() / dof;
}

// FNV-1a over the raw bytes of every position and velocity, in particle order
std::uint64_t System::stateHash() const {

    // Hash of every position and velocity bit pattern: equal hashes mean bitwise-identical states
    std::uint64_t hash = 14695981039346656037ull;
    auto mix = [&hash](double value) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (bits >> (8 * byte)) & 0xff;
            hash *= 1099511628211ull;
        }
    };

    // Mix in the position and velocity of every particle
    for (const auto& particle : particles_) {
        for (int k = 0; k < 3; ++k) {
            mix(particle.position[k]);
            mix(particle.velocity[k]);
        }
    }
    return hash;
}
