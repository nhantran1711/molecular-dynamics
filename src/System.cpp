#include "System.hpp"
#include <cmath>
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
    double kinetic = 0.0;

    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        const Particle& particle = particles_[i];
        double v2 = particle.velocity[0] * particle.velocity[0] + particle.velocity[1] * particle.velocity[1] + particle.velocity[2] * particle.velocity[2];
        kinetic += 0.5 * particle.mass * v2;
    }
    return kinetic;
}

// Temperature from equipartition: KE = (dof / 2) * T, with 3 dof removed for zero total momentum
double System::temperature() const {
    double dof = 3.0 * static_cast<double>(particles_.size()) - 3.0;
    return 2.0 * kineticEnergy() / dof;
}
