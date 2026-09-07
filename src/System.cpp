#include "System.hpp"
#include <cmath>

// Constructor for the System class
System::System(std::vector<Particle> particles, double boxLength): 
    particles_(std::move(particles)), boxLength_(boxLength) {}


// Using PBC, find the minimum image distance between two particles
std::array<double, 3> System::minimumImageDistance(const std::array<double, 3>& ri, const std::array<double, 3>& rj) const {
    std::array<double, 3> dr {};

    for (int k = 0; k < 3; ++k) {
        dr[k] = ri[k] - rj[k];

        // PBC 
        dr[k] -= boxLength_ * std::round(dr[k] / boxLength_);
    }
    return dr;
}

// Wrap position of a particle back into the simulation box
void System::wrapPosition(std::array<double, 3>& r) const {
    for (int k = 0; k < 3; ++k) {
        r[k] -= boxLength_ * std::floor(r[k] / boxLength_);
    }
}

// Calculate the kinetic energy of the system
double System::kineticEnergy() const {
    double kinetic = 0.0;

    for (const auto& particle: particles_) {
        double v2 = particle.velocity[0] * particle.velocity[0] + particle.velocity[1] * particle.velocity[1] + particle.velocity[2] * particle.velocity[2];
        kinetic += 0.5 * particle.mass * v2;
    }
    return kinetic;
}