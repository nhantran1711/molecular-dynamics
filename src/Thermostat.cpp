#include "Thermostat.hpp"
#include <cmath>

// Rescale the velocities of all particles in the system to achieve the target temperature
void rescaleVelocities(System& system, double targetTemperature) {
    // Calculate the current temperature of the system
    double current = system.temperature();

    // If the current temperature is zero or negative, we cannot rescale velocities
    if (current <= 0.0) {
        return;
    }

    // Calculate the scaling factor to achieve the target temperature
    double scale = std::sqrt(targetTemperature / current);

    // Rescale the velocities of all particles in the system
    for (auto& particle : system.getParticles()) {
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] *= scale;
        }
    }
}

BussiThermostat::BussiThermostat(double temperature, double tau, std::uint64_t seed)
    : temperature_(temperature), tau_(tau), rng_(seed) {}

void BussiThermostat::apply(System& system, double dt) {
    const double kinetic = system.kineticEnergy();
    if (kinetic <= 0.0) {
        return;
    }

    // Degrees of freedom, with 3 removed for the conserved total momentum
    const double dof = 3.0 * static_cast<double>(system.getParticles().size()) - 3.0;
    const double target = 0.5 * dof * temperature_;
    const double c = std::exp(-dt / tau_);

    // One Gaussian, plus the sum of squares of the other dof - 1 Gaussians, drawn as a chi-squared variable
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::gamma_distribution<double> gamma(0.5 * (dof - 1.0), 2.0);
    const double r1 = gaussian(rng_);
    const double sumSquares = gamma(rng_);

    // New kinetic energy (equation A7 of the paper)
    const double newKinetic = kinetic + (1.0 - c) * (target * (sumSquares + r1 * r1) / dof - kinetic)
        + 2.0 * r1 * std::sqrt(target / dof * kinetic * (1.0 - c) * c);

    const double scale = std::sqrt(newKinetic / kinetic);
    for (auto& particle : system.getParticles()) {
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] *= scale;
        }
    }
}
