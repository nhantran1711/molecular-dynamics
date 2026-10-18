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
