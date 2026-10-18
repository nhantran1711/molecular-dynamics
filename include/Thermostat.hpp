#pragma once

#include "System.hpp"
#include <cstdint>
#include <random>

// Rescale the velocities of all particles in the system to achieve the target temperature
void rescaleVelocities(System& system, double targetTemperature);

// A Bussi thermostat, which rescales the velocities of all particles in the system to achieve the target temperature, while preserving the canonical distribution of velocities. See Bussi et al., J. Chem. Phys. 126, 014101 (2007).
class BussiThermostat {
public:
    BussiThermostat(double temperature, double tau, std::uint64_t seed);

    // Call once per timestep, after the integrator step
    void apply(System& system, double dt);

    void setTemperature(double temperature) {
        temperature_ = temperature;
    };

private:
    double temperature_;
    double tau_;
    std::mt19937_64 rng_;
};
