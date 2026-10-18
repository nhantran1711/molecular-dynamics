#pragma once

#include "System.hpp"

// Rescale the velocities of all particles in the system to achieve the target temperature
void rescaleVelocities(System& system, double targetTemperature);
