#pragma once

#include "Particle.hpp"
#include "System.hpp"
#include <cstdint>
#include <vector>

// Place 4 * cellsPerSide^3 particles on an FCC lattice with the given lattice constant
std::vector<Particle> createFCCLattice(int cellsPerSide, double latticeConstant);

// Assign Maxwell-Boltzmann velocities, remove centre-of-mass drift and rescale to the target temperature
void initializeVelocities(System& system, double temperature, std::uint32_t seed);
