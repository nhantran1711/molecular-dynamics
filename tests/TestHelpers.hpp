#pragma once

#include "Forcefield.hpp"
#include "Initialisation.hpp"
#include "Integrator.hpp"
#include "NeighbourList.hpp"
#include "System.hpp"
#include "Thermostat.hpp"
#include <array>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace test {

// Same liquid state point as md_sim, in reduced Lennard-Jones units
constexpr double density = 0.8442;
constexpr double temperature = 0.728;
constexpr double cutOff = 2.5;
constexpr double skin = 0.2;
constexpr double dt = 0.005;

inline double latticeConstant() {
    return std::cbrt(4.0 / density);
}

// FCC lattice with Maxwell-Boltzmann velocities: 4 * cells^3 particles
inline System makeLattice(int cells, unsigned seed = 42) {
    // Create a system with particles arranged in an FCC lattice and initialize their velocities
    System system(createFCCLattice(cells, latticeConstant()), cells * latticeConstant());
    initializeVelocities(system, temperature, seed);

    return system;
}

// A disordered liquid configuration: melt the lattice for a few hundred thermostatted steps
inline System makeLiquid(int cells, int steps = 300) {
    // Create a system with particles arranged in an FCC lattice and initialize their velocities
    System system = makeLattice(cells);
    LennardJonesForceField forceField(1.0, 1.0, cutOff);
    VelocityVerletIntegrator integrator(dt);
    NeighbourList list(cutOff, skin);

    // Compute the initial forces and potential energy, then run a few steps to melt the lattice
    list.build(system);
    forceField.computeForces(system, list);

    // Run a few steps of the simulation with the thermostat on to melt the lattice into a liquid configuration
    for (int s = 1; s <= steps; ++s) {
        integrator.step(system, forceField, list);
        if (s % 10 == 0) {
            rescaleVelocities(system, temperature);
        }
    }
    return system;
}

inline std::array<double, 3> totalMomentum(const System& system) {
    // Calculate the total momentum of the system by summing the momentum of each particle
    std::array<double, 3> p {0.0, 0.0, 0.0};

    // Sum the momentum of each particle in the system
    for (const auto& particle : system.getParticles()) {
        for (int k = 0; k < 3; ++k) {
            p[k] += particle.mass * particle.velocity[k];
        }
    }
    return p;
}

// Run a block of code with a given OpenMP thread count, restoring the previous count afterwards
template <typename F>
void withThreads(int threads, F&& body) {
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(threads);
    body();
    omp_set_num_threads(previous);
#else
    (void)threads;
    body();
#endif
}

}
