#include "Forcefield.hpp"
#include "Initialisation.hpp"
#include "Integrator.hpp"
#include "System.hpp"
#include <cmath>
#include <cstdio>

// All quantities are in reduced Lennard-Jones units (epsilon = sigma = mass = k_B = 1)
int main() {
    const int cellsPerSide = 5;          // 4 * 5^3 = 500 particles
    const double density = 0.8442;       // Liquid argon near the triple point
    const double temperature = 0.728;
    const double cutOff = 2.5;
    const double dt = 0.005;
    const int nSteps = 2000;
    const int printEvery = 100;

    // FCC has 4 particles per unit cell, so a^3 = 4 / density
    const double latticeConstant = std::cbrt(4.0 / density);
    const double boxLength = cellsPerSide * latticeConstant;

    // Create the system and initialize velocities
    System system(createFCCLattice(cellsPerSide, latticeConstant), boxLength);
    initializeVelocities(system, temperature, 42);

    // Create the force field and integrator
    LennardJonesForceField forceField(1.0, 1.0, cutOff);
    VelocityVerletIntegrator integrator(dt);

    // Compute the initial forces and potential energy
    double potential = forceField.computeForces(system);
    const double initialEnergy = system.kineticEnergy() + potential;
    const double n = static_cast<double>(system.getParticles().size());

    // Print the header for the output table
    std::printf("N = %zu, L = %.4f, rho = %.4f, dt = %.4f\n", system.getParticles().size(), boxLength, density, dt);
    std::printf("%8s %12s %12s %12s %12s %12s\n", "step", "KE/N", "PE/N", "E/N", "T", "dE/E0");

    // Run the simulation for nSteps, printing every printEvery steps
    for (int step = 0; step <= nSteps; ++step) {
        // Advance the system by one timestep and compute the new potential energy
        if (step > 0) {
            potential = integrator.step(system, forceField);
        }

        // Print the current state of the system every printEvery steps
        if (step % printEvery == 0) {
            double kinetic = system.kineticEnergy();
            double total = kinetic + potential;
            std::printf("%8d %12.6f %12.6f %12.6f %12.6f %12.2e\n",
                step, kinetic / n, potential / n, total / n, system.temperature(),
                (total - initialEnergy) / std::abs(initialEnergy));
        }
    }
    return 0;
}
