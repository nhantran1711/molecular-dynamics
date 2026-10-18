#include "Forcefield.hpp"
#include "Initialisation.hpp"
#include "Integrator.hpp"
#include "System.hpp"
#include "Thermostat.hpp"
#include <cmath>
#include <cstdio>

// All quantities are in reduced Lennard-Jones units (epsilon = sigma = mass = k_B = 1)
int main() {
    const int cellsPerSide = 5;          // 4 * 5^3 = 500 particles
    const double density = 0.8442;       // Liquid argon near the triple point
    const double temperature = 0.728;
    const double cutOff = 2.5;
    const double dt = 0.005;
    const int equilibrationSteps = 2000;
    const int rescaleEvery = 10;
    const int productionSteps = 5000;
    const int printEvery = 250;

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
    const double n = static_cast<double>(system.getParticles().size());

    std::printf("N = %zu, L = %.4f, rho = %.4f, dt = %.4f\n", system.getParticles().size(), boxLength, density, dt);

    // Equilibration: melt the lattice while rescaling velocities to hold the target temperature
    std::printf("\nEquilibration (%d steps, rescale every %d)\n", equilibrationSteps, rescaleEvery);
    std::printf("%8s %12s %12s %12s %12s\n", "step", "KE/N", "PE/N", "E/N", "T");

    for (int step = 1; step <= equilibrationSteps; ++step) {
        potential = integrator.step(system, forceField);

        if (step % rescaleEvery == 0) {
            rescaleVelocities(system, temperature);
        }

        if (step % printEvery == 0) {
            // Print the kinetic energy, potential energy, total energy, and temperature per particle
            double kinetic = system.kineticEnergy();
            std::printf("%8d %12.6f %12.6f %12.6f %12.6f\n",
                step, kinetic / n, potential / n, (kinetic + potential) / n, system.temperature());
        }
    }

    // Production: run the simulation in the NVE ensemble and measure the temperature and energy drift
    const double initialEnergy = system.kineticEnergy() + potential;
    double sumT = 0.0;
    double sumT2 = 0.0;
    double maxDrift = 0.0;

    std::printf("\nProduction (%d steps, NVE)\n", productionSteps);
    std::printf("%8s %12s %12s %12s %12s %12s\n", "step", "KE/N", "PE/N", "E/N", "T", "dE/E0");

    // Run the production steps, compute the kinetic energy, potential energy, total energy, temperature, and energy drift
    for (int step = 1; step <= productionSteps; ++step) {
        potential = integrator.step(system, forceField);

        double kinetic = system.kineticEnergy();
        double total = kinetic + potential;
        double t = system.temperature();
        double drift = (total - initialEnergy) / std::abs(initialEnergy);

        // Accumulate the temperature and its square for mean and standard deviation calculations
        sumT += t;
        sumT2 += t * t;
        maxDrift = std::max(maxDrift, std::abs(drift));
        
        // Print the kinetic energy, potential energy, total energy, temperature, and energy drift per particle
        if (step % printEvery == 0) {
            std::printf("%8d %12.6f %12.6f %12.6f %12.6f %12.2e\n",
                step, kinetic / n, potential / n, total / n, t, drift);
        }
    }

    // Calculate the mean and standard deviation of the temperature over the production steps
    double meanT = sumT / productionSteps;
    double stdT = std::sqrt(sumT2 / productionSteps - meanT * meanT);
    std::printf("\n<T> = %.4f +/- %.4f (target %.4f), max |dE/E0| = %.2e\n", meanT, stdT, temperature, maxDrift);
    return 0;
}
