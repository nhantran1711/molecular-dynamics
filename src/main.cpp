#include "Forcefield.hpp"
#include "Initialisation.hpp"
#include "Integrator.hpp"
#include "NeighbourList.hpp"
#include "System.hpp"
#include "Thermostat.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <string_view>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Options {
    int cellsPerSide = 5;              // 4 * 5^3 = 500 particles
    bool useNeighbourList = true;      // --brute switches to the O(N^2) all-pairs force loop
    double dt = 0.005;
    int equilibrationSteps = 2000;
    int productionSteps = 5000;
    const char* csvPath = nullptr;     // per-step energies and temperature, for plotting
    bool quiet = false;                // only print the RESULT line
};

void printUsage() {
    std::fprintf(stderr,
        "Usage: md_sim [cellsPerSide] [--brute] [--dt X] [--equil N] [--steps N] [--csv PATH] [--quiet]\n");
}

// Parse command line arguments, returning false on anything unrecognised
bool parseOptions(int argc, char** argv, Options& options) {
    for (int a = 1; a < argc; ++a) {
        std::string_view arg = argv[a];
        bool hasValue = a + 1 < argc;

        if (arg == "--brute") {
            options.useNeighbourList = false;
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--dt" && hasValue) {
            options.dt = std::atof(argv[++a]);
        } else if (arg == "--equil" && hasValue) {
            options.equilibrationSteps = std::atoi(argv[++a]);
        } else if (arg == "--steps" && hasValue) {
            options.productionSteps = std::atoi(argv[++a]);
        } else if (arg == "--csv" && hasValue) {
            options.csvPath = argv[++a];
        } else if (!arg.empty() && arg[0] != '-') {
            options.cellsPerSide = std::atoi(argv[a]);
        } else {
            return false;
        }
    }
    return options.cellsPerSide > 0 && options.dt > 0.0 && options.productionSteps > 0 && options.equilibrationSteps >= 0;
}

}

// Main function for the molecular dynamics simulation
int main(int argc, char** argv) {
    Options options;
    if (!parseOptions(argc, argv, options)) {
        printUsage();
        return 1;
    }

    const int cellsPerSide = options.cellsPerSide;
    const bool useNeighbourList = options.useNeighbourList;
    const bool quiet = options.quiet;

    const double density = 0.8442;       // Liquid argon near the triple point
    const double temperature = 0.728;
    const double cutOff = 2.5;
    const double skin = 0.3;
    const double dt = options.dt;
    const int equilibrationSteps = options.equilibrationSteps;
    const int rescaleEvery = 10;
    const int productionSteps = options.productionSteps;
    const int printEvery = 250;

    // One row per step across both phases, so the plots can show equilibration settling as well as production
    std::FILE* csv = nullptr;
    if (options.csvPath) {
        csv = std::fopen(options.csvPath, "w");
        if (!csv) {
            std::fprintf(stderr, "Could not open %s: %s\n", options.csvPath, std::strerror(errno));
            return 1;
        }
        std::fprintf(csv, "step,time,phase,ke_per_particle,pe_per_particle,e_per_particle,temperature\n");
    }

    // FCC has 4 particles per unit cell, so a^3 = 4 / density
    const double latticeConstant = std::cbrt(4.0 / density);
    const double boxLength = cellsPerSide * latticeConstant;

    // Create the system and initialize velocities
    System system(createFCCLattice(cellsPerSide, latticeConstant), boxLength);
    initializeVelocities(system, temperature, 42);

    // Create the force field and integrator
    LennardJonesForceField forceField(1.0, 1.0, cutOff);
    VelocityVerletIntegrator integrator(dt);
    NeighbourList neighbourList(cutOff, skin);

    // Advance one timestep with whichever force path was selected
    auto advance = [&]() {
        return useNeighbourList ? integrator.step(system, forceField, neighbourList)
                                : integrator.step(system, forceField);
    };

    // Compute the initial forces and potential energy
    double potential = forceField.computeForces(system);
    neighbourList.build(system);
    const double n = static_cast<double>(system.getParticles().size());

#ifdef _OPENMP
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif

    if (!quiet) {
        std::printf("N = %zu, L = %.4f, rho = %.4f, dt = %.4f, forces = %s, threads = %d\n", system.getParticles().size(), boxLength, density, dt,
            useNeighbourList ? "neighbour list" : "all pairs", threads);

        // Equilibration: melt the lattice while rescaling velocities to hold the target temperature
        std::printf("\nEquilibration (%d steps, rescale every %d)\n", equilibrationSteps, rescaleEvery);
        std::printf("%8s %12s %12s %12s %12s\n", "step", "KE/N", "PE/N", "E/N", "T");
    }

    auto writeRow = [&](int step, const char* phase, double kinetic, double potentialEnergy, double t) {
        std::fprintf(csv, "%d,%.6f,%s,%.8f,%.8f,%.8f,%.8f\n", step, step * dt, phase,
            kinetic / n, potentialEnergy / n, (kinetic + potentialEnergy) / n, t);
    };

    for (int step = 1; step <= equilibrationSteps; ++step) {
        potential = advance();

        if (step % rescaleEvery == 0) {
            rescaleVelocities(system, temperature);
        }

        if (csv) {
            writeRow(step, "equilibration", system.kineticEnergy(), potential, system.temperature());
        }

        if (!quiet && step % printEvery == 0) {
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

    if (!quiet) {
        std::printf("\nProduction (%d steps, NVE)\n", productionSteps);
        std::printf("%8s %12s %12s %12s %12s %12s\n", "step", "KE/N", "PE/N", "E/N", "T", "dE/E0");
    }

    // Time only the production phase, so setup and equilibration don't skew the benchmarks
    const auto start = std::chrono::steady_clock::now();

    // Run the production steps, compute the kinetic energy, potential energy, total energy, temperature, and energy drift
    for (int step = 1; step <= productionSteps; ++step) {
        potential = advance();

        double kinetic = system.kineticEnergy();
        double total = kinetic + potential;
        double t = system.temperature();
        double drift = (total - initialEnergy) / std::abs(initialEnergy);

        // Accumulate the temperature and its square for mean and standard deviation calculations
        sumT += t;
        sumT2 += t * t;
        maxDrift = std::max(maxDrift, std::abs(drift));

        if (csv) {
            writeRow(equilibrationSteps + step, "production", kinetic, potential, t);
        }

        // Print the kinetic energy, potential energy, total energy, temperature, and energy drift per particle
        if (!quiet && step % printEvery == 0) {
            std::printf("%8d %12.6f %12.6f %12.6f %12.6f %12.2e\n",
                step, kinetic / n, potential / n, total / n, t, drift);
        }
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (csv) {
        std::fclose(csv);
    }

    // Calculate the mean and standard deviation of the temperature over the production steps
    double meanT = sumT / productionSteps;
    double stdT = std::sqrt(std::max(0.0, sumT2 / productionSteps - meanT * meanT));

    if (!quiet) {
        std::printf("\n<T> = %.4f +/- %.4f (target %.4f), max |dE/E0| = %.2e\n", meanT, stdT, temperature, maxDrift);

        // If the neighbour list was used, print the number of rebuilds and pairs in the list
        if (useNeighbourList) {
            int totalSteps = equilibrationSteps + productionSteps;
            std::printf("Neighbour list: %d rebuilds in %d steps (every %.1f steps), %zu pairs\n",
                neighbourList.getRebuildCount(), totalSteps,
                static_cast<double>(totalSteps) / neighbourList.getRebuildCount(), neighbourList.getPairCount());
        }
        std::printf("Production took %.3f s (%.4f ms per step)\n", seconds, 1000.0 * seconds / productionSteps);
    }

    // Machine-readable summary for the benchmark scripts
    std::printf("RESULT n=%zu threads=%d forces=%s dt=%g steps=%d seconds=%.6f ms_per_step=%.6f mean_T=%.6f std_T=%.6f max_drift=%.6e\n",
        system.getParticles().size(), threads, useNeighbourList ? "list" : "brute", dt, productionSteps,
        seconds, 1000.0 * seconds / productionSteps, meanT, stdT, maxDrift);
    return 0;
}
