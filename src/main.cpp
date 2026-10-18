#include "CommandLine.hpp"
#include "Forcefield.hpp"
#include "Initialisation.hpp"
#include "Integrator.hpp"
#include "NeighbourList.hpp"
#include "RadialDistribution.hpp"
#include "System.hpp"
#include "Thermostat.hpp"
#include "Trajectory.hpp"
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
    int cellsPerSide = 5;         // 4 * 5^3 = 500 particles
    bool useNeighbourList = true; // --brute switches to the O(N^2) all-pairs force loop
    double dt = 0.005;
    double skin = 0.2;   // neighbour list buffer beyond the cutoff (fastest in bench/ skin sweep)
    double cutOff = 2.5; // Lennard-Jones interaction cutoff
    int equilibrationSteps = 2000;
    int productionSteps = 5000;
    const char* csvPath = nullptr; // per-step energies and temperature, for plotting
    bool quiet = false;            // only print the RESULT line
    bool deterministic = false;    // bitwise-identical results for any thread count
    bool mixed = false;            // float pair maths, double positions and accumulation
    double density = 0.8442;       // liquid argon near the triple point
    double temperature = 0.728;
    bool nvt = false;              // keep a (Bussi) thermostat on during production
    const char* rdfPath = nullptr; // write g(r) averaged over production
    const char* xyzPath = nullptr; // extended XYZ trajectory of the production phase
    int xyzEvery = 100;
    unsigned seed = 42; // initial velocities and thermostat noise
    bool melt = false;  // spend the first third of equilibration at T = 2 to melt the lattice
};

void printUsage() {
    std::fprintf(
        stderr,
        "Usage: md_sim [cellsPerSide] [--brute] [--dt X] [--skin X] [--cutoff X] [--equil N] [--steps N] [--density X] "
        "[--temperature X] [--nvt] [--seed N] [--melt]\n"
        "              [--csv PATH] [--rdf PATH] [--xyz PATH] [--xyz-every N] [--deterministic] [--mixed] [--quiet]\n");
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
        } else if (arg == "--deterministic") {
            options.deterministic = true;
        } else if (arg == "--mixed") {
            options.mixed = true;
        } else if (arg == "--nvt") {
            options.nvt = true;
        } else if (arg == "--melt") {
            options.melt = true;
        } else if (arg == "--seed" && hasValue) {
            if (!parseNumber(argv[++a], options.seed)) {
                return false;
            }
        } else if (arg == "--density" && hasValue) {
            if (!parseNumber(argv[++a], options.density)) {
                return false;
            }
        } else if (arg == "--temperature" && hasValue) {
            if (!parseNumber(argv[++a], options.temperature)) {
                return false;
            }
        } else if (arg == "--rdf" && hasValue) {
            options.rdfPath = argv[++a];
        } else if (arg == "--xyz" && hasValue) {
            options.xyzPath = argv[++a];
        } else if (arg == "--xyz-every" && hasValue) {
            if (!parseNumber(argv[++a], options.xyzEvery)) {
                return false;
            }
        } else if (arg == "--dt" && hasValue) {
            if (!parseNumber(argv[++a], options.dt)) {
                return false;
            }
        } else if (arg == "--skin" && hasValue) {
            if (!parseNumber(argv[++a], options.skin)) {
                return false;
            }
        } else if (arg == "--cutoff" && hasValue) {
            if (!parseNumber(argv[++a], options.cutOff)) {
                return false;
            }
        } else if (arg == "--equil" && hasValue) {
            if (!parseNumber(argv[++a], options.equilibrationSteps)) {
                return false;
            }
        } else if (arg == "--steps" && hasValue) {
            if (!parseNumber(argv[++a], options.productionSteps)) {
                return false;
            }
        } else if (arg == "--csv" && hasValue) {
            options.csvPath = argv[++a];
        } else if (!arg.empty() && arg[0] != '-') {
            if (!parseNumber(argv[a], options.cellsPerSide)) {
                return false;
            }
        } else {
            return false;
        }
    }
    return options.cellsPerSide > 0 && options.dt > 0.0 && options.skin > 0.0 && options.cutOff > 0.0 &&
           options.density > 0.0 && options.temperature > 0.0 && options.xyzEvery > 0 && options.productionSteps > 0 &&
           options.equilibrationSteps >= 0;
}

} // namespace

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

    const double density = options.density;
    const double temperature = options.temperature;
    const double cutOff = options.cutOff;
    const double skin = options.skin;
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
        std::fprintf(
            csv, "step,time,phase,ke_per_particle,pe_per_particle,e_per_particle,temperature,pressure,inside_pairs\n");
    }

    // FCC has 4 particles per unit cell, so a^3 = 4 / density
    const double latticeConstant = std::cbrt(4.0 / density);
    const double boxLength = cellsPerSide * latticeConstant;

    // Create the system and initialize velocities
    System system(createFCCLattice(cellsPerSide, latticeConstant), boxLength);
    initializeVelocities(system, temperature, options.seed);

    // Create the force field and integrator
    LennardJonesForceField forceField(1.0, 1.0, cutOff);
    VelocityVerletIntegrator integrator(dt);
    NeighbourList neighbourList(cutOff, skin, options.deterministic);
    forceField.setDeterministic(options.deterministic);
    forceField.setMixedPrecision(options.mixed);
    if (options.mixed && (options.deterministic || !useNeighbourList)) {
        std::fprintf(stderr, "--mixed applies to the default neighbour list path only\n");
        return 1;
    }

    // Advance one timestep with whichever force path was selected
    auto advance = [&]() {
        return useNeighbourList ? integrator.step(system, forceField, neighbourList)
                                : integrator.step(system, forceField);
    };

    // Compute the initial forces and potential energy
    double potential = forceField.computeForces(system);
    neighbourList.build(system);
    const double n = static_cast<double>(system.size());

#ifdef _OPENMP
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif

    if (!quiet) {
        std::printf("N = %zu, L = %.4f, rho = %.4f, dt = %.4f, forces = %s, threads = %d\n", system.size(), boxLength,
                    density, dt, useNeighbourList ? "neighbour list" : "all pairs", threads);

        // Equilibration: melt the lattice while rescaling velocities to hold the target temperature
        std::printf("\nEquilibration (%d steps, rescale every %d)\n", equilibrationSteps, rescaleEvery);
        std::printf("%8s %12s %12s %12s %12s\n", "step", "KE/N", "PE/N", "E/N", "T");
    }

    // Virial pressure P = (2 KE + W) / 3V, without tail corrections
    const double volume = boxLength * boxLength * boxLength;
    auto pressure = [&](double kinetic) { return (2.0 * kinetic + forceField.getVirial()) / (3.0 * volume); };

    auto writeRow = [&](int step, const char* phase, double kinetic, double potentialEnergy, double t) {
        std::fprintf(csv, "%d,%.6f,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.0f\n", step, step * dt, phase, kinetic / n,
                     potentialEnergy / n, (kinetic + potentialEnergy) / n, t, pressure(kinetic),
                     forceField.getInsidePairs());
    };

    for (int step = 1; step <= equilibrationSteps; ++step) {
        potential = advance();

        if (step % rescaleEvery == 0) {
            // Near the freezing line the starting FCC lattice may never melt at the target temperature, so
            // --melt first makes it a hot liquid, then cools it
            const bool melting = options.melt && step <= equilibrationSteps / 3;
            rescaleVelocities(system, melting ? 2.0 : temperature);
        }

        if (csv) {
            writeRow(step, "equilibration", system.kineticEnergy(), potential, system.temperature());
        }

        if (!quiet && step % printEvery == 0) {
            // Print the kinetic energy, potential energy, total energy, and temperature per particle
            double kinetic = system.kineticEnergy();
            std::printf("%8d %12.6f %12.6f %12.6f %12.6f\n", step, kinetic / n, potential / n,
                        (kinetic + potential) / n, system.temperature());
        }
    }

    // Production: NVE by default (energy drift is then meaningful), or NVT with the Bussi thermostat
    BussiThermostat thermostat(temperature, 0.5, options.seed);
    RadialDistribution rdf(cutOff, 150);
    if (options.rdfPath && !useNeighbourList) {
        std::fprintf(stderr, "--rdf needs the neighbour list (drop --brute)\n");
        return 1;
    }

    const double initialEnergy = system.kineticEnergy() + potential;
    double sumT = 0.0;
    double sumT2 = 0.0;
    double maxDrift = 0.0;

    if (!quiet) {
        std::printf("\nProduction (%d steps, %s)\n", productionSteps, options.nvt ? "NVT, Bussi thermostat" : "NVE");
        std::printf("%8s %12s %12s %12s %12s %12s\n", "step", "KE/N", "PE/N", "E/N", "T", "dE/E0");
    }

    std::FILE* xyz = nullptr;
    if (options.xyzPath) {
        xyz = std::fopen(options.xyzPath, "w");
        if (!xyz) {
            std::fprintf(stderr, "Could not open %s: %s\n", options.xyzPath, std::strerror(errno));
            return 1;
        }
    }

    // Time only the production phase, so setup and equilibration don't skew the benchmarks
    const auto start = std::chrono::steady_clock::now();

    // Run the production steps, compute the kinetic energy, potential energy, total energy, temperature, and energy drift
    for (int step = 1; step <= productionSteps; ++step) {
        potential = advance();

        if (options.nvt) {
            thermostat.apply(system, dt);
        }
        if (options.rdfPath && step % 10 == 0) {
            rdf.sample(system, neighbourList);
        }
        if (xyz && step % options.xyzEvery == 0) {
            writeXyzFrame(xyz, system, (equilibrationSteps + step) * dt);
        }

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
            std::printf("%8d %12.6f %12.6f %12.6f %12.6f %12.2e\n", step, kinetic / n, potential / n, total / n, t,
                        drift);
        }
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (csv) {
        std::fclose(csv);
    }
    if (xyz) {
        std::fclose(xyz);
    }
    if (options.rdfPath && !rdf.write(options.rdfPath)) {
        std::fprintf(stderr, "Could not write %s: %s\n", options.rdfPath, std::strerror(errno));
        return 1;
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
                        static_cast<double>(totalSteps) / neighbourList.getRebuildCount(),
                        neighbourList.getPairCount());
        }
        std::printf("Production took %.3f s (%.4f ms per step)\n", seconds, 1000.0 * seconds / productionSteps);
    }

    // Machine-readable summary for the benchmark scripts
    std::printf("RESULT n=%zu threads=%d forces=%s dt=%g steps=%d seconds=%.6f ms_per_step=%.6f mean_T=%.6f std_T=%.6f "
                "max_drift=%.6e skin=%g cutoff=%g rebuilds=%d deterministic=%d state_hash=%016llx\n",
                system.size(), threads, useNeighbourList ? "list" : "brute", dt, productionSteps, seconds,
                1000.0 * seconds / productionSteps, meanT, stdT, maxDrift, skin, cutOff,
                neighbourList.getRebuildCount(), options.deterministic ? 1 : 0,
                static_cast<unsigned long long>(system.stateHash()));
    return 0;
}
