#include "Initialisation.hpp"
#include <cmath>
#include <random>

// Place particles on an FCC lattice: 4 basis atoms per cubic unit cell
std::vector<Particle> createFCCLattice(int cellsPerSide, double latticeConstant) {
    // Basis vectors for the FCC lattice
    static constexpr double basis[4][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5},
    };

    std::vector<Particle> particles;

    // Reserve space for the expected number of particles to avoid reallocations
    particles.reserve(4 * cellsPerSide * cellsPerSide * cellsPerSide);


    // Loop over the unit cells and place the basis atoms
    for (int ix = 0; ix < cellsPerSide; ++ix) {
        for (int iy = 0; iy < cellsPerSide; ++iy) {
            for (int iz = 0; iz < cellsPerSide; ++iz) {

                // Loop over the basis atoms and place them in the unit cell
                for (const auto& b : basis) {
                    Particle particle;

                    // Offset by a quarter cell so no particle sits exactly on the box boundary
                    particle.position = {
                        (ix + b[0] + 0.25) * latticeConstant,
                        (iy + b[1] + 0.25) * latticeConstant,
                        (iz + b[2] + 0.25) * latticeConstant,
                    };

                    // Add the particle to the vector
                    particles.push_back(particle);
                }
            }
        }
    }
    return particles;
}

// Draw Gaussian velocities, remove net momentum, then rescale to hit the target temperature exactly
void initializeVelocities(System& system, double temperature, std::uint32_t seed) {
    
    // Get a reference to the particles in the system
    auto& particles = system.getParticles();
    std::mt19937 rng(seed);

    // Draw Gaussian velocities for each particle and calculate the total momentum and mass
    std::array<double, 3> momentum {0.0, 0.0, 0.0};
    double totalMass = 0.0;

    // Draw Gaussian velocities for each particle and calculate the total momentum and mass
    for (auto& particle : particles) {
        std::normal_distribution<double> dist(0.0, std::sqrt(temperature / particle.mass));
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] = dist(rng);
            momentum[k] += particle.mass * particle.velocity[k];
        }
        totalMass += particle.mass;
    }

    // Remove centre-of-mass velocity so the system doesn't drift
    for (auto& particle : particles) {
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] -= momentum[k] / totalMass;
        }
    }

    // Rescale velocities to hit the target temperature exactly
    double scale = std::sqrt(temperature / system.temperature());
    for (auto& particle : particles) {
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] *= scale;
        }
    }
}