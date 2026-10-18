#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "Forcefield.hpp"
#include "Parallel.hpp"

LennardJonesForceField::LennardJonesForceField(double epsilon, double sigma, double cut_off)
    : epsilon_(epsilon), sigma_(sigma), cut_off_(cut_off), cut_off_squared_(cut_off * cut_off) {

        // Calculate the energy shift to ensure the potential goes to zero at the cutoff distance
        double sr6 = std::pow(sigma_ / cut_off_, 6);
        double sr12 = sr6 * sr6;
        energy_shift_ = 4.0 * epsilon_ * (sr12 - sr6);
}

// Minimum image convention is only valid if the cutoff is under half the box length
void LennardJonesForceField::checkCutOff(const System& system) const {
    if (2.0 * cut_off_ > system.getBoxLength()) {
        throw std::runtime_error("Cutoff must be less than half the box length");
    }
}

// For a pair inside the cutoff, return F/r and the pair's shifted potential energy
LennardJonesForceField::PairTerms LennardJonesForceField::pairTerms(double r2) const {
    // Compute the inverse squared distance and the Lennard-Jones potential terms
    double inv_r2 = 1.0 / r2;
    double sr2 = sigma_ * sigma_ * inv_r2;
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;

    // F = -dV/dr * (dr / r), written in terms of 1/r^2 to avoid a sqrt
    double f_over_r = 24.0 * epsilon_ * (2.0 * sr12 - sr6) * inv_r2;

    return {f_over_r, 4.0 * epsilon_ * (sr12 - sr6) - energy_shift_};
}

// O(N^2) pair loop: compute LJ forces on all particles and return the total potential energy
double LennardJonesForceField::computeForces(System& system) const {
    checkCutOff(system);

    auto& particles = system.getParticles();
    const std::size_t n = particles.size();

    for (auto& particle : particles) {
        particle.force = {0.0, 0.0, 0.0};
    }

    double potential = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            std::array<double, 3> dr = system.minimumImageDistance(particles[i].position, particles[j].position);

            // Compute the squared distance between the two particles, skipping pairs beyond the cutoff
            double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            if (r2 >= cut_off_squared_) {
                continue;
            }

            auto [f_over_r, energy] = pairTerms(r2);

            // Newton's third law: equal and opposite force on each particle
            for (int k = 0; k < 3; ++k) {
                particles[i].force[k] += f_over_r * dr[k];
                particles[j].force[k] -= f_over_r * dr[k];
            }
            potential += energy;
        }
    }
    return potential;
}

// Neighbour list loop: only visit pairs that were within cutoff + skin at the last list rebuild
double LennardJonesForceField::computeForces(System& system, const NeighbourList& neighbourList) const {
    checkCutOff(system);

    // Get a reference to the particles in the system
    auto& particles = system.getParticles();
    const std::size_t n = particles.size();

    double potential = 0.0;

    // Half neighbour list, so each pair is computed once and its force applied to both particles. Two threads
    // can share a particle j, so each thread accumulates into its own force array, and the arrays are summed
    // per particle at the end
    #pragma omp parallel reduction(+:potential)
    {
        // single has an implicit barrier, so every thread sees the resized buffer
        #pragma omp single
        threadForces_.resize(threadCount() * n);

        std::array<double, 3>* forces = threadForces_.data() + threadId() * n;
        std::fill(forces, forces + n, std::array<double, 3> {0.0, 0.0, 0.0});

        // Implicit barrier at the end, so all threads finish before any sums are taken
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            // Accumulate particle i's force in registers, writing it to memory once
            std::array<double, 3> force_i {0.0, 0.0, 0.0};

            for (std::size_t j : neighbourList.neighboursOf(i)) {
                std::array<double, 3> dr = system.minimumImageDistance(particles[i].position, particles[j].position);

                double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if (r2 >= cut_off_squared_) {
                    continue;
                }

                auto [f_over_r, energy] = pairTerms(r2);

                // Newton's third law: equal and opposite force on each particle
                for (int k = 0; k < 3; ++k) {
                    force_i[k] += f_over_r * dr[k];
                    forces[j][k] -= f_over_r * dr[k];
                }
                potential += energy;
            }

            for (int k = 0; k < 3; ++k) {
                forces[i][k] += force_i[k];
            }
        }

        // Sum every thread's contribution for each particle. Each thread owns distinct particles, so no races
        const int nThreads = threadCount();

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            std::array<double, 3> total {0.0, 0.0, 0.0};
            for (int t = 0; t < nThreads; ++t) {
                const auto& f = threadForces_[t * n + i];
                for (int k = 0; k < 3; ++k) {
                    total[k] += f[k];
                }
            }
            particles[i].force = total;
        }
    }
    return potential;
}
