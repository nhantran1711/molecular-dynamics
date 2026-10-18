#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
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

    if (deterministic_) {
        return computeForcesDeterministic(system, neighbourList);
    }

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

        // Per-thread scratch for one particle's neighbours, one array per component (struct of arrays) so the
        // compute loop below reads and writes contiguous doubles
        std::vector<double> xj, yj, zj, fxj, fyj, fzj;

        const double boxLength = system.getBoxLength();
        const double inverseBoxLength = system.getInverseBoxLength();

        // Implicit barrier at the end, so all threads finish before any sums are taken
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const auto neighbours = neighbourList.neighboursOf(i);
            const std::size_t m = neighbours.size();

            if (xj.size() < m) {
                for (auto* buffer : {&xj, &yj, &zj, &fxj, &fyj, &fzj}) {
                    buffer->resize(m);
                }
            }

            // Gather: copy neighbour positions into contiguous arrays. Scalar loads, since NEON has no gather
            for (std::size_t k = 0; k < m; ++k) {
                const auto& r = particles[neighbours[k]].position;
                xj[k] = r[0];
                yj[k] = r[1];
                zj[k] = r[2];
            }

            // Compute: the pair maths, vectorised across neighbours. Branch-free, since SIMD lanes can't take
            // different branches: pairs beyond the cutoff (still in the list thanks to the skin) are computed
            // anyway and masked to zero
            const auto& ri = particles[i].position;
            double fix = 0.0, fiy = 0.0, fiz = 0.0, energy_i = 0.0;

            #pragma omp simd reduction(+:fix, fiy, fiz, energy_i)
            for (std::size_t k = 0; k < m; ++k) {
                // Minimum image convention, as in System::minimumImageDistance
                double dx = ri[0] - xj[k];
                double dy = ri[1] - yj[k];
                double dz = ri[2] - zj[k];
                dx -= boxLength * std::round(dx * inverseBoxLength);
                dy -= boxLength * std::round(dy * inverseBoxLength);
                dz -= boxLength * std::round(dz * inverseBoxLength);

                double r2 = dx * dx + dy * dy + dz * dz;
                auto [f_over_r, energy] = pairTerms(r2);

                const bool inside = r2 < cut_off_squared_;
                f_over_r = inside ? f_over_r : 0.0;
                energy_i += inside ? energy : 0.0;

                // Newton's third law: + on particle i here, - on particle j in the scatter below
                fix += f_over_r * dx;
                fiy += f_over_r * dy;
                fiz += f_over_r * dz;
                fxj[k] = f_over_r * dx;
                fyj[k] = f_over_r * dy;
                fzj[k] = f_over_r * dz;
            }

            // Scatter: equal and opposite forces on the neighbours. Scalar stores, since NEON has no scatter
            for (std::size_t k = 0; k < m; ++k) {
                auto& f = forces[neighbours[k]];
                f[0] -= fxj[k];
                f[1] -= fyj[k];
                f[2] -= fzj[k];
            }

            forces[i][0] += fix;
            forces[i][1] += fiy;
            forces[i][2] += fiz;
            potential += energy_i;
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

// Deterministic neighbour list loop: every pair's force is stored, then each particle sums its own contributions in a fixed order, so the result is bitwise identical for any thread count
double LennardJonesForceField::computeForcesDeterministic(System& system, const NeighbourList& neighbourList) const {
    
    // Check that the neighbour list has a reverse index, which is needed for deterministic forces. The reverse index records, for every particle j, the pairs in which it is the second particle. The default mode doesn't need it, so it is off by default
    if (!neighbourList.hasReverseIndex()) {
        throw std::runtime_error("Deterministic forces need a NeighbourList built with reverseIndex = true");
    }

    // Get a reference to the particles in the system
    auto& particles = system.getParticles();
    const std::size_t n = particles.size();
    const std::size_t pairs = neighbourList.getPairCount();

    // Resize the pair force and energy buffers to hold one entry per pair, indexed like the neighbour list. The buffers are kept between calls to avoid reallocating, which also means computeForces must not run concurrently on one force field
    for (auto* buffer : {&pairFx_, &pairFy_, &pairFz_, &pairEnergy_}) {
        buffer->resize(pairs);
    }

    const double boxLength = system.getBoxLength();
    const double inverseBoxLength = system.getInverseBoxLength();

    #pragma omp parallel
    {
        std::vector<double> xj, yj, zj;

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const auto neighbours = neighbourList.neighboursOf(i);
            const std::size_t m = neighbours.size();

            if (xj.size() < m) {
                for (auto* buffer : {&xj, &yj, &zj}) {
                    buffer->resize(m);
                }
            }

            // Gather neighbour positions into contiguous arrays, as in the default kernel
            for (std::size_t k = 0; k < m; ++k) {
                const auto& r = particles[neighbours[k]].position;
                xj[k] = r[0];
                yj[k] = r[1];
                zj[k] = r[2];
            }

            // Compute the pair maths, vectorised across neighbours. Each pair writes its own slot in the force and energy buffers, so there is no reduction whose order could vary
            const std::size_t first = neighbourList.firstPairOf(i);
            double* fx = pairFx_.data() + first;
            double* fy = pairFy_.data() + first;
            double* fz = pairFz_.data() + first;
            double* e = pairEnergy_.data() + first;
            const auto& ri = particles[i].position;

            // Same branch-free pair maths as the default kernel, but every pair writes its own slot instead of
            // being summed, so there is no reduction whose order could vary
            #pragma omp simd
            for (std::size_t k = 0; k < m; ++k) {
                double dx = ri[0] - xj[k];
                double dy = ri[1] - yj[k];
                double dz = ri[2] - zj[k];
                dx -= boxLength * std::round(dx * inverseBoxLength);
                dy -= boxLength * std::round(dy * inverseBoxLength);
                dz -= boxLength * std::round(dz * inverseBoxLength);

                double r2 = dx * dx + dy * dy + dz * dz;
                auto [f_over_r, energy] = pairTerms(r2);

                const bool inside = r2 < cut_off_squared_;
                f_over_r = inside ? f_over_r : 0.0;
                fx[k] = f_over_r * dx;
                fy[k] = f_over_r * dy;
                fz[k] = f_over_r * dz;
                e[k] = inside ? energy : 0.0;
            }
        }
    }

    // Each particle owns its force, and sums its contributions in a fixed order: + as particle i, - as particle j
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        std::array<double, 3> force {0.0, 0.0, 0.0};
        const std::size_t first = neighbourList.firstPairOf(i);
        const std::size_t last = first + neighbourList.neighboursOf(i).size();
        for (std::size_t k = first; k < last; ++k) {
            force[0] += pairFx_[k];
            force[1] += pairFy_[k];
            force[2] += pairFz_[k];
        }
        for (std::size_t k : neighbourList.pairsAsSecond(i)) {
            force[0] -= pairFx_[k];
            force[1] -= pairFy_[k];
            force[2] -= pairFz_[k];
        }
        particles[i].force = force;
    }

    return deterministicSum(pairs, [this](std::size_t k) {
        return pairEnergy_[k];
    });
}
