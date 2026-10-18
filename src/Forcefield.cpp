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

    const std::size_t n = system.size();
    std::array<double*, 3> force {system.force(0).data(), system.force(1).data(), system.force(2).data()};

    for (int k = 0; k < 3; ++k) {
        std::fill(force[k], force[k] + n, 0.0);
    }

    double potential = 0.0;
    double virial = 0.0;
    double insidePairs = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            std::array<double, 3> dr = system.minimumImageDistance(i, j);

            // Compute the squared distance between the two particles, skipping pairs beyond the cutoff
            double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            if (r2 >= cut_off_squared_) {
                continue;
            }

            auto [f_over_r, energy] = pairTerms(r2);

            // Newton's third law: equal and opposite force on each particle
            for (int k = 0; k < 3; ++k) {
                force[k][i] += f_over_r * dr[k];
                force[k][j] -= f_over_r * dr[k];
            }
            potential += energy;
            virial += f_over_r * r2; // r . F = (F / r) r^2
            insidePairs += 1.0;
        }
    }
    virial_ = virial;
    insidePairs_ = insidePairs;
    return potential;
}

// Neighbour list loop: only visit pairs that were within cutoff + skin at the last list rebuild
double LennardJonesForceField::computeForces(System& system, const NeighbourList& neighbourList) const {
    checkCutOff(system);

    if (deterministic_) {
        return computeForcesDeterministic(system, neighbourList);
    }
    if (mixedPrecision_) {
        return computeForcesMixed(system, neighbourList);
    }

    const std::size_t n = system.size();
    const double* r = system.packedPositions();
    constexpr std::size_t stride = System::positionStride;
    double* fxOut = system.force(0).data();
    double* fyOut = system.force(1).data();
    double* fzOut = system.force(2).data();

    double potential = 0.0;
    double virial = 0.0;
    double insidePairs = 0.0;

    // Half neighbour list, so each pair is computed once and its force applied to both particles. Two threads
    // can share a particle j, so each thread accumulates into its own force array, and the arrays are summed
    // per particle at the end
    #pragma omp parallel reduction(+ : potential, virial, insidePairs)
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
                const double* rj = r + stride * neighbours[k];
                xj[k] = rj[0];
                yj[k] = rj[1];
                zj[k] = rj[2];
            }

            // Compute: the pair maths, vectorised across neighbours. Branch-free, since SIMD lanes can't take
            // different branches: pairs beyond the cutoff (still in the list thanks to the skin) are computed
            // anyway and masked to zero
            const std::array<double, 3> ri {r[stride * i], r[stride * i + 1], r[stride * i + 2]};
            double fix = 0.0, fiy = 0.0, fiz = 0.0, energy_i = 0.0, virial_i = 0.0, inside_i = 0.0;

            #pragma omp simd reduction(+ : fix, fiy, fiz, energy_i, virial_i, inside_i)
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
                virial_i += f_over_r * r2;
                inside_i += inside ? 1.0 : 0.0;

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
            virial += virial_i;
            insidePairs += inside_i;
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
            fxOut[i] = total[0];
            fyOut[i] = total[1];
            fzOut[i] = total[2];
        }
    }
    virial_ = virial;
    insidePairs_ = insidePairs;
    return potential;
}

// Deterministic neighbour list loop: every pair's force is stored, then each particle sums its own contributions in a fixed order, so the result is bitwise identical for any thread count
double LennardJonesForceField::computeForcesDeterministic(System& system, const NeighbourList& neighbourList) const {

    // Check that the neighbour list has a reverse index, which is needed for deterministic forces. The reverse index records, for every particle j, the pairs in which it is the second particle. The default mode doesn't need it, so it is off by default
    if (!neighbourList.hasReverseIndex()) {
        throw std::runtime_error("Deterministic forces need a NeighbourList built with reverseIndex = true");
    }

    const std::size_t n = system.size();
    const double* r = system.packedPositions();
    constexpr std::size_t stride = System::positionStride;
    double* fxOut = system.force(0).data();
    double* fyOut = system.force(1).data();
    double* fzOut = system.force(2).data();
    const std::size_t pairs = neighbourList.getPairCount();

    // Resize the pair force and energy buffers to hold one entry per pair, indexed like the neighbour list. The buffers are kept between calls to avoid reallocating, which also means computeForces must not run concurrently on one force field
    for (auto* buffer : {&pairFx_, &pairFy_, &pairFz_, &pairEnergy_, &pairVirial_}) {
        buffer->resize(pairs);
    }

    const double boxLength = system.getBoxLength();
    const double inverseBoxLength = system.getInverseBoxLength();

    // Counting pairs in the cutoff: sums of whole numbers are exact in any order, so a reduction stays deterministic
    double insidePairs = 0.0;

    #pragma omp parallel reduction(+ : insidePairs)
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
                const double* rj = r + stride * neighbours[k];
                xj[k] = rj[0];
                yj[k] = rj[1];
                zj[k] = rj[2];
            }

            // Compute the pair maths, vectorised across neighbours. Each pair writes its own slot in the force and energy buffers, so there is no reduction whose order could vary
            const std::size_t first = neighbourList.firstPairOf(i);
            double* fx = pairFx_.data() + first;
            double* fy = pairFy_.data() + first;
            double* fz = pairFz_.data() + first;
            double* e = pairEnergy_.data() + first;
            double* w = pairVirial_.data() + first;
            const std::array<double, 3> ri {r[stride * i], r[stride * i + 1], r[stride * i + 2]};

            // Same branch-free pair maths as the default kernel, but every pair writes its own slot instead of
            // being summed, so there is no reduction whose order could vary
            double inside_i = 0.0;

            #pragma omp simd reduction(+ : inside_i)
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
                w[k] = f_over_r * r2;
                inside_i += inside ? 1.0 : 0.0;
            }
            insidePairs += inside_i;
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
        fxOut[i] = force[0];
        fyOut[i] = force[1];
        fzOut[i] = force[2];
    }

    virial_ = deterministicSum(pairs, [this](std::size_t k) { return pairVirial_[k]; });
    insidePairs_ = insidePairs;

    return deterministicSum(pairs, [this](std::size_t k) { return pairEnergy_[k]; });
}

// Mixed-precision neighbour list loop: same structure as the default path (half list, per-thread force arrays),
// but the separation vectors are computed in double during the gather and then converted to float, so the
// SIMD loop works on 4 floats per instruction instead of 2 doubles. Separations are at most the list radius,
// so float keeps about 7 significant digits of each one; the per-particle sums over ~40 neighbours stay in
// float, and everything accumulated across particles is double
double LennardJonesForceField::computeForcesMixed(System& system, const NeighbourList& neighbourList) const {
    const std::size_t n = system.size();
    const double* r = system.packedPositions();
    constexpr std::size_t stride = System::positionStride;
    double* fxOut = system.force(0).data();
    double* fyOut = system.force(1).data();
    double* fzOut = system.force(2).data();

    const double boxLength = system.getBoxLength();
    const double inverseBoxLength = system.getInverseBoxLength();
    const float sigma2 = static_cast<float>(sigma_ * sigma_);
    const float epsilon = static_cast<float>(epsilon_);
    const float cutOff2 = static_cast<float>(cut_off_squared_);
    const float shift = static_cast<float>(energy_shift_);

    double potential = 0.0;
    double virial = 0.0;
    double insidePairs = 0.0;

    #pragma omp parallel reduction(+ : potential, virial, insidePairs)
    {
        #pragma omp single
        threadForces_.resize(threadCount() * n);

        std::array<double, 3>* forces = threadForces_.data() + threadId() * n;
        std::fill(forces, forces + n, std::array<double, 3> {0.0, 0.0, 0.0});

        std::vector<float> dxs, dys, dzs, fxj, fyj, fzj;

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const auto neighbours = neighbourList.neighboursOf(i);
            const std::size_t m = neighbours.size();

            if (dxs.size() < m) {
                for (auto* buffer : {&dxs, &dys, &dzs, &fxj, &fyj, &fzj}) {
                    buffer->resize(m);
                }
            }

            // Gather + minimum image in double, then narrow the (small) separations to float
            const double* ri = r + stride * i;
            for (std::size_t k = 0; k < m; ++k) {
                const double* rj = r + stride * neighbours[k];
                double dx = ri[0] - rj[0];
                double dy = ri[1] - rj[1];
                double dz = ri[2] - rj[2];
                dx -= boxLength * std::round(dx * inverseBoxLength);
                dy -= boxLength * std::round(dy * inverseBoxLength);
                dz -= boxLength * std::round(dz * inverseBoxLength);
                dxs[k] = static_cast<float>(dx);
                dys[k] = static_cast<float>(dy);
                dzs[k] = static_cast<float>(dz);
            }

            float fix = 0.0f, fiy = 0.0f, fiz = 0.0f, energy_i = 0.0f, virial_i = 0.0f, inside_i = 0.0f;

            #pragma omp simd reduction(+ : fix, fiy, fiz, energy_i, virial_i, inside_i)
            for (std::size_t k = 0; k < m; ++k) {
                const float dx = dxs[k];
                const float dy = dys[k];
                const float dz = dzs[k];
                const float r2 = dx * dx + dy * dy + dz * dz;

                const float inv_r2 = 1.0f / r2;
                const float sr2 = sigma2 * inv_r2;
                const float sr6 = sr2 * sr2 * sr2;
                const float sr12 = sr6 * sr6;

                const bool inside = r2 < cutOff2;
                const float f_over_r = inside ? 24.0f * epsilon * (2.0f * sr12 - sr6) * inv_r2 : 0.0f;
                energy_i += inside ? 4.0f * epsilon * (sr12 - sr6) - shift : 0.0f;
                virial_i += f_over_r * r2;
                inside_i += inside ? 1.0f : 0.0f;

                fix += f_over_r * dx;
                fiy += f_over_r * dy;
                fiz += f_over_r * dz;
                fxj[k] = f_over_r * dx;
                fyj[k] = f_over_r * dy;
                fzj[k] = f_over_r * dz;
            }

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
            virial += virial_i;
            insidePairs += inside_i;
        }

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
            fxOut[i] = total[0];
            fyOut[i] = total[1];
            fzOut[i] = total[2];
        }
    }
    virial_ = virial;
    insidePairs_ = insidePairs;
    return potential;
}
