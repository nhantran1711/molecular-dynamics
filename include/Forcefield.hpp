#pragma once

#include "NeighbourList.hpp"
#include "System.hpp"

class LennardJonesForceField {
public:
    LennardJonesForceField(double epsilon, double sigma, double cut_off);

    // Compute the forces on every particle by checking all pairs (O(N^2)) and return the total potential energy
    double computeForces(System& system) const;

    // Compute the forces using only the pairs in the neighbour list (O(N)), in parallel with OpenMP and
    // with the pair maths vectorised (SIMD), and return the total potential energy
    double computeForces(System& system, const NeighbourList& neighbourList) const;

    double getCutOff() const {
        return cut_off_;
    };

    // The constant subtracted from every pair energy inside the cutoff, V(r_c), so the potential is zero there
    double getEnergyShift() const {
        return energy_shift_;
    };

    // The virial W = sum(r_ij . F_ij) over every pair inside the cutoff, used to compute the pressure
    double getVirial() const {
        return virial_;
    };

    double getInsidePairs() const {
        return insidePairs_;
    };

    // Set whether to use the deterministic force mode, which produces bitwise-identical forces for any thread count
    // Mixed precision: positions, minimum image and force accumulation stay double, but the pair maths
    // (r^2, the LJ terms, F/r) runs in float, so each SIMD instruction handles 4 pairs instead of 2.
    // Applies to the default neighbour list path only
    void setMixedPrecision(bool mixed) {
        mixedPrecision_ = mixed;
    };

    void setDeterministic(bool deterministic) {
        deterministic_ = deterministic;
    };

private:
    // For a pair at squared distance r2 inside the cutoff, return F/r and the pair's shifted potential energy
    struct PairTerms {
        double f_over_r;
        double energy;
    };
    PairTerms pairTerms(double r2) const;

    // Check that the cutoff is less than half the box length, throwing an exception if not
    void checkCutOff(const System& system) const;

    double computeForcesDeterministic(System& system, const NeighbourList& neighbourList) const;
    double computeForcesMixed(System& system, const NeighbourList& neighbourList) const;

    // One force array per thread for the neighbour list loop, summed at the end. Kept between calls to
    // avoid reallocating, which also means computeForces must not run concurrently on one force field
    mutable std::vector<std::array<double, 3>> threadForces_;

    // Deterministic mode: force components and energy of every pair, indexed like the neighbour list
    mutable std::vector<double> pairFx_, pairFy_, pairFz_, pairEnergy_, pairVirial_;

    mutable double virial_ {0.0};
    mutable double insidePairs_ {0.0};
    bool deterministic_ {false};
    bool mixedPrecision_ {false};

    double epsilon_;
    double sigma_;
    double cut_off_;
    double cut_off_squared_;
    double energy_shift_;
};
