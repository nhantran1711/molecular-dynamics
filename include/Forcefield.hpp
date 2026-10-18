#pragma once

#include "NeighbourList.hpp"
#include "System.hpp"


class LennardJonesForceField {
public:
    LennardJonesForceField(double epsilon, double sigma, double cut_off);

    // Compute the forces on every particle by checking all pairs (O(N^2)) and return the total potential energy
    double computeForces(System& system) const;

    // Compute the forces using only the pairs in the neighbour list (O(N)) and return the total potential energy
    double computeForces(System& system, const NeighbourList& neighbourList) const;

    double getCutOff() const {
        return cut_off_;
    };

private:
    // Add the force between one pair to both particles and return the pair's potential energy (0 beyond the cutoff)
    double addPairForce(Particle& pi, Particle& pj, const std::array<double, 3>& dr) const;

    // Check that the cutoff is less than half the box length, throwing an exception if not
    void checkCutOff(const System& system) const;

    double epsilon_;
    double sigma_;
    double cut_off_;
    double cut_off_squared_;
    double energy_shift_;
};
