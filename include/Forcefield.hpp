#pragma once

#include "System.hpp"


class LennardJonesForceField {
public:
    LennardJonesForceField(double epsilon, double sigma, double cut_off);

    // Compute the forces on every particle and return the total potential energy
    double computeForces(System& system) const;

    double getCutOff() const {
        return cut_off_;
    };

private:
    double epsilon_;
    double sigma_;
    double cut_off_;
    double cut_off_squared_;
    double energy_shift_;
};
