#pragma once

#include "System.hpp"


class LennardJonesForceField {
public:
    LennardJonesForceField(double epsilon, double sigma, double cut_off);


private:
    double epsilon_;
    double sigma_;
    double cut_off_;
    double cut_off_squared_;
    double energy_shift_;
};