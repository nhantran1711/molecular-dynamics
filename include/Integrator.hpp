#pragma once

#include "Forcefield.hpp"
#include "System.hpp"


class VelocityVerletIntegrator {
public:
    explicit VelocityVerletIntegrator(double dt);


    // Advance the system by one timestep and return the new potential energy.
    double step(System& system, const LennardJonesForceField& forceField) const;

    double getTimestep() const {
        return dt_;
    };

private:
    double dt_;
};
