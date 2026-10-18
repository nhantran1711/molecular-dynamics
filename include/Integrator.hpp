#pragma once

#include "Forcefield.hpp"
#include "NeighbourList.hpp"
#include "System.hpp"


class VelocityVerletIntegrator {
public:
    explicit VelocityVerletIntegrator(double dt);


    // Advance the system by one timestep and return the new potential energy.
    double step(System& system, const LennardJonesForceField& forceField) const;

    // Same as above, but updates the neighbour list after the drift and uses it for the force calculation
    double step(System& system, const LennardJonesForceField& forceField, NeighbourList& neighbourList) const;

    double getTimestep() const {
        return dt_;
    };

private:
    // Half kick the velocities, then drift and wrap the positions by a full timestep
    void kickAndDrift(System& system) const;

    // Half kick the velocities using the newly computed forces
    void kick(System& system) const;

    double dt_;
};
