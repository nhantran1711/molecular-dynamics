#include "Integrator.hpp"

VelocityVerletIntegrator::VelocityVerletIntegrator(double dt) : dt_(dt) {}

// Velocity Verlet: half kick, drift, recompute forces, half kick
double VelocityVerletIntegrator::step(System& system, const LennardJonesForceField& forceField) const {
    
    // Half kick the velocities, then drift and wrap the positions by a full timestep
    kickAndDrift(system);

    // Compute the forces and potential energy using all pairs
    double potential = forceField.computeForces(system);
    kick(system);
    return potential;
}

double VelocityVerletIntegrator::step(System& system, const LennardJonesForceField& forceField, NeighbourList& neighbourList) const {
    
    // Half kick the velocities, then drift and wrap the positions by a full timestep
    kickAndDrift(system);
    neighbourList.update(system);
    
    // Compute the forces and potential energy using the neighbour list
    double potential = forceField.computeForces(system, neighbourList);
    kick(system);


    return potential;
}

void VelocityVerletIntegrator::kickAndDrift(System& system) const {
    const double half_dt = 0.5 * dt_;

    // Half kick: update velocities by half a timestep, then drift positions by a full timestep
    for (auto& particle : system.getParticles()) {
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] += half_dt * particle.force[k] / particle.mass;
            particle.position[k] += dt_ * particle.velocity[k];
        }
        system.wrapPosition(particle.position);
    }
}

void VelocityVerletIntegrator::kick(System& system) const {
    const double half_dt = 0.5 * dt_;

    // Half kick: update velocities by half a timestep
    for (auto& particle : system.getParticles()) {
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] += half_dt * particle.force[k] / particle.mass;
        }
    }
}
