#include "Integrator.hpp"

VelocityVerletIntegrator::VelocityVerletIntegrator(double dt) : dt_(dt) {}

// Velocity Verlet: half kick, drift, recompute forces, half kick
double VelocityVerletIntegrator::step(System& system, const LennardJonesForceField& forceField) const {


    auto& particles = system.getParticles();
    const double half_dt = 0.5 * dt_;

    // Half kick: update velocities by half a timestep
    for (auto& particle : particles) {
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] += half_dt * particle.force[k] / particle.mass;
            particle.position[k] += dt_ * particle.velocity[k];
        }
        system.wrapPosition(particle.position);
    }

    double potential = forceField.computeForces(system);

    // Half kick: update velocities by half a timestep
    for (auto& particle : particles) {
        for (int k = 0; k < 3; ++k) {
            particle.velocity[k] += half_dt * particle.force[k] / particle.mass;
        }
    }
    return potential;
}
