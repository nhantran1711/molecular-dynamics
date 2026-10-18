#include "Integrator.hpp"

VelocityVerletIntegrator::VelocityVerletIntegrator(double dt) : dt_(dt) {
}

// Velocity Verlet: half kick, drift, recompute forces, half kick
double VelocityVerletIntegrator::step(System& system, const LennardJonesForceField& forceField) const {

    // Half kick the velocities, then drift and wrap the positions by a full timestep
    kickAndDrift(system);

    // Compute the forces and potential energy using all pairs
    double potential = forceField.computeForces(system);
    kick(system);
    return potential;
}

double VelocityVerletIntegrator::step(System& system, const LennardJonesForceField& forceField,
                                      NeighbourList& neighbourList) const {

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
    const std::size_t n = system.size();
    const double* m = system.mass().data();

    // Half kick: update velocities by half a timestep, then drift positions by a full timestep and wrap them.
    // One loop per component; positions are packed, so x is read with a stride
    constexpr std::size_t stride = System::positionStride;
    double* r = system.packedPositions();

    #pragma omp parallel
    for (int k = 0; k < 3; ++k) {
        double* v = system.velocity(k).data();
        const double* f = system.force(k).data();

        #pragma omp for simd schedule(static) nowait
        for (std::size_t i = 0; i < n; ++i) {
            v[i] += half_dt * f[i] / m[i];
            r[stride * i + k] = system.wrapCoordinate(r[stride * i + k] + dt_ * v[i]);
        }
    }
}

void VelocityVerletIntegrator::kick(System& system) const {
    const double half_dt = 0.5 * dt_;
    const std::size_t n = system.size();
    const double* m = system.mass().data();

    // Half kick: update velocities by half a timestep
    #pragma omp parallel
    for (int k = 0; k < 3; ++k) {
        double* v = system.velocity(k).data();
        const double* f = system.force(k).data();

        #pragma omp for simd schedule(static) nowait
        for (std::size_t i = 0; i < n; ++i) {
            v[i] += half_dt * f[i] / m[i];
        }
    }
}
