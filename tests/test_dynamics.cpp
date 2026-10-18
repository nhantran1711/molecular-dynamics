#include "TestHelpers.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <vector>

TEST(Initialisation, FccLatticeHasExpectedCountAndSpacing) {
    const int cells = 3;
    const double a = 1.5;
    auto particles = createFCCLattice(cells, a);
    ASSERT_EQ(particles.size(), 4u * cells * cells * cells);

    // Nearest neighbours on an FCC lattice sit a / sqrt(2) apart
    System system(particles, cells * a);
    double minDistance = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < particles.size(); ++i) {
        for (std::size_t j = i + 1; j < particles.size(); ++j) {
            auto dr = system.minimumImageDistance(particles[i].position, particles[j].position);
            minDistance = std::min(minDistance, std::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]));
        }
    }
    EXPECT_NEAR(minDistance, a / std::sqrt(2.0), 1e-12);
}

TEST(Initialisation, VelocitiesHitTargetTemperatureWithZeroMomentum) {
    System system = test::makeLattice(5);
    EXPECT_NEAR(system.temperature(), test::temperature, 1e-12);
    for (double p : test::totalMomentum(system)) {
        EXPECT_NEAR(p, 0.0, 1e-12);
    }
}

TEST(Thermostat, RescalesToTarget) {
    System system = test::makeLattice(4);
    rescaleVelocities(system, 1.7);
    EXPECT_NEAR(system.temperature(), 1.7, 1e-12);
}

// NVE with the thermostat off: total energy must stay put. A missed pair, a race or a wrong force shows up as drift
TEST(Dynamics, ConservesEnergy) {
    System system = test::makeLiquid(5, 500);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    VelocityVerletIntegrator integrator(test::dt);
    NeighbourList list(test::cutOff, test::skin);
    list.build(system);
    double potential = forceField.computeForces(system, list);
    const double initial = system.kineticEnergy() + potential;

    double maxDrift = 0.0;
    for (int s = 0; s < 2000; ++s) {
        potential = integrator.step(system, forceField, list);
        maxDrift = std::max(maxDrift, std::abs((system.kineticEnergy() + potential - initial) / initial));
    }
    EXPECT_LT(maxDrift, 3e-4);
    EXPECT_GT(list.getRebuildCount(), 1);
}

// Pair forces are equal and opposite, so total momentum (zero at the start) must stay zero
TEST(Dynamics, ConservesMomentum) {
    System system = test::makeLiquid(5, 0);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    VelocityVerletIntegrator integrator(test::dt);
    NeighbourList list(test::cutOff, test::skin);
    list.build(system);
    forceField.computeForces(system, list);

    for (int s = 0; s < 1000; ++s) {
        integrator.step(system, forceField, list);
    }
    for (double p : test::totalMomentum(system)) {
        EXPECT_NEAR(p, 0.0, 1e-10);
    }
}

// Velocity Verlet is time-reversible: run forward, flip velocities, run the same number of steps, and every
// particle returns to its start. Uses the all-pairs path, which keeps particle order fixed
TEST(Dynamics, IsTimeReversible) {
    System system = test::makeLiquid(4, 200);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    VelocityVerletIntegrator integrator(test::dt);
    forceField.computeForces(system);

    std::vector<std::array<double, 3>> start;
    for (const auto& p : system.getParticles()) {
        start.push_back(p.position);
    }

    const int steps = 100;
    for (int s = 0; s < steps; ++s) {
        integrator.step(system, forceField);
    }
    for (auto& p : system.getParticles()) {
        for (double& v : p.velocity) {
            v = -v;
        }
    }
    for (int s = 0; s < steps; ++s) {
        integrator.step(system, forceField);
    }

    double maxError = 0.0;
    const auto& particles = system.getParticles();
    for (std::size_t i = 0; i < particles.size(); ++i) {
        auto dr = system.minimumImageDistance(particles[i].position, start[i]);
        maxError = std::max({maxError, std::abs(dr[0]), std::abs(dr[1]), std::abs(dr[2])});
    }
    EXPECT_LT(maxError, 1e-8);
}
