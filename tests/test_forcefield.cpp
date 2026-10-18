#include "TestHelpers.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

// Analytic Lennard-Jones force magnitude (positive = repulsive) and shifted energy, epsilon = sigma = 1
double ljForce(double r) {
    return 24.0 * (2.0 * std::pow(r, -13) - std::pow(r, -7));
}

double ljEnergy(double r, double cutOff) {
    auto v = [](double x) { return 4.0 * (std::pow(x, -12) - std::pow(x, -6)); };
    return v(r) - v(cutOff);
}

System pairAt(double x0, double x1, double boxLength) {
    Particle a;
    Particle b;
    a.position = {x0, 5.0, 5.0};
    b.position = {x1, 5.0, 5.0};
    return System({a, b}, boxLength);
}

} // namespace

TEST(ForceField, TwoParticlesMatchAnalyticForceAndEnergy) {
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    for (double r : {0.95, 1.12, 1.5, 2.0, 2.45}) {
        System system = pairAt(5.0, 5.0 + r, 20.0);
        double energy = forceField.computeForces(system);

        // Particle 0 sits at lower x, so a repulsive force pushes it towards -x
        EXPECT_NEAR(system.forceOf(0)[0], -ljForce(r), 1e-10) << "r = " << r;
        EXPECT_NEAR(system.forceOf(1)[0], ljForce(r), 1e-10) << "r = " << r;
        EXPECT_NEAR(system.forceOf(0)[1], 0.0, 1e-12);
        EXPECT_NEAR(energy, ljEnergy(r, test::cutOff), 1e-12) << "r = " << r;
    }
}

TEST(ForceField, ForceVanishesAtPotentialMinimum) {
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    System system = pairAt(5.0, 5.0 + std::pow(2.0, 1.0 / 6.0), 20.0);
    forceField.computeForces(system);
    EXPECT_NEAR(system.forceOf(0)[0], 0.0, 1e-12);
}

TEST(ForceField, NothingBeyondCutoff) {
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    System system = pairAt(5.0, 5.0 + test::cutOff + 1e-9, 20.0);
    EXPECT_DOUBLE_EQ(forceField.computeForces(system), 0.0);
    EXPECT_DOUBLE_EQ(system.forceOf(0)[0], 0.0);
}

TEST(ForceField, InteractsAcrossPeriodicBoundary) {
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    // 19 apart directly, 1 apart through the boundary: particle 0 is pushed towards +x, away from its image
    System system = pairAt(0.5, 19.5, 20.0);
    double energy = forceField.computeForces(system);
    EXPECT_NEAR(system.forceOf(0)[0], ljForce(1.0), 1e-10);
    EXPECT_NEAR(energy, ljEnergy(1.0, test::cutOff), 1e-12);
}

TEST(ForceField, RejectsCutoffBeyondHalfBox) {
    LennardJonesForceField forceField(1.0, 1.0, 2.5);
    System system = pairAt(1.0, 2.0, 4.0);
    EXPECT_THROW(forceField.computeForces(system), std::runtime_error);
}

// The neighbour list path (parallel, SIMD) must reproduce the O(N^2) reference, for any thread count,
// including counts that don't divide the work evenly
TEST(ForceField, NeighbourListMatchesAllPairs) {
    System system = test::makeLiquid(8);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);

    for (int threads : {1, 3, 8}) {
        test::withThreads(threads, [&] {
            NeighbourList list(test::cutOff, test::skin);
            list.build(system); // also re-sorts the particles, so do it before taking the reference

            double listEnergy = forceField.computeForces(system, list);
            std::vector<std::array<double, 3>> listForces;
            listForces.reserve(system.size());
            for (std::size_t i = 0; i < system.size(); ++i) {
                listForces.push_back(system.forceOf(i));
            }

            double referenceEnergy = forceField.computeForces(system);
            EXPECT_NEAR(listEnergy, referenceEnergy, 1e-9 * std::abs(referenceEnergy)) << threads << " threads";

            double maxDiff = 0.0;
            for (std::size_t i = 0; i < listForces.size(); ++i) {
                for (int k = 0; k < 3; ++k) {
                    maxDiff = std::max(maxDiff, std::abs(listForces[i][k] - system.forceOf(i)[k]));
                }
            }
            EXPECT_LT(maxDiff, 1e-11) << threads << " threads";
        });
    }
}

TEST(ForceField, TotalForceIsZero) {
    System system = test::makeLiquid(6);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    NeighbourList list(test::cutOff, test::skin);
    list.build(system);
    forceField.computeForces(system, list);

    std::array<double, 3> total {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
        for (double f : system.force(k)) {
            total[k] += f;
        }
    }
    for (double f : total) {
        EXPECT_NEAR(f, 0.0, 1e-10);
    }
}

// Mixed precision computes the pair maths in float: forces and energy must agree with the double path to
// float accuracy, and dynamics must still conserve energy
TEST(ForceField, MixedPrecisionMatchesDouble) {
    System system = test::makeLiquid(8);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    NeighbourList list(test::cutOff, test::skin);
    list.build(system);

    double energy = forceField.computeForces(system, list);
    std::vector<std::array<double, 3>> reference;
    reference.reserve(system.size());
    for (std::size_t i = 0; i < system.size(); ++i) {
        reference.push_back(system.forceOf(i));
    }

    forceField.setMixedPrecision(true);
    double mixedEnergy = forceField.computeForces(system, list);
    EXPECT_NEAR(mixedEnergy, energy, 1e-5 * std::abs(energy));

    double maxDiff = 0.0;
    double maxForce = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            maxDiff = std::max(maxDiff, std::abs(system.forceOf(i)[k] - reference[i][k]));
            maxForce = std::max(maxForce, std::abs(reference[i][k]));
        }
    }
    EXPECT_LT(maxDiff, 1e-5 * maxForce);
}

TEST(ForceField, MixedPrecisionConservesEnergy) {
    System system = test::makeLiquid(5, 500);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    forceField.setMixedPrecision(true);
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
}
