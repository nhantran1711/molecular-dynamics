#include "TestHelpers.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

// Melt a lattice with the thermostat on (which feeds kinetic energy back into the trajectory), then run NVE,
// with enough steps for many neighbour list rebuilds. Returns the hash of the final state
std::uint64_t runAndHash(int threads, bool deterministic) {
    std::uint64_t hash = 0;
    test::withThreads(threads, [&] {
        System system = test::makeLattice(6);
        LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
        forceField.setDeterministic(deterministic);
        VelocityVerletIntegrator integrator(test::dt);
        NeighbourList list(test::cutOff, test::skin, deterministic);
        list.build(system);
        forceField.computeForces(system, list);

        for (int s = 1; s <= 400; ++s) {
            integrator.step(system, forceField, list);
            if (s <= 200 && s % 10 == 0) {
                rescaleVelocities(system, test::temperature);
            }
        }
        hash = system.stateHash();
    });
    return hash;
}

}

// Default mode: the same thread count must reproduce the same trajectory bit for bit
TEST(Determinism, DefaultModeIsReproducibleAtFixedThreadCount) {
    for (int threads : {3, 4}) {
        const std::uint64_t first = runAndHash(threads, false);
        for (int run = 0; run < 3; ++run) {
            EXPECT_EQ(runAndHash(threads, false), first) << threads << " threads, run " << run;
        }
    }
}

// Deterministic mode: any thread count must give the same trajectory bit for bit
TEST(Determinism, DeterministicModeIsIndependentOfThreadCount) {
    const std::uint64_t reference = runAndHash(1, true);
    for (int threads : {2, 3, 4, 8}) {
        EXPECT_EQ(runAndHash(threads, true), reference) << threads << " threads";
    }
}

TEST(Determinism, DeterministicForcesMatchAllPairs) {
    System system = test::makeLiquid(8);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);

    for (int threads : {1, 3, 8}) {
        test::withThreads(threads, [&] {
            NeighbourList list(test::cutOff, test::skin, true);
            list.build(system);

            forceField.setDeterministic(true);
            double energy = forceField.computeForces(system, list);
            std::vector<std::array<double, 3>> forces;
            for (const auto& p : system.getParticles()) {
                forces.push_back(p.force);
            }

            double reference = forceField.computeForces(system);
            EXPECT_NEAR(energy, reference, 1e-9 * std::abs(reference)) << threads << " threads";

            double maxDiff = 0.0;
            for (std::size_t i = 0; i < forces.size(); ++i) {
                for (int k = 0; k < 3; ++k) {
                    maxDiff = std::max(maxDiff, std::abs(forces[i][k] - system.getParticles()[i].force[k]));
                }
            }
            EXPECT_LT(maxDiff, 1e-11) << threads << " threads";
        });
    }
}

TEST(Determinism, RequiresReverseIndex) {
    System system = test::makeLattice(6);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    forceField.setDeterministic(true);
    NeighbourList list(test::cutOff, test::skin);   // no reverse index
    list.build(system);
    EXPECT_THROW(forceField.computeForces(system, list), std::runtime_error);
}

TEST(Determinism, ReverseIndexListsEveryPairOnce) {
    System system = test::makeLiquid(6);
    NeighbourList list(test::cutOff, test::skin, true);
    list.build(system);

    std::vector<int> seen(list.getPairCount(), 0);
    for (std::size_t j = 0; j < system.getParticles().size(); ++j) {
        std::size_t previous = 0;
        bool firstPair = true;
        for (std::size_t k : list.pairsAsSecond(j)) {
            ++seen[k];
            EXPECT_TRUE(firstPair || k > previous) << "reverse pairs should be in increasing order";
            previous = k;
            firstPair = false;
        }
    }
    for (int count : seen) {
        EXPECT_EQ(count, 1);
    }
}
