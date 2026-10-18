#include "RadialDistribution.hpp"
#include "TestHelpers.hpp"
#include <gtest/gtest.h>
#include <random>

TEST(Observables, TwoParticleVirialAndInsideCount) {
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    Particle a;
    Particle b;
    a.position = {5.0, 5.0, 5.0};
    b.position = {6.3, 5.0, 5.0};
    System system({a, b}, 20.0);
    forceField.computeForces(system);

    // W = r . F = r * F(r), F(r) = 24 (2 r^-13 - r^-7)
    const double r = 1.3;
    EXPECT_NEAR(forceField.getVirial(), r * 24.0 * (2.0 * std::pow(r, -13) - std::pow(r, -7)), 1e-12);
    EXPECT_DOUBLE_EQ(forceField.getInsidePairs(), 1.0);
}

// Every force path must report the same virial and inside-pair count as the all-pairs reference
TEST(Observables, VirialAndInsideCountMatchAcrossForcePaths) {
    System system = test::makeLiquid(8);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    NeighbourList list(test::cutOff, test::skin, true);
    list.build(system);

    forceField.computeForces(system);
    const double virial = forceField.getVirial();
    const double inside = forceField.getInsidePairs();

    for (bool deterministic : {false, true}) {
        forceField.setDeterministic(deterministic);
        forceField.computeForces(system, list);
        EXPECT_NEAR(forceField.getVirial(), virial, 1e-9 * std::abs(virial)) << "deterministic = " << deterministic;
        EXPECT_DOUBLE_EQ(forceField.getInsidePairs(), inside) << "deterministic = " << deterministic;
    }
}

// With no forces only the thermostat changes the velocities, so it alone must produce the canonical
// distribution: mean temperature at the target and Var(KE) = dof * T^2 / 2
TEST(Observables, BussiThermostatSamplesCanonicalKineticEnergy) {
    // One thread: the system is tiny, so per-call OpenMP overhead would dominate. A short tau gives a new
    // independent kinetic energy every ~10 steps, so 100k steps hold ~10k independent samples (Var error ~1.4%)
    test::withThreads(1, [] {
        System system = test::makeLattice(5);
        const double target = 0.85;
        const double dof = 3.0 * static_cast<double>(system.size()) - 3.0;
        BussiThermostat thermostat(target, 0.05, 7);

        double sumK = 0.0;
        double sumK2 = 0.0;
        int samples = 0;
        for (int s = 0; s < 100000; ++s) {
            thermostat.apply(system, 0.005);
            if (s >= 1000) {
                const double k = system.kineticEnergy();
                sumK += k;
                sumK2 += k * k;
                ++samples;
            }
        }
        const double meanK = sumK / samples;
        const double varK = sumK2 / samples - meanK * meanK;
        EXPECT_NEAR(meanK / (0.5 * dof), target, 0.005 * target);
        EXPECT_NEAR(varK / (0.5 * dof * target * target), 1.0, 0.05);
    });
}

// Uniformly random positions are an ideal gas, whose g(r) is 1 at every distance
TEST(Observables, RadialDistributionOfIdealGasIsOne) {
    const double boxLength = 12.0;
    const int n = 2000;
    std::mt19937 rng(3); // NOLINT(bugprone-random-generator-seed): fixed seed keeps the test reproducible
    std::uniform_real_distribution<double> uniform(0.0, boxLength);

    RadialDistribution rdf(test::cutOff, 25);
    for (int sample = 0; sample < 40; ++sample) {
        std::vector<Particle> particles(n);
        for (auto& p : particles) {
            p.position = {uniform(rng), uniform(rng), uniform(rng)};
        }
        System system(particles, boxLength);
        NeighbourList list(test::cutOff, test::skin);
        list.build(system);
        rdf.sample(system, list);
    }

    const auto r = rdf.radii();
    const auto g = rdf.values();
    for (std::size_t b = 0; b < g.size(); ++b) {
        if (r[b] > 1.0) { // small shells hold few pairs, so their noise is large
            EXPECT_NEAR(g[b], 1.0, 0.06) << "r = " << r[b];
        }
    }
}
