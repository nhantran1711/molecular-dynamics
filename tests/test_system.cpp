#include "TestHelpers.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

namespace {

System emptyBox(double boxLength) {
    return System({}, boxLength);
}

}

TEST(System, MinimumImagePicksNearestPeriodicCopy) {
    System system = emptyBox(10.0);

    // 9 apart directly, but only 1 apart through the boundary
    auto dr = system.minimumImageDistance({0.5, 0.5, 0.5}, {9.5, 9.5, 9.5});
    for (double d : dr) {
        EXPECT_NEAR(d, 1.0, 1e-12);
    }

    // Already the nearest copy: unchanged
    dr = system.minimumImageDistance({3.0, 4.0, 5.0}, {1.0, 6.0, 5.0});
    EXPECT_NEAR(dr[0], 2.0, 1e-12);
    EXPECT_NEAR(dr[1], -2.0, 1e-12);
    EXPECT_NEAR(dr[2], 0.0, 1e-12);
}

TEST(System, MinimumImageComponentsNeverExceedHalfBox) {
    System system = emptyBox(7.0);
    for (double x = -20.0; x <= 20.0; x += 0.37) {
        auto dr = system.minimumImageDistance({x, 2.0 * x, -x}, {0.0, 0.0, 0.0});
        for (double d : dr) {
            EXPECT_LE(std::abs(d), 3.5 + 1e-12);
        }
    }
}

TEST(System, WrapPositionMapsIntoBox) {
    System system = emptyBox(10.0);

    std::array<double, 3> r {-0.5, 10.0, 23.0};
    system.wrapPosition(r);
    EXPECT_NEAR(r[0], 9.5, 1e-12);
    EXPECT_NEAR(r[1], 0.0, 1e-12);
    EXPECT_NEAR(r[2], 3.0, 1e-12);

    for (double x = -35.0; x <= 35.0; x += 0.73) {
        std::array<double, 3> p {x, -x, 0.5 * x};
        system.wrapPosition(p);
        for (double c : p) {
            EXPECT_GE(c, 0.0);
            EXPECT_LT(c, 10.0);
        }
    }
}

TEST(System, KineticEnergyAndTemperature) {
    Particle a;
    Particle b;
    a.velocity = {1.0, 0.0, 0.0};
    b.velocity = {-1.0, 0.0, 0.0};
    b.mass = 2.0;
    System system({a, b}, 10.0);

    // KE = 0.5 * 1 * 1 + 0.5 * 2 * 1
    EXPECT_DOUBLE_EQ(system.kineticEnergy(), 1.5);
    // T = 2 KE / (3N - 3)
    EXPECT_DOUBLE_EQ(system.temperature(), 2.0 * 1.5 / 3.0);
}

TEST(System, ReorderPermutesParticles) {
    std::vector<Particle> particles(4);
    for (int i = 0; i < 4; ++i) {
        particles[i].position = {static_cast<double>(i), 0.0, 0.0};
    }
    System system(particles, 10.0);

    const std::vector<std::size_t> order {2, 0, 3, 1};
    system.reorder(order);

    for (std::size_t k = 0; k < order.size(); ++k) {
        EXPECT_DOUBLE_EQ(system.getParticles()[k].position[0], static_cast<double>(order[k]));
    }
}
