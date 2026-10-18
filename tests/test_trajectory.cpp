#include "TestHelpers.hpp"
#include "Trajectory.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Parse one extended XYZ frame back into positions, in file order
std::vector<std::array<double, 3>> readFrame(std::FILE* file, std::size_t& n, std::vector<std::size_t>& ids) {
    char line[512];
    std::fgets(line, sizeof(line), file);
    n = std::stoul(line);
    std::fgets(line, sizeof(line), file); // header
    std::vector<std::array<double, 3>> positions;
    for (std::size_t k = 0; k < n; ++k) {
        std::fgets(line, sizeof(line), file);
        std::istringstream in(line);
        std::string species;
        std::array<double, 3> r {};
        std::array<double, 3> v {};
        std::size_t id = 0;
        in >> species >> r[0] >> r[1] >> r[2] >> v[0] >> v[1] >> v[2] >> id;
        positions.push_back(r);
        ids.push_back(id);
    }
    return positions;
}

} // namespace

TEST(Trajectory, IdsFollowParticlesThroughReordering) {
    System system = test::makeLiquid(6, 0);
    std::vector<std::array<double, 3>> byId(system.size());
    for (std::size_t i = 0; i < system.size(); ++i) {
        byId[system.ids()[i]] = system.positionOf(i);
    }

    NeighbourList list(test::cutOff, test::skin);
    list.build(system); // sorts particles by cell

    std::vector<std::size_t> ids(system.ids().begin(), system.ids().end());
    EXPECT_FALSE(std::is_sorted(ids.begin(), ids.end())) << "the build should have re-sorted the particles";
    std::sort(ids.begin(), ids.end());
    for (std::size_t k = 0; k < ids.size(); ++k) {
        EXPECT_EQ(ids[k], k) << "ids must stay a permutation";
    }
    for (std::size_t i = 0; i < system.size(); ++i) {
        EXPECT_EQ(system.positionOf(i), byId[system.ids()[i]]);
    }
}

// Line k of every frame must be the same particle: between two frames one step apart, with a re-sort in
// between, every line moves by much less than the particle spacing
TEST(Trajectory, FramesAreConsistentAcrossRebuilds) {
    System system = test::makeLiquid(6);
    LennardJonesForceField forceField(1.0, 1.0, test::cutOff);
    VelocityVerletIntegrator integrator(test::dt);
    NeighbourList list(test::cutOff, test::skin);
    list.build(system);
    forceField.computeForces(system, list);

    // Closed automatically, including when an ASSERT returns early
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> owner(std::tmpfile(), &std::fclose);
    std::FILE* file = owner.get();
    ASSERT_NE(file, nullptr);
    writeXyzFrame(file, system, 0.0);
    integrator.step(system, forceField, list);
    list.build(system); // force a re-sort between the frames
    writeXyzFrame(file, system, test::dt);
    ASSERT_EQ(std::fseek(file, 0, SEEK_SET), 0);

    std::size_t n1 = 0;
    std::size_t n2 = 0;
    std::vector<std::size_t> ids1;
    std::vector<std::size_t> ids2;
    auto first = readFrame(file, n1, ids1);
    auto second = readFrame(file, n2, ids2);

    ASSERT_EQ(n1, system.size());
    ASSERT_EQ(n2, system.size());
    double maxMove = 0.0;
    for (std::size_t k = 0; k < n1; ++k) {
        EXPECT_EQ(ids1[k], k);
        EXPECT_EQ(ids2[k], k);
        auto d = system.minimumImageDistance(first[k], second[k]);
        maxMove = std::max(maxMove, std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
    }
    EXPECT_LT(maxMove, 0.1);
}
