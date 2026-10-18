#include "TestHelpers.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace {

using PairSet = std::set<std::pair<std::size_t, std::size_t>>;

PairSet pairsInList(const NeighbourList& list, std::size_t n) {
    PairSet pairs;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j : list.neighboursOf(i)) {
            EXPECT_GT(j, i) << "half list should only store j > i";
            pairs.insert({i, j});
        }
    }
    return pairs;
}

PairSet pairsWithin(const System& system, double radius) {
    PairSet pairs;
    const auto& p = system.getParticles();
    for (std::size_t i = 0; i < p.size(); ++i) {
        for (std::size_t j = i + 1; j < p.size(); ++j) {
            auto dr = system.minimumImageDistance(p[i].position, p[j].position);
            if (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2] < radius * radius) {
                pairs.insert({i, j});
            }
        }
    }
    return pairs;
}

}

TEST(NeighbourList, ContainsExactlyThePairsWithinListRadius) {
    System system = test::makeLiquid(8);
    for (int threads : {1, 3, 8}) {
        test::withThreads(threads, [&] {
            NeighbourList list(test::cutOff, test::skin);
            list.build(system);
            const auto expected = pairsWithin(system, test::cutOff + test::skin);
            EXPECT_EQ(pairsInList(list, system.getParticles().size()), expected) << threads << " threads";
            EXPECT_EQ(list.getPairCount(), expected.size());
        });
    }
}

// 256 particles in a 6.7 sigma box: fewer than 3 cells per side, so the build falls back to all pairs
TEST(NeighbourList, SmallBoxFallbackIsCorrect) {
    System system = test::makeLiquid(4);
    NeighbourList list(test::cutOff, test::skin);
    list.build(system);
    EXPECT_EQ(pairsInList(list, system.getParticles().size()), pairsWithin(system, test::cutOff + test::skin));
}

TEST(NeighbourList, RebuildsOnlyAfterHalfSkinDisplacement) {
    System system = test::makeLiquid(6);
    NeighbourList list(test::cutOff, test::skin);
    list.build(system);
    const int builds = list.getRebuildCount();

    auto& particle = system.getParticles()[0];
    particle.position[0] += 0.4 * test::skin;
    system.wrapPosition(particle.position);
    EXPECT_FALSE(list.update(system));
    EXPECT_EQ(list.getRebuildCount(), builds);

    particle.position[0] += 0.2 * test::skin;   // now 0.6 skin from where it was at the last build
    system.wrapPosition(particle.position);
    EXPECT_TRUE(list.update(system));
    EXPECT_EQ(list.getRebuildCount(), builds + 1);
}

TEST(NeighbourList, RejectsListRadiusBeyondHalfBox) {
    System system = test::makeLattice(3);   // box ~5 sigma, list radius 2.7 > 2.5
    NeighbourList list(test::cutOff, test::skin);
    EXPECT_THROW(list.build(system), std::runtime_error);
}

TEST(CellList, EveryParticleInExactlyOneCellMatchingItsPosition) {
    System system = test::makeLiquid(8);
    CellList cells(test::cutOff + test::skin);
    cells.build(system);

    const int n = cells.getCellsPerSide();
    ASSERT_GE(n, 3);
    const double cellSize = system.getBoxLength() / n;

    std::vector<int> seen(system.getParticles().size(), 0);
    for (int c = 0; c < n * n * n; ++c) {
        for (std::size_t i : cells.particlesIn(c)) {
            ++seen[i];
            EXPECT_EQ(cells.cellOf(i), c);
            const auto& r = system.getParticles()[i].position;
            int expected = 0;
            for (int k = 0; k < 3; ++k) {
                expected = expected * n + std::min(static_cast<int>(r[k] / cellSize), n - 1);
            }
            EXPECT_EQ(expected, c);
        }
    }
    for (int count : seen) {
        EXPECT_EQ(count, 1);
    }
}

TEST(CellList, NeighbourCellsAreDistinctAndWrap) {
    System system = test::makeLattice(8);
    CellList cells(test::cutOff + test::skin);
    cells.build(system);
    const int n = cells.getCellsPerSide();
    ASSERT_GE(n, 3);

    for (int c : {0, n * n * n - 1, n * n + n + 1}) {
        auto neighbours = cells.neighbourCells(c);
        std::set<int> unique(neighbours.begin(), neighbours.end());
        EXPECT_EQ(unique.size(), 27u);
        EXPECT_TRUE(unique.count(c));
        for (int nb : neighbours) {
            EXPECT_GE(nb, 0);
            EXPECT_LT(nb, n * n * n);
        }
    }
}
