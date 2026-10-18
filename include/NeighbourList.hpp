#pragma once

#include "CellList.hpp"
#include "System.hpp"
#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

// A neighbour list for efficient force computation. 
// Each particle has a list of its neighbours within a cutoff distance.
class NeighbourList {
public:
    NeighbourList(double cutOff, double skin);

    // Rebuild only if a particle has moved more than skin / 2 since the last build. Returns true if rebuilt.
    bool update(System& system);

    // Unconditionally rebuild the list for the current positions. Also sorts the particles by cell,
    // so particle indices change on every rebuild
    void build(System& system);

    // Full list: every neighbour j != i of particle i, so each pair appears twice (i -> j and j -> i).
    // Costs double the pair work, but a thread handling particle i only ever writes to particle i.
    std::span<const std::size_t> neighboursOf(std::size_t i) const {
        return {neighbours_.data() + offsets_[i], offsets_[i + 1] - offsets_[i]};
    };
    
    int getRebuildCount() const {
        return rebuildCount_;
    };

    std::size_t getPairCount() const {
        return neighbours_.size() / 2;
    };

private:
    double listRadius_;
    double listRadiusSquared_;
    double maxDisplacementSquared_;
    CellList cells_;

    // CSR layout: neighbours of particle i are neighbours_[offsets_[i] .. offsets_[i + 1])
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> neighbours_;

    // Visit every candidate j within the 27 surrounding cells (or every particle, if the box is too small)
    template <typename Visit>
    void forEachCandidate(std::size_t i, std::size_t n, bool useCells, Visit&& visit) const;

    // Per-thread buffers of half-list pairs (i < j), kept between builds to avoid reallocating
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> threadPairs_;
    std::vector<std::size_t> fill_;

    // Positions at the last build, used to decide when the list goes stale
    std::vector<std::array<double, 3>> referencePositions_;
    int rebuildCount_ {0};
};
