#pragma once

#include "CellList.hpp"
#include "System.hpp"
#include <array>
#include <cstddef>
#include <span>
#include <vector>

// A neighbour list for efficient force computation. 
// Each particle has a list of its neighbours within a cutoff distance.
class NeighbourList {
public:
    NeighbourList(double cutOff, double skin);

    // Rebuild only if a particle has moved more than skin / 2 since the last build. Returns true if rebuilt.
    bool update(const System& system);

    // Unconditionally rebuild the list for the current positions
    void build(const System& system);

    // Half list: neighbours j > i of particle i, so each pair appears exactly once
    std::span<const std::size_t> neighboursOf(std::size_t i) const {
        return {neighbours_.data() + offsets_[i], offsets_[i + 1] - offsets_[i]};
    };
    
    int getRebuildCount() const {
        return rebuildCount_;
    };

    std::size_t getPairCount() const {
        return neighbours_.size();
    };

private:
    double listRadius_;
    double listRadiusSquared_;
    double maxDisplacementSquared_;
    CellList cells_;

    // CSR layout: neighbours of particle i are neighbours_[offsets_[i] .. offsets_[i + 1])
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> neighbours_;

    // Positions at the last build, used to decide when the list goes stale
    std::vector<std::array<double, 3>> referencePositions_;
    int rebuildCount_ {0};
};
