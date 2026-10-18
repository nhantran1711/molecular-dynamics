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
    // Construct a neighbour list with a given cutoff and skin distance. The skin distance is the extra distance beyond the cutoff that allows the list to remain valid for multiple timesteps without rebuilding. If reverseIndex is true, the list will also store the pairs in which each particle is the second particle, which is needed for deterministic forces.
    NeighbourList(double cutOff, double skin, bool reverseIndex = false);

    // Rebuild only if a particle has moved more than skin / 2 since the last build. Returns true if rebuilt.
    bool update(System& system);

    // Unconditionally rebuild the list for the current positions. Also sorts the particles by cell,
    // so particle indices change on every rebuild
    void build(System& system);

    // Half list: neighbours j > i of particle i, so each pair appears exactly once
    std::span<const std::size_t> neighboursOf(std::size_t i) const {
        return {neighbours_.data() + offsets_[i], offsets_[i + 1] - offsets_[i]};
    };

    // Global index of particle i's first pair: its pairs are [firstPairOf(i), firstPairOf(i) + neighboursOf(i).size())
    std::size_t firstPairOf(std::size_t i) const {
        return offsets_[i];
    };

    bool hasReverseIndex() const {
        return reverseIndex_;
    };

    // Global indices of the pairs (i, j) with j == this particle, in increasing order. Needs reverseIndex
    std::span<const std::size_t> pairsAsSecond(std::size_t j) const {
        return {reversePairs_.data() + reverseOffsets_[j], reverseOffsets_[j + 1] - reverseOffsets_[j]};
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
    bool reverseIndex_;
    CellList cells_;

    // CSR layout: neighbours of particle i are neighbours_[offsets_[i] .. offsets_[i + 1])
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> neighbours_;

    // Reverse CSR: pair indices where particle j is second are reversePairs_[reverseOffsets_[j] .. reverseOffsets_[j + 1])
    std::vector<std::size_t> reverseOffsets_;
    std::vector<std::size_t> reversePairs_;

    // Visit every candidate j within the 27 surrounding cells (or every particle, if the box is too small)
    template <typename Visit>
    void forEachCandidate(std::size_t i, std::size_t n, bool useCells, Visit&& visit) const;

    // Per-thread buffers of neighbour indices, kept between builds to avoid reallocating
    std::vector<std::vector<std::size_t>> threadNeighbours_;

    // Positions at the last build, used to decide when the list goes stale
    std::vector<std::array<double, 3>> referencePositions_;
    int rebuildCount_ {0};
};
