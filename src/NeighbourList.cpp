#include "NeighbourList.hpp"
#include <stdexcept>

NeighbourList::NeighbourList(double cutOff, double skin)
    : listRadius_(cutOff + skin),
      listRadiusSquared_((cutOff + skin) * (cutOff + skin)),
      maxDisplacementSquared_(0.25 * skin * skin),
      cells_(cutOff + skin) {}

// Rebuild only if a particle has moved more than skin / 2 since the last build. Returns true if rebuilt.
bool NeighbourList::update(const System& system) {

    // Get a reference to the particles in the system
    const auto& particles = system.getParticles();


    // If the number of particles has changed since the last build, we must rebuild the neighbour list
    if (referencePositions_.size() != particles.size()) {
        build(system);
        return true;
    }

    // Check if any particle has moved more than skin / 2 since the last build.
    // An OR reduction instead of an early return, since OpenMP loops can't break out early
    bool stale = false;

    #pragma omp parallel for reduction(||:stale) schedule(static)
    for (std::size_t i = 0; i < particles.size(); ++i) {
        // Minimum image, since a particle may have wrapped across the box since the last build
        std::array<double, 3> d = system.minimumImageDistance(particles[i].position, referencePositions_[i]);
        stale = stale || (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > maxDisplacementSquared_);
    }

    if (stale) {
        build(system);
    }
    return stale;
}

template <typename Visit>
void NeighbourList::forEachCandidate(std::size_t i, std::size_t n, bool useCells, Visit&& visit) const {
    if (useCells) {
        for (int cell : cells_.neighbourCells(cells_.cellOf(i))) {
            for (std::size_t j : cells_.particlesIn(cell)) {
                visit(j);
            }
        }
    } else {
        // Box too small for a 3x3x3 grid, so the 27 cells would overlap: fall back to checking every pair
        for (std::size_t j = 0; j < n; ++j) {
            visit(j);
        }
    }
}

// Unconditionally rebuild the list for the current positions
void NeighbourList::build(const System& system) {
    // Beyond half the box a particle could interact with two images of the same neighbour
    if (2.0 * listRadius_ > system.getBoxLength()) {
        throw std::runtime_error("Cutoff + skin must be less than half the box length");
    }

    // Get a reference to the particles in the system
    const auto& particles = system.getParticles();
    const std::size_t n = particles.size();

    // Build the cell list for the current particle positions
    cells_.build(system);
    const bool useCells = cells_.getCellsPerSide() >= 3;

    auto isNeighbour = [&](std::size_t i, std::size_t j) {
        if (j == i) {
            return false;
        }
        std::array<double, 3> dr = system.minimumImageDistance(particles[i].position, particles[j].position);
        return dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2] < listRadiusSquared_;
    };

    // Two passes so threads can fill the list in parallel without sharing a push_back:
    // first count each particle's neighbours, then a prefix sum gives every particle its own slice to write
    offsets_.assign(n + 1, 0);

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t count = 0;
        forEachCandidate(i, n, useCells, [&](std::size_t j) {
            count += isNeighbour(i, j);
        });
        offsets_[i + 1] = count;
    }

    for (std::size_t i = 0; i < n; ++i) {
        offsets_[i + 1] += offsets_[i];
    }
    neighbours_.resize(offsets_[n]);

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t next = offsets_[i];
        forEachCandidate(i, n, useCells, [&](std::size_t j) {
            if (isNeighbour(i, j)) {
                neighbours_[next++] = j;
            }
        });
    }

    referencePositions_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        referencePositions_[i] = particles[i].position;
    }
    ++rebuildCount_;
}
