#include "NeighbourList.hpp"
#include "Parallel.hpp"
#include <stdexcept>


NeighbourList::NeighbourList(double cutOff, double skin, bool reverseIndex)
    : listRadius_(cutOff + skin),
      listRadiusSquared_((cutOff + skin) * (cutOff + skin)),
      maxDisplacementSquared_(0.25 * skin * skin),
      reverseIndex_(reverseIndex),
      cells_(cutOff + skin) {}

// Rebuild only if a particle has moved more than skin / 2 since the last build. Returns true if rebuilt.
bool NeighbourList::update(System& system) {

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


    // Parallel loop to check if any particle has moved more than skin / 2 since the last build
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
void NeighbourList::build(System& system) {
    // Beyond half the box a particle could interact with two images of the same neighbour
    if (2.0 * listRadius_ > system.getBoxLength()) {
        throw std::runtime_error("Cutoff + skin must be less than half the box length");
    }

    // Get a reference to the particles in the system
    const auto& particles = system.getParticles();
    const std::size_t n = particles.size();

    // Build the cell list for the current particle positions
    cells_.build(system);

    // Reorder the particles in the system according to the cell list's sorted order, so that particles in the same cell are contiguous in memory. This improves cache locality and performance when accessing neighbouring particles.
    system.reorder(cells_.sortedOrder());
    cells_.build(system);
    const bool useCells = cells_.getCellsPerSide() >= 3;

    offsets_.assign(n + 1, 0);

    // Pass 1 (parallel): each thread finds the j > i neighbours of its own particles, appending them to
    // a private buffer and recording each particle's count. Distinct i per thread, so no races
    #pragma omp parallel
    {
        // single has an implicit barrier, so every thread sees the resized buffers
        #pragma omp single
        threadNeighbours_.resize(threadCount());

        auto& local = threadNeighbours_[threadId()];
        local.clear();

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t before = local.size();
            forEachCandidate(i, n, useCells, [&](std::size_t j) {
                if (j <= i) {
                    return;
                }
                std::array<double, 3> dr = system.minimumImageDistance(particles[i].position, particles[j].position);
                if (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2] < listRadiusSquared_) {
                    local.push_back(j);
                }
            });
            offsets_[i + 1] = local.size() - before;
        }
    }

    // Pass 2 (serial, a plain copy): prefix sum the counts into CSR offsets, then concatenate the buffers.
    // schedule(static) hands thread t the t-th contiguous block of i, so concatenating in thread order
    // lines every particle's neighbours up with its offset
    for (std::size_t i = 0; i < n; ++i) {
        offsets_[i + 1] += offsets_[i];
    }

    neighbours_.clear();
    neighbours_.reserve(offsets_[n]);
    for (const auto& local : threadNeighbours_) {
        neighbours_.insert(neighbours_.end(), local.begin(), local.end());
    }

    // If requested, build the reverse index, which records for every particle j the pairs in which it is the second particle. This is needed for deterministic forces, but not for the default mode, so it is off by default
    if (reverseIndex_) {
        reverseOffsets_.assign(n + 1, 0);
        for (std::size_t j : neighbours_) {
            ++reverseOffsets_[j + 1];
        }
        for (std::size_t i = 0; i < n; ++i) {
            reverseOffsets_[i + 1] += reverseOffsets_[i];
        }
        reversePairs_.resize(neighbours_.size());
        std::vector<std::size_t> fill(reverseOffsets_.begin(), reverseOffsets_.end() - 1);
        for (std::size_t k = 0; k < neighbours_.size(); ++k) {
            reversePairs_[fill[neighbours_[k]]++] = k;
        }
    }

    referencePositions_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        referencePositions_[i] = particles[i].position;
    }
    ++rebuildCount_;
}
