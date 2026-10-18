#include "NeighbourList.hpp"
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int threadCount() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

int threadId() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

}

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

    // Pass 1 (parallel): find the half list. Each pair is distance-checked once (j > i), and each thread
    // collects the pairs it finds in its own buffer, so threads never share a push_back
    #pragma omp parallel
    {
        // single has an implicit barrier, so every thread sees the resized buffers
        #pragma omp single
        threadPairs_.resize(threadCount());

        auto& local = threadPairs_[threadId()];
        local.clear();

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            forEachCandidate(i, n, useCells, [&](std::size_t j) {
                if (j <= i) {
                    return;
                }
                std::array<double, 3> dr = system.minimumImageDistance(particles[i].position, particles[j].position);
                if (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2] < listRadiusSquared_) {
                    local.emplace_back(i, j);
                }
            });
        }
    }

    // Pass 2 (serial, no distance checks): expand to the full list. Each half pair i-j puts j in i's
    // list and i in j's list. Writing both ends would race across threads, but this is only O(pairs)
    offsets_.assign(n + 1, 0);
    for (const auto& pairs : threadPairs_) {
        for (auto [i, j] : pairs) {
            ++offsets_[i + 1];
            ++offsets_[j + 1];
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        offsets_[i + 1] += offsets_[i];
    }
    neighbours_.resize(offsets_[n]);

    fill_.assign(offsets_.begin(), offsets_.end() - 1);
    for (const auto& pairs : threadPairs_) {
        for (auto [i, j] : pairs) {
            neighbours_[fill_[i]++] = j;
            neighbours_[fill_[j]++] = i;
        }
    }

    referencePositions_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        referencePositions_[i] = particles[i].position;
    }
    ++rebuildCount_;
}
