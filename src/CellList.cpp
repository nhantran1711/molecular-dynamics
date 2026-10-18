#include "CellList.hpp"
#include <algorithm>
#include <cmath>

CellList::CellList(double minCellSize) : minCellSize_(minCellSize) {}

// Bin every particle into its cell with a counting sort, which keeps each cell's particles contiguous in memory
void CellList::build(const System& system) {
    // Beyond half the box a particle could interact with two images of the same neighbour
    const auto& particles = system.getParticles();
    const double boxLength = system.getBoxLength();

    // Compute the number of cells along one side of the box, and the inverse cell size
    cellsPerSide_ = std::max(1, static_cast<int>(boxLength / minCellSize_));

    // Compute the total number of cells and the inverse cell size for mapping positions to cell indices
    const int nCells = cellsPerSide_ * cellsPerSide_ * cellsPerSide_;
    const double inverseCellSize = cellsPerSide_ / boxLength;

    // Resize the vectors to hold the cell indices and particle indices
    particleCell_.resize(particles.size());
    cellStart_.assign(nCells + 1, 0);
    cellParticles_.resize(particles.size());

    // Find each particle's cell and count how many particles land in each cell
    for (std::size_t i = 0; i < particles.size(); ++i) {

        // Map the particle's position to a cell index in each dimension, clamping to ensure it stays within bounds
        std::array<int, 3> c {};

        // Use the inverse cell size to convert position to cell index, and clamp to ensure it stays within bounds
        for (int k = 0; k < 3; ++k) {
            // Clamp guards against a position of exactly boxLength after floating point wrapping
            c[k] = std::min(static_cast<int>(particles[i].position[k] * inverseCellSize), cellsPerSide_ - 1);
        }
        
        // Compute the linear cell index from the 3D cell coordinates and store it for the particle
        particleCell_[i] = (c[0] * cellsPerSide_ + c[1]) * cellsPerSide_ + c[2];
        ++cellStart_[particleCell_[i] + 1];
    }

    // Prefix sum turns the counts into start offsets
    for (int c = 0; c < nCells; ++c) {
        cellStart_[c + 1] += cellStart_[c];
    }

    // Scatter particle indices into their cell's slot
    std::vector<std::size_t> fill(cellStart_.begin(), cellStart_.end() - 1);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        cellParticles_[fill[particleCell_[i]]++] = i;
    }
}

// The 3x3x3 block of cells around the given cell, wrapping across the periodic boundaries
std::array<int, 27> CellList::neighbourCells(int cell) const {
    
    // Convert the linear cell index to 3D coordinates (cx, cy, cz) in the grid of cells
    const int n = cellsPerSide_;
    const int cx = cell / (n * n);
    const int cy = (cell / n) % n;
    const int cz = cell % n;

    std::array<int, 27> neighbours {};

    int idx = 0;

    // Loop over the 3x3x3 block of cells around the given cell, wrapping across periodic boundaries
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {

                // Wrap the coordinates around the periodic boundaries using modulo arithmetic
                int x = (cx + dx + n) % n;
                int y = (cy + dy + n) % n;
                int z = (cz + dz + n) % n;

                // Compute the linear index of the neighbouring cell and store it in the neighbours array
                neighbours[idx++] = (x * n + y) * n + z;
            }
        }
    }
    return neighbours;
}
