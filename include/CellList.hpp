#pragma once

#include "System.hpp"
#include <array>
#include <cstddef>
#include <span>
#include <vector>

// A grid of cubic cells for efficient neighbour searching. 
// Each cell contains a list of the particles in it.
class CellList {

public:
    explicit CellList(double minCellSize);

    // Build the cell list from the current particle positions in the system
    void build(const System& system);

    // Get the number of cells along one side of the cubic box
    int getCellsPerSide() const {
        return cellsPerSide_;
    };

    // Get the cell index of a given particle
    int cellOf(std::size_t particle) const {
        return particleCell_[particle];
    };

    // Get the indices of the 27 neighbouring cells (including the cell itself) of a given cell
    std::array<int, 27> neighbourCells(int cell) const;

    // Get the indices of the particles in a given cell
    std::span<const std::size_t> particlesIn(int cell) const {
        return {cellParticles_.data() + cellStart_[cell], cellStart_[cell + 1] - cellStart_[cell]};
    };

private:
    double minCellSize_;
    int cellsPerSide_ {0};

    // Counting-sort layout: particles of cell c are cellParticles_[cellStart_[c] .. cellStart_[c + 1])
    std::vector<std::size_t> cellStart_;
    std::vector<std::size_t> cellParticles_;
    std::vector<int> particleCell_;
};
