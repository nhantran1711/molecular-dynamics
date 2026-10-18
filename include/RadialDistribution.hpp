#pragma once

#include "NeighbourList.hpp"
#include "System.hpp"
#include <vector>

// A radial distribution function g(r) calculator. It bins the distances between every pair of particles in the system, and normalises by the number of samples and the ideal gas density to produce g(r).
class RadialDistribution {
public:
    RadialDistribution(double rMax, int bins);

    // Add the current configuration's pair distances to the histogram
    void sample(const System& system, const NeighbourList& neighbourList);

    // Bin centres and normalised g(r) averaged over every sample so far
    std::vector<double> radii() const;
    std::vector<double> values() const;

    // Write "r,g" CSV rows
    bool write(const char* path) const;

private:
    double rMax_;
    double binWidth_;
    std::vector<double> histogram_;
    int samples_ {0};
    double particles_ {0.0};
    double density_ {0.0};
};
