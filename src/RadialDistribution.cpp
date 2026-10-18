#include "RadialDistribution.hpp"
#include <cmath>
#include <cstdio>
#include <numbers>

RadialDistribution::RadialDistribution(double rMax, int bins)
    : rMax_(rMax), binWidth_(rMax / bins), histogram_(static_cast<std::size_t>(bins), 0.0) {
}

// Reset the histogram and sample count to start a new calculation of g(r)
void RadialDistribution::sample(const System& system, const NeighbourList& neighbourList) {
    const std::size_t n = system.size();
    const double rMax2 = rMax_ * rMax_;

    // Half list: each pair is counted once
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j : neighbourList.neighboursOf(i)) {
            auto dr = system.minimumImageDistance(i, j);
            double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            if (r2 < rMax2) {
                histogram_[static_cast<std::size_t>(std::sqrt(r2) / binWidth_)] += 1.0;
            }
        }
    }

    const double volume = std::pow(system.getBoxLength(), 3);
    particles_ = static_cast<double>(n);
    density_ = particles_ / volume;
    ++samples_;
}

std::vector<double> RadialDistribution::radii() const {
    std::vector<double> r(histogram_.size());
    for (std::size_t b = 0; b < r.size(); ++b) {
        r[b] = (static_cast<double>(b) + 0.5) * binWidth_;
    }
    return r;
}

// g(r) = pairs in the shell / pairs an ideal gas at the same density would put there:
// (N / 2) * density * shell volume per sample, since the half list counts each pair once
std::vector<double> RadialDistribution::values() const {
    std::vector<double> g(histogram_.size(), 0.0);
    if (samples_ == 0) {
        return g;
    }
    for (std::size_t b = 0; b < g.size(); ++b) {
        const double rLow = static_cast<double>(b) * binWidth_;
        const double rHigh = rLow + binWidth_;
        const double shell = 4.0 / 3.0 * std::numbers::pi * (rHigh * rHigh * rHigh - rLow * rLow * rLow);
        g[b] = histogram_[b] / (samples_ * 0.5 * particles_ * density_ * shell);
    }
    return g;
}

bool RadialDistribution::write(const char* path) const {
    std::FILE* file = std::fopen(path, "w");
    if (!file) {
        return false;
    }
    const auto r = radii();
    const auto g = values();
    std::fprintf(file, "r,g\n");
    for (std::size_t b = 0; b < g.size(); ++b) {
        std::fprintf(file, "%.6f,%.6f\n", r[b], g[b]);
    }
    std::fclose(file);
    return true;
}
