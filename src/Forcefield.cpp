#include <cmath>
#include <stdexcept>
#include "Forcefield.hpp"

LennardJonesForceField::LennardJonesForceField(double epsilon, double sigma, double cut_off)
    : epsilon_(epsilon), sigma_(sigma), cut_off_(cut_off), cut_off_squared_(cut_off * cut_off) {

        // Calculate the energy shift to ensure the potential goes to zero at the cutoff distance
        double sr6 = std::pow(sigma_ / cut_off_, 6);
        double sr12 = sr6 * sr6;
        energy_shift_ = 4.0 * epsilon_ * (sr12 - sr6);
}

// O(N^2) pair loop: compute LJ forces on all particles and return the total potential energy
double LennardJonesForceField::computeForces(System& system) const {
    // Minimum image convention is only valid if the cutoff is under half the box length
    if (2.0 * cut_off_ > system.getBoxLength()) {
        throw std::runtime_error("Cutoff must be less than half the box length");
    }

    auto& particles = system.getParticles();
    const std::size_t n = particles.size();
    const double sigma2 = sigma_ * sigma_;

    for (auto& particle : particles) {
        particle.force = {0.0, 0.0, 0.0};
    }

    double potential = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            std::array<double, 3> dr = system.minimumImageDistance(particles[i].position, particles[j].position);
            double r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

            if (r2 >= cut_off_squared_) {
                continue;
            }

            double inv_r2 = 1.0 / r2;
            double sr2 = sigma2 * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            // F = -dV/dr * (dr / r), written in terms of 1/r^2 to avoid a sqrt
            double f_over_r = 24.0 * epsilon_ * (2.0 * sr12 - sr6) * inv_r2;

            for (int k = 0; k < 3; ++k) {
                particles[i].force[k] += f_over_r * dr[k];
                particles[j].force[k] -= f_over_r * dr[k];
            }

            potential += 4.0 * epsilon_ * (sr12 - sr6) - energy_shift_;
        }
    }
    return potential;
}
