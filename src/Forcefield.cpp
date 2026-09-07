#include <cmath>
#include <Forcefield.hpp>

LennardJonesForceField::LennardJonesForceField(double epsilon, double sigma, double cut_off)
    : epsilon_(epsilon), sigma_(sigma), cut_off_squared_(cut_off * cut_off) {
    
        // Calculate the energy shift to ensure the potential goes to zero at the cutoff distance
        double sr6 = std::pow(sigma_ / cut_off_, 6);
        double sr12 = sr6 * sr6;
        energy_shift_ = 4.0 * epsilon_ * (sr12 - sr6);
}