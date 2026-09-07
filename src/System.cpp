#include "System.hpp"
#include <cmath>

// Constructor for the System class
System::System(std::vector<Particle> particles, double boxLength): 
    particles_(std::move(particles)), boxLength_(boxLength) {}


// Wrap position of a particle back into the simulation box
void System::wrapPosition(std::array<double, 3>& r) const {
    for (int k = 0; k < 3; ++k) {
        r[k] -= boxLength_ * std::floor(r[k] / boxLength_);
    }