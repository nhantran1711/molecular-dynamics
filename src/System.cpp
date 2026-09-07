#include "System.hpp"
#include <cmath>

// Constructor for the System class
System::System(std::vector<Particle> particles, double boxLength): 
    particles_(std::move(particles)), boxLength_(boxLength) {}

