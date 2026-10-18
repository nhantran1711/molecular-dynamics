#pragma once

#include "System.hpp"
#include <cstdio>

// Write the current configuration of the system to an XYZ file, with the given time in reduced Lennard-Jones units. The file must be opened for writing before calling this function, and closed afterwards.
void writeXyzFrame(std::FILE* file, const System& system, double time);
