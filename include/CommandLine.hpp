#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>

// Parse a number from a string, returning false if the string is not a valid number or is out of range for the output type. The output type can be double, long, int or unsigned.
inline bool parseNumber(const char* text, double& out) {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || errno != 0) {
        return false;
    }
    out = value;
    return true;
}


// Parse a number from a string, returning false if the string is not a valid number or is out of range for the output type. The output type can be double, long, int or unsigned.
inline bool parseNumber(const char* text, long& out) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno != 0) {
        return false;
    }
    out = value;
    return true;
}

// Parse a number from a string, returning false if the string is not a valid number or is out of range for the output type. The output type can be double, long, int or unsigned.
inline bool parseNumber(const char* text, int& out) {
    long value = 0;
    if (!parseNumber(text, value) || value < INT_MIN || value > INT_MAX) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

// Parse a number from a string, returning false if the string is not a valid number or is out of range for the output type. The output type can be double, long, int or unsigned.
inline bool parseNumber(const char* text, unsigned& out) {
    long value = 0;
    if (!parseNumber(text, value) || value < 0 || value > static_cast<long>(UINT_MAX)) {
        return false;
    }
    out = static_cast<unsigned>(value);
    return true;
}
