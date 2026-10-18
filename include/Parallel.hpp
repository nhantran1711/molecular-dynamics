#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// OpenMP thread queries that fall back to a single thread when built without OpenMP

// Number of threads in the current parallel region
inline int threadCount() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

// This thread's index within the current parallel region
inline int threadId() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Run a parallel loop with a deterministic sum, so the result is bitwise identical for any thread count. The function f(i) is called for every i in [0, n), and its return values are summed. The summation is done in blocks of size blockSize, so the order of summation is fixed.
template <typename F>
double deterministicSum(std::size_t n, F&& f) {
    constexpr std::size_t blockSize = 1024;
    const std::size_t blocks = (n + blockSize - 1) / blockSize;
    std::vector<double> partial(blocks);

    // Parallel loop over blocks, each thread sums its block in a fixed order
    #pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::size_t end = std::min(n, (b + 1) * blockSize);
        double sum = 0.0;

        // Each thread sums its block in a fixed order, so the result is bitwise identical for any thread count
        for (std::size_t i = b * blockSize; i < end; ++i) {
            sum += f(i);
        }
        partial[b] = sum;
    }

    // Sum the partial sums in a fixed order, so the result is bitwise identical for any thread count
    double total = 0.0;
    for (double p : partial) {
        total += p;
    }
    return total;
}
