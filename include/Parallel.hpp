#pragma once

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
