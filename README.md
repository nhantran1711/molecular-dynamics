# Molecular Dynamics

A molecular dynamics engine in C++20 that simulates a Lennard-Jones fluid with Velocity Verlet integration and periodic boundary conditions. Cell lists and Verlet neighbour lists make each step O(N) instead of O(N²), and OpenMP spreads the step across cores. 
A benchmark suite measures energy conservation, temperature, stability and scaling.

All quantities are in reduced Lennard-Jones units (ε = σ = m = k_B = 1).

## Features

- **Lennard-Jones potential**, truncated at r_c = 2.5σ and shifted so the energy is zero at the cutoff
- **Velocity Verlet** integration (time-reversible, second order)
- **Periodic boundary conditions** with the minimum image convention
- **Cell lists** (counting sort into cells ≥ r_c + skin wide) used to build **Verlet neighbour lists**, rebuilt
  only when some particle has moved more than half the skin
- **OpenMP** parallel force calculation, neighbour list build, integration and reductions
- **SIMD-vectorised force kernel** (`#pragma omp simd`, confirmed with compiler vectorisation reports)
- Particles **re-sorted by cell** on every list rebuild, so spatial neighbours are neighbours in memory
- **FCC lattice** initialisation with Maxwell-Boltzmann velocities, and a velocity-rescaling equilibration
  phase followed by NVE production
- A **benchmark suite** (`bench/benchmark.py`) that runs the sweeps below and regenerates every plot

## Build and run

Requires CMake ≥ 3.16 and a C++20 compiler. 
OpenMP is optional (the build falls back to a single thread).
On macOS, Apple Clang needs `brew install libomp`.

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
./build/release/md_sim # 500 particles: 2000 equilibration + 5000 production steps
```

```
md_sim [cellsPerSide] [--brute] [--dt X] [--skin X] [--cutoff X] [--equil N] [--steps N] [--csv PATH] [--quiet]
```

| Option | Default | Meaning |
|---|---|---|
| `cellsPerSide` | 5 | FCC unit cells per side: N = 4 × cellsPerSide³ |
| `--brute` | off | O(N²) all-pairs forces instead of the neighbour list |
| `--dt` | 0.005 | timestep |
| `--skin` | 0.2 | neighbour list skin |
| `--cutoff` | 2.5 | interaction cutoff |
| `--equil` / `--steps` | 2000 / 5000 | equilibration / production steps |
| `--csv` | – | write per-step energies and temperature |
| `--quiet` | off | print only the machine-readable `RESULT` line |

Set the thread count with `OMP_NUM_THREADS`.

## Benchmarks

Measured on an Apple M1 MacBook Air (4 performance + 4 efficiency cores, fanless). Timings are the fastest of 3 runs. To regenerate everything (results go to `bench/results/`, plots to `bench/plots/`):

```bash
python bench/benchmark.py            # full suite, a few minutes
python bench/benchmark.py --quick    # smoke test, seconds
```

### Neighbour lists: O(N²) → O(N)

![Cost per step vs system size](bench/plots/size_scaling.png)

| N = 4000 | ms / step | vs all pairs |
|---|---|---|
| All pairs, 1 thread | 12.9 | 1× |
| Neighbour list, 1 thread | 1.49 | **8.7×** |
| Neighbour list, 4 threads | 0.60 | **21.6×** |

From 256 to 32,000 particles, the neighbour-list cost grows as about N^1.07.

### OpenMP scaling

![OpenMP strong scaling](bench/plots/thread_scaling.png)

The speedup at 4 threads is 2.5× for N = 4000 and 2.4× for N = 13,500. It levels off beyond 4 threads, as
expected on the M1: `schedule(static)` gives every thread an equal share, so once the slower efficiency cores
join, the other threads wait for them.

### Energy conservation

![Energy conservation](bench/plots/energy_conservation.png)

With the thermostat off, total energy stays within about 10⁻⁴ of its starting value, with no long-term drift.

![Energy fluctuation vs timestep](bench/plots/timestep_error.png)

Velocity Verlet is second order, so the energy error should scale as dt². With a 3.5σ cutoff it does
(measured slope 1.98). With 2.5σ the slope is 1.69 and the curve flattens at small dt: the force drops
abruptly to zero at the cutoff, and every pair that crosses it adds error that doesn't shrink with dt. Raising
the cutoff makes that jump about 10× smaller (0.039 → 0.0037), and the excess error disappears. Each point is the mean of 3
runs, with whiskers showing the spread.

### Temperature stability

![Temperature](bench/plots/temperature.png)

During equilibration, velocities are rescaled to the target temperature every 10 steps. The early dip is the
FCC lattice melting: kinetic energy turns into potential energy faster than the rescaling can correct. After
the thermostat is switched off, the temperature fluctuates around the target as expected for a closed system.

### Neighbour list skin

![Skin sweep](bench/plots/skin_sweep.png)

A small skin means frequent rebuilds. A large skin means more pairs to check every step. 0.2σ was fastest
here, and also at N = 13,500 and on a single thread, so it is the default.

## Design notes

**Parallel forces without races.** The force loop visits each pair once (a half neighbour list) and applies
equal and opposite forces to both particles. Two threads can share a particle `j`, so each thread adds forces
into its own array, and a second parallel loop sums the arrays for each particle.

My first version used a *full* neighbour list instead: every pair stored twice, so each thread writes only
its own particle and needs no extra arrays. Profiling and timing showed that this doubled the force work and roughly quadrupled the
list-build work (both directions, each searched twice). It was about 2.4× slower than the serial code on one thread.
Switching to a half list with per-thread arrays made a single thread 1.3× faster than the original serial
code. Full lists remain the better fit for GPUs, where thousands of threads would need thousands of force
arrays.

**Parallel list build.** Each thread finds the neighbours of its own contiguous block of particles and
appends them to a private buffer. A prefix sum over the per-particle counts gives the compact (CSR) offsets,
and the buffers are concatenated in thread order.

**SIMD force kernel.** The compiler would not vectorise the original pair loop: its cost model judged the
`if (r2 >= rc²) continue` branch and the scattered writes to `forces[j]` too expensive. The kernel now
handles each particle in three stages:

1. **gather** the neighbours' positions into contiguous per-component arrays (scalar loads; ARM NEON has no
   gather instruction);
2. **compute** minimum image, r², the LJ terms and the cutoff mask in a branch-free `#pragma omp simd` loop.
   Pairs beyond the cutoff (still in the list because of the skin) are computed and masked to zero;
3. **scatter** the equal and opposite forces back to each `j` (scalar stores).

Clang's `-Rpass=loop-vectorize` confirms that the compute loop is vectorised at width 2, the full NEON
width for doubles. Measured at N = 4000 on one thread, from builds of identical source with SIMD on and off:

| Kernel | ms / step | |
|---|---|---|
| Original scalar loop | 1.83 | |
| Gather / compute / scatter, vectorisation off | 1.68 | 1.09× (branch-free, contiguous) |
| Gather / compute / scatter, SIMD on | 1.46 | 1.15× from SIMD, **1.25× total** |

This is far below the 2× ideal: NEON holds only 2 doubles, the gather and scatter stay scalar, and list
rebuilds and integration aren't vectorised. A struct-of-arrays particle layout would remove the gather and
scatter copies, and single precision (4 lanes) or AVX-512 on x86 (8 doubles) would give more.

**Checking correctness.** The neighbour-list forces match the O(N²) reference to within 2×10⁻¹⁴, at several
thread counts including uneven ones. Energy conservation acts as an end-to-end check: a race or a missed pair
shows up as drift.

## Project layout

```
include/, src/
  System          particles, box, periodic boundaries, kinetic energy, temperature
  Forcefield      Lennard-Jones forces: O(N²) reference and parallel neighbour-list version
  CellList        particles binned into cells for neighbour search
  NeighbourList   Verlet list: parallel build, displacement-based rebuild
  Integrator      Velocity Verlet
  Initialisation  FCC lattice, Maxwell-Boltzmann velocities
  Thermostat      velocity rescaling, used during equilibration only
  Parallel.hpp    OpenMP thread helpers with single-thread fallbacks
  main.cpp        command-line driver
bench/benchmark.py  benchmark runner and plots
```

