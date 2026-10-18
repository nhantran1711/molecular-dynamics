# Molecular Dynamics

A molecular dynamics engine in C++20 that simulates a Lennard-Jones fluid with Velocity Verlet integration and periodic boundary conditions. Cell lists and Verlet neighbour lists make each step O(N) instead of O(N²), and OpenMP spreads the step across cores. 
A benchmark suite measures energy conservation, temperature, stability and scaling.

![md_vis: 2048 particles of Lennard-Jones liquid, coloured by speed](docs/visualiser.png)

All quantities are in reduced Lennard-Jones units (ε = σ = m = k_B = 1).

## Features

- **Lennard-Jones potential**, truncated at r_c = 2.5σ and shifted so the energy is zero at the cutoff
- **Velocity Verlet** integration (time-reversible, second order)
- **Periodic boundary conditions** with the minimum image convention
- **Cell lists** (counting sort into cells ≥ r_c + skin wide) used to build **Verlet neighbour lists**, rebuilt
  only when some particle has moved more than half the skin
- **OpenMP** parallel force calculation, neighbour list build, integration and reductions
- **Reproducible results:** bitwise-identical runs at a fixed thread count by default, and at *any* thread
  count with `--deterministic`
- **SIMD-vectorised force kernel** (`#pragma omp simd`, confirmed with compiler vectorisation reports)
- **Real-time OpenGL visualiser** (`md_vis`): particles drawn as sphere-shaded point sprites coloured by
  speed, with live temperature control to melt or freeze the system
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
| `--deterministic` | off | bitwise-identical results for any thread count (costs 6–36%) |
| `--quiet` | off | print only the machine-readable `RESULT` line |

Set the thread count with `OMP_NUM_THREADS`.

### Visualiser

`md_vis` is built when GLFW is installed (`brew install glfw`, or `libglfw3-dev` on Linux). It needs
OpenGL 4.1 and has only been tested on macOS.

```bash
./build/release/md_vis 8          # 4 × 8³ = 2048 particles
```

| Input | Action |
|---|---|
| drag / scroll | orbit / zoom |
| Space | pause |
| ↑ / ↓ | more / fewer steps per frame |
| H / C | heat / cool by 10% (sets the thermostat target) |
| T | thermostat on / off |
| R | reset to the lattice |

Temperature, energy and frame rate are shown in the window title. `--screenshot out.ppm [--frames N]`
renders N frames off-screen, saves the last one and exits.

## Testing

```bash
ctest --test-dir build/release --output-on-failure     # 29 GoogleTest tests, ~4 s
```

The tests cover every component:

- **Forces against analytic values:** two-particle force and energy, the zero-force point at 2^(1/6)σ,
  nothing beyond the cutoff, interaction across a periodic boundary.
- **Neighbour-list forces against the O(N²) reference**, at 1, 3 and 8 threads (3 checks uneven work
  division); total force ≈ 0.
- **Neighbour list** contains exactly the pairs within r_c + skin, including the small-box fallback, and
  rebuilds only after a particle moves more than half the skin.
- **Cell list:** each particle is in exactly one cell, matching its position; the 27 neighbour cells are
  distinct and wrap around the box.
- **Dynamics:** energy conservation, momentum conservation, and **time reversibility** (run forward, flip
  velocities, run back, and every particle returns to within 10⁻⁸ of its start).
- **Determinism:** identical state hashes across repeated runs at a fixed thread count, and across 1–8
  threads in deterministic mode.

The tests were checked by deliberately breaking the code: skipping neighbour cells, or flipping the sign of
the SIMD scatter, makes 6 and 7 tests fail respectively.

CI (GitHub Actions) builds and tests on Linux and macOS, and runs the tests under AddressSanitizer +
UndefinedBehaviorSanitizer and under ThreadSanitizer (with LLVM's Archer, so OpenMP's own synchronisation is
understood). For a sanitizer build locally: `-DMD_SANITIZE=address,undefined` or `-DMD_SANITIZE=thread`.

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

**Reproducibility.** Every run prints a hash of the final positions and velocities (`state_hash` in the
`RESULT` line), so two runs can be compared bit for bit. At first, 6 runs at 4 threads gave 4 different final
states. The cause was `reduction(+)` in the kinetic energy: OpenMP combines the per-thread partial sums in
an unspecified order, so the last bits can differ, and the thermostat feeds that value back into the
trajectory. MD is chaotic, so a last-bit difference grows into a completely different trajectory. Two fixes,
at two strengths:

- **Default: identical at a fixed thread count.** Energies use a fixed-order blocked sum
  (`deterministicSum`: fixed-size blocks summed in parallel, then combined in order). The force arrays were
  already summed in thread order. No measurable cost.
- **`--deterministic`: identical at any thread count.** The per-thread force arrays group each particle's
  contributions differently depending on the thread count. So in this mode every pair's force goes to its
  own slot, and each particle then adds up its own contributions in an order fixed by the neighbour list
  (pairs where it is first, then a reverse index of pairs where it is second). The extra memory traffic
  costs:

| | default | `--deterministic` | |
|---|---|---|---|
| N = 4000, 1 thread | 1.44 ms | 1.68 ms | +17% |
| N = 4000, 4 threads | 0.70 ms | 0.74 ms | +6% |
| N = 13,500, 1 thread | 5.30 ms | 6.17 ms | +16% |
| N = 13,500, 4 threads | 1.92 ms | 2.61 ms | +36% |

The worst case is large N on 4 threads: the per-pair arrays (~13 MB) no longer fit in cache, and the
reverse-index pass reads them from scattered addresses. Restoring the old reduction makes the reproducibility test fail in 3 of 5
runs.

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
vis/
  Renderer        OpenGL drawing: point-sprite spheres, box outline, orbit camera
  main.cpp        md_vis window, input and live simulation loop
tests/              GoogleTest unit tests
bench/benchmark.py  benchmark runner and plots
.github/workflows/  CI: Linux + macOS build and test, ASan/UBSan and TSan jobs
```

