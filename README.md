# Molecular Dynamics

[![CI](https://github.com/nhantran1711/molecular-dynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/nhantran1711/molecular-dynamics/actions/workflows/ci.yml)

A molecular dynamics engine in C++20 that simulates a Lennard-Jones fluid with Velocity Verlet integration and periodic boundary conditions. Cell lists and Verlet neighbour lists make each step O(N) instead of O(N²), and OpenMP spreads the step across cores. 
Results are validated against NIST reference data, and a benchmark suite measures energy conservation,
temperature stability and scaling.

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
- **SIMD-vectorised force kernel** (`#pragma omp simd`, confirmed with compiler vectorisation reports), with
  an optional **mixed-precision** mode (float pair maths, double positions and accumulation)
- **Hot/cold memory layout:** positions packed per particle for the neighbour gathers, everything else in
  separate arrays per component, chosen by measuring it against both pure layouts
- **Validated against NIST** reference data for the Lennard-Jones fluid (energy and virial pressure), with
  error bars from independent replicas
- **Bussi stochastic velocity rescaling** thermostat for correct canonical (NVT) sampling
- **Observables:** virial pressure, radial distribution function g(r), and **extended XYZ trajectories** with
  stable particle ids (readable by OVITO, VMD and ASE)
- **Real-time OpenGL visualiser** (`md_vis`): particles drawn as sphere-shaded point sprites coloured by
  speed, with live temperature control to melt or freeze the system
- Particles **re-sorted by cell** on every list rebuild, so spatial neighbours are neighbours in memory
- **FCC lattice** initialisation with Maxwell-Boltzmann velocities, a velocity-rescaling equilibration phase
  (optionally melting at T = 2 first), then NVE or NVT production
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
md_sim [cellsPerSide] [--brute] [--dt X] [--skin X] [--cutoff X] [--equil N] [--steps N] [--density X]
       [--temperature X] [--nvt] [--seed N] [--melt] [--csv PATH] [--rdf PATH] [--xyz PATH] [--xyz-every N]
       [--deterministic] [--mixed] [--quiet]
```

| Option | Default | Meaning |
|---|---|---|
| `cellsPerSide` | 5 | FCC unit cells per side: N = 4 × cellsPerSide³ |
| `--brute` | off | O(N²) all-pairs forces instead of the neighbour list |
| `--dt` | 0.005 | timestep |
| `--skin` | 0.2 | neighbour list skin |
| `--cutoff` | 2.5 | interaction cutoff |
| `--equil` / `--steps` | 2000 / 5000 | equilibration / production steps |
| `--density` / `--temperature` | 0.8442 / 0.728 | state point |
| `--nvt` | off | keep the Bussi thermostat on during production (default production is NVE) |
| `--melt` | off | spend the first third of equilibration at T = 2, so dense states start from a liquid |
| `--seed` | 42 | initial velocities and thermostat noise |
| `--csv` | – | write per-step energies, temperature, pressure and pairs inside the cutoff |
| `--rdf` | – | write g(r) averaged over production |
| `--xyz` / `--xyz-every` | – / 100 | write an extended XYZ trajectory every N production steps |
| `--deterministic` | off | bitwise-identical results for any thread count (costs 6–36%) |
| `--mixed` | off | float pair maths (4 SIMD lanes), double positions and accumulation |
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
ctest --test-dir build/release --output-on-failure     # 37 GoogleTest tests, ~5 s
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
- **Observables:** the virial and pair counts agree across all force paths; the Bussi thermostat alone
  reproduces the canonical mean and variance of the kinetic energy; g(r) of an ideal gas is 1.
- **Mixed precision** matches the double-precision forces to float accuracy and still conserves energy.
- **Trajectories:** particle ids survive re-sorting, and line k of every XYZ frame is the same particle.

The tests were checked by deliberately breaking the code: skipping neighbour cells, or flipping the sign of
the SIMD scatter, makes 6 and 7 tests fail respectively.

CI (GitHub Actions) builds and tests on Linux and macOS with warnings as errors (`-DMD_WERROR=ON`), runs the
tests under AddressSanitizer + UndefinedBehaviorSanitizer and under ThreadSanitizer (with LLVM's Archer, so
OpenMP's own synchronisation is understood), and checks formatting (`clang-format`, pinned to 23.1.3) and
static analysis (`clang-tidy`: bug-prone patterns, performance and the clang static analyser). For a sanitizer
build locally: `-DMD_SANITIZE=address,undefined` or `-DMD_SANITIZE=thread`.

clang-tidy found two real bugs when it was added: `atoi`/`atof` turned bad arguments such as `--steps 12abc`
into 0 without complaint (now rejected), and a pixel offset in the screenshot code was computed in `int`
before widening, which overflows for large images.

## Validation against NIST

The engine is checked against the NIST Standard Reference Simulation Website's Lennard-Jones benchmarks
([NVT Monte Carlo](https://mmlapps.nist.gov/srs/LJ_PURE/mc.htm), N = 500, cutoff 3σ, standard long-range
corrections, virial pressure). Run it with `python bench/validate.py` (about 25 minutes).

Each state point is 8 independent NVT runs (Bussi thermostat; 50 t* equilibration starting from a lattice
melted at T = 2, then 100 t* production). Energies are converted to NIST's convention: the shift is removed
using the number of pairs inside the cutoff, and the standard tail corrections are added.

| T | ρ | U (this engine) | U (NIST) | z | P (this engine) | P (NIST) | z |
|---|---|---|---|---|---|---|---|
| 0.85 | 0.776 | −5.5120 ± 0.0009 | −5.5121 ± 0.0005 | 0.1 | −0.0002 ± 0.0048 | 0.0068 ± 0.0018 | −1.4 |
| 0.85 | 0.780 | −5.5382 ± 0.0013 | −5.5386 ± 0.0007 | 0.2 | 0.0413 ± 0.0076 | 0.0479 ± 0.0032 | −0.8 |
| 0.85 | 0.820 | −5.7921 ± 0.0013 | −5.7947 ± 0.0006 | 1.8 | 0.5588 ± 0.0074 | 0.5535 ± 0.0041 | 0.6 |
| 0.85 | 0.860 | −6.0276 ± 0.0011 | −6.0305 ± 0.0024 | 1.1 | 1.2768 ± 0.0060 | 1.2660 ± 0.0136 | 0.7 |
| 0.85 | 0.900 | −6.2355 ± 0.0015 | −6.2391 ± 0.0053 | 0.7 | 2.2427 ± 0.0083 | 2.2314 ± 0.0272 | 0.4 |
| 0.90 | 0.776 | −5.4671 ± 0.0015 | −5.4689 ± 0.0004 | 1.2 | 0.2412 ± 0.0076 | 0.2406 ± 0.0027 | 0.1 |
| 0.90 | 0.780 | −5.4950 ± 0.0013 | −5.4956 ± 0.0008 | 0.4 | 0.2756 ± 0.0078 | 0.2785 ± 0.0030 | −0.4 |
| 0.90 | 0.820 | −5.7453 ± 0.0009 | −5.7456 ± 0.0008 | 0.3 | 0.8186 ± 0.0042 | 0.8239 ± 0.0029 | −1.0 |
| 0.90 | 0.860 | −5.9742 ± 0.0011 | −5.9753 ± 0.0006 | 0.9 | 1.5762 ± 0.0062 | 1.5781 ± 0.0033 | −0.3 |
| 0.90 | 0.900 | −6.1733 ± 0.0015 | −6.1773 ± 0.0016 | 1.9 | 2.5995 ± 0.0087 | 2.5848 ± 0.0095 | 1.1 |

z = (this engine − NIST) / combined standard error. All 20 comparisons have |z| < 2, and χ² = 17.2 for 20
comparisons, which is what purely statistical differences produce.

![Energy vs density](bench/plots/validation_energy.png)
![Pressure vs density](bench/plots/validation_pressure.png)
![Radial distribution function](bench/plots/rdf.png)

**Getting the error bars right took three attempts**, and the first two were wrong in instructive ways:

1. With one run per state point and errors from block averaging, every point came out slightly *above* NIST
   (z ≈ 2–3 throughout). Two hypotheses for a real bias were tested and ruled out: the timestep (results
   unchanged from dt = 0.005 to 0.00125) and the potential shift changing which configurations are sampled
   (reweighting the samples to the unshifted potential moved U by at most 0.0015).
2. The cause was statistical. The blocking curve never reaches a plateau, so block averaging underestimated
   the error; the spread of independent replicas is 30–40% larger. And every state point had used the same
   seed, so they shared thermostat noise and their errors were correlated, which made them all lean the same
   way. Independent replicas with distinct seeds removed the "bias".
3. Then at ρ = 0.90 some replicas froze: they started from an FCC lattice near the freezing line and never
   melted. Melting at T = 2 before cooling to the target fixed it.

![Block averaging](bench/plots/blocking.png)

NIST also publishes NVE molecular dynamics data at T ≈ 0.85. Those values differ from NIST's own Monte Carlo
data by more than their quoted error bars at several points (most clearly the pressure at ρ = 0.90: 2.544 from
MD, 2.231 from MC). This engine's results agree with the Monte Carlo data.

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
| All pairs, 1 thread | 12.7 | 1× |
| Neighbour list, 1 thread | 1.44 | **8.8×** |
| Neighbour list, 4 threads | 0.58 | **21.7×** |

From 256 to 32,000 particles, the neighbour-list cost grows as about N^1.08.

### OpenMP scaling

![OpenMP strong scaling](bench/plots/thread_scaling.png)

The speedup at 4 threads is 2.4× for N = 4000 and 2.7× for N = 13,500. It levels off beyond 4 threads, as
expected on the M1: `schedule(static)` gives every thread an equal share, so once the slower efficiency cores
join, the other threads wait for them.

### Energy conservation

![Energy conservation](bench/plots/energy_conservation.png)

With the thermostat off, total energy stays within about 10⁻⁴ of its starting value, with no long-term drift.

![Energy fluctuation vs timestep](bench/plots/timestep_error.png)

Velocity Verlet is second order, so the energy error should scale as dt². With a 3.5σ cutoff it does
(measured slope 1.92). With 2.5σ the slope is 1.73 and the curve flattens at small dt: the force drops
abruptly to zero at the cutoff, and every pair that crosses it adds error that doesn't shrink with dt. Raising
the cutoff makes that jump about 10× smaller (0.039 → 0.0037), and the excess error disappears. Each point is the mean of 3
runs with different seeds, with whiskers showing the spread.

### Temperature stability

![Temperature](bench/plots/temperature.png)

During equilibration, velocities are rescaled to the target temperature every 10 steps. The early dip is the
FCC lattice melting: kinetic energy turns into potential energy faster than the rescaling can correct. After
the thermostat is switched off, the temperature fluctuates around the target as expected for a closed system.

### Neighbour list skin

![Skin sweep](bench/plots/skin_sweep.png)

A small skin means frequent rebuilds; a large skin means more pairs to check every step. 0.1σ is clearly too
small (rebuilds every 3 steps). From 0.2σ to 0.6σ the cost is flat to within about 5%, which is the size of
run-to-run noise on this machine. An earlier run, before the kernel and layout changes, had 0.2σ ahead by
12%; it remains the default.

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
rebuilds and integration aren't vectorised.

**Memory layout.** Three layouts were measured on the same code: one struct per particle (array of structs),
one array per component (struct of arrays), and a hot/cold split (positions packed as `x y z pad`, everything
else one array per component). All three produce bit-identical trajectories, checked with the state hash.
Timings are medians over runs that rotate the order of the binaries, because on a fanless laptop whichever
runs first is faster; an unrotated comparison had made the hybrid look 9% slower.

| | N = 4000, 1 thread | N = 4000, 4 threads | N = 32,000, 4 threads |
|---|---|---|---|
| Array of structs | 1.000× | 1.000× | 1.000× |
| Struct of arrays | 0.967× | 0.967× | 0.941× |
| **Hot/cold split** (used) | **1.012×** | **1.016×** | **1.008×** |

The hot loop is a gather: for each neighbour j it loads x[j], y[j], z[j]. Pure struct of arrays turns that
into 3 cache lines per neighbour instead of 1, so it is slower even though the integrator now vectorises.
Packing just the positions keeps the gather to one 32-byte load per neighbour. The effect is small either way:
the step is mostly compute-bound and a 32,000-particle system fits in the M1's caches.

**Mixed precision.** With `--mixed`, positions, the minimum image and all sums across particles stay double,
and the separations are narrowed to float for the pair maths. Separations are at most the list radius, so
float keeps about 7 significant digits of each. The compute loop then vectorises at width 4 instead of 2:

| | double | mixed | max energy drift (double / mixed) |
|---|---|---|---|
| N = 4000, 1 thread | 1.445 ms | 1.320 ms (1.09×) | 4.5×10⁻⁵ / 4.1×10⁻⁵ |
| N = 4000, 4 threads | 0.588 ms | 0.544 ms (1.08×) | 3.2×10⁻⁵ / 4.1×10⁻⁵ |
| N = 32,000, 4 threads | 4.892 ms | 4.494 ms (1.09×) | 1.5×10⁻⁵ / 1.9×10⁻⁵ |

Energy conservation is unchanged. The step gains only 9% because the compute loop is no longer the whole
cost: the scalar gather and scatter and the list rebuilds take the same time in both modes.

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
  Particle        plain particle record for setup and I/O
  System          particle storage (hot/cold layout), stable ids, box, periodic boundaries, kinetic energy
  Forcefield      Lennard-Jones forces: O(N²) reference, parallel SIMD, deterministic and mixed-precision
                  neighbour-list versions; virial and pair counts
  CellList        particles binned into cells for neighbour search
  NeighbourList   Verlet list: parallel build, displacement-based rebuild
  Integrator      Velocity Verlet
  Initialisation  FCC lattice, Maxwell-Boltzmann velocities
  Thermostat      velocity rescaling (equilibration) and Bussi stochastic rescaling (NVT)
  RadialDistribution  g(r) from neighbour-list pairs
  Trajectory      extended XYZ output in particle-id order
  CommandLine.hpp checked number parsing for the command-line drivers
  Parallel.hpp    OpenMP thread helpers with single-thread fallbacks
  main.cpp        command-line driver
vis/
  Renderer        OpenGL drawing: point-sprite spheres, box outline, orbit camera
  main.cpp        md_vis window, input and live simulation loop
tests/              GoogleTest unit tests
bench/benchmark.py  benchmark runner and plots
bench/validate.py   validation against NIST reference data
.github/workflows/  CI: Linux + macOS build and test (-Werror), ASan/UBSan and TSan, clang-format and clang-tidy
```


## References

**Methods**

- L. Verlet, "Computer 'experiments' on classical fluids. I. Thermodynamical properties of Lennard-Jones
  molecules", *Phys. Rev.* **159**, 98 (1967). The neighbour list with a skin, and the Verlet integrator.
- W. C. Swope, H. C. Andersen, P. H. Berens and K. R. Wilson, *J. Chem. Phys.* **76**, 637 (1982). The
  velocity form of the Verlet integrator.
- M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed., Oxford University Press (2017).
  Periodic boundaries and the minimum image convention, cell lists, the truncated-and-shifted potential,
  long-range (tail) corrections for energy and pressure, the virial pressure, and g(r).
- D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed., Academic Press (2002). Reduced
  Lennard-Jones units and the standard liquid state point (ρ = 0.8442, T = 0.728).
- G. Bussi, D. Donadio and M. Parrinello, "Canonical sampling through velocity rescaling", *J. Chem. Phys.*
  **126**, 014101 (2007). The stochastic velocity-rescaling thermostat (equation A7 of the paper).
- H. Flyvbjerg and H. G. Petersen, "Error estimates on averages of correlated data", *J. Chem. Phys.* **91**,
  461 (1989). Block averaging, and why it needs a plateau to be trusted.
- S. Páll and B. Hess, "A flexible algorithm for calculating pair interactions on SIMD architectures",
  *Comput. Phys. Commun.* **184**, 2641 (2013). Background for the SIMD kernel, packed positions and mixed
  precision, as used in GROMACS.

**Reference data**

- NIST Standard Reference Simulation Website, Lennard-Jones fluid benchmarks:
  [index](https://mmlapps.nist.gov/srs/LJ_PURE/index.htm),
  [NVT Monte Carlo](https://mmlapps.nist.gov/srs/LJ_PURE/mc.htm) (used for the validation table) and
  [NVE molecular dynamics](https://mmlapps.nist.gov/srs/LJ_PURE/md.htm).

**Tools and formats**

- [OpenMP API specifications](https://www.openmp.org/specifications/): `parallel for`, `simd`, reductions,
  and the unspecified order in which reductions combine partial results.
- [Archer](https://github.com/llvm/llvm-project/tree/main/openmp/tools/archer): LLVM's OpenMP data race
  detection tool, which lets ThreadSanitizer understand OpenMP synchronisation in CI.
- [clang-format style options](https://clang.llvm.org/docs/ClangFormatStyleOptions.html) (including
  `IndentPPDirectives: Leave`) and the [clang-tidy check list](https://clang.llvm.org/extra/clang-tidy/checks/list.html).
- [GoogleTest](https://github.com/google/googletest): unit test framework.
- [Extended XYZ](https://github.com/libAtoms/extxyz): trajectory file format.
