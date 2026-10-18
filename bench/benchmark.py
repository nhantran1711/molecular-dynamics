#!/usr/bin/env python3
"""Benchmark suite for md_sim: energy conservation, temperature stability, thread scaling and N scaling.

Runs the Release binary, writes raw results to bench/results/ and plots to bench/plots/.

    python bench/benchmark.py              # run every benchmark, then plot
    python bench/benchmark.py --quick      # smaller sizes and fewer steps, to check the pipeline works
    python bench/benchmark.py energy       # run (and plot) only some benchmarks
    python bench/benchmark.py --plot-only  # re-plot existing results without re-running

Timings on a laptop vary with thermals and background load: every timing is the fastest of --repeats runs.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "build" / "release" / "md_sim"
RESULTS = ROOT / "bench" / "results"
PLOTS = ROOT / "bench" / "plots"

# Reference palette (first three categorical slots, which stay colourblind-safe for any pairing)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------- running md_sim

def run_md(cells, threads, *flags):
    """Run md_sim once and return its RESULT line as a dict."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = [str(BINARY), str(cells), "--quiet", *map(str, flags)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"md_sim failed ({' '.join(cmd[1:])}):\n{proc.stderr}")
    out = proc.stdout
    line = next(l for l in out.splitlines() if l.startswith("RESULT "))
    result = dict(field.split("=", 1) for field in line.split()[1:])
    return {k: (v if k == "forces" else float(v)) for k, v in result.items()}


def fastest_of(repeats, cells, threads, *flags):
    runs = [run_md(cells, threads, *flags) for _ in range(repeats)]
    return min(runs, key=lambda r: r["seconds"])


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}")


def read_rows(path):
    with open(path) as f:
        return [{k: (v if k in ("phase", "forces") else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


# ---------------------------------------------------------------- benchmarks

DTS = [0.0025, 0.005, 0.01]


def run_energy(quick, repeats):
    """Same physical time at three timesteps: Velocity Verlet's energy error should scale as dt^2."""
    equil_time, prod_time = (2.0, 5.0) if quick else (10.0, 50.0)
    for dt in DTS:
        print(f"  energy: dt={dt}")
        run_md(5, 4, "--dt", dt, "--equil", round(equil_time / dt), "--steps", round(prod_time / dt),
               "--csv", RESULTS / f"energy_dt{dt}.csv")


def run_threads(quick, repeats):
    """Strong scaling: fixed N, more threads."""
    sizes = [10] if quick else [10, 15]
    threads = [1, 2, 4] if quick else [1, 2, 4, 6, 8]
    steps = 100 if quick else 500
    rows = []
    for cells in sizes:
        for t in threads:
            r = fastest_of(repeats, cells, t, "--equil", 100, "--steps", steps)
            print(f"  threads: N={int(r['n'])} threads={t} {r['ms_per_step']:.3f} ms/step")
            rows.append({"n": int(r["n"]), "threads": t, "ms_per_step": r["ms_per_step"]})
    write_rows(RESULTS / "thread_scaling.csv", rows)


def run_sizes(quick, repeats):
    """Cost per step vs N: all pairs (serial) vs neighbour list at 1 and 4 threads."""
    list_cells = [4, 6, 8, 10] if quick else [4, 5, 6, 8, 10, 12, 15, 20]
    brute_cells = [4, 6, 8] if quick else [4, 5, 6, 8, 10]
    steps = 50 if quick else 200
    configs = [("brute", 1, brute_cells, ["--brute"]), ("list", 1, list_cells, []), ("list", 4, list_cells, [])]
    rows = []
    for forces, t, cells_list, extra in configs:
        for cells in cells_list:
            r = fastest_of(repeats, cells, t, *extra, "--equil", 50, "--steps", steps)
            print(f"  sizes: {forces} threads={t} N={int(r['n'])} {r['ms_per_step']:.3f} ms/step")
            rows.append({"forces": forces, "threads": t, "n": int(r["n"]), "ms_per_step": r["ms_per_step"]})
    write_rows(RESULTS / "size_scaling.csv", rows)


TIMESTEPS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]


TIMESTEP_CUTOFFS = [2.5, 3.5]


def relative_rms_energy(path):
    energies = [r["e_per_particle"] for r in read_rows(path) if r["phase"] == "production"]
    mean = sum(energies) / len(energies)
    return (sum((e - mean) ** 2 for e in energies) / len(energies)) ** 0.5 / abs(mean)


def run_timestep(quick, repeats):
    """Energy fluctuation vs dt at fixed physical time: Velocity Verlet is second order, so it should scale as dt^2.

    Two cutoffs, because the force jumps to zero at the cutoff and adds error that doesn't scale as dt^2; a
    larger cutoff shrinks that jump. Each point is the mean of several runs: OpenMP reductions sum in a
    different order every run, so trajectories (and a single run's RMS) differ run to run.
    """
    dts = TIMESTEPS[::2] if quick else TIMESTEPS
    equil_time, prod_time = (2.0, 5.0) if quick else (10.0, 20.0)
    rows = []
    for cutoff in TIMESTEP_CUTOFFS:
        for dt in dts:
            path = RESULTS / f"timestep_rc{cutoff}_dt{dt}.csv"
            samples = []
            for _ in range(repeats):
                run_md(5, 4, "--cutoff", cutoff, "--dt", dt, "--equil", round(equil_time / dt),
                       "--steps", round(prod_time / dt), "--csv", path)
                samples.append(relative_rms_energy(path))
            mean = sum(samples) / len(samples)
            print(f"  timestep: rc={cutoff} dt={dt} rms(E)/|E| = {mean:.3e} "
                  f"(runs: {', '.join(f'{x:.2e}' for x in samples)})")
            rows.append({"cutoff": cutoff, "dt": dt, "relative_rms_energy": mean,
                         "min": min(samples), "max": max(samples)})
    write_rows(RESULTS / "timestep_error.csv", rows)


SKINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def run_skin(quick, repeats):
    """Neighbour list skin: small skins rebuild often, large skins carry more pairs through every force step."""
    skins = SKINS[::2] if quick else SKINS
    equil, steps = (100, 200) if quick else (200, 1000)
    rows = []
    for skin in skins:
        r = fastest_of(repeats, 10, 4, "--skin", skin, "--equil", equil, "--steps", steps)
        steps_per_rebuild = (equil + steps) / r["rebuilds"]
        print(f"  skin: {skin} {r['ms_per_step']:.3f} ms/step, rebuild every {steps_per_rebuild:.1f} steps")
        rows.append({"skin": skin, "ms_per_step": r["ms_per_step"], "steps_per_rebuild": steps_per_rebuild})
    write_rows(RESULTS / "skin_sweep.csv", rows)


# ---------------------------------------------------------------- plotting

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="semibold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)


def figure():
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / name, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {(PLOTS / name).relative_to(ROOT)}")


def plot_energy():
    # Energy drift, production phase only, relative to the energy at the start of production
    fig, ax = figure()
    # Largest dt drawn first, so the smaller (quieter) curves stay visible on top
    for dt, color in reversed(list(zip(DTS, SERIES))):
        rows = [r for r in read_rows(RESULTS / f"energy_dt{dt}.csv") if r["phase"] == "production"]
        t0, e0 = rows[0]["time"], rows[0]["e_per_particle"]
        times = [r["time"] - t0 for r in rows]
        drift = [1e4 * (r["e_per_particle"] - e0) / abs(e0) for r in rows]
        ax.plot(times, drift, color=color, linewidth=1.2, label=f"dt = {dt}")
    style(ax, "Energy conservation (NVE, N = 500)", "time (LJ units)", "(E − E₀) / |E₀|   (×10⁻⁴)")
    # Legend below the plot area, in dt order, so it never covers the data
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, labelcolor=TEXT_SECONDARY, ncols=3,
              loc="upper left", bbox_to_anchor=(0, -0.14))
    save(fig, "energy_conservation.png")

    # Temperature across equilibration (thermostat on) and production (thermostat off), at the default dt
    rows = read_rows(RESULTS / "energy_dt0.005.csv")
    fig, ax = figure()
    times = [r["time"] for r in rows]
    ax.plot(times, [r["temperature"] for r in rows], color=SERIES[0], linewidth=1.2)
    ax.axhline(0.728, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    switch = next(r["time"] for r in rows if r["phase"] == "production")
    ax.axvline(switch, color=GRID, linewidth=1.5)
    ax.annotate("thermostat off →", (switch, 0.728), xytext=(6, 28), textcoords="offset points",
                fontsize=9, color=TEXT_SECONDARY)
    ax.annotate("target T = 0.728", (times[-1], 0.728), xytext=(0, 6), textcoords="offset points",
                ha="right", fontsize=9, color=TEXT_SECONDARY)
    style(ax, "Temperature (N = 500, dt = 0.005)", "time (LJ units)", "temperature (LJ units)")
    save(fig, "temperature.png")


def plot_threads():
    rows = read_rows(RESULTS / "thread_scaling.csv")
    sizes = sorted({int(r["n"]) for r in rows})
    fig, ax = figure()
    max_threads = max(r["threads"] for r in rows)
    ax.plot([1, max_threads], [1, max_threads], color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)),
            label="ideal")
    for n, color in zip(sizes, SERIES):
        pts = sorted((r["threads"], r["ms_per_step"]) for r in rows if int(r["n"]) == n)
        base = pts[0][1]
        ax.plot([t for t, _ in pts], [base / ms for _, ms in pts], color=color, linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"N = {n}")
    style(ax, "OpenMP strong scaling (neighbour list)", "threads", "speedup vs 1 thread")
    ax.set_xticks(sorted({int(r["threads"]) for r in rows}))
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="upper left")
    save(fig, "thread_scaling.png")


def plot_sizes():
    rows = read_rows(RESULTS / "size_scaling.csv")
    series = [("brute", 1, "all pairs, 1 thread"), ("list", 1, "neighbour list, 1 thread"),
              ("list", 4, "neighbour list, 4 threads")]
    fig, ax = figure()
    for (forces, t, label), color in zip(series, SERIES):
        pts = sorted((r["n"], r["ms_per_step"]) for r in rows if r["forces"] == forces and int(r["threads"]) == t)
        ax.plot([n for n, _ in pts], [ms for _, ms in pts], color=color, linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Label log axes with plain numbers at the measured sizes rather than powers of ten
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.set_xticks(sorted({int(r["n"]) for r in rows}))
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(plain)
    style(ax, "Cost per step vs system size", "particles N (log)", "ms per step (log)")
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="upper left")
    save(fig, "size_scaling.png")


def plot_timestep():
    rows = read_rows(RESULTS / "timestep_error.csv")
    fig, ax = figure()

    for cutoff, color in zip(TIMESTEP_CUTOFFS, SERIES):
        pts = sorted((r["dt"], r["relative_rms_energy"], r["min"], r["max"]) for r in rows if r["cutoff"] == cutoff)
        dts = [p[0] for p in pts]
        # Whiskers show the spread across repeated runs
        ax.vlines(dts, [p[2] for p in pts], [p[3] for p in pts], color=color, linewidth=1, alpha=0.6)
        ax.plot(dts, [p[1] for p in pts], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"cutoff {cutoff:g}σ")

    # Reference slope-2 line through the largest cutoff's middle point: parallel to the data means error ∝ dt²
    ref_pts = sorted((r["dt"], r["relative_rms_energy"]) for r in rows if r["cutoff"] == TIMESTEP_CUTOFFS[-1])
    dts = [p[0] for p in ref_pts]
    mid_dt, mid_err = ref_pts[len(ref_pts) // 2]
    ref = [mid_err * (dt / mid_dt) ** 2 for dt in dts]
    ax.plot(dts, ref, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), label="∝ dt²")

    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.set_xticks(dts)
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    style(ax, "Energy fluctuation vs timestep (N = 500)", "timestep dt (log)", "RMS(E) / |E|  (log)")
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="upper left")
    save(fig, "timestep_error.png")


def plot_skin():
    rows = read_rows(RESULTS / "skin_sweep.csv")
    skins = [r["skin"] for r in rows]
    ms = [r["ms_per_step"] for r in rows]
    fig, ax = figure()
    ax.plot(skins, ms, color=SERIES[0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)

    # Label the fastest point and the two ends with how often the list was rebuilt
    best = min(range(len(rows)), key=lambda i: ms[i])
    for i in sorted({0, best, len(rows) - 1}):
        text = f"rebuild every {rows[i]['steps_per_rebuild']:.1f} steps"
        if i == best:
            text = f"fastest: skin {skins[i]:g}\n{text}"
        # The minimum sits in a valley, so its label goes below the point to stay clear of the line
        offset, va = ((0, -14), "top") if i == best else ((0, 12), "bottom")
        # End labels anchor inward so they don't run into the axis or off the plot
        ha = "center" if i == best else ("left" if i == 0 else "right")
        ax.annotate(text, (skins[i], ms[i]), xytext=offset, textcoords="offset points", ha=ha, va=va,
                    fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylim(0, max(ms) * 1.3)
    style(ax, "Neighbour list skin vs cost (N = 4000, 4 threads)", "skin (σ)", "ms per step")
    save(fig, "skin_sweep.png")


# ---------------------------------------------------------------- entry point

BENCHMARKS = {
    "energy": (run_energy, plot_energy),
    "threads": (run_threads, plot_threads),
    "sizes": (run_sizes, plot_sizes),
    "timestep": (run_timestep, plot_timestep),
    "skin": (run_skin, plot_skin),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", nargs="*", help=f"benchmarks to run: {', '.join(BENCHMARKS)} (default: all)")
    parser.add_argument("--quick", action="store_true", help="small sizes and few steps, to test the pipeline")
    parser.add_argument("--repeats", type=int, default=3, help="take the fastest of this many timing runs")
    parser.add_argument("--plot-only", action="store_true", help="re-plot existing results without running")
    args = parser.parse_args()

    unknown = [name for name in args.which if name not in BENCHMARKS]
    if unknown:
        parser.error(f"unknown benchmark {', '.join(unknown)} (choose from {', '.join(BENCHMARKS)})")

    if not args.plot_only and not BINARY.exists():
        sys.exit(f"{BINARY.relative_to(ROOT)} not found. Build it first:\n"
                 "  cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release && cmake --build build/release")

    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in args.which or list(BENCHMARKS):
        run, plot = BENCHMARKS[name]
        print(f"{name}:")
        if not args.plot_only:
            run(args.quick, 1 if args.quick else args.repeats)
        plot()


if __name__ == "__main__":
    main()
