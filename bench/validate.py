#!/usr/bin/env python3
"""Validate md_sim against the NIST Lennard-Jones benchmark data (Standard Reference Simulation Website).

    python bench/validate.py              # run every state point (several minutes), then compare and plot
    python bench/validate.py --plot-only  # re-analyse existing runs

Reference data: https://mmlapps.nist.gov/srs/LJ_PURE/md.htm (NVE MD) and .../mc.htm (NVT Monte Carlo).
Both use N = 500, a 3 sigma cutoff, standard long-range corrections, and the virial pressure. The MD runs
also use Velocity Verlet with dt = 0.005, like this engine.

Which comparison is like for like: a constant shift of the potential does not change forces, so MD with a
truncated potential (NIST MD) and MD with a truncated-and-shifted one (this engine) follow the same dynamics
and sample the same configurations. Monte Carlo with the truncated, unshifted potential weights
configurations very slightly differently, so a small systematic offset from the MC data is expected.
"""

import argparse
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import (  # noqa: E402
    BINARY, GRID, RESULTS, SERIES, SURFACE, TEXT_SECONDARY, figure, read_rows, save, style, write_rows,
)

CUTOFF = 3.0
DT = 0.005
EQUIL_STEPS = 10000        # 50 time units, as NIST
PROD_STEPS = 20000         # 100 time units, as NIST
REPLICAS = 8               # independent runs per state point; error bars come from their spread
BLOCK_TIME = 4.0           # block length for the blocking plot, in LJ time units (NIST used 2-4)

# NIST NVE MD (T is their measured mean): (T, rho, U per particle, sigma U, P, sigma P)
NIST_MD = [
    (0.851, 0.776, -5.517, 0.001, 0.030, 0.006),
    (0.853, 0.780, -5.533, 0.001, 0.072, 0.005),
    (0.852, 0.820, -5.803, 0.001, 0.573, 0.005),
    (0.851, 0.840, -5.909, 0.001, 0.910, 0.006),
    (0.849, 0.860, -6.027, 0.001, 1.282, 0.005),
    (0.851, 0.900, -6.234, 0.001, 2.544, 0.006),
]

# NIST NVT Monte Carlo, liquid densities: (T, rho, U, sigma U, P, sigma P)
NIST_MC = [
    (0.85, 0.776, -5.5121, 4.55e-4, 6.7714e-3, 1.77e-3),
    (0.85, 0.780, -5.5386, 7.26e-4, 4.7924e-2, 3.18e-3),
    (0.85, 0.820, -5.7947, 6.03e-4, 5.5355e-1, 4.13e-3),
    (0.85, 0.860, -6.0305, 2.38e-3, 1.2660, 1.36e-2),
    (0.85, 0.900, -6.2391, 5.27e-3, 2.2314, 2.72e-2),
    (0.90, 0.776, -5.4689, 4.20e-4, 2.4056e-1, 2.74e-3),
    (0.90, 0.780, -5.4956, 7.86e-4, 2.7851e-1, 2.97e-3),
    (0.90, 0.820, -5.7456, 7.51e-4, 8.2386e-1, 2.85e-3),
    (0.90, 0.860, -5.9753, 5.53e-4, 1.5781, 3.29e-3),
    (0.90, 0.900, -6.1773, 1.57e-3, 2.5848, 9.54e-3),
]


def shift_energy(rc):
    return 4.0 * (rc ** -12 - rc ** -6)


def tail_energy(rho, rc):
    """Standard long-range correction to U per particle, assuming g(r) = 1 beyond the cutoff."""
    return 8.0 / 3.0 * math.pi * rho * (rc ** -9 / 3.0 - rc ** -3)


def tail_pressure(rho, rc):
    return 16.0 / 3.0 * math.pi * rho ** 2 * (2.0 / 3.0 * rc ** -9 - rc ** -3)


def block_error(series, block_len):
    """Standard error of the mean from non-overlapping blocks. Correlated samples make the naive error
    (block_len = 1) too small; with blocks longer than the correlation time, block means are independent."""
    blocks = len(series) // block_len
    means = [sum(series[b * block_len:(b + 1) * block_len]) / block_len for b in range(blocks)]
    mean = sum(means) / blocks
    var = sum((m - mean) ** 2 for m in means) / (blocks - 1)
    return math.sqrt(var / blocks)


def run_path(temperature, rho, replica):
    return RESULTS / f"validate_T{temperature}_rho{rho}_r{replica}.csv"


def rdf_path(temperature, rho):
    return RESULTS / f"validate_T{temperature}_rho{rho}_rdf.csv"


def run_state(temperature, rho, state_index):
    """Run REPLICAS independent simulations in parallel, one thread each. Every replica of every state point gets
    its own seed: sharing seeds would share thermostat noise, correlating the errors of different state points"""
    env = dict(os.environ, OMP_NUM_THREADS="1")
    procs = []
    for r in range(REPLICAS):
        cmd = [str(BINARY), "5", "--quiet", "--density", str(rho), "--temperature", str(temperature),
               "--cutoff", str(CUTOFF), "--dt", str(DT), "--nvt", "--seed", str(1000 * state_index + r + 1),
               "--equil", str(EQUIL_STEPS), "--steps", str(PROD_STEPS), "--csv", str(run_path(temperature, rho, r))]
        if r == 0:
            cmd += ["--rdf", str(rdf_path(temperature, rho))]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
    for proc in procs:
        _, err = proc.communicate()
        if proc.returncode != 0:
            sys.exit(f"md_sim failed: {err}")


def replica_series(temperature, rho, replica):
    """U per particle and P per step, with NIST's conventions: truncated (unshifted) energy + tail corrections"""
    rows = [r for r in read_rows(run_path(temperature, rho, replica)) if r["phase"] == "production"]
    shift = shift_energy(CUTOFF)
    u = [r["pe_per_particle"] + r["inside_pairs"] * shift / 500 + tail_energy(rho, CUTOFF) for r in rows]
    p = [r["pressure"] + tail_pressure(rho, CUTOFF) for r in rows]
    return u, p


def analyse(temperature, rho):
    """Mean over replicas, with the standard error from the spread of replica means. Block averaging of one run
    underestimates the error here: the energy stays correlated for longer than the blocks (see blocking.png)"""
    u_means, p_means = [], []
    for r in range(REPLICAS):
        u, p = replica_series(temperature, rho, r)
        u_means.append(sum(u) / len(u))
        p_means.append(sum(p) / len(p))
    return {
        "u": statistics.mean(u_means), "u_err": statistics.stdev(u_means) / math.sqrt(REPLICAS),
        "p": statistics.mean(p_means), "p_err": statistics.stdev(p_means) / math.sqrt(REPLICAS),
    }


def compare(reference, label):
    rows = []
    print(f"\n{label}")
    print(f"{'T':>6} {'rho':>6} | {'U (ours)':>17} {'U (NIST)':>17} {'z':>5} | {'P (ours)':>15} {'P (NIST)':>15} {'z':>5}")
    for t, rho, u_ref, u_ref_err, p_ref, p_ref_err in reference:
        a = analyse(t, rho)
        zu = (a["u"] - u_ref) / math.hypot(a["u_err"], u_ref_err)
        zp = (a["p"] - p_ref) / math.hypot(a["p_err"], p_ref_err)
        print(f"{t:6.3f} {rho:6.3f} | {a['u']:9.4f} ±{a['u_err']:.4f} {u_ref:9.4f} ±{u_ref_err:.4f} {zu:5.1f} | "
              f"{a['p']:7.4f} ±{a['p_err']:.4f} {p_ref:7.4f} ±{p_ref_err:.4f} {zp:5.1f}")
        rows.append({"reference": label, "T": t, "rho": rho, "U": a["u"], "U_err": a["u_err"], "U_nist": u_ref,
                     "U_nist_err": u_ref_err, "U_z": zu, "P": a["p"], "P_err": a["p_err"], "P_nist": p_ref,
                     "P_nist_err": p_ref_err, "P_z": zp})
    return rows


# ---------------------------------------------------------------- plots

def plot_eos(rows):
    md = [r for r in rows if r["reference"] == "NIST MD"]
    for key, ylabel, name in (("U", "U per particle (LJ units)", "validation_energy.png"),
                              ("P", "pressure (LJ units)", "validation_pressure.png")):
        fig, ax = figure()
        rho = [r["rho"] for r in md]
        ax.errorbar(rho, [r[f"{key}_nist"] for r in md], yerr=[r[f"{key}_nist_err"] for r in md], color=SERIES[1],
                    linestyle="none", marker="s", markersize=9, markerfacecolor="none", markeredgewidth=1.5,
                    capsize=3, label="NIST MD (NVE, T ≈ 0.85)")
        ax.errorbar(rho, [r[key] for r in md], yerr=[r[f"{key}_err"] for r in md], color=SERIES[0], linewidth=2,
                    marker="o", markersize=5, capsize=3, label="this engine (NVT, same T)")
        style(ax, f"{'Energy' if key == 'U' else 'Pressure'} vs density, N = 500, r_c = 3σ", "density ρ", ylabel)
        ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="upper left" if key == "P" else "upper right")
        save(fig, name)


def plot_rdf():
    fig, ax = figure()
    for rho, color in ((0.776, SERIES[0]), (0.900, SERIES[1])):
        rows = read_rows(rdf_path(0.85, rho))
        ax.plot([r["r"] for r in rows], [r["g"] for r in rows], color=color, linewidth=1.8, label=f"ρ = {rho}")
    ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    style(ax, "Radial distribution function, T = 0.85", "r (σ)", "g(r)")
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="upper right")
    save(fig, "rdf.png")


def plot_blocking():
    """Error estimate vs block length for one run. It should rise while blocks are shorter than the correlation
    time, then level off at the true error. Here it never levels off, which is why the error bars come from
    independent replicas instead. The dashed line is the replica-based error, in the same units."""
    u, p = replica_series(0.85, 0.82, 0)
    a = analyse(0.85, 0.82)
    fig, ax = figure()
    lengths = [2 ** k for k in range(0, 13) if len(u) // 2 ** k >= 8]
    for series, label, color in ((u, "U per particle", SERIES[0]), (p, "P", SERIES[1])):
        errors = [block_error(series, b) for b in lengths]
        ax.plot([b * DT for b in lengths], [e / errors[-1] for e in errors], color=color, linewidth=2, marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    # A single run's error is sqrt(REPLICAS) times the error of the mean over replicas
    replica_u = a["u_err"] * math.sqrt(REPLICAS) / block_error(u, lengths[-1])
    ax.axhline(replica_u, color=SERIES[0], linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("U error from replica spread", (lengths[0] * DT, replica_u), xytext=(0, 5),
                textcoords="offset points", fontsize=9, color=TEXT_SECONDARY)
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    style(ax, "Block averaging, T = 0.85, ρ = 0.82 (one run)", "block length (LJ time, log)",
          "error estimate / largest-block estimate")
    ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="upper left")
    save(fig, "blocking.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="re-analyse existing runs without re-running")
    args = parser.parse_args()

    states = sorted({(t, rho) for t, rho, *_ in NIST_MD + NIST_MC})
    if not args.plot_only:
        RESULTS.mkdir(parents=True, exist_ok=True)
        for i, (t, rho) in enumerate(states, 1):
            print(f"  [{i}/{len(states)}] T = {t}, rho = {rho} ({REPLICAS} replicas)", flush=True)
            run_state(t, rho, i)

    rows = compare(NIST_MD, "NIST MD") + compare(NIST_MC, "NIST MC")
    write_rows(RESULTS / "validation.csv", rows)
    plot_eos(rows)
    plot_rdf()
    plot_blocking()


if __name__ == "__main__":
    main()
