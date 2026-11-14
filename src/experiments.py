"""Run all experiments and write plots, a metrics table and trajectory JSON to results/.

Usage (from the project root):  python src/experiments.py
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cartpole import Params
from controllers import LQGController, PIDController
from simulate import Scenario, metrics, run

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
NOMINAL = Params()
COLORS = {"LQG": "#1f77b4", "PID": "#d62728"}


def make(name: str, ts: float):
    """Both controllers are always DESIGNED on the nominal model."""
    return LQGController(NOMINAL, ts) if name == "LQG" else PIDController(ts)


NAMES = ["LQG", "PID"]


def survives(name: str, sc: Scenario) -> bool:
    return not run(make(name, sc.ts), sc).fell


def success_rate(name: str, sc: Scenario, seeds=range(8)) -> float:
    return float(np.mean([survives(name, replace(sc, seed=s)) for s in seeds]))


# ----------------------------------------------------------------------------
def exp_nominal():
    sc = Scenario()
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    rows = []
    for n in NAMES:
        r = run(make(n, sc.ts), sc)
        ax[0].plot(r.t, np.degrees(r.states[:, 2]), color=COLORS[n], label=n)
        ax[1].plot(r.t, r.states[:, 0], color=COLORS[n], label=n)
        ax[2].plot(r.t, r.force, color=COLORS[n], label=n)
        rows.append({"controller": n, **metrics(r)})
        json.dump({"t": r.t.round(4).tolist(), "x": r.states[:, 0].round(5).tolist(),
                   "theta": r.states[:, 2].round(5).tolist(),
                   "force": r.force.round(4).tolist()},
                  open(RESULTS / f"trajectory_{n.lower()}.json", "w"))
    ax[0].set_ylabel("Pendulum angle (deg)")
    ax[1].set_ylabel("Cart position (m)")
    ax[2].set_ylabel("Applied force (N)")
    ax[2].set_xlabel("Time (s)")
    for a in ax:
        a.axvline(4.0, color="gray", ls=":", lw=1)
        a.grid(alpha=0.3)
        a.legend(loc="upper right")
    ax[0].set_title("Nominal scenario: 11.5° initial tilt, 3 N push at t = 4 s (dotted line)")
    fig.tight_layout()
    fig.savefig(RESULTS / "01_nominal.png", dpi=150)
    plt.close(fig)
    with open(RESULTS / "metrics_nominal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


def exp_region_of_attraction():
    """Largest initial tilt each controller can recover, versus actuator force limit."""
    f_limits = [2, 3, 4, 6, 8, 10, 15, 20]
    angles = np.arange(0.05, 1.31, 0.05)
    out = {n: [] for n in NAMES}
    for fmax in f_limits:
        for n in NAMES:
            best = 0.0
            for a in angles:
                sc = Scenario(x0=(0, 0, a, 0), f_max=fmax, push=None, t_end=6.0)
                if survives(n, sc):
                    best = a
                else:
                    break
            out[n].append(np.degrees(best))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in NAMES:
        ax.plot(f_limits, out[n], "o-", color=COLORS[n], label=n)
    ax.set_xlabel("Actuator force limit (N)")
    ax.set_ylabel("Max recoverable initial tilt (deg)")
    ax.set_title("Effect of actuator saturation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "02_saturation.png", dpi=150)
    plt.close(fig)
    return {"f_limits": f_limits, **out}


def sweep(xs, make_scenario, fname, xlabel, title, log_x=False):
    rates = {n: [success_rate(n, make_scenario(x)) * 100 for x in xs] for n in NAMES}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in NAMES:
        ax.plot(xs, rates[n], "o-", color=COLORS[n], label=n)
    ax.set_ylim(-5, 105)
    if log_x:
        if min(xs) <= 0:
            ax.set_xscale("symlog", linthresh=1)
        else:
            ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)
    return {"x": list(map(float, xs)), **rates}


def exp_delay():
    xs = list(range(0, 13, 1))  # samples of 10 ms
    return sweep(xs, lambda d: Scenario(delay_steps=d, push=None, t_end=6.0),
                 "03_delay.png", "Command delay (samples of 10 ms)",
                 "Robustness to actuation delay")


def exp_mass_mismatch():
    xs = [0.1, 0.2, 0.5, 1, 2, 4, 8, 15, 30]
    return sweep(xs, lambda f: Scenario(plant=NOMINAL.scaled(m=f), push=None, t_end=6.0),
                 "04_mass_mismatch.png", "True pendulum mass / design mass",
                 "Robustness to model error (controllers designed for 0.2 kg)", log_x=True)


def exp_length_mismatch():
    xs = [0.2, 0.3, 0.5, 1, 2, 3, 4]
    return sweep(xs, lambda f: Scenario(plant=NOMINAL.scaled(l=f), push=None, t_end=6.0),
                 "05_length_mismatch.png", "True pendulum length / design length",
                 "Robustness to model error (controllers designed for l = 0.3 m)", log_x=True)


def exp_noise():
    xs = [0, 1, 5, 10, 30, 60, 100, 200]
    return sweep(xs, lambda k: Scenario(meas_std=(0.001 * k, 0.002 * k), push=None, t_end=6.0),
                 "06_noise.png", "Sensor noise multiplier (1 = 1 mm, 0.11°)",
                 "Robustness to sensor noise", log_x=True)


if __name__ == "__main__":
    summary = {}
    print("Nominal scenario...")
    summary["nominal"] = exp_nominal()
    for row in summary["nominal"]:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print("Saturation / region of attraction...")
    summary["saturation"] = exp_region_of_attraction()
    print("Delay...")
    summary["delay"] = exp_delay()
    print("Mass mismatch...")
    summary["mass"] = exp_mass_mismatch()
    print("Length mismatch...")
    summary["length"] = exp_length_mismatch()
    print("Noise...")
    summary["noise"] = exp_noise()
    json.dump(summary, open(RESULTS / "summary.json", "w"), indent=2, default=float)
    print("Done. Results in", RESULTS)
