"""Closed-loop simulation with realistic non-idealities.

The plant is the NONLINEAR model integrated with RK4 at a fine step (dt).
The controller runs at a slower sample time (ts) with a zero-order hold, and sees:
  * noisy measurements of x and theta only
  * a pure transport delay of `delay_steps` samples on the command
  * actuator saturation at +/- f_max
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cartpole import Params, rk4_step


@dataclass
class Scenario:
    x0: tuple = (0.0, 0.0, 0.2, 0.0)         # initial state (0.2 rad ~ 11.5 deg)
    t_end: float = 8.0
    dt: float = 0.001                         # physics step (s)
    ts: float = 0.01                          # controller sample time (s) = 100 Hz
    f_max: float = 10.0                       # actuator limit (N)
    meas_std: tuple = (0.001, 0.002)          # sensor noise std: x (m), theta (rad)
    delay_steps: int = 0                      # command delay in controller samples
    push: tuple | None = (4.0, 3.0, 0.05)     # (time s, force N, duration s) or None
    plant: Params = field(default_factory=Params)
    seed: int = 0


@dataclass
class Result:
    t: np.ndarray
    states: np.ndarray      # (N, 4)
    force: np.ndarray       # applied force (after saturation)
    command: np.ndarray     # controller command (before saturation)
    fell: bool
    scenario: Scenario


FALL_ANGLE = 1.4  # rad (~80 deg): beyond this the pendulum is considered fallen


def run(controller, sc: Scenario) -> Result:
    rng = np.random.default_rng(sc.seed)
    n_sub = int(round(sc.ts / sc.dt))
    n_ctrl = int(round(sc.t_end / sc.ts))
    controller.reset()

    state = np.array(sc.x0, dtype=float)
    queue = deque([0.0] * sc.delay_steps)
    u_applied = 0.0
    t_log, s_log, f_log, c_log = [], [], [], []
    fell = False

    for k in range(n_ctrl):
        t = k * sc.ts
        y = np.array([state[0], state[2]]) + rng.normal(0.0, sc.meas_std)
        cmd = controller.update(y, t, u_applied)
        queue.append(cmd)
        cmd_delayed = queue.popleft()
        u_applied = float(np.clip(cmd_delayed, -sc.f_max, sc.f_max))

        t_log.append(t)
        s_log.append(state.copy())
        f_log.append(u_applied)
        c_log.append(cmd)

        for j in range(n_sub):
            f = u_applied
            if sc.push is not None:
                t_now = t + j * sc.dt
                if sc.push[0] <= t_now < sc.push[0] + sc.push[2]:
                    f += sc.push[1]
            state = rk4_step(state, f, sc.plant, sc.dt)
        if abs(state[2]) > FALL_ANGLE or not np.all(np.isfinite(state)):
            fell = True
            break

    return Result(np.array(t_log), np.array(s_log), np.array(f_log),
                  np.array(c_log), fell, sc)


def metrics(res: Result) -> dict:
    """Performance numbers for one run."""
    th, x, f = res.states[:, 2], res.states[:, 0], res.force
    out = {"fell": res.fell,
           "max_angle_deg": float(np.degrees(np.max(np.abs(th)))),
           "peak_force_N": float(np.max(np.abs(f))),
           "sat_fraction": float(np.mean(np.abs(f) >= res.scenario.f_max - 1e-9))}
    if res.fell:
        out.update(settling_time_s=np.nan, rms_force_N=np.nan, max_cart_dev_m=np.nan)
        return out
    # Settling time after the initial condition: last time |theta| > 1 deg (before the push).
    t_stop = res.scenario.push[0] if res.scenario.push else res.t[-1]
    mask = res.t < t_stop
    bad = np.where(np.abs(th[mask]) > np.radians(1.0))[0]
    out["settling_time_s"] = float(res.t[mask][bad[-1]]) if bad.size else 0.0
    out["rms_force_N"] = float(np.sqrt(np.mean(f**2)))
    out["max_cart_dev_m"] = float(np.max(np.abs(x)))
    return out
