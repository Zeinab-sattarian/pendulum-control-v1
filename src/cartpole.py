"""Nonlinear cart-pole (inverted pendulum on a cart) model.

State vector: [x, x_dot, theta, theta_dot]
  x      : cart position (m), positive to the right
  theta  : pendulum angle from the UPRIGHT position (rad), positive = leaning right
Input: horizontal force F on the cart (N).

Derived from the Lagrangian L = T - V with the pendulum modelled as a uniform rod
of mass m whose centre of mass is at distance l from the pivot:

  (M + m) x'' + m l cos(th) th''  = F - b x' + m l sin(th) th'^2
  m l cos(th) x'' + (I + m l^2) th'' = m g l sin(th)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

N_STATES = 4


@dataclass(frozen=True)
class Params:
    M: float = 0.5   # cart mass (kg)
    m: float = 0.2   # pendulum mass (kg)
    l: float = 0.3   # pivot to pendulum centre of mass (m)
    g: float = 9.81  # gravity (m/s^2)
    b: float = 0.1   # cart viscous friction (N.s/m)

    @property
    def inertia(self) -> float:
        """Moment of inertia of a uniform rod about its centre of mass."""
        return self.m * self.l**2 / 3.0

    def scaled(self, **factors: float) -> "Params":
        """Return a copy with some parameters multiplied, e.g. scaled(m=1.3)."""
        return replace(self, **{k: getattr(self, k) * f for k, f in factors.items()})


def dynamics(state: np.ndarray, force: float, p: Params) -> np.ndarray:
    """Continuous-time nonlinear dynamics: returns d(state)/dt."""
    _, xd, th, thd = state
    s, c = np.sin(th), np.cos(th)
    a11, a12 = p.M + p.m, p.m * p.l * c
    a22 = p.inertia + p.m * p.l**2
    r1 = force - p.b * xd + p.m * p.l * s * thd**2
    r2 = p.m * p.g * p.l * s
    det = a11 * a22 - a12 * a12          # closed-form 2x2 solve (faster than linalg.solve)
    xdd = (a22 * r1 - a12 * r2) / det
    thdd = (a11 * r2 - a12 * r1) / det
    return np.array([xd, xdd, thd, thdd])


def rk4_step(state: np.ndarray, force: float, p: Params, dt: float) -> np.ndarray:
    """One 4th-order Runge-Kutta step with the force held constant."""
    k1 = dynamics(state, force, p)
    k2 = dynamics(state + 0.5 * dt * k1, force, p)
    k3 = dynamics(state + 0.5 * dt * k2, force, p)
    k4 = dynamics(state + dt * k3, force, p)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def linearize(p: Params, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Linearize about the upright equilibrium (state = 0, F = 0).

    Uses central finite differences of the nonlinear model, so the linear model
    can never drift out of sync with the simulation.
    """
    x0 = np.zeros(N_STATES)
    A = np.zeros((N_STATES, N_STATES))
    for i in range(N_STATES):
        d = np.zeros(N_STATES)
        d[i] = eps
        A[:, i] = (dynamics(x0 + d, 0.0, p) - dynamics(x0 - d, 0.0, p)) / (2 * eps)
    B = ((dynamics(x0, eps, p) - dynamics(x0, -eps, p)) / (2 * eps)).reshape(-1, 1)
    return A, B


# Sensors measure cart position and pendulum angle only (no velocity sensors).
C_MEAS = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])
