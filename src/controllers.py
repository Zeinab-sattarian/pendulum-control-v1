"""Controllers for the cart-pole.

All controllers share one interface:
    reset()
    update(y, t, u_prev) -> commanded force (N)
where y = [x, theta] is the (noisy) measurement and u_prev is the force that was
actually applied during the previous sample (after saturation / delay).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.signal import cont2discrete

from cartpole import C_MEAS, Params, linearize


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Continuous-time LQR gain K such that u = -K x minimises  integral(x'Qx + u'Ru)."""
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


class LQGController:
    """LQR state feedback + steady-state Kalman filter.

    The plant only provides x and theta. The Kalman filter reconstructs the
    velocities from those noisy measurements and the known model.
    """

    name = "LQG"

    def __init__(self, design_params: Params, ts: float,
                 q_diag=(10.0, 1.0, 100.0, 1.0), r: float = 0.1,
                 meas_std=(0.001, 0.002), process_std: float = 0.05,
                 x_ref: float = 0.0):
        self.ts = ts
        self.x_ref = x_ref
        A, B = linearize(design_params)
        self.K = lqr_gain(A, B, np.diag(q_diag), np.array([[r]])).flatten()

        # Kalman filter designed on the zero-order-hold discretised model.
        Ad, Bd, *_ = cont2discrete((A, B, C_MEAS, np.zeros((2, 1))), ts, method="zoh")
        self.Ad, self.Bd = Ad, Bd
        Qn = (process_std**2) * (Bd @ Bd.T) + 1e-9 * np.eye(4)
        Rn = np.diag(np.square(meas_std))
        P = solve_discrete_are(Ad.T, C_MEAS.T, Qn, Rn)
        self.L = P @ C_MEAS.T @ np.linalg.inv(C_MEAS @ P @ C_MEAS.T + Rn)
        self.reset()

    def reset(self) -> None:
        self.xhat = np.zeros(4)
        self._first = True

    def update(self, y, t, u_prev):
        y = np.asarray(y, dtype=float)
        if self._first:  # initialise the estimate from the first measurement
            self.xhat = np.array([y[0], 0.0, y[1], 0.0])
            self._first = False
        else:
            pred = self.Ad @ self.xhat + self.Bd.flatten() * u_prev
            self.xhat = pred + self.L @ (y - C_MEAS @ pred)
        err = self.xhat - np.array([self.x_ref, 0.0, 0.0, 0.0])
        return float(-self.K @ err)


class PIDController:
    """Cascaded-style PID: an angle PID plus a cart-position/velocity PD term.

    A single angle PID balances the pendulum but lets the cart drift away, so a
    position term is added. Velocities come from low-pass-filtered derivatives of
    the noisy measurements (no model, no observer) - the classical approach.
    """

    name = "PID"

    def __init__(self, ts: float, kp=45.0, ki=0.0, kd=4.0,
                 kx=2.0, kxd=3.0, tau: float = 0.02, x_ref: float = 0.0):
        self.ts, self.tau = ts, tau
        self.kp, self.ki, self.kd, self.kx, self.kxd = kp, ki, kd, kx, kxd
        self.x_ref = x_ref
        self.reset()

    def reset(self) -> None:
        self._prev = None
        self._xd = 0.0
        self._thd = 0.0
        self._int = 0.0

    def update(self, y, t, u_prev):
        x, th = float(y[0]), float(y[1])
        if self._prev is None:
            self._prev = (x, th)
        a = self.ts / (self.tau + self.ts)  # first-order low-pass on the derivative
        self._xd += a * ((x - self._prev[0]) / self.ts - self._xd)
        self._thd += a * ((th - self._prev[1]) / self.ts - self._thd)
        self._prev = (x, th)
        self._int += th * self.ts
        self._int = float(np.clip(self._int, -1.0, 1.0))  # crude anti-windup
        return (self.kp * th + self.ki * self._int + self.kd * self._thd
                + self.kx * (x - self.x_ref) + self.kxd * self._xd)
