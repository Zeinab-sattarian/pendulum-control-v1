"""Physics sanity checks: if these fail, nothing downstream can be trusted."""
import numpy as np

from cartpole import Params, dynamics, linearize, rk4_step


def energy(s, p):
    _, xd, th, thd = s
    kinetic = (0.5 * (p.M + p.m) * xd**2 + p.m * p.l * np.cos(th) * xd * thd
               + 0.5 * (p.inertia + p.m * p.l**2) * thd**2)
    return kinetic + p.m * p.g * p.l * np.cos(th)


def test_energy_is_conserved_without_friction_or_force():
    p = Params(b=0.0)
    s = np.array([0.0, 0.0, 2.5, 0.0])  # pendulum swinging down from a large angle
    e0 = energy(s, p)
    for _ in range(2000):
        s = rk4_step(s, 0.0, p, 0.001)
    assert abs(energy(s, p) - e0) / abs(e0) < 1e-6


def test_upright_equilibrium_is_unstable_and_controllable():
    A, B = linearize(Params())
    assert np.max(np.linalg.eigvals(A).real) > 1.0  # one clearly unstable pole
    ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(4)])
    assert np.linalg.matrix_rank(ctrb) == 4


def test_linear_model_matches_nonlinear_for_small_angles():
    p = Params()
    A, B = linearize(p)
    s_nl = np.array([0.0, 0.0, 0.01, 0.0])
    s_lin = s_nl.copy()
    dt = 0.001
    for _ in range(100):  # 0.1 s
        s_nl = rk4_step(s_nl, 0.0, p, dt)
        s_lin = s_lin + dt * (A @ s_lin)  # forward Euler is enough at this tolerance
    assert abs(s_nl[2] - s_lin[2]) < 1e-3


def test_pendulum_falls_without_control():
    p = Params()
    s = np.array([0.0, 0.0, 0.01, 0.0])
    for _ in range(1500):
        s = rk4_step(s, 0.0, p, 0.001)
    assert abs(s[2]) > 0.5


def test_dynamics_at_equilibrium_is_zero():
    assert np.allclose(dynamics(np.zeros(4), 0.0, Params()), 0.0)
