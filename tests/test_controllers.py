"""Regression tests for the controllers and the simulation harness."""
import numpy as np

from cartpole import Params, linearize
from controllers import LQGController, PIDController
from simulate import Scenario, run

NOMINAL = Params()


def test_lqr_closed_loop_is_stable():
    c = LQGController(NOMINAL, 0.01)
    A, B = linearize(NOMINAL)
    assert np.max(np.linalg.eigvals(A - B @ c.K.reshape(1, -1)).real) < 0


def test_kalman_filter_is_stable():
    c = LQGController(NOMINAL, 0.01)
    assert np.max(np.abs(np.linalg.eigvals(c.Ad - c.L @ (np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) @ c.Ad)))) < 1


def test_both_controllers_balance_the_nominal_case():
    sc = Scenario()
    for ctrl in (LQGController(NOMINAL, sc.ts), PIDController(sc.ts)):
        res = run(ctrl, sc)
        assert not res.fell
        assert abs(res.states[-1, 2]) < np.radians(0.5)


def test_actuator_limit_is_respected():
    sc = Scenario(f_max=3.0)
    res = run(LQGController(NOMINAL, sc.ts), sc)
    assert np.max(np.abs(res.force)) <= 3.0 + 1e-9


def test_lqg_tolerates_more_delay_than_pid():
    """Regression for a documented result: LQG survives 60 ms, PID does not."""
    sc = Scenario(delay_steps=6, push=None, t_end=6.0)
    assert not run(LQGController(NOMINAL, sc.ts), sc).fell
    assert run(PIDController(sc.ts), sc).fell


def test_simulation_is_reproducible():
    sc = Scenario(seed=3)
    a = run(LQGController(NOMINAL, sc.ts), sc).states
    b = run(LQGController(NOMINAL, sc.ts), sc).states
    assert np.array_equal(a, b)
