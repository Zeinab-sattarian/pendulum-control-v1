# Inverted pendulum on a cart: LQG vs PID under realistic constraints

A **simulation-only** study of balancing an unstable system (cart-pole) with two controllers, and of
*where each one breaks*. Everything is Python (NumPy/SciPy/Matplotlib) with automated tests.

> Scope note: this is a simulation. It does not capture friction nonlinearities, sensor drift, quantisation
> or real-time jitter. Those are listed under *Limitations*.

## What is modelled

- **Plant:** full nonlinear cart-pole derived from the Lagrangian, integrated with RK4 (1 ms step).
  Verified by tests: energy conservation (frictionless), match with the linear model for small angles.
- **Controller loop:** 100 Hz sampling with zero-order hold, **noisy** sensors for cart position and
  angle only (no velocity sensors), **actuator saturation**, optional **command delay**, external **push**.
- **LQG:** LQR state feedback (Q = diag(10, 1, 100, 1), R = 0.1) + steady-state **Kalman filter**
  that reconstructs velocities from noisy x and θ.
- **PID:** angle PID + cart position/velocity PD, velocities from low-pass-filtered derivatives
  (the classical model-free approach).
- Both controllers are **designed on the nominal model** and then tested on perturbed plants.

## Results (nominal plant: M = 0.5 kg, m = 0.2 kg, l = 0.3 m)

| Test | LQG | PID |
|---|---|---|
| Settling time of angle (|θ| < 1°, start 11.5°) | 0.90 s | 0.35 s |
| Cart returns to origin after push | yes (smooth) | slowly |
| Max recoverable initial tilt, force limit 10 N | 45.8° | 48.7° |
| Max recoverable initial tilt, force limit 2 N | 11.5° | 14.3° |
| Max command delay tolerated | 60 ms | 30 ms |
| Max pendulum-mass error tolerated (true / design) | 4× | 15× |
| Max sensor-noise level tolerated (× nominal 1 mm / 0.115°) | 100× | 30× (75% success at 60×) |

**Take-aways**
1. The model-based **LQG** wins where *estimation* matters: noise (the Kalman filter smooths it) and delay.
   Its force signal is much smoother.
2. The **PID** wins where *model error* matters (it does not rely on the model) and is a bit faster on the angle.
   Its force is jittery because it differentiates noisy measurements.
3. There is no universally better controller; the right choice depends on which non-ideality dominates.

Plots are in `results/` (`01_nominal.png` ... `06_noise.png`), raw numbers in `results/summary.json`.

Success = pendulum never exceeds 80° during the run; sweeps use 8 noise seeds per point.

## Run it

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell. On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/experiments.py          # regenerates every plot in results/ (about 1-2 minutes)
python -m pytest                   # physics + controller tests
```

## Layout

```
src/cartpole.py      nonlinear dynamics, RK4, numerical linearisation
src/controllers.py   LQR/Kalman (LQG) and PID
src/simulate.py      closed-loop simulation, scenarios, metrics
src/experiments.py   all experiments, plots, JSON export
tests/               11 tests (physics sanity + controller regression)
results/             plots, metrics, trajectory JSON (for the web dashboard)
```

## Limitations (what a real build would add)

- PID gains and LQR weights were tuned by hand, not optimised; a fairer comparison would optimise both.
- No friction nonlinearity (stiction), encoder quantisation, motor dynamics or timing jitter.
- Delay is not compensated in the Kalman filter (a delay-aware predictor would extend the LQG limit).
- Only stabilisation near upright; no swing-up from hanging position yet.

## Planned next steps

- Linear **MPC** with input constraints, compared against LQG/PID.
- **Energy-based swing-up** + switch to LQR.
- **React dashboard** animating the pendulum from the exported trajectories, with live gain sliders.
- GitHub Actions CI running the tests.
