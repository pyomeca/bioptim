# MHE: moving horizon estimation (anim_mhe.py, scene MHEWindow, about 12.5 s)

## What it teaches
A window of 0.5 s (10 intervals of 0.05 s) slides over noisy measurements of the pendulum angle. At each new
measurement the window's `TRACK_STATE` target is updated (`update_objectives_target`), the OCP is solved again (warm
started from the previous window) and the estimate is the first node of the window. Plotted: truth (dashed), noisy
measurements (red dots, they arrive one per window), the fit over the current window (orange), the estimate (green),
and, on its own axis, the error of each measurement and of the estimate with respect to the truth, plus running RMS
readouts computed from the data.

## Real content
`generate_mhe_data.py`:
- Truth: cart-pendulum (`examples/models/cart_pendulum.bioMod`) simulated with `solve_ivp` from theta = pi/2 under a
  known cart force 2 sin(2 pi t / 1.5). SYNTHETIC noise: Gaussian, sigma = 0.1 rad, seed 0 (stated on screen).
- Estimator: `MovingHorizonEstimator` (`get_all_iterations=True`), 40 real IPOPT solves, ALL status 0 (6-7 iterations).
  The estimator does not know the force: `MINIMIZE_CONTROL` (weight 1) regularises `tau` (bounded to +-5 N).
- Numbers quoted (computed from `data/mhe_results.npz`, t = 0 to 1.95 s, 40 samples): RMS measurement error 0.079 rad
  (sample RMS of the noise, below sigma = 0.1), RMS estimate error 0.027 rad; maximum |error| 0.045 rad (estimate) vs
  0.233 rad (measurement).
- Code lines on screen match the script (`meas`, `n_windows`, `N`, `dt`, `objectives` are the script's names
  simplified; the `x_bounds` / `u_bounds` definitions are not shown).

## Commands (from the repo root / docs/animations)
    PYTHONPATH=. python docs/animations/generate_mhe_data.py         # conda env captury_biobuddy
    cd docs/animations && manim render -qh anim_mhe.py MHEWindow

## Caveats
- The estimate shown is the FIRST node of each window: it uses 0.5 s of FUTURE measurements (fixed-lag smoothing), so
  it is delayed by one window length. The last node (filter-like, no future data) is less accurate: RMS 0.032 rad
  (printed by the generation script, not shown in the video).
- No arrival cost and no explicit noise model: the weights (1000 on the angle, 1 on the force) are hand-picked.
- Only theta is measured (cart position is estimated implicitly). Model is exact (same as the simulator): only the
  measurement noise is corrected, not a model mismatch.
- The RMS readouts over the first windows use very few samples (window 1: a single sample), so they are noisy.
- The dot at each window appears when its measurement "arrives" (last node of the window); the estimate at time t is
  drawn when the window starting at t is solved, i.e. 0.5 s of measurements later.
- Solve times are not reported and no real-time claim is made.

## Exercises
1. Increase the noise (sigma = 0.3) or shorten the window (N = 4): how do the RMS errors of estimate and measurement
   change?
2. Plot the last-node estimate (`pred_theta[:, -1]`) instead of the first node and compare its error and delay.
3. Add a model mismatch (e.g. wrong pendulum mass in the estimator) and see what the window can and cannot correct.
