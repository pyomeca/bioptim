# Accuracy check: re-integrating an optimal solution (`AccuracyCheck`, ~15 s)

## What it teaches
An OCP solution satisfies the dynamics only through its transcription (RK4 steps or collocation defects). Calling
`sol.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.SCIPY_DOP853)` re-integrates the optimal
controls from the initial state without ever resetting the state (Shooting.SINGLE: reset only at each phase). The gap
between the re-integrated and the optimised trajectory is the drift. Four real solves of the same pendulum swing-up
(N = 30, T = 1 s): `RK4(n_integration_steps=1)`, `RK4(n_integration_steps=5)`, `COLLOCATION(polynomial_degree=3)` and
`COLLOCATION(polynomial_degree=5)`. Final bar plot: max |dtheta| (log scale) with the IPOPT solve time as cost.

Measured (max |dtheta| over the nodes, rad / solve time): RK4 1 step 1.27 / 0.19 s; RK4 5 steps 8.6e-4 / 0.86 s;
collocation deg 3 1.00 / 0.26 s; collocation deg 5 0.26 / 0.53 s (all IPOPT status 0).

## Generate and render
```bash
# repo root, conda env captury_biobuddy on the PATH
PYTHONPATH=. python docs/animations/generate_accuracy_data.py        # writes data/accuracy_pendulum.npz
# from docs/animations
manim render -qh anim_accuracy.py AccuracyCheck                      # 1080p60
```

## Caveats (honest)
- This bioptim version has no `Shooting.SINGLE_CONTINUOUS`; `Shooting.SINGLE` is the "never reset" mode
  (see `Shooting` in `bioptim/misc/enums.py`). `integrator` must not be `SolutionIntegrator.OCP` for collocation.
- `sol.integrate` calls scipy with its default tolerances (rtol = 1e-3). The bar values use an independent
  reference (`tight_reference` in the generator: same dynamics function, same constant controls, DOP853 with
  rtol = 1e-10). The red curve in the animation is exactly what `sol.integrate` returns; the two differ by
  <= 2.4e-2 on the state norm at the last node, invisible at the plotted scale but not negligible for RK4 5 steps
  (8.6e-4 rad with the tight reference).
- With `SolutionIntegrator.OCP` the RK4 solutions re-integrate to their own optimum within 1e-10 (stored as
  `*_final_state_error_ocp`): "consistent with its own scheme" is not "consistent with the real dynamics".
- For COLLOCATION the intermediate points returned by `integrate` carry time labels that do not match the values (uniform
  sampling vs collocation times), so only the values at shooting nodes are used, and the curves are drawn through nodes.
- Cost is solve time on one laptop, a single run (noisy: not a benchmark). Errors depend on this problem (large
  torque, fast swing-up); the pendulum amplifies small defects over the trajectory. N = 10 versions were tried first
  and drift by several rad for every scheme, so N = 30 was chosen for the comparison.

## Exercises
1. Add `RK4(n_integration_steps=2, 3, 10)` and plot the drift versus solve time; what is the slope on a log-log plot
   (expected about 4 for RK4 at fixed dt)?
2. Re-run with `method="radau"` for the collocation; does the ranking of degrees 3 and 5 change?
3. Use `Shooting.MULTIPLE` instead of `Shooting.SINGLE`: what does the error measure now, and why is it much smaller?
