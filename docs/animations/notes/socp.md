# Stochastic optimal control: robust path constraint (anim_socp.py, scene `RobustPath`, 13.3 s)

## What it teaches
A stochastic OCP (`StochasticOptimalControlProgram`, `SocpType.COLLOCATION`) optimises the mean trajectory AND the
state covariance P at every node (`cov` is a control, `M` an algebraic state, both created by the stochastic model).
The covariance is propagated by `ConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION`
(P_k+1 = M (dg/dx P dg/dx' + dg/dw sigma_w dg/dw') M', Gillis 2013) and made periodic by
`PhaseTransitionFcn.COVARIANCE_CYCLIC`. The path constraint h(q) >= 0 of the example
(`obstacle_avoidance_direct_collocation.py`, two super-ellipse obstacles, time-optimal periodic loop of a mass point
attached to a guide point by a spring) can be applied to the mean only (`is_robustified=False`) or robustified
(`is_robustified=True`): `out -= gamma * sqrt(dh_dx @ cov @ dh_dx.T)`, gamma = 1.

Left: the two real mean trajectories with the 1-sigma position-covariance ellipse taken from the optimised `cov` at each
node (red = the tube overlaps an obstacle, i.e. z < 1). Right: z = h / sqrt(dh P dh') along the loop (computed from
the npz, min over the two obstacles, capped at 3 for display). Mean-only: min z = 0.00 (the mean touches the obstacle,
26 of 41 nodes have z < 1). Robustified: min z = 1.00 (constraint active, by construction) and the loop takes
1.319 s -> 1.372 s (+4.0 %).

## Commands (from the repo root, conda env captury_biobuddy)
```
PYTHONPATH=. python docs/animations/generate_socp_data.py 40 100      # ~7 min, writes data/socp_results.npz
cd docs/animations && manim render -qh anim_socp.py RobustPath
```
Arguments: n_shooting (40) and motor noise magnitude (100). Three real IPOPT solves are stored: `det` (plain OCP, not
shown), `nonrobust` (SOCP, mean-only constraint), `robust` (SOCP, robustified). Data are stored under the keys
`<case>_q`, `_qdot`, `_u`, `_cov` (16 x 41, column-major 4x4), `_tf`, `_status`, `_iterations`, `_seconds`.
The generator imports `prepare_socp` from the bioptim example (unchanged), so the model and constraints are the example's.

## IPOPT status (honest)
All three solves: status 0 (Optimal Solution Found). Nonrobust 94 iterations, robust 97 iterations, deterministic 84;
roughly 100-170 s each on this machine with n_shooting = 40, polynomial degree 5 (Legendre), use_sx=True.

## Caveats
- Noise level: the example's default motor noise magnitude is 1, which gives a position standard deviation of only
  ~0.013 m, invisible next to the obstacles and with almost identical trajectories (tf 1.3288 vs 1.3331 s). I raised it
  to 100 (std ~0.13 m, 0.124-0.129 along the whole loop) so the effect is visible. Bioptim uses `diag(magnitude)` as the
  noise covariance sigma_w in the collocation covariance constraint (i.e. a variance), whereas the example's plotting
  function treats it as a standard deviation: do not compare noise values across those two conventions.
- The covariance is almost constant along the loop (stable spring + drag dynamics, no feedback). The ellipses therefore
  do not shrink or grow visibly; what changes with the robustification is the position of the mean, not the size of P.
  There is no feedback gain K and no sensory noise in this example (`sensory_noise_magnitude` is empty); K and sensory
  noise appear in the arm-reaching examples, which are far slower and were not attempted here.
- "Mean only" is the same SOCP with `is_robustified=False`, not the plain OCP, so that the covariance comes from the same
  machinery. The plain OCP (stored as `det_*`) reaches tf = 1.3288 s at every noise level, while the mean-only SOCP
  reaches 1.3194 s at noise 25 and 100 (it lands in a slightly different local minimum, mean trajectories differ by up
  to 0.24 m) and 1.3288 s at noise 1. Both are valid IPOPT solutions of nonconvex problems.
- The 1-sigma margin is a linearised (first-order) criterion, not an exact probability; for a Gaussian and a single
  linearised constraint it corresponds to about 84 % one-sided satisfaction at each node. The scene does not run a Monte
  Carlo check (the bioptim example does, with `sol.noisy_integrate`).
- The time-axis of the clearance plot uses each solution's own loop time (t = linspace(0, tf, 41)).

## Exercises
1. Change gamma in `path_constraint` from 1 to 2 (or 0.5) and re-solve: how do the loop time and the minimum z change?
2. Set the motor noise magnitude to 1, 25 and 100 and plot (loop time robust - loop time mean-only) against noise.
3. Replace `SocpType.COLLOCATION(...)` by `SocpType.TRAPEZOIDAL(...)` (see the arm-reaching examples for the needed model
   changes) and compare accuracy and solve time.
