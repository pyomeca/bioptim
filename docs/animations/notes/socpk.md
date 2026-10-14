# Stochastic control: the feedback gain K (anim_socpk.py, scene `StochasticArmFeedback`)

## What the scene teaches
In a stochastic optimal control problem (SOCP, `StochasticOptimalControlProgram`) the optimiser does not only return the
mean motion: it also returns the state covariance P at every node and the feedback gain K that turns the noisy sensory
input into a torque correction, `tau = tau_nominal + K (y - ref + sensory_noise) + motor_noise`
(`bioptim/models/biorbd/stochastic_biorbd_model.py`, lines 18-38). Both P (`cov`, 16 values) and K (`k`, 8 values = 2
torques x 4 sensory inputs: hand x, y, vx, vy) are CONTROLS of the problem.

Setting (all from `data/socpk_results.npz`): Leuven two-link arm (`LeuvenArmModel.bioMod`), reach in T = 0.8 s,
N = 40 intervals, `SocpType.COLLOCATION(polynomial_degree=3, method="legendre")`, `StochasticTorqueBiorbdModel`, motor
noise std 0.05, sensory noise std 3e-4 m (hand position) and 2.4e-3 m/s (hand velocity), `sensory_reference` = hand
position and velocity. The problem is the unchanged `prepare_socp` of
`arm_reaching_torque_driven_collocations.py`.

Beat 1 (low sensory noise = the example's values): mean hand path (yellow) with the 2-standard-deviation covariance
ellipses every 4th node (hand covariance = J P_qq J', J = Jacobian of marker 2), and the largest hand standard
deviation over time. Printed by the generator: peak 11.3 mm (mid reach); at the target the standard deviation is
4.00 mm along x and along y (the bound of the example, 0.004 m on each, is active) and 4.1 mm along the major axis of
the ellipse, which is what the plot shows; IPOPT status 0 in 52 iterations. The mean path of the deterministic solve (dashed, plain OCP with
`TorqueBiorbdModel`, same cost and constraints, 3 iterations, objective 51.643) is 0.02 mm away from the stochastic mean
path at most: the noise changes the covariance and the cost, not the mean path.

Beat 2: K as a heat map (rows: shoulder torque x [x, y, vx, vy] then elbow torque x [x, y, vx, vy]; columns: nodes)
for sensory noise x1 and x3 (2 real solves), and their difference (own colour scale, max |difference| 13.1 while the
maps saturate at 34.6). Numbers on screen: mean |K| 4.93 -> 5.77, cost 52.3 -> 58.5, peak hand std 11.3 -> 11.7 mm,
IPOPT 52 then 155 iterations (both status 0).

## Commands (from the repo root, conda env `captury_biobuddy` on the PATH, see STANDARD.md section 6)
```
PYTHONPATH=. python docs/animations/generate_socpk_data.py        # about 15 min: 5 solves, writes data/socpk_results.npz
python docs/animations/render_series.py anim_socpk.py StochasticArmFeedback --lang both --strict
```
Optional arguments of the generator: `N` (40) and the high sensory-noise multiplier (3).

## Honest caveats
- **The example does not fix the initial covariance.** In `arm_reaching_torque_driven_collocations.py` the control `cov`
  is bounded by +-inf and `initial_cov` only feeds the automatic initial guess (`auto_initialization`), which the example
  does not use. Run as is, IPOPT converges (36 iterations) to a solution whose P has negative variances at node 0, the
  stochastic constraints are then vacuous, the objective (51.643) equals the deterministic one and K is meaningless. The
  generator therefore wraps `StochasticOptimalControlProgram` to add `ConstraintFcn.TRACK_CONTROL` on `cov` at
  `Node.START` with the example's own `initial_cov = diag(1e-4, 1e-4, 1e-7, 1e-7)`; everything else is the example.
  This constraint is not in the code panel (see the footer of the scene) but is in the generator.
- Code panel: `sens = ...` and `sensory = cas.vertcat(pos, vel)` simplify the argument and the return value of `noises(sens)` in
  the generator; the bio_model call is the example's `prepare_socp` with its remaining arguments replaced by `...`;
  the last line of beat 2 is bioptim's `_compute_torques_from_noise_and_feedback_default`.
- Reduced size: N = 40 (dt = 0.02 s) instead of the example's N = 80 (dt = 0.01 s); noise magnitudes are computed as
  std^2 / dt with the smaller N so the noise per second is unchanged. Solver settings are the example's, with a limit of
  2000 iterations.
- No warm start; the deterministic solve is NOT used to initialise the stochastic one.
- Local minima: not explored (single start). The solutions are IPOPT local optima of a non-convex problem.
- **High sensory noise is x3 only.** Multipliers x5 and x10 ended with IPOPT status 1 (restoration failed) after 320 and
  415 iterations and are not shown (only their status is stored in the npz). Their failure is probably due to the fixed
  4 mm bound at the target becoming (nearly) infeasible with that much sensory noise; this was not investigated.
- Both solutions end with a hand standard deviation of 4.00 mm (x and y) at the target because that bound is active: the effect of
  sensory noise appears in K, in the cost and slightly in the peak std, not in the final ellipse.
- The last column of K (node N) is not used by the dynamics (control_type CONSTANT_WITH_LAST_NODE): it keeps its initial guess 0.01, hence the dark last column.
- The hand covariance only uses the position block of P (P_qq) through the Jacobian; the velocity block is not shown.
- The covariance is the linearised propagation of Gillis 2013 as implemented in bioptim, not a Monte Carlo check.
- Solve times on this machine: 46 s (x1) and 113 s (x3), MUMPS.

## Exercises
1. Change the motor noise std (`MOTOR_STD = 0.05`) to 0.1: does the problem stay feasible, and how does the peak hand std
   change?
2. Loosen the bound of the example (`max_bound` 0.004**2 on the hand position at `Node.END`) to 0.008**2 and compare K for
   the x1 and x3 sensory noises: is the end ellipse still equal in both solutions?
3. Replace `SocpType.COLLOCATION(polynomial_degree=3)` by a degree 4 collocation and compare the iteration count and the
   cost with the values of this note.
