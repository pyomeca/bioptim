# Variable scaling (anim_scaling.py)

## What the scene teaches
`VariableScalingList` (`x_scaling`, `u_scaling` of `OptimalControlProgram`) changes only what the NLP solver sees:
the decision variable is `x / scale`, bounds and objectives are still written in physical units. The scene shows
(1) the largest |value| of q, qdot, tau in the solution, physical vs divided by the scaling factors (3.14 / 76.4 / 1000
become 1.05 / 0.90 / 1.11), and (2) the real IPOPT primal infeasibility (`inf_pr`) history of the same OCP without and
with scaling: 374 vs 65 iterations, same optimum.

Problem: `bioptim/examples/toy_examples/feature_examples/example_variable_scaling.py` (pendulum, n_shooting=30,
final_time=0.1 s, RK4, tau in [-1000, 1000], qdot in [-314, 314]). Scaling values are the example's own:
q=[1, 3], qdot=[85, 85], tau=[900, 1]. The "unscaled" run is the same `prepare_ocp` with every factor forced to 1.

## Commands
    # from the repo root, in the conda env with bioptim
    PYTHONPATH=. python docs/animations/generate_scaling_data.py      # -> data/scaling_pendulum.npz (~10 s)
    # from docs/animations
    manim render -qh anim_scaling.py Scaling                         # ~14 s video

## Real vs simplified
- All numbers come from `data/scaling_pendulum.npz` (iteration history parsed from IPOPT's own output file, cost,
  trajectories). Both runs: IPOPT status 0 (Optimal Solution Found). Costs 31609.834083 vs 31609.834068; max
  |tau_unscaled - tau_scaled| = 3e-5 N, so the same solution.
- The bar chart uses the largest |value| over the solution of the scaled run divided by the factors; this is a
  picture of the magnitudes, not a dump of the internal solver vector.
- The u_bounds tau <= 1000 is active (tau saturates), so the problem is a bang-like torque profile.
- IPOPT's own `nlp_scaling_method` (default "gradient-based") was left on. Re-running with
  `solver.set_nlp_scaling_method("none")` gave the same iteration counts (374 / 65) in `generate_scaling_data.py`;
  I did not verify that IPOPT actually receives the option through bioptim's option passing, so do not read this as a
  claim about IPOPT's internal scaling.
- Single problem, single run per case (deterministic solver, so reproducible), not a statistical benchmark. On
  the same pendulum with final_time=1.0 the effect was not favourable to scaling (72 vs 133 iterations and different
  costs 41.58 / 38.25, i.e. different local minima), so scaling is not a guaranteed speed-up; it is a conditioning aid.
- The inf_pr curve plateau near 1 with spikes in the unscaled run are IPOPT restoration phases (iterations marked "r"
  in its log).

## Exercises
1. Set every factor of one variable to a wrong value (e.g. tau=[1, 1] or qdot=[1000, 1000]) and compare iterations.
2. Change `final_time` to 1.0 and to 0.5: does scaling still help, and are the costs the same?
3. Scale the parameters too (see `example_parameter_scaling.py`) and add a parameter of order 1e3.
