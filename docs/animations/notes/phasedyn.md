# PhaseDynamics: SHARED_DURING_THE_PHASE vs ONE_PER_NODE (`PhaseDynamicsScene`, ~16 s)

## What it teaches
`DynamicsOptions(phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE | ONE_PER_NODE)` (enum in `bioptim/misc/enums.py`,
default SHARED). Verified in `bioptim/dynamics/ode_solver_base.py`: SHARED builds one integrator and does
`dynamics = dynamics * nlp.ns` (same Function object at every node); ONE_PER_NODE calls `initialize_integrator(...,
node_index=node_index)` for every node.
1. Diagram: 1 vs 30 distinct integrator `casadi.Function` objects (`len({id(f) for f in nlp.dynamics})`, measured).
2. Measured build / IPOPT time for N = 30, 60, 120 (median of 3) and identical optimum.
3. Where ONE_PER_NODE is required: a multinode constraint on 7 nodes of one phase.

## Data (real; `generate_phasedyn_data.py`, `data/phasedyn_{bench,hold,series}.npz`)
Problem: `TimeDependentModel` of `example_pendulum_time_dependent.py` (torque scaled by sin(t)), RK4, SX, T = 1 s,
minimise tau, swing to 3.14 rad with only the cart actuated.
- Distinct integrators: 1 (SHARED) vs N (ONE_PER_NODE) for N = 30, 60, 120.
- The constraint vector g has the SAME number of SX nodes in both options (49552 / 99089 / 198161): the NLP is identical,
  only the Python-side model construction differs. Same iterations (50 / 77 / 94) and same cost (2.3477 / 2.0973 /
  2.0641); q identical to 0 (max |dq| = 0).
- Timings (median of 3, s): build 0.6 / 1.0 / 1.6 (SHARED) vs 1.0 / 2.0 / 6.8 (ONE_PER_NODE); solve 2.2 / 4.8 / 9.2 vs
  2.2 / 4.6 / 13.2.
- Required case: `MultinodeConstraintFcn.CUSTOM` (`hold_rotation`: rotation equal to that of the first node) on nodes
  24 to 30 of one phase. SHARED raises at OCP construction
  `ValueError: Valid values for setting the cx is 0, 1 or 2 ... more penalties than available in a multinode constraint ...
  use phase_dynamics=PhaseDynamics.ONE_PER_NODE`. ONE_PER_NODE: IPOPT status 0, 172 iterations, cost 10.28 (free: 2.35),
  rotation = 3.14 on nodes 24 to 30.
- `numerical_data_timeseries`: `example_external_forces.py` (cube, per-node external forces) solves with BOTH options and
  gives the same cost (7067.85, max |dq| = 0), and the time-dependent pendulum also works with SHARED. So in this
  version a node-dependent numerical time series or time-dependent dynamics does NOT require ONE_PER_NODE (the numerical
  series are inputs of the function, filled per node). The README sentence "different external force at each node" is
  therefore not a hard requirement here.

## Generate and render
```bash
# repo root, conda env captury_biobuddy on the PATH (about 10 min: the benchmark solves 18 OCPs)
PYTHONPATH=. python docs/animations/generate_phasedyn_data.py
# from docs/animations
manim render -qh anim_phasedyn.py PhaseDynamicsScene
```

## Caveats (honest)
- Timings were taken on a machine busy with other jobs: the spread is large (e.g. ONE_PER_NODE build N = 120: 6.9, 4.2,
  9.0 s; first build of a run carries warm-up). Read the ratios (build 1.6x to 4x), not the absolute seconds. The
  solve-time gap at N = 120 (9.2 vs 13.2 s) was not seen at N <= 60 and may partly be noise, although the runs are in the
  same order. Identical NLP graphs suggest no real difference in the solver, only in construction.
- A first attempt used `STATES_EQUALITY` on 4 nodes: it is implemented as the SUM of differences to the first node (one
  weak equation), not pairwise equality, hence the custom `hold_rotation`. Holding for nodes 26 to 30, or 22 to 30, ended
  with IPOPT status 1 (poor problems); 24 to 30 converges.
- ONE_PER_NODE is also needed by collocation with several multinode penalties, and stochastic OCPs force it; ACADOS
  requires SHARED (`acados_interface.py`). Not shown.

## Exercises
1. Change `HOLD_NODES` to only 3 nodes (25, 27, 29): does SHARED still work? (It should: the limit is 3 nodes per phase.)
2. Measure `nlp.dynamics_func` graph size vs N with `use_sx=False` (MX) for both options.
3. Add `n_threads=2` with ONE_PER_NODE (see `configure_new_variable.py`): what error do you get?
