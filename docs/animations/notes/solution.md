# The Solution object and its accessors (anim_solution.py, scene SolutionTour, about 14 s)

## What it teaches
One real solve (2-phase pendulum, multiple shooting with RK4, IPOPT status 0, 38 iterations), then the SAME solution read
through four accessors. Each marker on the time axis is one column of the array really returned, drawn at its real time
(`sol.decision_time` / `sol.stepwise_time` / evenly spaced), and the states are also dropped on the q_rot curve.

| accessor (code shown in the video)                                   | shape (from the npz)          |
|----------------------------------------------------------------------|-------------------------------|
| `sol.decision_states(to_merge=SolutionMerge.ALL)`                    | (4, 9) = 5 + 4 nodes          |
| `sol.stepwise_states(to_merge=SolutionMerge.ALL)`                    | (4, 27) = 17 + 10             |
| `sol.interpolate(100)["q"]`                                          | (2, 100)                      |
| `sol.integrate(to_merge=[SolutionMerge.KEYS, SolutionMerge.NODES])`  | [(4, 17), (4, 10)]            |

Also shown: the `SolutionMerge` options (KEYS stacks q and qdot, NODES concatenates nodes, PHASES concatenates phases,
ALL does the three), `sol.cost` (62.11), `sol.detailed_cost` (55.47 for phase 0, 6.64 for phase 1; sum checked), and
`sol.print_cost()`, `sol.graphs()`, `sol.animate()` (named only, not run).

Problem: `pendulum.bioMod`, phase 0 = 4 intervals over 0.6 s (RK4, 3 steps), phase 1 = 3 intervals over 0.4 s (RK4,
2 steps), q(0)=0, q_rot(end)=1 rad, only the sliding translation actuated, minimise squared tau (Lagrange).

## Commands (repo root, env captury_biobuddy; render from docs/animations)
    PYTHONPATH=. python docs/animations/generate_solution_data.py
    manim render -qh anim_solution.py SolutionTour

The generator prints every shape it stores; the scene asserts them against the video text.

## Honest caveats (verified in this version of the source)
- Without `to_merge`, `decision_states()` returns a list per phase of a dict of lists per node (arrays (nq, n_sub)); with
  one phase the outer list is dropped. `to_merge=SolutionMerge.ALL` returns a single array (keys stacked).
- `decision_states` = the decision variables (nodes only for RK4; for COLLOCATION it also holds the collocation points,
  so decision and stepwise then have the same shape). `stepwise_states` re-integrates RK4 substeps and repeats the
  interval end (each node's block ends on the next node's start), hence 17 = 4 x 4 + 1 rather than 13 unique times.
- Merging phases keeps both copies of t = 0.6 s (end of phase 0, start of phase 1): 9 = 5 + 4 columns.
- `interpolate(100)` uses piecewise-linear interpolation of the stepwise states after removing duplicated times.
- `integrate` here uses the default integrator (`SolutionIntegrator.OCP`, the RK4 of the OCP) and `Shooting.SINGLE`, so the
  gap to the stepwise states is round-off (3e-12). With `OdeSolver.COLLOCATION` phases `SolutionIntegrator.OCP` is refused
  (ValueError): a scipy integrator must be given. In an exploratory run with a collocation phase (degree 3, only 0.13 s
  intervals) the re-integrated states differed by up to 0.3 rad from the collocation states and the times returned for
  the collocation phase did not line up with the samples, so that variant was NOT used in the video.
- `duplicated_times=False` in `stepwise_time` looked unreliable in this version (dropped the last time of each phase); not used.
- Cost value is large (62) because the swing in 1 s with a passive rotation needs big torques; not the point of the scene.
- Parameters are not shown (this OCP has none: `sol.parameters` is `{}`).

## Exercises
1. Change `STEPS` to (6, 4) in the generator: which shapes change and which do not? Why does `decision_states` keep (4, 9)?
2. Replace RK4 by `OdeSolver.COLLOCATION(polynomial_degree=3)` and compare `decision_states` and `stepwise_states` shapes.
   Then call `sol.integrate(shooting_type=Shooting.MULTIPLE, integrator=SolutionIntegrator.SCIPY_RK45)` and measure the gap.
3. Use `to_merge=SolutionMerge.PHASES` and `to_merge=[SolutionMerge.KEYS, SolutionMerge.PHASES]`; predict the return type first.
