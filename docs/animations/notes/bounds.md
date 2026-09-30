# Bounds and initial guess (anim_bounds.py)

## What the scene teaches
Bounds and initial guesses are given to Bioptim as small arrays whose number of columns depends on the
`InterpolationType`; the initial guess changes the IPOPT run (iterations, and which local minimum is reached), not the
problem. Setting: pendulum swing-up (`pendulum.bioMod`, N = 20, T = 1 s, RK4 with 5 steps, `MINIMIZE_CONTROL` on `tau`,
rotation from 0 to 3.14 rad, |tau| <= 100 on the translation, torque on the rotation fixed to 0).

Beat 1 (real `InitialGuess` / `Bounds` objects, no solve): the guess of the rotation `q_rot` with four types, array shape
passed by the user and value at each of the 21 nodes obtained with `evaluate_at`:

| InterpolationType | array passed | shape |
|---|---|---|
| `CONSTANT` | `[[0.0]]` | (1, 1) |
| `CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT` | `[[0, 1.57, 3.14]]` | (1, 3) |
| `LINEAR` | `[[0, 3.14]]` | (1, 2) |
| `EACH_FRAME` | one value per node (the coarse solution below) | (1, 21) |

The bounds are the real `Bounds` of `x_bounds["q"]` (default interpolation of `Bounds` is
`CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT`): array shape (2, 3) for the two rows of `q`, pinned to 0 at the first node and to
3.14 (rotation) at the last node, +-6.28 rad (the ranges of the model) in between.

Beat 2 (three real IPOPT solves, all status 0), values printed by `generate_bounds_data.py`:

| initial guess | interpolation | iterations | cost | peak abs(tau) |
|---|---|---|---|---|
| zeros | `CONSTANT` | 67 | 39.93 | 21.2 N |
| straight line start -> end pose (velocities and torques 0) | `LINEAR` | 33 | 112.87 | 39.1 N |
| coarse solve (5 intervals) interpolated on the 21 nodes | `EACH_FRAME` | 13 | 51.28 | 22.0 N |

The coarse solve (N = 5, from zeros) took 115 iterations, cost 42.14, status 0.

## Commands
Environment: conda env `captury_biobuddy` (see STANDARD.md, section 6). From the repository root:
`PYTHONPATH=. python docs/animations/generate_bounds_data.py` (writes `data/bounds_guess.npz`), then
`python docs/animations/render_series.py anim_bounds.py BoundsInitialGuess --lang both --strict`
(catalog entry in `catalog_part_bounds.json`, French in `i18n/fr_bounds.json`).

## Honest caveats
- Warm start: the third guess comes from a first solve of a coarse problem (N = 5). Its 115 iterations are NOT counted in
  the 13 shown; counting them the total is larger than the run from zeros. The point is that a good guess makes the
  final solve short, not that it is free.
- Local minima: the three runs end in three different local minima (costs 39.93, 112.87, 51.28). The zeros run has the
  lowest cost here but nothing proves it is the global minimum; a lower cost was not searched for. The "good" guess is
  good for the number of iterations, not for the cost. Only one problem and one discretization were solved.
- The three costs are IPOPT costs of the same objective (sum of tau squared integrated with RK4, weight 1).
- The guess is applied to `q`, `qdot` and `tau` with the same `InterpolationType` (code shown for `q` only, a comment
  says so); the line guess has zero velocities and torques, the coarse guess is piecewise linear (states) and piecewise
  constant (torque) between the 5 coarse intervals.
- Beat 1 shows one guess curve per type but no ghost/difference axis: the four types are alternative ways to write a guess,
  not a perturbation of one reference. The `SPLINE` and `CUSTOM` types exist and are not shown. In this version
  `SPLINE` is evaluated with `scipy.interpolate.interp1d(self.t, self)` whose default `kind` is linear
  (`path_conditions.py`, `evaluate_at`), although the enum comment says "cubic spline": that is why it was left out.
  `ALL_POINTS` acts as `EACH_FRAME` for multiple shooting (only different for direct collocation).
- Wide versus tight bounds were not compared (not shown, not computed).
- The code shown uses the alias `IT = InterpolationType` (also defined in the generator) and `guess` for `x[:2]`, to keep
  lines short; the linked example (`custom_initial_guess.py`) uses the full names.

## Exercises
1. In `generate_bounds_data.py`, replace `IT.LINEAR` of the second run by `IT.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT` with a
   `(4, 3)` array (start pose, middle pose, end pose). Do the iterations and the cost change?
2. Change `N_COARSE` from 5 to 10: how many iterations does the coarse solve need, and how many the final one? Is it worth it?
3. Tighten the bounds of the rotation (for example `x_bounds["q"][1, 1:-1] = -0.5, 3.64`) and solve again from zeros: do
   the iterations, the cost or the local minimum change?
