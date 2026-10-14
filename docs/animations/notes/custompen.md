# Custom objective and custom constraint (anim_custompen.py)

## What the scene teaches

A user function `f(controller, **extra)` that returns a casadi expression can replace an `ObjectiveFcn` (then
`custom_type=ObjectiveFcn.Lagrange` or `ObjectiveFcn.Mayer` is required) or a `ConstraintFcn` (nothing else is needed).
Bioptim wraps it itself in `ObjectiveFcn.Lagrange.CUSTOM` / `ObjectiveFcn.Mayer.CUSTOM` / `ConstraintFcn.CUSTOM`
(`bioptim/limits/objective_functions.py` lines 54-64, `bioptim/limits/constraints.py` lines 72-75). Inside the function the
`PenaltyController` gives the model (`controller.model`), the states, controls and parameters *at the current node*
(`controller.states["q"].cx`, `controller.controls["tau"].cx`, `controller.parameters.cx`); extra keyword arguments of
`add(...)` (here `marker="tip"`) are forwarded to the function.

Setting (real solves, `generate_custompen_data.py`, `data/custompen_pendulum.npz`): one-link pendulum
(`models/custompen_pendulum.bioMod`, 1 kg point mass at the end of a 1 m rod, one torque at the pivot), swing from hanging
(q = 0) to upright (q = pi), at rest at both ends, N = 30 intervals, T = 2 s, RK4 with 5 steps per interval, torque
bounds +-40 N.m, `TorqueBiorbdModel`, IPOPT. Three independent solves with the same linear initial guess:

* plain: `MINIMIZE_CONTROL` on `tau`, weight 0.01 (IPOPT status 0, 17 iterations, peak |tau| = 10.6 N.m, mean tip height
  0.20 m, peak |power| = 30.95 W);
* custom objective: plain + `tip_height` (custom Lagrange objective, weight 1.0, `quadratic=False`; cost = integral of
  the tip height, read with `controller.model.markers()`): status 0, 10 iterations, peak |tau| = 13.1 N.m, mean tip
  height -0.37 m;
* custom constraint: plain + `power` = `tau * qdot` (read from `controller.controls` and `controller.states`) kept
  between -p_max and +p_max at `Node.ALL_SHOOTING`, p_max = 0.6 x the peak |power| of the plain solve = 18.57 W: status 0,
  21 iterations, peak |power| = 18.57 W, the limit is reached at 11 of the 30 shooting nodes (tolerance 1e-3 W), the plain
  solve is above it at 7 nodes.

All numbers on screen (mean heights, peaks, p_max, margin, node counts) are computed in the scene from the npz.
The "margin" is p_max minus the peak of the constrained solve: it is about 2.5e-5 W (IPOPT keeps the iterate strictly
inside its bound), printed as 0.00 W.

## Commands

```bash
E=/c/Users/<you>/miniconda3/envs/captury_biobuddy
export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
PYTHONIOENCODING=utf-8 PYTHONPATH=. python docs/animations/generate_custompen_data.py
python docs/animations/render_series.py anim_custompen.py CustomPenalty --lang both --strict
```

## Honest caveats

* Simplified model: a single rigid pendulum with a point mass and a torque at the pivot, fixed duration, fixed N = 30.
* Three independent solves, no warm start, no continuation. The problem is non-convex: each solve returns *a* local
  minimum (nothing proves it is global). The custom-objective solution first rises a little, falls back and then swings
  up: this pumping motion is a feature of that local solution, not a general result.
* The custom objective is linear in the tip height (`quadratic=False`) and its weight (1.0) is large compared with the
  torque weight (0.01): it was chosen to make the effect visible, not tuned.
* `p_max` is derived from the plain solve (60 %), so the constraint is guaranteed to be active; the constrained solve
  reaches the same final state, so it is a slower, flatter power profile, not a cheaper motion (its cost is higher).
* The constraint is enforced at the shooting nodes only (`Node.ALL_SHOOTING`), not between them.
* Mechanical power is signed: only |P| <= p_max is constrained, the negative side is never active here.
* The code panels rewrap the calls of the generator on two lines and omit the imports and the bounds.
* The scene shows only the last solve of each beat; the IPOPT line refers to that solve (all three converged with status 0).

## Exercises

1. In `prepare_ocp`, change `FRACTION` from 0.6 to 0.4 (then 0.9). What happens to the number of active nodes and to the
   iteration count? At which fraction does the problem become infeasible for T = 2 s?
2. Make the objective quadratic: `quadratic=True` in the `objectives.add(tip_height, ...)` call and return
   `tip[2, ...] + 1` (height above the lowest point). How does the shape of the tip height change?
3. Change the constraint bounds to `min_bound=0, max_bound=p_max` (no negative power, i.e. no braking by the motor). Is the
   problem still feasible for a swing-up that must stop at q = pi? Look at the sign of the power in the last nodes of the
   plain solve.
