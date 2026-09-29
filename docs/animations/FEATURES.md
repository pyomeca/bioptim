# Bioptim feature animations (part 2)

Six short Manim (Community Edition) scenes on bioptim features, on the same pendulum swing-up as `dms_vs_dc.py`
(`bioptim/examples/models/pendulum.bioMod`: `q = (y, theta)`, only the sideways force on `y` is actuated). The last
scene (`Impact`) uses a small dedicated model, `models/point_floor.bioMod` (a point mass with one rigid contact).
Every curve is a REAL bioptim + IPOPT result stored in `data/features_*.npz`; the code shown beside the curves is the
code used in `generate_features_data.py`.

| Scene | Length | Data file | What it teaches |
|-------|--------|-----------|-----------------|
| `ObjectivesNodes` | ~35 s | `features_objectives.npz` | Lagrange vs Mayer objectives, the `Node` enum, the effect of a weight |
| `ConstraintsBounds` | ~31 s | `features_constraints.npz` | `u_bounds` (control bounds) shrinking, `x_bounds` on a state |
| `MultiphaseTransitions` | ~17 s | `features_multiphase.npz` | several phases with different durations, `PhaseTransitionFcn` |
| `FreeTime` | ~19 s | `features_time.npz` | `ObjectiveFcn.Mayer.MINIMIZE_TIME`: the phase duration is optimized |
| `Parameters` | ~35 s | `features_parameters.npz` | `ParameterList`: a scalar (peak torque) optimized with the trajectory, `parameter_bounds/init/objectives`, layout of the decision vector |
| `Impact` | ~24 s | `features_impact.npz` | `PhaseTransitionFcn.IMPACT` on a point mass hitting the floor (velocity jump), and why `CONTINUOUS` fails there |

## Regenerate the data

The environment needs bioptim, biorbd, casadi and IPOPT on the `PATH` (on Windows with conda, the `Library/bin`
folder of the environment). From the repository root:

```bash
PYTHONPATH=. python docs/animations/generate_features_data.py
```

It runs 5 + 5 + 2 + 3 + 4 + 2 = 21 OCPs (RK4, 20-40 shooting nodes), about 15 s in total, and writes six small
`.npz` files (8-16 kB each). IPOPT is deterministic, so the same bioptim version gives the same numbers. To
regenerate only some experiments, name them: `python docs/animations/generate_features_data.py parameters impact`
(names: `objectives constraints multiphase time parameters impact`).

## Render

Manim (0.18+ / 0.21) is needed, with the fonts Segoe UI and Consolas on Windows (DejaVu elsewhere). No LaTeX.
From `docs/animations`:

```bash
manim -qh features_scenes.py ObjectivesNodes ConstraintsBounds MultiphaseTransitions FreeTime Parameters Impact  # 1080p60
manim -ql features_scenes.py FreeTime                                                              # quick 480p15 preview
manim -pqm features_scenes.py ConstraintsBounds                                                    # 720p30 + open
```

## Scene 1 - `ObjectivesNodes`

* Lagrange (`ObjectiveFcn.Lagrange.*`) is an integral over the N intervals: allowed nodes are `Node.ALL_SHOOTING`
  (default) and `Node.ALL`. Mayer (`ObjectiveFcn.Mayer.*`) is evaluated at one node: `Node.END` by default.
* The node grid shows `Node.START`, `INTERMEDIATES`, `PENULTIMATE`, `END`, `ALL_SHOOTING`, `ALL`.
  Careful: in the source (`penalty_option.py`) `Node.INTERMEDIATES` is `range(1, ns - 1)`, i.e. nodes 1 to N-2
  (the penultimate node N-1 is excluded although the enum comment says "all but first and last"). The grid is
  schematic (N = 10); the real solves use N = 30.
* Five real solves with `Lagrange MINIMIZE_CONTROL (weight 1)` plus `Mayer TRACK_STATE` on the pendulum angle at
  `Node.END` (target 3.14) with the Mayer weight 0, 1, 30, 300, 10000. Final angle: -0.13, 0.73, 2.03, 3.00,
  3.14 rad. The end angle is free here (no bound), so the weight is what pulls the pendulum up.

## Scene 2 - `ConstraintsBounds`

* Same swing-up, fixed T = 1 s, minimum `MINIMIZE_CONTROL`, end state fixed by `x_bounds`. `u_bounds["tau"]` shrinks
  100 -> 20 -> 15 -> 12 N; the shaded band is the forbidden region. The 100 N bound is inactive (unconstrained peak
  24 N), 20/15/12 N are active. Costs: 40.3, 41.7, 48.6, 76.6.
* Second beat: a bound on the sideways position, `x_bounds["q"].min[0, 1:] = -0.4` and `.max[0, 1:] = +0.4`
  (cost 40 -> 686, peak torque 69 N).
* Bounds act on decision variables. General path constraints are `ConstraintFcn` terms (for example
  `ConstraintFcn.TRACK_STATE` with `min_bound`/`max_bound`); this is mentioned on screen but NOT solved in the
  animation (see the exercises).

## Scene 3 - `MultiphaseTransitions`

* Two phases, `n_shooting = (12, 18)`, `phase_time = (0.5, 1.0)`, drawn as coloured time segments. Phase 0 ends at
  1.57 rad (90 degrees) with free velocity, phase 1 ends upright at rest.
* `PhaseTransitionFcn.CONTINUOUS` (the default link between phases): the states at the end of phase 0 and the start
  of phase 1 are equal (omega = 0.49 rad/s on both sides). Then `PhaseTransitionFcn.DISCONTINUOUS` with phase 1
  starting at rest: nothing links the phases, so omega jumps 3.58 -> 0 and the sideways position jumps -1.00 -> 1.09 m
  (there is no cost on the jump: this is only meant to make the transition visible, not a physical model).
* `PhaseTransitionFcn.IMPACT` (rigid contact) is only mentioned here: it needs a contact model, which the pendulum does
  not have. It is demonstrated with real solves in scene 6 (`Impact`).

## Scene 4 - `FreeTime`

* `Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.1, max_bound=4.0)`: `phase_time=1.0` is only the initial
  guess. With `|tau| <= 100, 80, 60 N` the optimal durations are 0.328, 0.463, 0.601 s (N = 40).
* The torque sits on its bound most of the time (bang-bang like) with some chattering between nodes.

## Scene 5 - `Parameters`

* A parameter is a decision variable that does not depend on time (or on the node). Here the peak torque `max_tau`:
  `parameters.add("max_tau", no_model_change, size=1, ...)` (the function is a no-op: the parameter changes no model
  property, it only appears in a constraint and in an objective), `parameter_bounds` (`InterpolationType.CONSTANT`, 0 to
  100 N), `parameter_init` (50 N) and `parameter_objectives` (`ObjectiveFcn.Parameter.MINIMIZE_PARAMETER`, quadratic,
  the weight is what changes). Two custom constraints (`max_tau - tau >= 0`, `max_tau + tau >= 0`, `Node.ALL_SHOOTING`)
  link the parameter to the 30 controls; this is the same min-max idea as
  `toy_examples/torque_driven_ocp/minimize_maximum_torque_by_extra_parameter.py`.
* Decision vector picture (sizes read from the real OCP, `ocp.vector_layout.total_size = 186`):
  `[dt (1) | X = 31 nodes x 4 | U = 30 nodes x 2 | parameters (1)]`. With the default `OrderingStrategy.VARIABLE_MAJOR`
  the order is dt, states, controls, algebraic states, parameters (see `optimization/vector_layout.py`); `dt` is itself
  one scalar per phase, pinned by its bounds when the phase duration is fixed (free with `MINIMIZE_TIME`).
* Four real solves (pendulum swing-up, T = 1 s, N = 30, minimum `MINIMIZE_CONTROL`) with weight 0.001, 0.01, 0.03,
  0.1: `max_tau* = 23.74, 21.55, 17.93, 14.92 N`; `int tau^2 dt = 40.3, 40.8, 43.5, 48.1`; total cost 40.8, 45.5, 53.2,
  70.4. `max_tau*` always equals the peak of `|tau|` (the bound is active): the parameter is a horizontal band.
* The model is not modified by the parameter here. Parameters that modify the model (gravity, mass: see
  `getting_started/custom_parameters.py`) use the same declaration; the function given to `parameters.add` then applies
  the value to the biorbd model. Parameters are global: the same vector entry is seen by all the phases.

## Scene 6 - `Impact`

* Model `models/point_floor.bioMod`: a 1 kg point in the x-z plane (`translations xz`) with one rigid contact along
  the z axis (`contact Mass_contact ... axis z`). Two phases: phase 0 is a free fall from z = 1 m (duration
  sqrt(2 z0 / g) = 0.452 s, `tau_z = 0`, horizontal speed 1 m/s at the start), phase 1 (1 s) is the contact phase
  (`contact_types=[ContactType.RIGID_EXPLICIT]`) in which the point slides to x = 3 m and stops
  (`MINIMIZE_CONTROL` in both phases). The floor is the plane z = 0 because phase 0 is constrained to end at z = 0.
* `PhaseTransitionFcn.IMPACT` (source: `limits/phase_transition.py`) imposes `q_after = q_before` and
  `qdot_after = model.qdot_from_impact(q_before, qdot_before)`, that is biorbd `ComputeConstraintImpulsesDirect`
  with the contacts of the model after the transition. Real solve: `v_z` goes from -4.43 m/s to 0 (jump +4.43 m/s),
  `v_x` stays 2.71 m/s (frictionless contact along z only); the lost kinetic energy is 9.81 J = m g z0 (m = 1 kg).
  IPOPT: 7 iterations, `Solve_Succeeded`.
* Same problem with `PhaseTransitionFcn.CONTINUOUS`: IPOPT returns `Infeasible_Problem_Detected` (24 iterations,
  bioptim status 1). Continuity would impose `v_z = -4.43 m/s` at the first node of the contact phase, but the rigid
  contact keeps the vertical acceleration at 0, so the point would sink through the floor. The dashed red curve in
  the animation is the LAST IPOPT ITERATE, which satisfies neither the dynamics nor the continuity: it is only shown to
  illustrate the failure and must not be read as a solution.
* Simplifications (honest list): the impact is perfectly inelastic and frictionless along the contact axis (this is
  what the biorbd impulse model computes, no restitution coefficient); the mass is a point (no rotation); the phase 0
  duration is fixed to the free-fall time instead of being free; no unilateral constraint (`contact force >= 0`) is
  imposed in phase 1; the same `bioMod` (with its contact) is used in both phases, only the `contact_types` of the
  models differ.

## Honest remarks

* All IPOPT runs reported status 0 (converged) in the final data; the iteration counts shown in the scenes are the real
  ones. During development, solves with tighter bounds (10 N and 8 N; 50 and 30 N for the free-time problem) ended with
  IPOPT status 1 ("solved to acceptable level"), and the resulting costs/durations were not monotonic in the bound
  (for instance cost 70.9 at 10 N but 46.1 at 8 N; T* = 0.607 s at 50 N against 0.601 s at 60 N). These are different
  local minima of a non-convex problem, so the values selected for the animation (in `generate_features_data.py`) are the
  ones that converged cleanly and vary monotonically; they are one local solution each, not a proof of global optimality.
* The free-time result depends on the discretization (N = 30 gave 0.382 s at 100 N, N = 40 gives 0.328 s: bang-bang
  controls are poorly represented by piecewise constants on a coarse grid) and on the initial guess.
* Minimum-control cost with a cart bound (scene 2, second beat) was also sensitive: the solution with `|y| <= 0.5`
  had a higher cost than with 0.4 in one intermediate run, again a sign of local minima.
* The oscillations in the multiphase angular velocity and in the free-time torque are real solver output (the problem
  has no smoothness term).
* `Parameters`: the swing-up with a minimum-effort cost has several local minima. The four solves use a continuation:
  the first one (weight 0.001) starts from the default guess, each following one from the previous solution (states,
  controls and parameter). Solved independently from the default guess, the weights gave max_tau* = 23.74, 20.99
  (total cost 45.12, slightly lower than the 45.45 shown), 18.01, 17.53 (cost 72.0 against 58.8 with continuation for
  the neighbouring weight 0.05) and, for weight 0.1, 23.32 (cost 162.8 against 70.4): other, worse local minima (and a
  slightly better one at 0.01). A linear initial guess for theta did not help either. The values shown are therefore
  one local solution per weight, not a proof of global optimality. All shown solves have IPOPT status `Solve_Succeeded`.
* `Impact`: in a first version the vertical control `tau_z` was free in the contact phase. The contact imposes
  `z'' = 0` whatever `tau_z` is (checked by evaluating the dynamics function), so `tau_z` cannot change the motion, yet
  IPOPT returned a constant `tau_z` = 10.1 N (far from the optimal 0), which inflated the cost from 21 to 123 without
  changing the trajectory. This looks like a numerical artefact of a degenerate variable; the final problem bounds
  `tau_z` to 0 in both phases (`u_bounds[p]["tau"][1, :] = 0`). Also note that the example
  `toy_examples/torque_driven_ocp/example_rigid_contact.py` passes `contact_types` to `DynamicsOptions` and the
  arguments in another order than the current `OptimalControlProgram` signature; here `contact_types` is given to the
  model (`TorqueBiorbdModel(path, contact_types=[...])`), as in `getting_started/example_inequality_constraint.py`.

## Exercises

1. In `ObjectivesNodes`, replace the Mayer term by `ObjectiveFcn.Mayer.MINIMIZE_STATE` on `qdot` at `Node.END`: what
   happens to the final velocity? Then use `Node.ALL_SHOOTING` for a Lagrange `MINIMIZE_STATE` on `qdot`.
2. Verify the `Node.INTERMEDIATES` remark: add an objective on `Node.INTERMEDIATES` in a small OCP and print
   `ocp.nlp[0].J` (or the plot of the penalty) to see which nodes are used.
3. In `ConstraintsBounds`, keep `|tau| <= 100` but add a `ConstraintFcn.TRACK_STATE` (key `q`, index 0) with
   `min_bound`/`max_bound` at `Node.ALL` instead of the `x_bounds` trick. Do you get the same trajectory?
4. Find the smallest torque bound for which IPOPT still finds a swing-up in 1 s (try 10, 9, 8 N) and compare the
   returned status with the cost: why is the cost not monotonic?
5. In `MultiphaseTransitions`, add a cost on the jump with a `PhaseTransitionFcn.CONTINUOUS` on the controls
   (`CONTINUOUS_CONTROLS`), or a third phase.
6. In `FreeTime`, combine `MINIMIZE_TIME` (weight 1) with `MINIMIZE_CONTROL` (small weight): sweep the ratio and draw the
   trade-off between duration and effort. Try `n_shooting = 60` and see how T* changes.
7. In `Parameters`, add a second parameter `min_tau` (asymmetric torque limits) as in
   `minimize_maximum_torque_by_extra_parameter.py`, or use a parameter shared by two phases. Then try a parameter that
   modifies the model: optimize the gravity as in `custom_parameters.py` and see what `parameter_bounds` do.
8. In `Parameters`, solve the weight 0.1 problem from the default guess and compare the cost with the continuation
   result: which local minimum do you get, and what does the torque look like?
9. In `Impact`, make the phase 0 duration free (`ObjectiveFcn.Mayer.MINIMIZE_TIME` with the constraint z = 0 at its end)
   and check that it converges to sqrt(2 z0 / g). Then start with a non-zero vertical speed and see how the impact
   velocity and the lost energy change.
10. In `Impact`, replace the contact axis `z` by `xz` in the bioMod: the horizontal velocity is now also removed by the
    impact (a point that sticks to the floor). How does the phase 1 problem change?
