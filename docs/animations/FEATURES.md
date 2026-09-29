# Bioptim feature animations (part 2)

Four short Manim (Community Edition) scenes on bioptim features, on the same pendulum swing-up as `dms_vs_dc.py`
(`bioptim/examples/models/pendulum.bioMod`: `q = (y, theta)`, only the sideways force on `y` is actuated).
Every curve is a REAL bioptim + IPOPT result stored in `data/features_*.npz`; the code shown beside the curves is the
code used in `generate_features_data.py`.

| Scene | Length | Data file | What it teaches |
|-------|--------|-----------|-----------------|
| `ObjectivesNodes` | ~35 s | `features_objectives.npz` | Lagrange vs Mayer objectives, the `Node` enum, the effect of a weight |
| `ConstraintsBounds` | ~31 s | `features_constraints.npz` | `u_bounds` (control bounds) shrinking, `x_bounds` on a state |
| `MultiphaseTransitions` | ~17 s | `features_multiphase.npz` | several phases with different durations, `PhaseTransitionFcn` |
| `FreeTime` | ~19 s | `features_time.npz` | `ObjectiveFcn.Mayer.MINIMIZE_TIME`: the phase duration is optimized |

## Regenerate the data

The environment needs bioptim, biorbd, casadi and IPOPT on the `PATH` (on Windows with conda, the `Library/bin`
folder of the environment). From the repository root:

```bash
PYTHONPATH=. python docs/animations/generate_features_data.py
```

It runs 5 + 5 + 2 + 3 = 15 OCPs (RK4, 30-40 shooting nodes), about 10 s in total, and writes four small `.npz`
files (8-16 kB each). IPOPT is deterministic, so the same bioptim version gives the same numbers.

## Render

Manim (0.18+ / 0.21) is needed, with the fonts Segoe UI and Consolas on Windows (DejaVu elsewhere). No LaTeX.
From `docs/animations`:

```bash
manim -qh features_scenes.py ObjectivesNodes ConstraintsBounds MultiphaseTransitions FreeTime   # 1080p60
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
* `PhaseTransitionFcn.IMPACT` (rigid contact) is only mentioned: it needs a contact model, which the pendulum does not
  have.

## Scene 4 - `FreeTime`

* `Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.1, max_bound=4.0)`: `phase_time=1.0` is only the initial
  guess. With `|tau| <= 100, 80, 60 N` the optimal durations are 0.328, 0.463, 0.601 s (N = 40).
* The torque sits on its bound most of the time (bang-bang like) with some chattering between nodes.

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
