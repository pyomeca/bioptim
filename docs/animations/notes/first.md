# My first OCP, step by step (anim_first.py, scene FirstOCP, about 16 s of content at native speed)

## What the scene teaches
The pendulum swing-up of `bioptim/examples/getting_started/basic_ocp.py` (README "A first practical example") is built
line by line, and each block of code has its visual counterpart on the left:

| step | code shown | visual (all real) |
| --- | --- | --- |
| model + dynamics | `TorqueBiorbdModel("pendulum.bioMod")`, `DynamicsOptions(ode_solver=OdeSolver.RK4())` | stick figure at rest drawn from the two real biorbd markers (`marker_1` -> `marker_2`) |
| objective | `Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")` | the force axis and the cost `Σ (force)² Δt` |
| bounds | `BoundsList` for `q`, `qdot` (`bounds_from_ranges`, start and end fixed, `q1(T) = 3.14`) and `tau` (+-100 N, rotation passive) | red bands outside the cart range read from the bounds, red dots for the fixed poses, red bands beyond +-100 N, dashed target pose |
| initial guess | `InitialGuessList` with zeros for `q`, `qdot`, `tau` | dashed lines at zero |
| solve | `OptimalControlProgram(...)`, `ocp.solve(Solver.IPOPT())` | real q1(t), force(t), poses along the solution, tip path, cost and IPOPT status |

Setting: `pendulum.bioMod` (sliding translation q0 + rotation q1, only the translation is actuated), N = 30 intervals,
T = 1 s, RK4 multiple shooting, IPOPT default options, `use_sx=True`. Numbers shown (generator output): IPOPT status 0
(converged), 57 iterations, cost 41.6, peak force 22.0 N, q1(T) = 3.14 rad; the cart reaches its lower bound of -1 m.

## Commands
Environment: conda env `captury_biobuddy` (bioptim, biorbd, casadi, IPOPT), repo root:

    E=/c/Users/<you>/miniconda3/envs/captury_biobuddy
    export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python docs/animations/generate_first_data.py     # -> data/first_pendulum.npz
    python docs/animations/render_series.py anim_first.py FirstOCP --dry
    python docs/animations/render_series.py anim_first.py FirstOCP --lang both --strict

The generator prints `status 0 iterations 57 cost 41.58...`, the shapes of the stored bounds and the marker positions.

## Honest caveats
- One cold-start solve from an all-zero initial guess, no continuation. The problem is non-convex: IPOPT returns *a*
  local minimum (here the cart first backs up to -1, its lower bound, then swings the pole up); another guess (or the
  mirror image, cart going right) could give another swing.
- The code on screen is a simplification of `prepare_ocp` of the generator (imports omitted, the model path shortened to
  `"pendulum.bioMod"`, `Solver.IPOPT()` where the generator switches off the online plot and the IPOPT print level, and
  `N`, `T` written 30 and 1 as in the generator). The reference is `basic_ocp.py`, which uses N = 400 and passes the
  options as arguments of `prepare_ocp`; the video uses N = 30 to keep the solve small.
- Not shown: `sol.print_cost()`, `sol.graphs()`, `sol.animate()` (they need a display), the `qdot` curves, and the
  difference axis of the ghost convention (the dashed line is the initial guess, and the solution is shown against it).
- The force axis is scaled to the +-100 N bound, so the real force (peak 22.0 N) looks small: the force bound is not
  active, whereas the cart lower bound is.
- The cost `Σ (force)² Δt` on the force axis is a schematic label of what `MINIMIZE_CONTROL` accumulates (Lagrange term);
  the value 41.6 is `sol.cost`.
- The stage poses are drawn from the real biorbd marker positions of the solution nodes (every 6th node plus the last).

## Exercises
1. Change `x_bounds["q"][1, -1] = 3.14` to `1.57` in the generator: which pose is reached, and how do the peak force and
   the cost change?
2. Set `u_bounds["tau"] = [-20, -20], [20, 20]` (keep `[1, :] = 0`): is the problem still feasible in T = 1 s? Read the
   IPOPT status printed by the generator.
3. Replace the all-zero `x_init["q"]` by a linear path from 0 to 3.14 rad for q1 (`InterpolationType.LINEAR` with a
   start and an end column) and compare the iteration count and the cart trajectory with the cold start.
