# IPOPT: iterates and multi-start (anim_solver.py)

Two short scenes (about 13 s and 9 s) on the pendulum swing-up (`bioptim/examples/models/pendulum.bioMod`, N = 30,
T = 1 s, only the translation is actuated, `MINIMIZE_CONTROL`, end upright at rest).

## What it teaches

1. `IpoptIterates`: an IPOPT iterate is generally NOT a feasible trajectory. Iterations 0, 1, 3, 8, 12, 40, 51 are shown
   with the objective and the primal infeasibility (`inf_pr`). The objective first stays small (about 5), then grows
   (48 at iteration 40) while `inf_pr` drops from 3 to 4e-2 and finally to 1.6e-12; the final cost is 40.28. Code shown:
   `Solver.IPOPT()`, `set_maximum_iterations(k)`, `set_tol(...)` (both verified in `bioptim/interfaces/ipopt_options.py`,
   lines 216 and 240).
2. `IpoptMultiStart`: from different initial guesses (`InitialGuessList.add("q", ..., interpolation=EACH_FRAME)`) IPOPT
   converges (`Solve_Succeeded`) to different local minima with costs 40.28, 42.70, 43.79, 48.41, 185.77, 302.89.

## Generate and render

```bash
E=/c/Users/micka/miniconda3/envs/captury_biobuddy
export PATH="$E:$E/Library/bin:$E/Library/mingw-w64/bin:$E/Scripts:$PATH"
PYTHONPATH=. python docs/animations/generate_solver_data.py     # ~1 min, writes data/solver_iterates.npz, data/solver_multistart.npz
cd docs/animations
manim render -qh anim_solver.py IpoptIterates
manim render -qh anim_solver.py IpoptMultiStart
```

## Honest caveats

- The k-th iterate is obtained by a separate solve with `set_maximum_iterations(k)` (IPOPT is deterministic). Objective and
  `inf_pr` in the readout are IPOPT's own history from the full solve; they agree with `sol.cost` of the truncated solves.
- Iteration 0 is the initial guess after IPOPT projects it into the bounds (q = 0 everywhere, except the last node fixed
  at 3.14), hence the jump at t = 1 s. Later iterates are infeasible (dynamics defects), that is the point.
- The plotted trajectories are the shooting-node values (linearly joined), not the integrated path.
- Multi-start is a plain loop over 12 random smooth guesses (seed 0, `np.random.default_rng(0)`), not bioptim's
  `MultiStart`. Of these, 11 converged (status 0); one stopped with status 1 (cost about 5000, not shown). Duplicates (185.77 and
  302.89 each appear twice) are shown once. The scene shows 6 of the distinct minima (the 4 best and the 2 worst below 400)
  and the zero-guess solution of scene 1. Other minima exist (52.5, 66.3, 82.8, 86.7 not drawn).
- These are local minima of the discretized problem, not proven distinct global structures; the lowest found (40.28) is not
  certified global.

## Exercises

1. Change `ITER_LIST` and the `show` list to look at iterations 15 to 40. Does the objective ever decrease monotonically?
   Why not (hint: infeasible iterates, filter line search)?
2. Loosen `set_tol` to 1e-3 and compare the iteration count and the final cost to 1e-8.
3. Increase `N_STARTS` or the amplitude of the random guess in `random_start`: how many distinct costs do you find, and does
   any beat 40.28?
