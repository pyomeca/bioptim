# Online plot (`OnlineIterates` ~11.7 s + `OfflineGraphs` ~4.8 s)

## What it teaches
`Solver.IPOPT(show_online_optim=True)` opens a live window refreshed at every IPOPT iteration (states, controls, custom
plots, optional IPOPT-output panel). Scene 1 re-draws that window (2x2: q, tau, custom "angle (deg)", IPOPT output) from
real iterates, next to the code: `show_online_optim=True` (= `online_optim=OnlineOptim.DEFAULT`; both together raise a
ValueError, `interface_utils.py`), `online_optim=OnlineOptim.MULTIPROCESS_SERVER` with `show_options=dict(show_bounds=True)`
(show_options go to `PlotOcp`; `host`/`port` for the SERVER variants), `ocp.add_plot(name, lambda t0, phases_dt, node_idx,
x, u, p, a, d: ..., plot_type=PlotType.PLOT)` and `ocp.add_plot_ipopt_outputs()`. Scene 2: the offline twin
`sol.graphs(show_bounds=True, show_now=False, save_name=...)` with two REAL matplotlib figures written by bioptim.

## Data (real)
`data/online_iterates.npz` from `generate_online_data.py`: pendulum swing-up (pendulum.bioMod, y actuated, N = 30, T = 1 s,
minimise tau, end upright at rest, RK4, SX). Iterate k (k = 0..51) = real solve with `set_maximum_iterations(k)`; the
inf_pr / inf_du curves are IPOPT's own history (`stats()["iterations"]`) of the full solve: Solve_Succeeded, 51 iterations,
cost 40.28. `data/online_graphs_*.png`: figures from `sol.graphs` on the Agg backend (regenerate only these:
`python docs/animations/generate_online_data.py graphs`).

## Commands
```bash
PYTHONPATH=. python docs/animations/generate_online_data.py        # ~3 min (52 solves)
# from docs/animations
manim render -qh anim_online.py OnlineIterates OfflineGraphs        # 1080p60, two mp4
```

## Caveats
- NOT a screen capture: the window is drawn in Manim (labelled on screen). The real online plot shows f, inf_pr and inf_du
  computed by bioptim's own callback (`ipopt_output_plot.py`: inf_pr = max |g|), here IPOPT's reported values are used, and
  f is the cost of the plotted iterate (`sol.cost`), shown as a number, not a curve.
- Iterates 13-15 and 26-28 are IPOPT restoration iterations: the reported objective of the history differs from `sol.cost`
  of the max_iter solve (which returns the last accepted point), so the trajectory stays put while inf_pr/inf_du move.
- The cost bump to ~75 around iterations 24-30 is real (the trajectory changes basin of the swing before converging).
- The IPOPT-output panel exists only live (`sol.graphs` sets `plot_ipopt_outputs = False`), so scene 2 shows only q and the
  custom plot. On Windows the default backend is MULTIPROCESS_SERVER (`OnlineOptim.MULTIPROCESS` is not available there).
- `show_online_optim` cannot be used in multi-start.

## Exercises
1. Add a custom plot of the kinetic energy or of tau^2 and check that it also appears in `sol.graphs()`.
2. Set `show_options=dict(show_bounds=False)`: how do the axes change during the solve?
3. Use `ocp.add_plot_penalty(CostType.ALL)` and compare the penalty plots with the cost printed by `sol.print_cost()`.
