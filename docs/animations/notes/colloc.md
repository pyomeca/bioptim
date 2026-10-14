# Collocation degree and point family (`CollocationDegree`, ~13.5 s)

## What it teaches
`OdeSolver.COLLOCATION(polynomial_degree=d, method="legendre"|"radau")`. Beat 1 shows the collocation points inside one
interval (rescaled to [0, 1]) for d = 2..6, read from `casadi.collocation_points(d, method)` (the integrator uses
`[0] + collocation_points(degree, method)`, `bioptim/dynamics/integrator.py`): Legendre points are all interior, Radau
ends on the next shooting node. Beat 2: error of each real solution against the dynamics, on a log axis, together with
the real number of decision variables (`sol.vector`) and IPOPT iterations / time (Legendre). Message: the error drops by
orders of magnitude with the degree while the vector grows linearly (+N*nx = +120 per degree).

## Data (real, `data/colloc_pendulum.npz`)
Cart-pendulum of `generate_accuracy_data.py` (`pendulum.bioMod`), sliding translation actuated, rotation 0 -> 1 rad in
T = 1 s, N = 30, minimise the integral of tau^2, common initial guess (linear q). All 10 solves: IPOPT status 0.
Error = norm of (final state re-integrated from x0 with the optimised piecewise-constant controls, DOP853 rtol 1e-10,
atol 1e-12, same `nlp.dynamics_func`) minus (optimised final state); never reset (single shooting).

| degree | legendre err / cost | radau err / cost | variables |
|---|---|---|---|
| 2 | 1.2e-1 / 7.60 | 2.8e-2 / 8.00 | 425 |
| 3 | 2.4e-2 / 7.81 | 2.1e-2 / 7.69 | 545 |
| 4 | 2.0e-3 / 7.78 | 1.2e-2 / 7.80 | 665 |
| 5 | 2.4e-4 / 7.78 | 3.2e-3 / 7.77 | 785 |
| 6 | 2.6e-4 / 7.78 | 5.0e-4 / 7.78 | 905 |

Legendre degree 2 -> 5: error / 490, variables x 1.85. IPOPT: 10-18 iterations, 0.1-0.7 s (Legendre, one run, noisy).

## Generate and render
```bash
# repo root, conda env captury_biobuddy on the PATH
PYTHONPATH=. python docs/animations/generate_colloc_data.py      # writes data/colloc_pendulum.npz
# from docs/animations
manim render -qh anim_colloc.py CollocationDegree                # 1080p60
```

## Caveats (honest)
- The swing-up to 3.14 rad (as in `generate_accuracy_data.py`) is NOT a good test: with it the optimiser exploits the
  discretisation, the costs differ by a factor 2-5 between runs and the errors do not decrease monotonically (errors of
  ~1 rad up to degree 6 at N = 30; also tried T = 2, 3 and actuating the rotation: same or worse, some non-converged).
  The gentler 1 rad target was chosen because every solution reaches the same cost (7.6-8.0) and the errors are
  comparable. This choice is a limitation: the ranking depends on the problem.
- Not monotonic: Radau is better than Legendre at degree 2 and 3 here (a fluke of the different optima; costs also
  differ), Legendre wins from degree 4. Legendre degree 6 (2.6e-4) is not better than degree 5 (2.4e-4): a plateau.
  It is not the IPOPT tolerance (tol and constr_viol_tol = 1e-10 give 2.4e-4 and 2.6e-4 again); at N = 60 the degree 6
  error drops to 6.7e-7 (degree 4: 6.9e-5), so the plateau comes from the optimum adapting to N = 30, not from the scheme.
- The theoretical orders (2d for Legendre, 2d-1 for Radau) are quoted from collocation theory, not measured here.
- `local_max` (each interval re-integrated from the optimised node state) is also stored; it is noisier than the drift
  (e.g. 1.1e-1 for Legendre degree 3) and is not plotted.
- `duplicate_starting_point` is not shown: it is not used by these solves.
- IPOPT time is one run on one laptop, not a benchmark; iterations are more reliable.

## Exercises
1. Change `COLLOC_TARGET=3.14` (env variable of the generator) and observe how the costs and errors stop being
   comparable; why can a coarser scheme reach a lower cost?
2. Run `N` = 15, 30, 60 (`COLLOC_N`) at fixed degree and estimate the convergence slope in h on a log-log plot.
3. Plot the error against the IPOPT time instead of the degree: which (degree, method) is the best trade-off?
