# Learning Bioptim with Manim: Direct Multiple Shooting vs Direct Collocation

A short animated course (about 2 minutes, 5 scenes) built with [Manim Community](https://www.manim.community/) that
explains

1. what an **optimal control problem** (OCP) is (state `x`, control `u`, dynamics `ẋ = f(x, u)`, Lagrange/Mayer cost,
   constraints, bounds) and how each piece maps to the bioptim API;
2. how bioptim **discretizes** it:
   * **Direct Multiple Shooting (DMS)**: `OdeSolver.RK4(n_integration_steps=5)`, continuity constraint `x_{k+1} = F(x_k, u_k)`;
   * **Direct Collocation (DC)**: `OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre")`, collocation points,
     defect constraints at the collocation points.

The running example is the bioptim pendulum swing-up (`bioptim/examples/toy_examples/sqp_method/pendulum.py`, same
problem as the README tutorial): start hanging (`q = 0`), finish upright (`q = 3.14`) at rest, minimum `∫ τ² dt`
(`ObjectiveFcn.Lagrange.MINIMIZE_CONTROL`).

## Content of this folder

| File | Role |
|------|------|
| `generate_pendulum_data.py` | Builds the pendulum OCP **with bioptim**, solves it with RK4 (DMS) and COLLOCATION (DC) and writes `data/pendulum_solutions.npz` |
| `data/pendulum_solutions.npz` | Small (about 50 kB) result file, committed so the animation renders without bioptim |
| `dms_vs_dc.py` | The Manim scenes (they only need `numpy` and `manim`, no LaTeX) |
| `.gitignore` | Ignores generated media (`media/`, `*.mp4`) |

### Provenance of the data: real bioptim output

`data/pendulum_solutions.npz` was produced by `generate_pendulum_data.py` with **bioptim (this repository) +
biorbd + casadi 3.7 + IPOPT**. There is no fallback/fake data. N = 20 shooting intervals, T = 1 s, initial guess = linear
interpolation of `q` from hanging to upright, zero controls. Content (`p` is `rk4` or `col`):

* `{p}_q_nodes`, `{p}_qdot_nodes` (2, N+1): converged states at the shooting nodes; `{p}_tau` (2, N): controls;
* `{p}_q_steps`, `{p}_qdot_steps`, `{p}_t_steps`: states/time *inside* each interval (the 6 RK4 integration points,
  or the node + the 3 collocation points);
* `{p}_n_decision_variables`, `{p}_cost`, `{p}_iterations`;
* `rk4_it_*` / `col_it_*`: same fields for an **early, not converged IPOPT iterate** (iteration 0 = the initial guess for both transcriptions
  (linear interpolation of q and q̇, zero controls). They are what the animation shows before "the optimizer closes the gaps": the gaps and
  slope mismatches drawn in red are the real constraint violations of that iterate.

Two honest remarks:

* RK4 and COLLOCATION are two independent IPOPT runs on a **non-convex** problem and may end in different local minima
  (the costs differ a lot, and depend on the initial guess). The animation therefore compares variable and iteration
  counts, not "accuracy". Exercise 5 below shows how to do a fair accuracy comparison.
* bioptim reports 125 (RK4) and 365 (COLLOCATION) decision variables. The counting `(N+1)·4 + N·2` gives 124 and
  `(N+1)·4 + N·3·4 + N·2` gives 364: one extra variable is present in the solver vector that I did not investigate.

## Installation

Rendering only (no bioptim needed, the data file is committed):

```bash
python -m venv .venv-manim
# Windows: .venv-manim\Scripts\activate      Linux/macOS: source .venv-manim/bin/activate
pip install manim black
```

Manim Community 0.21 bundles PyAV, so no separate `ffmpeg` install and no LaTeX are required (the scenes use `Text` /
`MarkupText` with Unicode math only). On Windows the scenes use the fonts *Segoe UI* and *Consolas* (DejaVu on other
systems).

Regenerating the data (needs bioptim and its dependencies: casadi, biorbd, ...):

```bash
conda env create -f environment.yml      # creates the "bioptim" environment (see repository root)
conda activate bioptim
```

On Windows, when calling a conda environment's `python.exe` directly (without `conda activate`), add the environment's
`Library\bin` folder to `PATH`, otherwise IPOPT's DLLs are not found (`Plugin 'ipopt' is not found`).

## Generate the data (optional)

From the repository root:

```bash
python docs/animations/generate_pendulum_data.py
```

It takes about one minute and prints, for each of the four solves, the number of decision variables, the cost and the
number of iterations.

## Render the animation

From `docs/animations/` (`-ql` = 480p15, quick preview; use `-qh` for 1080p60):

```bash
manim render -ql --media_dir ../../media_anim dms_vs_dc.py OCPStatement TimeGrid MultipleShooting DirectCollocation Comparison
```

Pick a media directory outside the repository (or keep the default `media/`, which is git-ignored). One scene only:

```bash
manim render -ql dms_vs_dc.py MultipleShooting
manim render -s  dms_vs_dc.py DirectCollocation      # still image of the last frame
```

The first render is slow (font cache); the following ones take about one minute per scene at `-ql`. To get a single
video, concatenate the five mp4 files (for instance with `ffmpeg -f concat`, or a video editor).

## What each scene teaches

1. **OCPStatement**: the OCP as a table *mathematics / meaning / bioptim code*: Lagrange cost
   (`ObjectiveFcn.Lagrange.MINIMIZE_CONTROL`), Mayer cost (`ObjectiveFcn.Mayer.MINIMIZE_STATE`), dynamics
   (`TorqueBiorbdModel`, `DynamicsOptions`), boundary values and bounds (`BoundsList`), path constraints
   (`Constraint(ConstraintFcn.TRACK_STATE, ...)`). Then the pendulum swing-up, animated from the real solution.
2. **TimeGrid**: `n_shooting` intervals of length `Δt = T/N`; states `x_k` are decision variables at the `N+1` nodes,
   controls `u_k` are piecewise constant on the `N` intervals.
3. **MultipleShooting**: inside each interval the dynamics are integrated with RK4 (5 steps) from `x_k`, giving `F(x_k, u_k)`.
   At the initial guess this does not reach `x_{k+1}`: the red gap is the **defect**. IPOPT moves `x_k, u_k` until all
   defects are zero, that is the **continuity constraint** `x_{k+1} = F(x_k, u_k)`.
4. **DirectCollocation**: where the Legendre and Radau collocation points lie in `[0, 1]` (computed with numpy, same values as
   `casadi.collocation_points`); then per interval a degree-3 polynomial through `x_k` and the 3 collocation states. At each
   collocation point the polynomial slope must equal the dynamics `f(x, u)` (**defects**, red vs white tangents) and the
   polynomial must reach `x_{k+1}` (continuity). No integration is performed: the NLP is larger but sparse.
5. **Comparison**: table DMS vs DC (extra unknowns, constraints, dynamics evaluations, order, NLP structure, and the
   variable/iteration counts of this pendulum) and the exact bioptim lines to switch from one to the other.

## Exercises to learn bioptim

1. **Change `polynomial_degree`** (2, 3, 4, 5) in `generate_pendulum_data.py`: how do the number of decision variables,
   iterations and solve time evolve? Where do the collocation points move (`collocation_points` in `dms_vs_dc.py`)?
2. **Change `n_shooting`** (10, 20, 40) for both transcriptions: how does the size of the NLP grow, and what happens to the gaps
   between nodes?
3. **Change the ODE solver**: `OdeSolver.RK4(n_integration_steps=1, 5, 10)`, `OdeSolver.RK8()`,
   `OdeSolver.COLLOCATION(method="radau")`, `OdeSolver.IRK()`. Which ones are DMS, which ones are DC-like?
4. **Plot the defects yourself**: after `sol = ocp.solve(...)`, integrate the optimal controls with `sol.integrate()` and plot `F(x_k, u_k) - x_{k+1}` for the RK4 problem; for collocation, evaluate the polynomial from
   `sol.stepwise_states()` and compare its slope with `q̇`.
5. **Fair accuracy comparison**: give both problems the same initial guess (or warm-start one from the other), then
   compare the optimal costs, and compare each solution with a fine simulation (`sol.integrate(...)`) of the optimal controls.
6. **Change the OCP, not the discretization**: add a Mayer objective (`ObjectiveFcn.Mayer.MINIMIZE_TIME` with a free
   `phase_time`), or `ConstraintFcn.TRACK_STATE` with bounds on `q`, and see which parts of the animation (nodes,
   constraints, cost) are affected.
7. **Add a second scene**: animate `Direct Single Shooting` (only `u_k` as unknowns, one long integration) and explain why
   it is more sensitive than DMS.
