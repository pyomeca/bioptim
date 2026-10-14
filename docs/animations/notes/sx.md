# use_sx: MX or SX graph (`SxVsMx`, ~13.4 s)

## What it teaches
`OptimalControlProgram(..., use_sx=True)` builds the CasADi graph with SX (scalar nodes) instead of MX (default,
`use_sx=False`; README, section "Performance"). Beat 1: a SCHEMATIC of the two graph types (labelled as such in the
video; it is not a dump of the real graph). Beat 2: measured times for the same OCP, as three groups of bars:
OCP build (`OptimalControlProgram(...)`), solver set-up (wall time of `ocp.solve` minus IPOPT time) and IPOPT time
(`sol.real_time_to_optimize`), with a readout computed from the data.

## Data (real, `data/sx_timings.npz`)
Cart-pendulum `pendulum.bioMod`, sliding translation actuated, rotation 0 -> 1 rad in T = 1 s, RK4,
minimise the integral of tau^2, default IPOPT (exact Hessian), linear initial guess. N = 30 and N = 50 stored; N = 50
shown. Each configuration is rebuilt and re-solved 3 times (interleaved MX, SX, MX, SX...; one uncounted warm-up first).
Medians (s), all runs IPOPT status 0, identical iteration count and cost (difference ~1e-14) between MX and SX:

| N | | build | set-up (ocp.solve minus IPOPT) | IPOPT | iterations | ms / iteration |
|---|---|---|---|---|---|---|
| 30 | MX | 0.17 | 0.29 | 0.39 | 29 | 13 |
| 30 | SX | 0.18 | 0.63 | 0.11 | 29 | 4 |
| 50 | MX | 0.25 | 0.48 | 0.54 | 24 | 22 |
| 50 | SX | 0.24 | 1.06 | 0.14 | 24 | 6 |

Spread over the 3 runs: a few percent on build and IPOPT time (e.g. N = 50 IPOPT MX 0.505-0.578 s, SX 0.140-0.157 s),
set-up nearly constant.

Where the extra SX set-up goes: profiling `ocp.solve` (N = 30, SX) shows about 75 % of the call inside
`casadi.nlpsol(...)`, i.e. CasADi building the Jacobian and Hessian graphs from the big SX expression, before IPOPT
starts. This is a profile of one run, not a benchmark.

## Generate and render
```bash
# repo root, conda env captury_biobuddy on the PATH
PYTHONPATH=. python docs/animations/generate_sx_data.py      # writes data/sx_timings.npz (about 1 min)
# from docs/animations
manim render -qh anim_sx.py SxVsMx                            # 1080p60
```

## Caveats (honest)
- Timings depend on the machine (one Windows laptop), on the CasADi/IPOPT versions and on load; only 3 repeats. The
  iteration count and the cost are the reliable columns. Do not quote the ratios as universal.
- For ONE solve of this small problem SX does NOT win: total (build + set-up + IPOPT) is 1.26 s (MX) vs 1.44 s (SX) at
  N = 50. SX wins on the IPOPT time (about 3.7x per iteration) and pays for it in set-up. The break-even shown in the
  video (about 36 iterations) is computed as extra set-up / saved time per iteration, assuming the per-iteration cost is
  constant; it is an estimate from the medians, not a measured crossing.
- Set-up cost, and RAM, grow with problem size (README: SX "requires more RAM"); larger models and N are where the
  gain in IPOPT time matters, but that was NOT measured here.
- MX here means the defaults: `DynamicsOptions.expand_dynamics=True` (default), so MX already expands the dynamics
  function to SX internally (`configure_problem.py`); that is part of why MX is not much slower. `expand_continuity`
  (default False) and the `expand` flag of individual objectives/constraints (default False) were NOT varied.
- Not covered: `n_threads`, and model functions that are not SX compatible (nothing verified; the README states that
  OdeSolver.IRK is not compatible with `use_sx=True`).

## Exercises
1. Add a third configuration, MX with `DynamicsOptions(expand_dynamics=False)`, and one with `expand_continuity=True`;
   which bar of the chart moves?
2. Repeat with N = 100 and 200 (or a muscle model): how do the set-up bar and the IPOPT bar of SX scale with N? Watch
   the memory.
3. In an MHE/NMPC loop the same NLP is solved many times: estimate from these numbers how many solves it takes before
   SX is better, and check whether bioptim rebuilds the NLP at every solve.
