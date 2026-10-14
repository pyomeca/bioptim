# Architecture overview

## Purpose

`bioptim` formulates and solves direct optimal-control problems for biomechanics.
Users declare models, dynamics, objectives, constraints, bounds, initial guesses
and solver options; the package constructs symbolic CasADi programs and exposes
their solutions.

## Observed execution path

```text
Public API / examples
    -> OptimalControlProgram and NonLinearProgram
    -> dynamics configuration + variables + penalties
    -> transcription and optimization vector
    -> solver adapter (Ipopt, Acados, SQP, MadNLP)
    -> Solution reconstruction, integration, plotting and export
```

This is an OCP-oriented architecture, not a strict layered application. Imports
between `optimization`, `dynamics`, `limits`, `models`, `interfaces` and `gui`
are currently interdependent; new layer rules must therefore be introduced only
when they describe a boundary that already exists.

## Main areas

| Area | Responsibility | Change with special care |
| --- | --- | --- |
| `bioptim/optimization` | OCP/NLP assembly, vectors, parameters, solutions, receding horizon | grid dimensions, phase linkage, warm starts, scaling |
| `bioptim/dynamics` | variable declaration, dynamics configuration and integrators | MX/SX graphs, defects, collocation nodes |
| `bioptim/limits` | objectives, constraints, bounds, initial guesses, penalties | target/bounds scaling, node selection, weights |
| `bioptim/interfaces` | solver-specific adapters and options | statuses, multipliers, control/bounds ordering, solver capabilities |
| `bioptim/models` | biomodel protocols and biorbd/pinocchio implementations | units, frames, DoF ordering, contacts and external forces |
| `bioptim/misc` | enums, mappings, types and small shared utilities | keep independent of higher-level packages where possible |
| `bioptim/gui` | plotting and online callbacks | avoid coupling computational core further to display code |
| `bioptim/examples` | executable API documentation and reference use cases | preserve pedagogical output and update matching tests |

## Critical invariants

- CasADi type (`SX`, `MX`, `DM`), shape and sparsity are semantic.
- Direct shooting and collocation use different grids (`EACH_FRAME`,
  `ALL_POINTS`, intermediate nodes); controls may be constant or continuous.
- Variable, bounds and target scaling must remain in clearly identified
  coordinates.
- Biomechanical units, signs, frames, quaternion/root-DoF conventions, contacts
  and external forces must be verified explicitly.
- Solver adapters do not share identical notions of convergence, warm start,
  multipliers or option support.

## Tests and CI

Tests are grouped into `tests/shard1` through `tests/shard6` for execution time,
not by package. CI creates the Conda environment from `environment.yml`; Acados
is installed only for Linux shard 1. Numerical regressions must first be compared
with the same test on `origin/master`, including the resolved package versions.

## Current architecture risks

1. Large orchestrators (`OptimalControlProgram`, `ConfigureProblem`, `Solution`
   and the penalty system) concentrate many responsibilities.
2. Dynamic dispatch, callbacks and CasADi function construction make exhaustive
   static call graphs incomplete; inferred edges must be marked uncertain.
3. CI dependencies are not fully pinned and the Python requirement differs
   between `setup.py` and `environment.yml`; numerical baselines can drift.
4. Plotting, solver adapters and scientific core remain closely coupled in some
   paths. Refactor only alongside a behavior-preserving, well-tested change.
