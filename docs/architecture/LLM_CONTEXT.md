# LLM change context

Read this guide to locate code; then inspect the source and tests themselves.

## Read first

| Task | Read first | Validate first |
| --- | --- | --- |
| Add or change an OCP option | `optimization/optimal_control_program.py`, the option class, an analogous example | closest shard test and an example import |
| Change dynamics or variables | `dynamics/configure_problem.py`, `dynamics/configure_variables.py`, the model class | MX/SX and RK/collocation paths affected |
| Change objective, constraint, bounds or targets | `limits/penalty*.py`, `limits/path_conditions.py`, controller/tests | scaling, rows/cols/nodes and grid dimensions |
| Change a solver option or adapter | `interfaces/`, `interfaces/abstract_options.py`, solver tests | affected solver status/options; do not assume adapter equivalence |
| Change a biomodel feature | `models/protocols/`, matching `models/biorbd/` or `models/pinocchio/` implementation | units, frame, DoF ordering, contacts/external forces |
| Change a solution, warm start or receding horizon | `optimization/solution/`, `receding_horizon_optimization.py` | solution shape, grid, status and phase/cycle handling |
| Change a plot or callback | `gui/`, the producing penalty/solution code | a focused smoke test and plot reference where applicable |
| Diagnose CI numerics | failing test and `origin/master` first | package versions, platform, solver status, feasibility and cost |

## Core flow

`OptimalControlProgram` builds phase-level `NonLinearProgram` objects. Dynamics
configuration creates symbolic variables and CasADi functions; limits create
penalties; vector helpers assemble the numerical problem; interfaces solve it;
`Solution` exposes and reconstructs results. Examples are executable API
documentation, while shards distribute test duration rather than ownership.

## Safe change procedure

1. Locate an analogous example and direct test.
2. Identify coordinate systems and dimensions before editing.
3. Make the smallest coherent source-and-test change.
4. Run focused tests, then broaden to each affected variant.
5. For changed numerical results, explain the scientific cause and compare
   constraints, cost and solver status against a reference.
6. Run Black for touched Python files and inspect the Git diff.

## Files requiring special care

- `bioptim/__init__.py`: public API facade.
- `optimization/optimal_control_program.py` and `dynamics/configure_problem.py`:
  high-fan-in orchestrators.
- `limits/penalty*.py` and `limits/path_conditions.py`: scaling and grid
  semantics.
- `interfaces/`: backend-specific behavior is intentional.
- `external/` and `c_generated_code/`: do not treat as editable core source.

## Architecture direction

Do not force a general `CLI -> domain -> infrastructure` model. Preserve the
current OCP-oriented boundaries. The first candidate for an explicit import
contract is that `bioptim.misc` stays independent of other `bioptim`
sub-packages; validate it before enforcing it in CI.
