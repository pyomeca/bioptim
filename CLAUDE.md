# Instructions for Claude and other coding assistants

Read and follow [AGENTS.md](AGENTS.md) before making changes. It is authoritative
(invariants, safe-change procedure); this file is only a navigation map.

For non-trivial work, also read:

- [architecture overview](docs/architecture/OVERVIEW.md)
- [LLM change context](docs/architecture/LLM_CONTEXT.md)

Source code and tests remain authoritative over any prose. Paths below are relative to `bioptim/`.

## What bioptim is

Python framework that builds and solves optimal control problems (OCP) with CasADi,
mostly for biomechanics (biorbd models, optionally Pinocchio). Multiple shooting or direct
collocation, solved by Ipopt, Fatrop, ACADOS or a home-made SQP. Version: `misc/__version__.py`.

## Package map

| Package | Key files | Main classes |
|---|---|---|
| `optimization/` | `optimal_control_program.py`, `non_linear_program.py`, `optimization_vector.py`, `vector_layout.py`, `parameters.py`, `optimization_variable.py`, `solution/solution.py` | `OptimalControlProgram`, `NonLinearProgram` (one per phase), `OptimizationVectorHelper`, `VectorLayout`, `Parameter`, `Solution` |
| same, variants | `stochastic_optimal_control_program.py`, `variational_optimal_control_program.py`, `receding_horizon_optimization.py`, `multi_start.py`, `problem_type.py` | `StochasticOptimalControlProgram`, `VariationalOptimalControlProgram`, `RecedingHorizonOptimization` (NMPC/MHE subclasses), `MultiStart`, `SocpType` |
| `dynamics/` | `configure_problem.py`, `configure_variables.py`, `ode_solvers.py`, `ode_solver_base.py`, `integrator.py`, `lagrange_interpolation.py`, `dynamics_functions.py`, `state_space_dynamics/`, `fatigue/` | `ConfigureProblem`, `DynamicsOptions(List)`, `OdeSolver`, `OdeSolverBase`, `Integrator`, `LagrangeInterpolation`, `DynamicsFunctions`; one class per dynamics type in `state_space_dynamics/` |
| `limits/` | `penalty_option.py`, `penalty.py`, `penalty_controller.py`, `objective_functions.py`, `constraints.py`, `path_conditions.py`, `phase_transition.py`, `multinode_penalty.py`, `multinode_constraint.py`, `multinode_objective.py` | `PenaltyOption`, `PenaltyFunctionAbstract`, `PenaltyController`, `Objective`/`ObjectiveFcn`, `Constraint`/`ConstraintFcn`, `Bounds`, `InitialGuess`, `PhaseTransition(Fcn)`, `MultinodePenalty` |
| `models/` | `protocols/` (`biomodel.py` + holonomic/stochastic/variational variants), `biorbd/`, `pinocchio/` | `BioModel` (Protocol), `BiorbdModel`, `PinocchioModel` |
| `interfaces/` | `__init__.py`, `interface_utils.py`, `*_interface.py`, `*_options.py` | `Solver` namespace (`IPOPT`, `FATROP`, `SQP_METHOD`, `ACADOS`), `SolverInterface`, `IpoptInterface`, `FatropInterface`, `SQPInterface`, `AcadosInterface` |
| `gui/` | `plot.py`, `graph.py`, `online_callback_*.py` | `PlotOcp`, `CustomPlot` |
| `misc/` | `enums.py`, `options.py`, `mapping.py`, `parameters_types.py`, `utils.py` | `ControlType`, `Node`, `Shooting`, `DefectType`, `OptionList`, `BiMapping`; type aliases (`CX`, `Int`, ...) |
| `examples/` | `__main__.py` (GUI launcher), `getting_started/`, `torque_driven_ocp/`, `muscle_driven_ocp/`, `stochastic_optimal_control/`, `acados/`, `moving_horizon_estimation/`, ... | Runnable scripts; `.bioMod` models next to them |

Public API is re-exported from `__init__.py`; new public classes must be added there.

## Where is X

- OCP construction: `OptimalControlProgram.__init__` -> `_check_arguments_and_build_nlp` -> `_prepare_dynamics`
  (calls `ConfigureProblem.initialize` per phase) -> `_prepare_bounds_and_init`
  (calls `OptimizationVectorHelper.declare_ocp_shooting_points`) -> `_declare_multi_node_penalties`
  -> `_finalize_penalties` (calls `_declare_continuity`) -> `_prepare_vector_layout`.
- Solving: `OptimalControlProgram.solve` -> solver interface `.solve()` ->
  `interface_utils.generic_solve` (builds NLP, calls the solver) -> `Solution.from_dict`.
  Bounds/objective/constraints dispatch: `generic_dispatch_bounds`, `generic_dispatch_obj_func`,
  `generic_get_all_penalties` in `interfaces/interface_utils.py`.
- Decision vector: `OptimizationVectorHelper.vector/bounds_vectors/init_vector`. Content is time (dt),
  states X, controls U, algebraic states A and parameters; the order is set by `VectorLayout` /
  `OrderingStrategy` (`VARIABLE_MAJOR` is the historical layout, `TIME_MAJOR` the alternative).
  Never index the vector by hand; use the helpers and `Solution`/`SolutionMerge`.
- Dynamics and variables of a phase: `ConfigureProblem.initialize` (via `AutoConfigure` in
  `configure_variables.py`, model-provided `*_configuration_functions`) and `state_space_dynamics/`.
- Penalties: user-facing `Objective`/`Constraint` -> `PenaltyOption` -> function in `PenaltyFunctionAbstract.Functions`
  (`penalty.py`) or the Fcn enum's own function; functions receive a `PenaltyController`.
- Models: interface in `models/protocols/biomodel.py`; add a backend by implementing it (see `docs/pinocchio_model.md`).
- Post-processing/plots: `Solution` (`solution/solution.py`, `solution_data.py`), `gui/plot.py`.

## Discretization

- Multiple shooting: `OdeSolver.RK1/RK2/RK4/RK8`, `IRK`, `CVODES`, `TRAPEZOIDAL`, `VARIATIONAL` (all nested classes
  of `OdeSolver` in `dynamics/ode_solvers.py`). RK classes use `rk_base.py`; integrators live in `dynamics/integrator.py`.
- Direct collocation: `OdeSolver.COLLOCATION(polynomial_degree=4, method="legendre" | "radau", duplicate_starting_point=...)`;
  `IRK` subclasses it (implicit shooting). Polynomials in `dynamics/lagrange_interpolation.py`,
  integrator class `COLLOCATION` in `dynamics/integrator.py`.
- Base contract: `OdeSolverBase` (`ode_solver_base.py`): `is_direct_collocation`, `is_direct_shooting`,
  `n_required_cx`, `initialize_integrator`, `defects_type` (`DefectType` in `misc/enums.py`).
- Continuity/defects: `OptimalControlProgram._declare_continuity` adds `ConstraintFcn.STATE_CONTINUITY`
  (or `ObjectiveFcn.Mayer.STATE_CONTINUITY` if `state_continuity_weight` is set) on `Node.ALL_SHOOTING`
  per phase, plus phase-transition continuity between phases. For collocation, defects are built in `integrator.COLLOCATION`.
- Per-node variables: `NonLinearProgram.declare_shooting_points` (states, controls, algebraic states, one symbol set per
  node; `n_states_decision_steps` gives the number of columns, which is larger with collocation).
  Control interpolation: `ControlType` (`CONSTANT`, `CONSTANT_WITH_LAST_NODE`, `LINEAR_CONTINUOUS`, `NONE`) in `misc/enums.py`;
  the number of control columns per node depends on it.

## Tests and examples

- `tests/shard1` .. `tests/shard6`: same suite split for CI (about 70 test files); shared helpers in `tests/utils.py`
  and `tests/test_utils_ocp.py`. `shard1/test__run_examples.py` runs the examples; plot reference images are in `tests/plot_reference_images/`.
- `bioptim/examples/`: start with `getting_started/` (`basic_ocp.py`, `custom_*.py`, `example_multiphase.py`, ...). GUI launcher: `python -m bioptim.examples`.
  The root `examples/` folder only holds a README.

## Conventions

- Format: `black . -l120 --exclude "external/*"`. NumPy-style docstrings.
- Type aliases (`CX`, `Int`, `Str`, `Bool`, `NpArray`, ...) come from `misc/parameters_types.py`; use them in annotations.
- PR titles: `[WIP]` while in progress, `[RTR]` ready to review, `[RTM]` ready to merge (`docs/contributing.md`).
- `external/` (acados, ...) is a submodule; do not edit it.

## Things worth knowing

- `OdeSolver.TRAPEZOIDAL` reports `is_direct_shooting=True` and `is_direct_collocation=False`, yet it is used mainly by the
  stochastic `SocpType.TRAPEZOIDAL_*` problems; it rejects piece-wise constant controls.
- `interfaces/fratrop_options.py` is spelled that way (holds `FATROP`); the file name is not a typo you should fix casually.
- Python requirement lives in `pyproject.toml` (`requires-python >= 3.10`), not `setup.py` (no such file).
