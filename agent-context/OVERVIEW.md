# bioptim overview for agents

Shared, agent-maintained map of the codebase. Rules for contributing are in `agent-context/RULES.md`.
Last verified against the code on 2026-10-15 (bioptim 3.5.0). The code wins over this file: when you find a
wrong fact, fix it here in the same PR.

Package paths below are relative to `bioptim/` (e.g. `misc/enums.py` is `bioptim/misc/enums.py`, `examples/models/`
is `bioptim/examples/models/`); paths that start with `tests/`, `docs/`, `external/` or `.github/` are relative to the
repository root. The root `examples/` folder only holds a README.

## What bioptim is

Python framework that builds and solves optimal control problems (OCP) with CasADi, mostly for biomechanics
(biorbd models, optionally Pinocchio). Multiple shooting or direct collocation, solved by Ipopt, Fatrop, ACADOS or
a home-made SQP. Version: `misc/__version__.py`. Dependencies are conda-only (`environment.yml`;
`pyproject.toml` declares `dependencies = []`).

## Package map

`*` marks classes that are internal (not re-exported from `bioptim/__init__.py`).

| Package | Key files | Main classes |
|---|---|---|
| `optimization/` | `optimal_control_program.py`, `non_linear_program.py`, `optimization_vector.py`, `vector_layout.py`, `bound_vector.py`, `init_vector.py`, `parameters.py`, `optimization_variable.py`, `variable_scaling.py`, `solution/solution.py`, `solution/solution_data.py` | `OptimalControlProgram`, `NonLinearProgram` (one per phase), `OptimizationVectorHelper`*, `VectorLayout`*, `OrderingStrategy`, `Parameter(List)`, `VariableScaling(List)`, `Solution`, `SolutionMerge`, `TimeAlignment` |
| same, variants | `stochastic_optimal_control_program.py`, `variational_optimal_control_program.py`, `receding_horizon_optimization.py`, `multi_start.py`, `problem_type.py` | `StochasticOptimalControlProgram`, `VariationalOptimalControlProgram`, `RecedingHorizonOptimization`* (base of `MovingHorizonEstimator`, `NonlinearModelPredictiveControl` and their `Cyclic*`/`MultiCyclic*` variants), `MultiStart`, `SocpType` |
| `dynamics/` | `configure_problem.py`, `configure_variables.py`, `configure_new_variable.py`, `dynamics_evaluation.py`, `dynamics_functions.py`, `ode_solvers.py`, `ode_solver_base.py`, `rk_base.py`, `integrator.py`, `lagrange_interpolation.py`, `state_space_dynamics/`, `fatigue/` | `ConfigureProblem`, `DynamicsOptions(List)`, `AutoConfigure`, `DynamicsEvaluation`, `DynamicsFunctions`, `OdeSolver`, `OdeSolverBase`, `Integrator`*, `LagrangeInterpolation`*; one dynamics mixin per file in `state_space_dynamics/` (`StateDynamics`, `TorqueDynamics`, `MusclesDynamics`, ...) |
| `limits/` | `objective_functions.py`, `constraints.py`, `penalty.py`, `penalty_option.py`, `penalty_controller.py`, `penalty_helpers.py`, `weight.py`, `path_conditions.py`, `phase_transition.py`, `phase_transtion_factory.py`, `multinode_penalty.py`, `multinode_constraint.py`, `multinode_objective.py` | `Objective(List)`/`ObjectiveFcn`, `Constraint(List)`/`ConstraintFcn`, `PenaltyFunctionAbstract`*, `PenaltyOption`*, `PenaltyController`, `PenaltyHelpers`, `ObjectiveWeight`, `ConstraintWeight`, `Bounds(List)`, `InitialGuess(List)`, `PhaseTransition(List)`/`PhaseTransitionFcn`, `MultinodePenalty`*, `MultinodeConstraint(List)`, `MultinodeObjective(List)` |
| `models/` | `protocols/biomodel.py` (+ `holonomic_biomodel.py`, `stochastic_biomodel.py`, `variational_biomodel.py`), `biorbd/biorbd_model.py` (+ `multi_biorbd_model.py`, `holonomic_biorbd_model.py`, `stochastic_biorbd_model.py`, `variational_biorbd_model.py`, `external_forces.py`), `biorbd/model_dynamics.py`, `pinocchio/pinocchio_model.py`, `pinocchio/model_dynamics.py`, viewers (`biorbd/viewer_bioviz.py`, `models/viewer_pyorerun.py`) | `BioModel` (a `typing.Protocol`), `BiorbdModel`, `MultiBiorbdModel`, `HolonomicBiorbdModel`, `StochasticBiorbdModel`, `VariationalBiorbdModel`, `PinocchioModel`; model+dynamics classes (`TorqueBiorbdModel`, `MusclesBiorbdModel`, `TorquePinocchioModel`, ...); `ExternalForceSetTimeSeries`, `ExternalForceSetVariables` |
| `interfaces/` | `__init__.py`, `solver_interface.py`, `interface_utils.py`, `ipopt_interface.py`, `fatrop_interface.py`, `sqp_interface.py`, `acados_interface.py`, `*_options.py`, `abstract_options.py`, `solve_ivp_interface.py` | `Solver` namespace (`IPOPT`, `FATROP`, `SQP_METHOD`, `ACADOS`), `GenericSolver`* (base of the option classes), `SolverInterface`*, `IpoptInterface`*, `FatropInterface`*, `SQPInterface`*, `AcadosInterface`*; `solve_ivp_interface` is used by `Solution.integrate` |
| `gui/` | `plot.py`, `graph.py`, `online_callback_*.py`, `ipopt_output_plot.py`, `check_conditioning.py` | `CustomPlot`, `PlotOcp`*, `PlottingServer` |
| `misc/` | `enums.py`, `fcn_enum.py`, `options.py`, `mapping.py`, `parameters_types.py`, `utils.py` | Enums (`ControlType`, `Node`, `PhaseDynamics`, `InterpolationType`, `Shooting`, `SolutionIntegrator`, `QuadratureRule`, `DefectType`, ...); `FcnEnum`* (base of the `*Fcn` enums); `OptionList`*; `BiMapping(List)`; type aliases (`CX`, `Int`, ...) |
| `examples/` | `__main__.py` (Qt GUI launcher), `getting_started/`, `toy_examples/` (16 topics: `torque_driven_ocp/`, `muscle_driven_ocp/`, `stochastic_optimal_control/`, `acados/`, `moving_horizon_estimation/`, `custom_model/`, ...), `biomechanics/gait_optimal_estimation/`, `models/` | Runnable scripts, most with `prepare_ocp(...)` + `main()`. All `.bioMod` files and meshes live in `examples/models/` |

Public API is re-exported from `bioptim/__init__.py`; new public classes must be added there.

## How models and dynamics fit together

- A usable model is a backend class combined with a dynamics mixin, e.g.
  `class TorqueBiorbdModel(BiorbdModel, TorqueDynamics)` in `models/biorbd/model_dynamics.py`
  (Pinocchio: `TorquePinocchioModel` in `models/pinocchio/model_dynamics.py`).
- The backend implements the `BioModel` protocol (`models/protocols/biomodel.py`): kinematics, dynamics, markers, ...
- The mixin derives from `StateDynamics` (`dynamics/state_space_dynamics/abstract_dynamics.py`). It declares
  `state_configuration_functions`, `control_configuration_functions`, `algebraic_configuration_functions`,
  `extra_configuration_functions` (consumed by `AutoConfigure`) and `dynamics(...)`, which returns a
  `DynamicsEvaluation(dxdt=..., defects=...)`: `dxdt` for shooting solvers, `defects` for implicit transcriptions
  (`COLLOCATION`, `IRK`).
- Custom dynamics: subclass a backend and `StateDynamics`; see `examples/toy_examples/custom_model/`
  (`custom_package/my_model.py`, `main.py`).

## Key OCP options

- `OptimalControlProgram(bio_model, n_shooting, phase_time, dynamics=..., ...)`; lists (one item per phase) make a
  multiphase OCP.
- `use_sx=False`: CasADi `MX` by default, `SX` when `True` (not supported by `OdeSolver.IRK`, see Discretization).
- `control_type=ControlType.CONSTANT` (see Discretization).
- `ordering_strategy=OrderingStrategy.VARIABLE_MAJOR`: layout of the decision vector.
- `DynamicsOptions(ode_solver=OdeSolver.RK4(), phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
  expand_dynamics=True, expand_continuity=False, skip_continuity=False, state_continuity_weight=ConstraintWeight(),
  numerical_data_timeseries=None)`. `SHARED_DURING_THE_PHASE` builds one integrator reused at every node;
  `ONE_PER_NODE` builds one per node (required by stochastic OCPs, except `SocpType.COLLOCATION`).
- `Node` (`misc/enums.py`): `START`, `MID`, `INTERMEDIATES`, `PENULTIMATE`, `END`, `ALL`, `ALL_SHOOTING`, `TRANSITION`,
  `MULTINODES`, `DEFAULT`, or integer node indices. For objectives only, `DEFAULT` becomes `ALL_SHOOTING` (Lagrange)
  or `END` (Mayer); constraints must be given an explicit node.
- After construction: `ocp.update_objectives`, `update_constraints`, `update_parameter_objectives`,
  `update_parameter_constraints`, `update_objectives_target`, `update_bounds`, `update_initial_guess`.

## Where is X

- OCP construction: `OptimalControlProgram.__init__` -> `_check_arguments_and_build_nlp` -> `_prepare_dynamics`
  (calls `ConfigureProblem.initialize` then `ode_solver.prepare_dynamic_integrator` per phase) ->
  `_prepare_bounds_and_init` (calls `OptimizationVectorHelper.declare_ocp_shooting_points`) ->
  `_declare_multi_node_penalties` -> `_finalize_penalties` (phase transitions, then `_declare_continuity`, then
  constraints and objectives) -> `_prepare_vector_layout`. `StochasticOptimalControlProgram` overrides
  `_prepare_bounds_and_init` and `_declare_multi_node_penalties`.
- Solving: `OptimalControlProgram.solve` -> solver interface `.solve()` -> `Solution.from_dict`.
  - Ipopt, Fatrop and SQP go through `interface_utils.generic_solve` (builds the NLP, calls the solver).
    Bounds/objective/constraints dispatch: `generic_dispatch_bounds`, `generic_dispatch_obj_func`,
    `generic_get_all_penalties` in `interfaces/interface_utils.py`.
  - ACADOS does not: `AcadosInterface` builds its own costs and constraints and calls `AcadosOcpSolver`.
    `Solution.from_dict` only fills cost and constraint values for Ipopt-like solvers (Ipopt, Fatrop).
- Decision vector: `OptimizationVectorHelper.vector/bounds_vectors/init_vector`. Content is time (dt), states X,
  controls U, algebraic states A and parameters; the order is set by `VectorLayout` / `OrderingStrategy`
  (`optimization/vector_layout.py`; `VARIABLE_MAJOR` is the historical default, `TIME_MAJOR` the alternative;
  `VectorLayout` also accepts a custom generator callable, although `OptimalControlProgram` only types
  `OrderingStrategy`). Never index the vector by hand; use the helpers and
  `Solution`/`SolutionMerge`.
- Dynamics and variables of a phase: `ConfigureProblem.initialize` -> `AutoConfigure` (`configure_variables.py`), fed
  by the `*_configuration_functions` of the model's dynamics mixin (see above).
- Penalties: user-facing `Objective`/`Constraint` -> `PenaltyOption.add_or_replace_to_penalty_pool` -> the Fcn enum
  member is called (`FcnEnum.__call__`, `misc/fcn_enum.py`), which runs a function of
  `PenaltyFunctionAbstract.Functions` (`penalty.py`) or the enum's own function (phase transitions, multinode
  penalties). Functions receive a `PenaltyController` and return a CasADi expression.
- Models: interface in `models/protocols/biomodel.py`. `docs/pinocchio_model.md` documents the Pinocchio backend
  (function equivalents with biorbd); use `models/pinocchio/` as the reference for a new backend.
- Post-processing: `Solution` (`solution/solution.py`): `decision_states/controls/parameters/algebraic_states`,
  `stepwise_states/controls`, `decision_time`/`stepwise_time`, `integrate`, `cost`, `print_cost`, `graphs`,
  `animate`. Merge options and time alignment in `solution/solution_data.py`. Plots in `gui/plot.py`.

## Discretization

- `OdeSolver` (`dynamics/ode_solvers.py`) nests `RK1`, `RK2`, `RK4`, `RK8`, `VARIATIONAL` (all based on `RK` in
  `dynamics/rk_base.py`), `TRAPEZOIDAL`, `COLLOCATION`, `IRK` and `CVODES`. The CasADi integrators they build live in
  `dynamics/integrator.py`.
- Direct collocation: `OdeSolver.COLLOCATION(polynomial_degree=4, method="legendre", defects_type=DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS)`;
  `method` can also be `"radau"`; `duplicate_starting_point=False` is an `OdeSolverBase` keyword. Polynomials in
  `dynamics/lagrange_interpolation.py`, integrator class `COLLOCATION` in `dynamics/integrator.py`.
- `IRK` subclasses `COLLOCATION` (implicit shooting) but reports `is_direct_shooting=True`,
  `is_direct_collocation=False`, and does not support `use_sx=True`.
- Base contract: `OdeSolverBase` (`ode_solver_base.py`): `is_direct_collocation`, `is_direct_shooting`,
  `n_required_cx`, `initialize_integrator`, `defects_type` (`DefectType`: `QDDOT_EQUALS_FORWARD_DYNAMICS`,
  `TAU_EQUALS_INVERSE_DYNAMICS`, `NOT_APPLICABLE`).
- Continuity/defects: `OptimalControlProgram._declare_continuity` adds `ConstraintFcn.STATE_CONTINUITY`
  (or `ObjectiveFcn.Mayer.STATE_CONTINUITY` when `state_continuity_weight` is not a `ConstraintWeight`) on
  `Node.ALL_SHOOTING` per phase (skipped with `skip_continuity=True`), plus phase-transition continuity between phases
  (except `DISCONTINUOUS` transitions). With collocation and `duplicate_starting_point`, the constraint branch also
  adds `FIRST_COLLOCATION_HELPER_EQUALS_STATE` (not when continuity is an objective). Collocation defects come from `integrator.COLLOCATION` and are concatenated
  in the `state_continuity` penalty (`limits/penalty.py`).
- Per-node variables: `NonLinearProgram.declare_shooting_points` (states, controls, algebraic states, one symbol set
  per node; `n_states_decision_steps` gives the number of columns: 1 for RK/IRK, `polynomial_degree + 1` for
  collocation, +1 with `duplicate_starting_point`, and 1 at the last node).
- Control interpolation: `ControlType` (`misc/enums.py`): `NONE`, `CONSTANT`, `CONSTANT_WITH_LAST_NODE`,
  `LINEAR_CONTINUOUS`. Decision controls are always one column per node; what changes is the number of control
  nodes (`n_shooting` for `CONSTANT`, `n_shooting + 1` for the two others) and how many points the integrator
  interpolates per interval (`ControlType.nb_interpolation_points`: 2 for `LINEAR_CONTINUOUS`).
- Initial guesses/targets: `InterpolationType` (`CONSTANT`, `LINEAR`, `EACH_FRAME`, `ALL_POINTS` (every collocation
  point; same as `EACH_FRAME` in shooting), `SPLINE`, `CUSTOM`, ...).

## Task -> read first -> validate

| Task | Read first | Run first |
|---|---|---|
| Objective, constraint, penalty | `limits/penalty.py`, `limits/objective_functions.py`, `limits/constraints.py`, `limits/penalty_option.py` | `tests/shard6/test_penalty.py`, `tests/shard6/test_multinode_constraints.py`, `tests/shard3/test_custom_weight.py` |
| Dynamics, variables | `dynamics/configure_problem.py`, `dynamics/configure_variables.py`, `dynamics/state_space_dynamics/` | `tests/shard5/test_dynamics.py`, `tests/shard5/test_configure_problem.py`, `tests/shard5/test_custom_model.py` |
| Model backend | `models/protocols/biomodel.py`, `models/biorbd/biorbd_model.py`, `models/pinocchio/pinocchio_model.py` | `tests/shard4/test_biorbd_model.py`, `tests/shard6/test_pinocchio_model.py`, `tests/shard5/test_dynamics_units.py` |
| ODE solver, transcription, defects | `dynamics/ode_solvers.py`, `dynamics/integrator.py`, `optimization/optimal_control_program.py` | `tests/shard6/test_defects.py`, `tests/shard6/test_continuity_linear_continuous.py`, `tests/shard2/test_global_getting_started.py` |
| Decision vector, bounds, initial guess | `optimization/optimization_vector.py`, `optimization/vector_layout.py`, `limits/path_conditions.py` | `tests/shard6/test_vector_layout.py`, `tests/shard6/test_bounds_accessor.py`, `tests/shard6/test_update_bounds_and_init.py` |
| Solver interface | `interfaces/interface_utils.py`, the `*_interface.py` and `*_options.py` of the solver | `tests/shard6/test_unit_solver_interface.py`, `tests/shard5/test_solver_options.py`, `tests/shard6/test_global_sqp.py`, `tests/shard1/test_acados_interface.py` |
| Solution, plots | `optimization/solution/solution.py`, `gui/plot.py` | `tests/shard5/test_solution.py`, `tests/shard1/test_simulate.py`, `tests/shard1/test_plot.py` |
| Stochastic OCP | `optimization/stochastic_optimal_control_program.py`, `optimization/problem_type.py` | `tests/shard5/test_global_stochastic_collocation.py`, `tests/shard6/test_global_stochastic_except_collocation.py` |
| Variational OCP | `optimization/variational_optimal_control_program.py`, `models/biorbd/variational_biorbd_model.py` | `tests/shard4/test_variational_biorbd_model.py`, `tests/shard4/test_variational_optimal_control_program.py` |
| Receding horizon (MHE/NMPC) | `optimization/receding_horizon_optimization.py` | `tests/shard1/test_mhe.py`, `tests/shard1/test_global_mhe.py`, `tests/shard1/test_global_nmpc_final.py` |
| New or changed example | the example and its `prepare_ocp` | `grep -rn "<module name>" tests` to find the tests that import it; `tests/shard4/test_prepare_all_examples.py` if it is listed there |

## Scientific invariants

- CasADi type (`SX`, `MX`, `DM`), shape and sparsity carry meaning; do not convert or reshape casually.
- Shooting and collocation use different grids (collocation points, `duplicate_starting_point`); controls may be
  constant or continuous (`ControlType`). Check sizes per node, not per phase.
- Scaling (`VariableScaling`, target and bound scaling) must stay in clearly identified scaled or unscaled
  coordinates (`Solution` accessors take `scaled=`).
- Units, signs, frames, DoF ordering, quaternion/root conventions, contacts and external forces must be checked
  explicitly against the model.
- Solver interfaces do not share the same notion of status, convergence, multipliers, warm start or option support.

## Tests and examples

- `tests/shard1` .. `tests/shard6`: one suite split for CI time, not by package; 75 test files, very uneven
  (shard2 is a single large file, shard6 has 33). Each shard is a CI job.
- Helpers: `tests/utils.py` (`TestUtils`: `assert_objective_value`, `compare_ocp_to_solve`, `deep_assert`,
  `simulate`, `mx_to_array`, `bioptim_folder`, `load_module`, ...) and `tests/test_utils_ocp.py` (a shared
  `prepare_ocp`). No `conftest.py`, no pytest config, no custom markers. Run from the repository root
  (tests import `tests.utils`).
- Typical pattern: import the example inside the test (`from bioptim.examples.getting_started import basic_ocp as
  ocp_module`), call its `prepare_ocp`, solve with `Solver.IPOPT()`, then `TestUtils.assert_objective_value(sol, expected)` and `npt.assert_almost_equal` on
  `sol.decision_states()[...]`. Tests are heavily parametrized over `phase_dynamics`, `ode_solver`, `use_sx`; invalid
  combinations are pruned with `pytest.skip`.
- `tests/shard4/test_prepare_all_examples.py` builds the `prepare_ocp` of many examples (no solve), but not all:
  some tests are `pass` stubs and several folders (`holonomic_constraints/`, `fatigue/`, `stochastic_optimal_control/`,
  `sqp_method/`, `biomechanics/`, ...) are not covered there. Check that your example is called by some test.
- `tests/shard1/test__run_examples.py` only tests the GUI launcher (example list, search); `test_run_examples` itself
  returns immediately on Linux.
- Plot reference images: `tests/plot_reference_images/`, used by `tests/shard1/test_plots.py`
  (`@pytest.mark.mpl_image_compare`). CI does not pass `--mpl`, so images are not compared there; see
  `docs/generating_plot_reference_images.md`.
- ACADOS tests (`tests/shard1/test_acados_interface.py`, ACADOS case in `tests/shard1/test_mhe.py`) need ACADOS,
  installed in CI only on Linux and macOS shard 1.
- CI (`.github/workflows/`): `isblacked.yml` (black check), `run_tests_linux.yml` (6 shards with coverage),
  `run_tests_osx_win.yml` (macOS and Windows, 6 shards each). The test workflows skip draft PRs.
- `bioptim/examples/`: start with `getting_started/` (`basic_ocp.py`, `custom_*.py`, `example_multiphase.py`, ...).
  GUI launcher: `python -m bioptim.examples`. The root `examples/` folder only holds a README.

## Conventions

- Format: `black . -l120 --exclude "external/*"` (CI installs an unpinned black). NumPy-style docstrings.
- Type aliases (`CX`, `Int`, `Str`, `Bool`, `NpArray`, ...) come from `misc/parameters_types.py`; use them in
  annotations (most of the package already does).
- Variable names that can be plural are plural (`docs/contributing.md`).
- PR titles: `[WIP]` while in progress, `[RTR]` ready to review, `[RTM]` ready to merge (`docs/contributing.md`;
  for agent PRs, `[RTM]` is set by a reviewer, see `agent-context/RULES.md`).
- `external/` (`acados`, `biomechanics_models`) holds git submodules; do not edit it.

## Things worth knowing

- `OdeSolver.TRAPEZOIDAL` reports `is_direct_shooting=True` and `is_direct_collocation=False`; it is used mainly by the
  stochastic `SocpType.TRAPEZOIDAL_IMPLICIT` problems and rejects `ControlType.CONSTANT`.
  `SocpType.TRAPEZOIDAL_EXPLICIT` still exists but raises in `stochastic_dynamics.py`.
- `OdeSolver.CVODES` exists but `initialize_integrator` raises `NotImplementedError`.
- File names with typos that are imported elsewhere; do not rename them casually: `interfaces/fratrop_options.py`
  (holds `FATROP`) and `limits/phase_transtion_factory.py`.
- `OrderingStrategy` values are functions, so it has no real enum members (`list(OrderingStrategy)` is empty);
  `VectorLayout` handles them as callables.
- `examples/getting_started/custom_dynamics.py` is a stub; the real example is `examples/toy_examples/custom_model/`.
- Python requirement lives in `pyproject.toml` (`requires-python >= 3.10`); there is no `setup.py`. In practice
  (2026-10-15), `import bioptim` fails with `NameError: GeneralizedCoordinates` before Python 3.14 (checked on
  3.11 to 3.13) and works on 3.14: `models/protocols/holonomic_biomodel.py` uses a `TYPE_CHECKING`-only import in an
  annotation evaluated at class creation. Use a Python 3.14 environment until this is fixed, and remove this note
  once it is.
