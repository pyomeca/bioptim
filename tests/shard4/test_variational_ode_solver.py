"""
Tests of OdeSolver.VARIATIONAL, that owns the discrete Euler-Lagrange functions of the variational integrator.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpy.testing as npt
import pytest
from bioptim import (
    BoundsList,
    ControlType,
    QuadratureRule,
    DynamicsOptions,
    InterpolationType,
    MultinodeConstraintFcn,
    MultinodeConstraintList,
    OdeSolver,
    OptimalControlProgram,
    Shooting,
    SolutionMerge,
    Solver,
    VariationalOptimalControlProgram,
    VariationalTorqueBiorbdModel,
)

from ..utils import TestUtils


def _prepare_variational_ocp(n_shooting: int = 5, use_sx: bool = False, **kwargs) -> VariationalOptimalControlProgram:
    bio_model = VariationalTorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")
    n_q = bio_model.nb_q

    q_bounds = BoundsList()
    q_bounds["q"] = bio_model.bounds_from_ranges("q")
    q_bounds["q"][:, [0, -1]] = 0
    q_bounds["q"][1, -1] = 3.14

    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_q, [100] * n_q
    u_bounds["tau"][1, :] = 0

    qdot_bounds = BoundsList()
    qdot_bounds.add("qdot_start", min_bound=[0] * n_q, max_bound=[0] * n_q, interpolation=InterpolationType.CONSTANT)
    qdot_bounds.add("qdot_end", min_bound=[0] * n_q, max_bound=[0] * n_q, interpolation=InterpolationType.CONSTANT)

    return VariationalOptimalControlProgram(
        bio_model,
        n_shooting=n_shooting,
        final_time=1,
        q_bounds=q_bounds,
        u_bounds=u_bounds,
        qdot_bounds=qdot_bounds,
        use_sx=use_sx,
        **kwargs,
    )


@pytest.mark.parametrize("use_sx", [False, True])
def test_variational_functions_are_owned_by_the_ode_solver(use_sx):
    nlp = _prepare_variational_ocp(use_sx=use_sx).nlp[0]

    assert nlp.variational_three_nodes_func.name() == "ThreeNodesIntegration"
    assert nlp.variational_first_node_func.name() == "TwoFirstNodesIntegration"
    assert nlp.variational_last_node_func.name() == "TwoLastNodesIntegration"

    # dt, q at nodes i-1, i, i+1 and tau at the same nodes
    assert nlp.variational_three_nodes_func.n_in() == 7
    assert nlp.variational_first_node_func.n_in() == 6
    assert nlp.variational_last_node_func.n_in() == 6
    assert nlp.variational_three_nodes_func.size_out(0) == (nlp.model.nb_q, 1)


def test_variational_functions_do_not_hijack_the_dynamics_defects():
    nlp = _prepare_variational_ocp().nlp[0]

    assert nlp.dynamics_defects_func is None
    assert nlp.extra_dynamics_defects_func == []


def test_variational_has_no_continuous_time_dynamics():
    nlp = _prepare_variational_ocp().nlp[0]

    assert not nlp.model.has_continuous_dynamics
    assert nlp.dynamics_func is None
    assert nlp.extra_dynamics_func == []


def test_variational_model_cannot_be_used_with_an_integrating_ode_solver():
    bio_model = VariationalTorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")

    with pytest.raises(ValueError, match="no continuous-time dynamics"):
        OptimalControlProgram(
            bio_model, n_shooting=5, phase_time=1, dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4())
        )


def test_variational_ode_solver_is_not_a_runge_kutta():
    ode_solver = OdeSolver.VARIATIONAL()

    assert not isinstance(ode_solver, OdeSolver.RK4)
    assert not hasattr(ode_solver, "n_integration_steps")
    assert ode_solver.is_direct_shooting and not ode_solver.is_direct_collocation
    assert "variational" in str(ode_solver)


@pytest.fixture(scope="module")
def solution():
    ocp = _prepare_variational_ocp(n_shooting=5)
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(100)
    solver.set_print_level(0)
    return ocp.solve(solver)


@pytest.mark.parametrize("shooting_type", [Shooting.SINGLE, Shooting.MULTIPLE, Shooting.SINGLE_DISCONTINUOUS_PHASE])
def test_variational_solution_cannot_be_integrated(solution, shooting_type):
    with pytest.raises(NotImplementedError, match="variational integrator"):
        solution.integrate(shooting_type=shooting_type)


def test_variational_stepwise_outputs_are_the_decision_nodes(solution):
    decision_states = solution.decision_states(to_merge=SolutionMerge.NODES)
    stepwise_states = solution.stepwise_states(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(stepwise_states["q"], decision_states["q"])
    npt.assert_almost_equal(
        solution.stepwise_time(to_merge=SolutionMerge.NODES), solution.decision_time(to_merge=SolutionMerge.NODES)
    )
    assert decision_states["q"].shape == (2, 6)


def test_variational_solution_outputs_still_work(solution):
    solution.print_cost()
    assert solution.parameters["qdot_start"].shape == (2,)
    solution.ocp.print(to_console=False, to_graph=False)
    solution.graphs(show_now=False, automatically_organize=False)


def test_variational_continuity_is_declared_by_the_ode_solver():
    n_shooting = 5
    nlp = _prepare_variational_ocp(n_shooting=n_shooting).nlp[0]

    names = [g.name.split("_Multinode")[0] for g in nlp.g_internal if g]

    assert names == (
        ["initial_node_equations"] + ["three_nodes_equations"] * (n_shooting - 1) + ["final_node_equations"]
    )
    assert not nlp.dynamics_type.skip_continuity


def test_variational_continuity_leaves_the_user_multinode_constraints_first():
    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(MultinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 0), nodes=(0, 5), key="q")
    bio_model = VariationalTorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")
    q_bounds = BoundsList()
    q_bounds["q"] = bio_model.bounds_from_ranges("q")
    qdot_bounds = BoundsList()
    for key in ("qdot_start", "qdot_end"):
        qdot_bounds.add(key, min_bound=[0, 0], max_bound=[0, 0], interpolation=InterpolationType.CONSTANT)

    ocp = VariationalOptimalControlProgram(
        bio_model,
        n_shooting=5,
        final_time=1,
        q_bounds=q_bounds,
        qdot_bounds=qdot_bounds,
        multinode_constraints=multinode_constraints,
    )

    names = [g.name.split("_Multinode")[0] for g in ocp.nlp[0].g_internal if g]
    assert names[0] == "STATES_EQUALITY"
    assert len(names) == 1 + 4 + 2


@pytest.mark.parametrize("control_type", [ControlType.CONSTANT, ControlType.NONE])
def test_variational_rejects_control_types_without_last_node(control_type):
    bio_model = VariationalTorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")

    with pytest.raises(NotImplementedError, match="not supported by OdeSolver.VARIATIONAL"):
        OptimalControlProgram(
            bio_model,
            n_shooting=5,
            phase_time=1,
            dynamics=DynamicsOptions(ode_solver=OdeSolver.VARIATIONAL()),
            control_type=control_type,
        )


def test_variational_accepts_constant_controls_with_last_node():
    bio_model = VariationalTorqueBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod")

    ocp = OptimalControlProgram(
        bio_model,
        n_shooting=5,
        phase_time=1,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.VARIATIONAL()),
        control_type=ControlType.CONSTANT_WITH_LAST_NODE,
    )

    assert ocp.nlp[0].variational_three_nodes_func is not None


def test_variational_ode_solver_options_are_validated():
    with pytest.raises(TypeError, match="discrete_approximation should be a QuadratureRule"):
        OdeSolver.VARIATIONAL(discrete_approximation="trapezoidal")
    with pytest.raises(TypeError, match="control_discrete_approximation should be a QuadratureRule"):
        OdeSolver.VARIATIONAL(control_discrete_approximation=1)
    with pytest.raises(TypeError, match="ode_solver must be of type OdeSolver.VARIATIONAL"):
        _prepare_variational_ocp(ode_solver=OdeSolver.RK4())


def _three_nodes_residual(ode_solver: OdeSolver.VARIATIONAL | None) -> np.ndarray:
    kwargs = {} if ode_solver is None else {"ode_solver": ode_solver}
    nlp = _prepare_variational_ocp(**kwargs).nlp[0]
    q_prev, q_cur, q_next = [0.1, 0.2], [0.3, 0.5], [0.8, 0.9]
    tau_prev, tau_cur, tau_next = [1.0, 0.0], [4.0, 0.0], [9.0, 0.0]
    return np.array(nlp.variational_three_nodes_func(0.1, q_prev, q_cur, q_next, tau_prev, tau_cur, tau_next))


def test_variational_default_options_are_trapezoidal_lagrangian_and_midpoint_forcing():
    default = OdeSolver.VARIATIONAL()

    assert default.discrete_approximation == QuadratureRule.TRAPEZOIDAL
    assert default.control_discrete_approximation == QuadratureRule.MIDPOINT
    npt.assert_almost_equal(
        _three_nodes_residual(None),
        _three_nodes_residual(
            OdeSolver.VARIATIONAL(
                discrete_approximation=QuadratureRule.TRAPEZOIDAL,
                control_discrete_approximation=QuadratureRule.MIDPOINT,
            )
        ),
    )


def test_variational_quadrature_options_reach_the_discrete_equations():
    default = _three_nodes_residual(None)

    lagrangian = _three_nodes_residual(OdeSolver.VARIATIONAL(discrete_approximation=QuadratureRule.MIDPOINT))
    forcing_left = _three_nodes_residual(
        OdeSolver.VARIATIONAL(control_discrete_approximation=QuadratureRule.RECTANGLE_LEFT)
    )
    forcing_right = _three_nodes_residual(
        OdeSolver.VARIATIONAL(control_discrete_approximation=QuadratureRule.RECTANGLE_RIGHT)
    )

    assert not np.allclose(lagrangian, default)
    assert not np.allclose(forcing_left, default)
    assert not np.allclose(forcing_right, forcing_left)
    # The forcing only enters the residual through the controls: left uses (prev, cur) and right uses (cur, next), so
    # they differ by dt / 2 * (tau_next - tau_prev) = 0.1 / 2 * (9 - 1)
    npt.assert_almost_equal((forcing_right - forcing_left).squeeze(), [0.4, 0.0])
