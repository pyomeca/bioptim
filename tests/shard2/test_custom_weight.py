from bioptim import (
    TorqueBiorbdModel,
    OdeSolver,
    OdeSolverBase,
    ControlType,
    QuadratureRule,
    OptimalControlProgram,
    Objective,
    ObjectiveFcn,
    DynamicsOptions,
    BoundsList,
    Solver,
    PhaseDynamics,
    SolutionMerge,
    InterpolationType,
    Node,
    ObjectiveWeight,
)
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils
from bioptim.examples.getting_started import custom_objective_weights as ocp_module


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    interpolation_type: InterpolationType,
    node: Node,
    control_type: ControlType,
    objective: str,
    integration_rule: QuadratureRule = QuadratureRule.DEFAULT,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    integration_rule: QuadratureRule
        The integration rule to use
    control_type: ControlType
        The type of control to use (constant or linear)
    objective: str
        The objective to minimize (torque or power)
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    final_time = 1
    bio_model = TorqueBiorbdModel(biorbd_model_path)

    # n_nodes
    if node == Node.START:
        n_nodes = 1
    elif node == Node.INTERMEDIATES:
        n_nodes = n_shooting - 2
    elif node == Node.ALL_SHOOTING:
        n_nodes = n_shooting
    else:
        n_nodes = len(node)

    # ObjectiveWeight
    if interpolation_type == InterpolationType.CONSTANT:
        weight = [1]
        weight = ObjectiveWeight(weight, interpolation=InterpolationType.CONSTANT)
    elif interpolation_type == InterpolationType.LINEAR:
        weight = [0, 1]
        weight = ObjectiveWeight(weight, interpolation=InterpolationType.LINEAR)
    elif interpolation_type == InterpolationType.EACH_FRAME:
        # It emulates a Linear interpolation
        weight = np.linspace(0, 1, n_nodes)
        weight = ObjectiveWeight(weight, interpolation=InterpolationType.EACH_FRAME)
    elif interpolation_type == InterpolationType.SPLINE:
        spline_time = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        spline_points = np.random.random((5,)) * (-10) - 5
        weight = ObjectiveWeight(spline_points, interpolation=InterpolationType.SPLINE, t=spline_time)
    elif interpolation_type == InterpolationType.CUSTOM:
        # The custom functions refer to the one at the beginning of the file.
        # For this particular instance, it emulates a Linear interpolation
        extra_params = {"n_nodes": n_nodes}
        weight = ObjectiveWeight(ocp_module.custom_weight, interpolation=InterpolationType.CUSTOM, **extra_params)
    else:
        raise NotImplementedError("Not implemented yet")

    # Add objective functions
    if objective == "lagrange":
        objective_functions = Objective(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            node=node,
            weight=weight,
            integration_rule=integration_rule,
        )
    elif objective == "mayer":
        objective_functions = Objective(
            ObjectiveFcn.Mayer.MINIMIZE_CONTROL,
            key="tau",
            node=node,
            weight=weight,
            integration_rule=integration_rule,
        )
    else:
        raise ValueError("Wrong objective")

    # Dynamics
    dynamics = DynamicsOptions(ode_solver=ode_solver, expand_dynamics=True, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0  # Prevent the model from actively rotate

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        phase_time=final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=True,
        n_threads=1,
        control_type=control_type,
    )


@pytest.mark.parametrize(
    "phase_dynamics",
    [
        PhaseDynamics.SHARED_DURING_THE_PHASE,
        PhaseDynamics.ONE_PER_NODE,
    ],
)
@pytest.mark.parametrize(
    "objective",
    [
        "lagrange",
        "mayer",
    ],
)
@pytest.mark.parametrize(
    "control_type",
    [
        ControlType.CONSTANT,
        ControlType.CONSTANT_WITH_LAST_NODE,
        ControlType.LINEAR_CONTINUOUS,
    ],
)
@pytest.mark.parametrize(
    "interpolation_type",
    [
        InterpolationType.CONSTANT,
        InterpolationType.LINEAR,
        InterpolationType.EACH_FRAME,
        InterpolationType.SPLINE,
        InterpolationType.CUSTOM,
    ],
)
@pytest.mark.parametrize(
    "node",
    [
        Node.START,
        Node.INTERMEDIATES,
        Node.ALL_SHOOTING,
        [0, 4, 6, 7],
    ],
)
def test_pendulum(control_type, interpolation_type, node, objective, phase_dynamics):

    bioptim_folder = TestUtils.module_folder(ocp_module)
    n_shooting = 30
    np.random.seed(42)  # For reproducibility of spline

    # Test the errors first
    if objective == "lagrange" and (node == Node.START or node == Node.INTERMEDIATES or isinstance(node, list)):
        with pytest.raises(
            RuntimeError, match="Lagrange objective are for Node.ALL_SHOOTING or Node.ALL, did you mean Mayer?"
        ):
            ocp = prepare_ocp(
                biorbd_model_path=bioptim_folder + "/../models/pendulum.bioMod",
                n_shooting=n_shooting,
                objective=objective,
                interpolation_type=interpolation_type,
                node=node,
                control_type=control_type,
                phase_dynamics=phase_dynamics,
            )
        return

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../models/pendulum.bioMod",
        n_shooting=n_shooting,
        objective=objective,
        interpolation_type=interpolation_type,
        node=node,
        control_type=control_type,
        phase_dynamics=phase_dynamics,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    j_printed = TestUtils.sum_cost_function_output(sol)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    tau = controls["tau"]
    ntau = tau.shape[0]
    dt = sol.t_span()[0][-1]

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    if interpolation_type == InterpolationType.CONSTANT:
        if node == Node.START:
            value = tau[:, 0]
            if objective == "mayer":
                npt.assert_almost_equal(f[0, 0], np.sum(value**2))
                npt.assert_almost_equal(j_printed, np.sum(value**2))
            # else raise error above
        elif node == Node.INTERMEDIATES:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = tau[:, 1:-1]
                    npt.assert_almost_equal(f[0, 0], np.sum(value**2))
                    npt.assert_almost_equal(j_printed, np.sum(value**2))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = tau[:, 1:-2]
                    npt.assert_almost_equal(f[0, 0], np.sum(value**2))
                    npt.assert_almost_equal(j_printed, np.sum(value**2))
                else:
                    value = tau[:, 1:-4:2]
                    npt.assert_almost_equal(f[0, 0], np.sum(value**2))
                    npt.assert_almost_equal(j_printed, np.sum(value**2))
            # else raise error above
        elif node == Node.ALL_SHOOTING:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau**2))
                    npt.assert_almost_equal(j_printed, np.sum(tau**2))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau**2))
                    npt.assert_almost_equal(j_printed, np.sum(tau**2))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau[:, 0:-1:2] ** 2))
                    npt.assert_almost_equal(j_printed, np.sum(tau[:, 0:-1:2] ** 2))
            else:
                if control_type == ControlType.CONSTANT:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau**2 * dt))
                    npt.assert_almost_equal(j_printed, np.sum(tau**2 * dt))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau**2 * dt))
                    npt.assert_almost_equal(j_printed, np.sum(tau**2 * dt))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau[:, 0:-1:2] ** 2 * dt))
                    npt.assert_almost_equal(j_printed, np.sum(tau[:, 0:-1:2] ** 2 * dt))
        else:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau[:, node] ** 2))
                    npt.assert_almost_equal(j_printed, np.sum(tau[:, node] ** 2))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau[:, node] ** 2))
                    npt.assert_almost_equal(j_printed, np.sum(tau[:, node] ** 2))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    npt.assert_almost_equal(f[0, 0], np.sum(tau[:, np.array(node) * 2] ** 2))
                    npt.assert_almost_equal(j_printed, np.sum(tau[:, np.array(node) * 2] ** 2))
            # else raise error above
    elif (
        interpolation_type == InterpolationType.LINEAR
        or interpolation_type == InterpolationType.CUSTOM
        or interpolation_type == InterpolationType.EACH_FRAME
    ):
        if node == Node.START:
            if objective == "mayer":
                npt.assert_almost_equal(f[0, 0], 0)
                npt.assert_almost_equal(j_printed, 0)
            # else raise error above
        elif node == Node.INTERMEDIATES:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = np.zeros((ntau, n_shooting - 2))
                    for i in range(ntau):
                        value[i, :] = tau[i, 1:-1] ** 2 * np.linspace(0, 1, n_shooting - 2)
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = np.zeros((ntau, n_shooting - 2))
                    for i in range(ntau):
                        value[i, :] = tau[i, 1:-2] ** 2 * np.linspace(0, 1, n_shooting - 2)
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = np.zeros((ntau, n_shooting - 2))
                    for i in range(ntau):
                        value[i, :] = tau[i, 1:-4:2] ** 2 * np.linspace(0, 1, n_shooting - 2)
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
            # else raise error above
        elif node == Node.ALL_SHOOTING:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, :] ** 2 * np.linspace(0, 1, n_shooting)
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, :-1] ** 2 * np.linspace(0, 1, n_shooting)
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, 0:-2:2] ** 2 * np.linspace(0, 1, n_shooting)
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
            else:
                if control_type == ControlType.CONSTANT:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, :] ** 2 * np.linspace(0, 1, n_shooting)
                    npt.assert_almost_equal(f[0, 0], np.sum(value * dt))
                    npt.assert_almost_equal(j_printed, np.sum(value * dt))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, :-1] ** 2 * np.linspace(0, 1, n_shooting)
                    npt.assert_almost_equal(f[0, 0], np.sum(value * dt))
                    npt.assert_almost_equal(j_printed, np.sum(value * dt))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, 0:-1:2] ** 2 * np.linspace(0, 1, n_shooting)
                    npt.assert_almost_equal(f[0, 0], np.sum((value * dt)))
                    npt.assert_almost_equal(j_printed, np.sum((value * dt)))
        else:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = np.zeros((ntau, len(node)))
                    for i in range(ntau):
                        value[i, :] = tau[i, node] ** 2 * np.linspace(0, 1, len(node))
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = np.zeros((ntau, len(node)))
                    for i in range(ntau):
                        value[i, :] = tau[i, node] ** 2 * np.linspace(0, 1, len(node))
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = np.zeros((ntau, len(node)))
                    for i in range(ntau):
                        value[i, :] = tau[i, np.array(node) * 2] ** 2 * np.linspace(0, 1, len(node))
                    npt.assert_almost_equal(f[0, 0], np.sum(value))
                    npt.assert_almost_equal(j_printed, np.sum(value))
            # else raise error above
    elif interpolation_type == InterpolationType.SPLINE:
        # Testing values because too complicated for my small brain
        if node == Node.START:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    npt.assert_almost_equal(f[0, 0], -4969.650216032278)
                    npt.assert_almost_equal(j_printed, -4969.650216032278)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    npt.assert_almost_equal(f[0, 0], -4969.650216032278)
                    npt.assert_almost_equal(j_printed, -4969.650216032278)
                else:
                    npt.assert_almost_equal(f[0, 0], -1368.4151423970898)
                    npt.assert_almost_equal(j_printed, -1368.4151423970898)
            # else raise error above
        elif node == Node.INTERMEDIATES:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    npt.assert_almost_equal(f[0, 0], -66855.99375991023)
                    npt.assert_almost_equal(j_printed, -66855.99375991023)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    npt.assert_almost_equal(f[0, 0], -66855.99375991023)
                    npt.assert_almost_equal(j_printed, -66855.99375991023)
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    npt.assert_almost_equal(f[0, 0], -62189.06584139419)
                    npt.assert_almost_equal(j_printed, -62189.06584139419)
            # else raise error above
        elif node == Node.ALL_SHOOTING:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    npt.assert_almost_equal(f[0, 0], -67592.34632346843)
                    npt.assert_almost_equal(j_printed, -67592.34632346843)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    npt.assert_almost_equal(f[0, 0], -67592.34632346843)
                    npt.assert_almost_equal(j_printed, -67592.34632346843)
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    npt.assert_almost_equal(f[0, 0], -68421.45149762284)
                    npt.assert_almost_equal(j_printed, -68421.45149762284)
            else:
                if control_type == ControlType.CONSTANT:
                    npt.assert_almost_equal(f[0, 0], -2069.98362328184)
                    npt.assert_almost_equal(j_printed, -2069.98362328184)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    npt.assert_almost_equal(f[0, 0], -2069.98362328184)
                    npt.assert_almost_equal(j_printed, -2069.98362328184)
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    npt.assert_almost_equal(f[0, 0], -3315.0557682774315)
                    npt.assert_almost_equal(j_printed, -3315.0557682774315)
        else:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    npt.assert_almost_equal(f[0, 0], -14340.247184500638)
                    npt.assert_almost_equal(j_printed, -14340.247184500638)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    npt.assert_almost_equal(f[0, 0], -14340.247184500638)
                    npt.assert_almost_equal(j_printed, -14340.247184500638)
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    npt.assert_almost_equal(f[0, 0], -14746.563287583314)
                    npt.assert_almost_equal(j_printed, -14746.563287583314)
            # else raise error above


@pytest.mark.parametrize(
    "control_type",
    [
        ControlType.CONSTANT,
        ControlType.CONSTANT_WITH_LAST_NODE,
        ControlType.LINEAR_CONTINUOUS,
    ],
)
@pytest.mark.parametrize(
    "interpolation_type",
    [
        InterpolationType.CONSTANT,
        InterpolationType.LINEAR,
        InterpolationType.EACH_FRAME,
        InterpolationType.SPLINE,
        InterpolationType.CUSTOM,
    ],
)
@pytest.mark.parametrize(
    "integration_rule",
    [QuadratureRule.APPROXIMATE_TRAPEZOIDAL, QuadratureRule.TRAPEZOIDAL],
)
def test_pendulum_integration_rule(control_type, interpolation_type, integration_rule):

    bioptim_folder = TestUtils.module_folder(ocp_module)
    n_shooting = 30
    np.random.seed(42)  # For reproducibility of spline

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=n_shooting,
        objective="lagrange",
        interpolation_type=interpolation_type,
        integration_rule=integration_rule,
        node=Node.ALL_SHOOTING,
        control_type=control_type,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    j_printed = TestUtils.sum_cost_function_output(sol)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    tau = controls["tau"]
    ntau = tau.shape[0]
    dt = sol.t_span()[0][-1]

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    if interpolation_type == InterpolationType.CONSTANT:
        if control_type == ControlType.CONSTANT or control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            npt.assert_almost_equal(f[0, 0], np.sum(tau**2 * dt))
            npt.assert_almost_equal(j_printed, np.sum(tau**2 * dt))
        else:
            if integration_rule == QuadratureRule.TRAPEZOIDAL:
                out = 0
                for i in range(round((tau[0, :].shape[0] - 1) / 2)):
                    out += (tau[0, i * 2] ** 2 + tau[0, i * 2 + 1] ** 2) / 2 * dt
                npt.assert_almost_equal(f[0, 0], out)
                npt.assert_almost_equal(j_printed, out)
            elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                npt.assert_almost_equal(f[0, 0], np.sum(tau[0, 0:-1:2] ** 2 * dt))
                npt.assert_almost_equal(j_printed, np.sum(tau[0, 0:-1:2] ** 2 * dt))
    elif (
        interpolation_type == InterpolationType.LINEAR
        or interpolation_type == InterpolationType.CUSTOM
        or interpolation_type == InterpolationType.EACH_FRAME
    ):
        if control_type == ControlType.CONSTANT:
            value = np.zeros((ntau, n_shooting))
            for i in range(ntau):
                value[i, :] = tau[i, :] ** 2 * np.linspace(0, 1, n_shooting)
            npt.assert_almost_equal(f[0, 0], np.sum(value * dt))
            npt.assert_almost_equal(j_printed, np.sum(value * dt))
        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            value = np.zeros((ntau, n_shooting))
            for i in range(ntau):
                value[i, :] = tau[i, :-1] ** 2 * np.linspace(0, 1, n_shooting)
            npt.assert_almost_equal(f[0, 0], np.sum(value * dt))
            npt.assert_almost_equal(j_printed, np.sum(value * dt))
        else:
            if integration_rule == QuadratureRule.TRAPEZOIDAL:
                out = 0
                weight = np.linspace(0, 1, n_shooting)
                for i in range(round((tau[0, :].shape[0] - 1) / 2)):
                    out += (tau[0, i * 2] ** 2 + tau[0, i * 2 + 1] ** 2) / 2 * dt * weight[i]
                npt.assert_almost_equal(f[0, 0], out)
                npt.assert_almost_equal(j_printed, out)
            elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                npt.assert_almost_equal(f[0, 0], 9.55437614886297)
                npt.assert_almost_equal(j_printed, 9.55437614886297)
    elif interpolation_type == InterpolationType.SPLINE:
        # Testing values because too complicated for my small brain
        if control_type == ControlType.CONSTANT or control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            npt.assert_almost_equal(f[0, 0], -2069.98362328184)
            npt.assert_almost_equal(j_printed, -2069.98362328184)
        else:
            if integration_rule == QuadratureRule.TRAPEZOIDAL:
                npt.assert_almost_equal(f[0, 0], -4120.763073947789)
                npt.assert_almost_equal(j_printed, -4120.763073947789)
            elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                npt.assert_almost_equal(f[0, 0], -3315.0557682774315)
                npt.assert_almost_equal(j_printed, -3315.0557682774315)


def test_bad_weights():

    bioptim_folder = TestUtils.module_folder(ocp_module)

    with pytest.raises(
        ValueError,
        match="The interpolation type ALL_POINTS is not allowed for Weight since the objective is evaluated only at the node and not at the collocation points. Use EACH_FRAME instead.",
    ):
        ObjectiveWeight([0, 1], interpolation=InterpolationType.ALL_POINTS)

    with pytest.raises(
        RuntimeError,
        match=r"Invalid number of column for InterpolationType.CONSTANT \(ncols = 2\), the expected number of column is 1",
    ):
        ObjectiveWeight([0, 1], interpolation=InterpolationType.CONSTANT)

    with pytest.raises(
        RuntimeError,
        match=r"Invalid number of column for InterpolationType.LINEAR \(ncols = 3\), the expected number of column is 2",
    ):
        ObjectiveWeight([0, 1, 2], interpolation=InterpolationType.LINEAR)

    with pytest.raises(RuntimeError, match=r"Value for InterpolationType.SPLINE must have at least 2 columns"):
        ObjectiveWeight([0], interpolation=InterpolationType.SPLINE)

    with pytest.raises(RuntimeError, match=r"Spline necessitate a time vector"):
        ObjectiveWeight([0, 1, 2, 3, 4, 5], interpolation=InterpolationType.SPLINE)

    with pytest.raises(
        RuntimeError, match=r"Spline necessitate a time vector which as the same length as column of data"
    ):
        ObjectiveWeight([0, 1, 2, 3, 4], interpolation=InterpolationType.SPLINE, t=[0, 1, 2, 3, 4, 5])

    with pytest.raises(RuntimeError, match=r"InterpolationType is not implemented yet"):
        ObjectiveWeight([0, 1, 2, 3, 4], interpolation="bad_type")

    # Finally set a weight
    weight = ObjectiveWeight([0, 1, 2, 3, 4], interpolation=InterpolationType.EACH_FRAME)

    with pytest.raises(
        RuntimeError, match=r"check_and_adjust_dimensions must be called at least once before evaluating at"
    ):
        weight.evaluate_at(0, 1)

    with pytest.raises(
        RuntimeError,
        match=r"Invalid number of column for InterpolationType.EACH_FRAME \(ncols = 5\), the expected number of column is 3 for coucou.",
    ):
        weight.check_and_adjust_dimensions(n_nodes=3, element_name="coucou")

    weight.check_and_adjust_dimensions(n_nodes=5, element_name="coucou")
    with pytest.raises(RuntimeError, match=r"index too high for evaluate at"):
        weight.evaluate_at(10, 1)
