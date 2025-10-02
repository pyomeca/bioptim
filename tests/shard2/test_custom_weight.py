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
    ConstraintWeight,
    ConstraintList,
    ConstraintFcn,
)
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils
from bioptim.examples.toy_examples.feature_examples import custom_objective_weights as objective_ocp_module
from bioptim.examples.toy_examples.feature_examples import custom_constraint_weights as constraint_ocp_module
from bioptim.limits.weight import Weight


def get_nb_nodes(node, n_shooting):
    """
    get the number of nodes depending on the type of node selected
    """
    if node == Node.START:
        n_nodes = 1
    elif node == Node.INTERMEDIATES:
        n_nodes = n_shooting - 2
    elif node == Node.ALL_SHOOTING:
        n_nodes = n_shooting
    else:
        n_nodes = len(node)
    return n_nodes


def get_weight(interpolation_type, n_nodes, final_time, weight_type="objective"):
    """
    Get the weight depending on the interpolation type and the type (objective or constraint).
    """
    if interpolation_type == InterpolationType.CONSTANT:
        weight = [1]
        if weight_type == "objective":
            weight = ObjectiveWeight(weight, interpolation=InterpolationType.CONSTANT)
        else:
            weight = ConstraintWeight(weight, interpolation=InterpolationType.CONSTANT)
    elif interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        weight = [0, 1, 2]
        if weight_type == "objective":
            weight = ObjectiveWeight(weight, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
        else:
            weight = ConstraintWeight(weight, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    elif interpolation_type == InterpolationType.LINEAR:
        weight = [0, 1]
        if weight_type == "objective":
            weight = ObjectiveWeight(weight, interpolation=InterpolationType.LINEAR)
        else:
            weight = ConstraintWeight(weight, interpolation=InterpolationType.LINEAR)
    elif interpolation_type == InterpolationType.EACH_FRAME:
        # It emulates a Linear interpolation
        weight = np.linspace(0, 1, n_nodes)
        if weight_type == "objective":
            weight = ObjectiveWeight(weight, interpolation=InterpolationType.EACH_FRAME)
        else:
            weight = ConstraintWeight(weight, interpolation=InterpolationType.EACH_FRAME)
    elif interpolation_type == InterpolationType.SPLINE:
        spline_time = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        spline_points = np.random.random((5,)) * (-10) - 5
        if weight_type == "objective":
            weight = ObjectiveWeight(spline_points, interpolation=InterpolationType.SPLINE, t=spline_time)
        else:
            weight = ConstraintWeight(spline_points, interpolation=InterpolationType.SPLINE, t=spline_time)
    elif interpolation_type == InterpolationType.CUSTOM:
        # The custom functions refer to the one at the beginning of the file.
        # For this particular instance, it emulates a Linear interpolation
        extra_params = {"n_nodes": n_nodes}
        if weight_type == "objective":
            weight = ObjectiveWeight(
                objective_ocp_module.custom_weight, interpolation=InterpolationType.CUSTOM, **extra_params
            )
        else:
            weight = ConstraintWeight(
                constraint_ocp_module.custom_weight, interpolation=InterpolationType.CUSTOM, **extra_params
            )
    else:
        raise NotImplementedError("Not implemented yet")
    return weight


def objective_prepare_ocp(
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

    n_nodes = get_nb_nodes(node, n_shooting)
    weight = get_weight(interpolation_type, n_nodes, final_time, weight_type="objective")

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


def constraint_prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    interpolation_type: InterpolationType,
    node: Node,
    control_type: ControlType,
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
    control_type: ControlType
        The type of control to use (constant or linear)
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

    n_nodes = get_nb_nodes(node, n_shooting)
    weight = get_weight(interpolation_type, n_nodes, final_time, weight_type="constraint")

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=node)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="tau", target=np.ones((2, 1)), node=node, weight=weight)

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
        constraints=constraints,
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
        InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
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
def test_pendulum_objective(control_type, interpolation_type, node, objective, phase_dynamics):

    bioptim_folder = TestUtils.bioptim_folder()
    n_shooting = 30
    np.random.seed(42)  # For reproducibility of spline

    # Test the errors first
    if objective == "lagrange" and (node == Node.START or node == Node.INTERMEDIATES or isinstance(node, list)):
        with pytest.raises(
            RuntimeError, match="Lagrange objective are for Node.ALL_SHOOTING or Node.ALL, did you mean Mayer?"
        ):
            ocp = objective_prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
                n_shooting=n_shooting,
                objective=objective,
                interpolation_type=interpolation_type,
                node=node,
                control_type=control_type,
                phase_dynamics=phase_dynamics,
            )
        return

    ocp = objective_prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
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
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    tau = controls["tau"]
    ntau = tau.shape[0]
    dt = sol.t_span()[0][-1]

    # Check objective function value
    if interpolation_type == InterpolationType.CONSTANT:
        if node == Node.START:
            value = tau[:, 0]
            if objective == "mayer":
                TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value**2))
            # else raise error above
        elif node == Node.INTERMEDIATES:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = tau[:, 1:-1]
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value**2))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = tau[:, 1:-2]
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value**2))
                else:
                    value = tau[:, 1:-4:2]
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value**2))
            # else raise error above
        elif node == Node.ALL_SHOOTING:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau**2))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau**2))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau[:, 0:-1:2] ** 2))
            else:
                if control_type == ControlType.CONSTANT:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau**2 * dt))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau**2 * dt))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau[:, 0:-1:2] ** 2 * dt))
        else:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau[:, node] ** 2))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau[:, node] ** 2))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau[:, np.array(node) * 2] ** 2))
            # else raise error above
    elif (
        interpolation_type == InterpolationType.LINEAR
        or interpolation_type == InterpolationType.CUSTOM
        or interpolation_type == InterpolationType.EACH_FRAME
    ):
        if node == Node.START:
            if objective == "mayer":
                TestUtils.assert_objective_value(sol=sol, expected_value=0)
            # else raise error above
        elif node == Node.INTERMEDIATES:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = np.zeros((ntau, n_shooting - 2))
                    for i in range(ntau):
                        value[i, :] = tau[i, 1:-1] ** 2 * np.linspace(0, 1, n_shooting - 2)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = np.zeros((ntau, n_shooting - 2))
                    for i in range(ntau):
                        value[i, :] = tau[i, 1:-2] ** 2 * np.linspace(0, 1, n_shooting - 2)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = np.zeros((ntau, n_shooting - 2))
                    for i in range(ntau):
                        value[i, :] = tau[i, 1:-4:2] ** 2 * np.linspace(0, 1, n_shooting - 2)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
            # else raise error above
        elif node == Node.ALL_SHOOTING:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, :] ** 2 * np.linspace(0, 1, n_shooting)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, :-1] ** 2 * np.linspace(0, 1, n_shooting)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, 0:-2:2] ** 2 * np.linspace(0, 1, n_shooting)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
            else:
                if control_type == ControlType.CONSTANT:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, :] ** 2 * np.linspace(0, 1, n_shooting)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value * dt))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, :-1] ** 2 * np.linspace(0, 1, n_shooting)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value * dt))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = np.zeros((ntau, n_shooting))
                    for i in range(ntau):
                        value[i, :] = tau[i, 0:-1:2] ** 2 * np.linspace(0, 1, n_shooting)
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum((value * dt)))
        else:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = np.zeros((ntau, len(node)))
                    for i in range(ntau):
                        value[i, :] = tau[i, node] ** 2 * np.linspace(0, 1, len(node))
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = np.zeros((ntau, len(node)))
                    for i in range(ntau):
                        value[i, :] = tau[i, node] ** 2 * np.linspace(0, 1, len(node))
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = np.zeros((ntau, len(node)))
                    for i in range(ntau):
                        value[i, :] = tau[i, np.array(node) * 2] ** 2 * np.linspace(0, 1, len(node))
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
            # else raise error above
    elif interpolation_type == InterpolationType.SPLINE:
        # Testing values because too complicated for my small brain
        if node == Node.START:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-4969.650216032278)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-4969.650216032278)
                else:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-1368.4151423970898)
            # else raise error above
        elif node == Node.INTERMEDIATES:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-66855.99375991023)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-66855.99375991023)
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-62189.06584139419)
            # else raise error above
        elif node == Node.ALL_SHOOTING:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-67592.34632346843)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-67592.34632346843)
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-68421.45149762284)
            else:
                if control_type == ControlType.CONSTANT:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-2069.98362328184)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-2069.98362328184)
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-3315.0557682774315)
        else:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-14340.247184500638)
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-14340.247184500638)
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    TestUtils.assert_objective_value(sol=sol, expected_value=-14746.563287583314)
            # else raise error above

    elif interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        if node == Node.START:
            if objective == "mayer":
                TestUtils.assert_objective_value(sol=sol, expected_value=0)
            # else raise error above
        elif node == Node.INTERMEDIATES:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = tau[:, 1:-1] ** 2
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = tau[:, 1:-2] ** 2
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                else:
                    value = tau[:, 1:-4:2] ** 2
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
            # else raise error above
        elif node == Node.ALL_SHOOTING:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = tau**2
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = tau**2
                    value[:, 0] *= 0  # First node has weight 0
                    # TODO: the last control (-1) should be weighted 2 in the line bellow. This is a problem originating
                    #  from a combination of ConstraintWeight and penalty_option.py
                    value[:, -2] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = tau[:, 0:-1:2] ** 2
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
            else:
                if control_type == ControlType.CONSTANT:
                    value = tau**2 * dt
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = tau**2 * dt
                    value[:, 0] *= 0  # First node has weight 0
                    # TODO: the last control (-1) should be weighted 2 in the line bellow. This is a problem originating
                    #  from a combination of ConstraintWeight and penalty_option.py
                    value[:, -2] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = tau[:, 0:-1:2] ** 2 * dt
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
        else:
            if objective == "mayer":
                if control_type == ControlType.CONSTANT:
                    value = tau[:, node] ** 2
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                    value = tau[:, node] ** 2
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    value = tau[:, np.array(node) * 2] ** 2
                    value[:, 0] *= 0  # First node has weight 0
                    value[:, -1] *= 2  # First node has weight 2
                    TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value))
            # else raise error above
    else:
        raise RuntimeError("Should not happen")


@pytest.mark.parametrize(
    "phase_dynamics",
    [
        PhaseDynamics.SHARED_DURING_THE_PHASE,
        PhaseDynamics.ONE_PER_NODE,
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
        InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
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
def test_pendulum_constraint(control_type, interpolation_type, node, phase_dynamics):

    bioptim_folder = TestUtils.bioptim_folder()
    n_shooting = 30
    np.random.seed(42)  # For reproducibility of spline

    ocp = constraint_prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        n_shooting=n_shooting,
        interpolation_type=interpolation_type,
        node=node,
        control_type=control_type,
        phase_dynamics=phase_dynamics,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(5)
    sol = ocp.solve(solver)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    tau = controls["tau"]
    ntau = tau.shape[0]

    # Keep only the user constraint (not the continuity)
    if node == Node.START:
        nodes = [0]
    elif node == Node.INTERMEDIATES:
        nodes = list(range(1, n_shooting - 1))
    elif node == Node.ALL_SHOOTING:
        nodes = list(range(0, n_shooting))
    elif node == [0, 4, 6, 7]:
        nodes = node
    else:
        raise ValueError("Wrong node")
    g_computed = np.ndarray((0, 1))
    starting_node = 0
    for i in range(0, n_shooting):
        increment = 2 * ntau
        if i in nodes:
            g_computed = np.concatenate(
                (
                    g_computed,
                    sol.constraints[starting_node + 2 * ntau : starting_node + 2 * ntau + 2],
                )
            )
            increment += 2
        starting_node += increment
    g_computed = g_computed.reshape(-1)

    # Check constraint function value
    if interpolation_type == InterpolationType.CONSTANT:
        if node == Node.START:
            value = tau[:, 0] - np.ones((ntau,))
            npt.assert_almost_equal(g_computed, value)
        elif node == Node.INTERMEDIATES:
            if control_type == ControlType.CONSTANT:
                value = tau[:, 1:-1] - np.ones((ntau, n_shooting - 2))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, 1:-2] - np.ones((ntau, n_shooting - 2))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = tau[:, 1:-4:2] - np.ones((ntau, n_shooting - 2))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
        elif node == Node.ALL_SHOOTING:
            if control_type == ControlType.CONSTANT:
                value = tau - np.ones((ntau, n_shooting))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, :-1] - np.ones((ntau, n_shooting))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = tau[:, 0:-1:2] - np.ones((ntau, n_shooting))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
        else:
            if control_type == ControlType.CONSTANT:
                value = tau[:, node] - np.ones((ntau, len(node)))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, node] - np.ones((ntau, len(node)))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = tau[:, np.array(node) * 2] - np.ones((ntau, len(node)))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
    elif interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        if node == Node.START:
            value = tau[:, 0] - np.ones((ntau,))
            npt.assert_almost_equal(g_computed, value)
        elif node == Node.INTERMEDIATES:
            if control_type == ControlType.CONSTANT:
                value = tau[:, 1:-1] - np.ones((ntau, n_shooting - 2))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, 1:-2] - np.ones((ntau, n_shooting - 2))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = tau[:, 1:-4:2] - np.ones((ntau, n_shooting - 2))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
        elif node == Node.ALL_SHOOTING:
            if control_type == ControlType.CONSTANT:
                value = tau - np.ones((ntau, n_shooting))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, :-1] - np.ones((ntau, n_shooting))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
                # TODO: @pariterre -> the last node seems to be missing from the constraint
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = tau[:, 0:-1:2] - np.ones((ntau, n_shooting))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
        else:
            if control_type == ControlType.CONSTANT:
                value = tau[:, node] - np.ones((ntau, len(node)))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, node] - np.ones((ntau, len(node)))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = tau[:, np.array(node) * 2] - np.ones((ntau, len(node)))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
    elif (
        interpolation_type == InterpolationType.LINEAR
        or interpolation_type == InterpolationType.CUSTOM
        or interpolation_type == InterpolationType.EACH_FRAME
    ):
        if node == Node.START:
            npt.assert_almost_equal(g_computed, np.zeros((ntau,)))
        elif node == Node.INTERMEDIATES:
            if control_type == ControlType.CONSTANT:
                value = np.zeros((ntau, n_shooting - 2))
                for i in range(ntau):
                    value[i, :] = (tau[i, 1:-1] - np.ones((n_shooting - 2,))) * np.linspace(0, 1, n_shooting - 2)
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = np.zeros((ntau, n_shooting - 2))
                for i in range(ntau):
                    value[i, :] = (tau[i, 1:-2] - np.ones((n_shooting - 2,))) * np.linspace(0, 1, n_shooting - 2)
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = np.zeros((ntau, n_shooting - 2))
                for i in range(ntau):
                    value[i, :] = (tau[i, 1:-4:2] - np.ones((n_shooting - 2,))) * np.linspace(0, 1, n_shooting - 2)
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
        elif node == Node.ALL_SHOOTING:
            if control_type == ControlType.CONSTANT:
                value = np.zeros((ntau, n_shooting))
                for i in range(ntau):
                    value[i, :] = (tau[i, :] - np.ones((n_shooting,))) * np.linspace(0, 1, n_shooting)
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = np.zeros((ntau, n_shooting))
                for i in range(ntau):
                    value[i, :] = (tau[i, :-1] - np.ones((n_shooting,))) * np.linspace(0, 1, n_shooting)
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = np.zeros((ntau, n_shooting))
                for i in range(ntau):
                    value[i, :] = (tau[i, 0:-2:2] - np.ones((n_shooting,))) * np.linspace(0, 1, n_shooting)
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
        else:
            if control_type == ControlType.CONSTANT:
                value = np.zeros((ntau, len(node)))
                for i in range(ntau):
                    value[i, :] = (tau[i, node] - np.ones((len(node),))) * np.linspace(0, 1, len(node))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = np.zeros((ntau, len(node)))
                for i in range(ntau):
                    value[i, :] = (tau[i, node] - np.ones((len(node),))) * np.linspace(0, 1, len(node))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = np.zeros((ntau, len(node)))
                for i in range(ntau):
                    value[i, :] = (tau[i, np.array(node) * 2] - np.ones((len(node),))) * np.linspace(0, 1, len(node))
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                raise NotImplementedError("Not implemented yet")
    elif interpolation_type == InterpolationType.SPLINE:
        # Testing values because too complicated for my small brain
        if node == Node.START:
            if control_type == ControlType.CONSTANT:
                npt.assert_almost_equal(g_computed, np.array([-3.42388421e-09, 1.09865848e01]))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                npt.assert_almost_equal(g_computed, np.array([-3.42388421e-09, 1.09865848e01]))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                npt.assert_almost_equal(g_computed, np.array([2.20904876e-09, 1.09865848e01]))
            else:
                raise NotImplementedError("Not implemented yet")
        elif node == Node.INTERMEDIATES:
            if control_type == ControlType.CONSTANT:
                npt.assert_almost_equal(
                    g_computed[[0, 5, 9, 14]], np.array([0.07896499, 10.1424261, 9.29826735, 0.65405287])
                )
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                npt.assert_almost_equal(
                    g_computed[[1, 6, 10, 15]], np.array([10.98658484, 0.14410163, 0.50068864, 8.03202923])
                )
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                npt.assert_almost_equal(
                    g_computed[[2, 7, 11, 16]],
                    np.array([-4.40919652e-03, 9.72034673e00, 8.87618798e00, 6.50661698e-01]),
                )
            else:
                raise NotImplementedError("Not implemented yet")
        elif node == Node.ALL_SHOOTING:
            if control_type == ControlType.CONSTANT:
                npt.assert_almost_equal(
                    g_computed[[3, 8, 12, 17]], np.array([10.59264409, 0.18613381, 0.36485732, 7.83505886])
                )
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                npt.assert_almost_equal(
                    g_computed[[4, 9, 13, 18]], np.array([1.1076412e-03, 9.4108218e00, 8.6229404e00, 4.6671626e-01])
                )
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                npt.assert_almost_equal(
                    g_computed[[5, 10, 14, 19]], np.array([10.19870335, 0.28270337, 0.43049137, 7.44111811])
                )
            else:
                raise NotImplementedError("Not implemented yet")
        else:
            if control_type == ControlType.CONSTANT:
                npt.assert_almost_equal(
                    g_computed,
                    np.array(
                        [4.6897301, 10.98658484, 3.42854943, 8.03202923, 2.80024294, 6.56010175, 2.76576921, 6.47934048]
                    ),
                )
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                npt.assert_almost_equal(
                    g_computed,
                    np.array([4.6897301, 10.9865848, 3.4285494, 8.0320292, 2.8002429, 6.5601017, 2.7657692, 6.4793405]),
                )
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                npt.assert_almost_equal(
                    g_computed,
                    np.array(
                        [
                            2.77224961,
                            10.98658484,
                            2.02672534,
                            8.03202923,
                            1.65531325,
                            6.56010175,
                            1.63493473,
                            6.47934048,
                        ]
                    ),
                )
            else:
                raise NotImplementedError("Not implemented yet")
    else:
        raise NotImplementedError("Not implemented yet")

    elif interpolation_type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        if node == Node.START:
            value = np.zeros((ntau,))
            npt.assert_almost_equal(g_computed, value)
        elif node == Node.INTERMEDIATES:
            if control_type == ControlType.CONSTANT:
                value = tau[:, 1:-1] - np.ones((ntau, n_shooting - 2))
                value[:, 0] *= 0  # First node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, 1:-2] - np.ones((ntau, n_shooting - 2))
                value[:, 0] *= 0  # First intermediate node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            else:
                value = tau[:, 1:-4:2] - np.ones((ntau, n_shooting - 2))
                value[:, 0] *= 0  # First intermediate node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
        elif node == Node.ALL_SHOOTING:
            if control_type == ControlType.CONSTANT:
                value = tau - np.ones((ntau, n_shooting))
                value[:, 0] *= 0  # First intermediate node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, :-1] - np.ones((ntau, n_shooting))
                value[:, 0] *= 0  # First intermediate node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = tau[:, 0:-1:2] - np.ones((ntau, n_shooting))
                value[:, 0] *= 0  # First intermediate node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
        else:
            if control_type == ControlType.CONSTANT:
                value = tau[:, node] - np.ones((ntau, len(node)))
                value[:, 0] *= 0  # First intermediate node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                value = tau[:, node] - np.ones((ntau, len(node)))
                value[:, 0] *= 0  # First intermediate node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                value = tau[:, np.array(node) * 2] - np.ones((ntau, len(node)))
                value[:, 0] *= 0  # First intermediate node has weight 0
                value[:, -1] *= 2  # Last node has weight 2
                npt.assert_almost_equal(g_computed, value.flatten(order="F"))
    else:
        raise RuntimeError("Should not happen")


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

    bioptim_folder = TestUtils.bioptim_folder()
    n_shooting = 30
    np.random.seed(42)  # For reproducibility of spline

    ocp = objective_prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
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
    if interpolation_type == InterpolationType.CONSTANT:
        if control_type == ControlType.CONSTANT or control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau**2 * dt))
        else:
            if integration_rule == QuadratureRule.TRAPEZOIDAL:
                out = 0
                for i in range(round((tau[0, :].shape[0] - 1) / 2)):
                    out += (tau[0, i * 2] ** 2 + tau[0, i * 2 + 1] ** 2) / 2 * dt
                TestUtils.assert_objective_value(sol=sol, expected_value=out)
            elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(tau[0, 0:-1:2] ** 2 * dt))
    elif (
        interpolation_type == InterpolationType.LINEAR
        or interpolation_type == InterpolationType.CUSTOM
        or interpolation_type == InterpolationType.EACH_FRAME
    ):
        if control_type == ControlType.CONSTANT:
            value = np.zeros((ntau, n_shooting))
            for i in range(ntau):
                value[i, :] = tau[i, :] ** 2 * np.linspace(0, 1, n_shooting)
            TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value * dt))
        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            value = np.zeros((ntau, n_shooting))
            for i in range(ntau):
                value[i, :] = tau[i, :-1] ** 2 * np.linspace(0, 1, n_shooting)
            TestUtils.assert_objective_value(sol=sol, expected_value=np.sum(value * dt))
        else:
            if integration_rule == QuadratureRule.TRAPEZOIDAL:
                out = 0
                weight = np.linspace(0, 1, n_shooting)
                for i in range(round((tau[0, :].shape[0] - 1) / 2)):
                    out += (tau[0, i * 2] ** 2 + tau[0, i * 2 + 1] ** 2) / 2 * dt * weight[i]
                TestUtils.assert_objective_value(sol=sol, expected_value=out)
            elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                TestUtils.assert_objective_value(sol=sol, expected_value=9.55437614886297)
    elif interpolation_type == InterpolationType.SPLINE:
        # Testing values because too complicated for my small brain
        if control_type == ControlType.CONSTANT or control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            TestUtils.assert_objective_value(sol=sol, expected_value=-2069.98362328184)
        else:
            if integration_rule == QuadratureRule.TRAPEZOIDAL:
                TestUtils.assert_objective_value(sol=sol, expected_value=-4120.763073947789)
            elif integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                TestUtils.assert_objective_value(sol=sol, expected_value=-3315.0557682774315)


def test_bad_weights():

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


def test_weight_instanciation():

    # Create a weight
    weight1 = Weight(value=1.0)
    weight1.check_and_adjust_dimensions(5, "weight1")

    # Create another weight
    weight2 = Weight(value=10.0)
    weight2.check_and_adjust_dimensions(10, "weight2")

    # weight1.n_nodes is still 5, weight2.n_nodes is 10
    assert weight1.n_nodes == 5
    assert weight2.n_nodes == 10
