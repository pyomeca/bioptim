from typing import Any

import pytest
import numpy as np

from casadi import MX

from bioptim import (
    TorqueBiorbdModel,
    OptimalControlProgram,
    DynamicsOptions,
    DynamicsOptionsList,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    ConstraintList,
    ConstraintFcn,
    Node,
    PenaltyController,
    PhaseTransitionList,
    PhaseTransitionFcn,
    ParameterList,
    ParameterObjectiveList,
    MultinodeObjectiveList,
    PhaseDynamics,
    VariableScaling,
)

from bioptim.gui.graph import OcpToGraph
from tests.utils import TestUtils


def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls["tau"].cx - post.controls["tau"].cx


def custom_func_track_markers(controller: PenaltyController, first_marker: str, second_marker: str) -> MX:
    # Get the index of the markers from their name
    marker_0_idx = controller.model.marker_index(first_marker)
    marker_1_idx = controller.model.marker_index(second_marker)
    markers = controller.model.markers()(controller.q, controller.parameters.cx)
    return markers[:, marker_1_idx] - markers[:, marker_0_idx]


def prepare_ocp_phase_transitions(
    biorbd_model_path: str,
    with_constraints: bool,
    with_mayer: bool,
    with_lagrange: bool,
    phase_dynamics: PhaseDynamics,
) -> OptimalControlProgram:
    # BioModel path
    bio_model = (
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
    )

    # Problem parameters
    n_shooting = (20, 20, 20, 20)
    final_time = (2, 5, 4, 2)
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    multinode_objectives = MultinodeObjectiveList()
    if with_lagrange:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=3)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_VELOCITY, weight=0, phase=3)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MARKERS, weight=0, phase=3, marker_index=[0, 1])

    if with_mayer:
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME)
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, phase=0, node=1)
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, weight=0, phase=3, marker_index=[0, 1])

        multinode_objectives.add(
            minimize_difference,
            weight=100,
            nodes_phase=(1, 2),
            nodes=(Node.PENULTIMATE, Node.START),
            quadratic=True,
        )

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(expand_dynamics=True, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(expand_dynamics=True, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(expand_dynamics=True, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(expand_dynamics=True, phase_dynamics=phase_dynamics))

    # Constraints
    constraints = ConstraintList()
    if with_constraints:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            node=Node.START,
            first_marker="m0",
            second_marker="m1",
            phase=0,
            list_index=1,
        )
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS, node=2, first_marker="m0", second_marker="m1", phase=0, list_index=2
        )
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            node=Node.END,
            first_marker="m0",
            second_marker="m2",
            phase=0,
            list_index=3,
        )
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            node=Node.END,
            first_marker="m0",
            second_marker="m1",
            phase=1,
            list_index=4,
        )
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            node=Node.END,
            first_marker="m0",
            second_marker="m2",
            phase=2,
            list_index=5,
        )
        for i in range(n_shooting[3]):
            constraints.add(
                custom_func_track_markers, node=i, first_marker="m0", second_marker="m1", phase=3, list_index=6
            )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds.add("q", bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds.add("q", bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bio_model[3].bounds_from_ranges("qdot"), phase=3)

    x_bounds[0]["q"][1, 0] = 0
    x_bounds[0]["qdot"][:, 0] = 0
    x_bounds[-1]["q"][1, 0] = 0
    x_bounds[-1]["qdot"][:, 0] = 0

    x_bounds[0]["q"][2, 0] = 0.0
    x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[1].nb_tau, max_bound=[tau_max] * bio_model[1].nb_tau, phase=1)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[2].nb_tau, max_bound=[tau_max] * bio_model[2].nb_tau, phase=2)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[3].nb_tau, max_bound=[tau_max] * bio_model[3].nb_tau, phase=3)

    # Define phase transitions
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=2)
    phase_transitions.add(PhaseTransitionFcn.CYCLIC)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transitions,
        multinode_objectives=multinode_objectives,
    )


def my_parameter_function(bio_model: TorqueBiorbdModel, value: MX, extra_value: Any):
    value[2] *= extra_value
    bio_model.set_gravity(value)


def set_mass(bio_model: TorqueBiorbdModel, value: MX):
    bio_model.segments[0].characteristics().setMass(value)


def my_target_function(controller: PenaltyController, key: str) -> MX:
    return controller.parameters[key].cx


def prepare_ocp_parameters(
    biorbd_model_path,
    final_time,
    n_shooting,
    optim_gravity,
    optim_mass,
    min_g,
    max_g,
    target_g,
    min_m,
    max_m,
    target_m,
    ode_solver=OdeSolver.RK4(),
    use_sx=False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    optim_gravity: bool
        If the gravity should be optimized
    optim_mass: bool
        If the mass should be optimized
    min_g: np.ndarray
        The minimal value for the gravity
    max_g: np.ndarray
        The maximal value for the gravity
    target_g: np.ndarray
        The target value for the gravity
    min_m: float
        The minimal value for the mass
    max_m: float
        The maximal value for the mass
    target_m: float
        The target value for the mass
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.

    Returns
    -------
    The ocp ready to be solved
    """

    # Define the parameter to optimize
    parameters = ParameterList(use_sx=use_sx)
    parameter_objectives = ParameterObjectiveList()
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    if optim_gravity:
        g_scaling = VariableScaling("gravity_xyz", np.array([1, 1, 10.0]))
        parameters.add(
            "gravity_xyz",  # The name of the parameter
            my_parameter_function,  # The function that modifies the biorbd model
            size=3,  # The number of elements this particular parameter vector has
            scaling=g_scaling,  # The scaling of the parameter
            extra_value=1,  # You can define as many extra arguments as you want
        )

        # Give the parameter some min and max bounds
        parameter_bounds.add("gravity_xyz", min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)

        # and an initial condition
        parameter_init["gravity_xyz"] = (min_g + max_g) / 2

        # and an objective function
        parameter_objectives.add(
            my_target_function,
            weight=1000,
            quadratic=True,
            custom_type=ObjectiveFcn.Parameter,
            target=target_g / g_scaling.scaling,  # Make sure your target fits the scaling
            key="gravity_xyz",
        )

    if optim_mass:
        m_scaling = VariableScaling("mass", np.array([10.0]))
        parameters.add(
            "mass",  # The name of the parameter
            set_mass,  # The function that modifies the biorbd model
            size=1,  # The number of elements this particular parameter vector has
            scaling=m_scaling,  # The scaling of the parameter
        )

        parameter_bounds.add("mass", min_bound=[min_m], max_bound=[max_m], interpolation=InterpolationType.CONSTANT)

        parameter_init["mass"] = (min_m + max_m) / 2

        parameter_objectives.add(
            my_target_function,
            weight=100,
            quadratic=True,
            custom_type=ObjectiveFcn.Parameter,
            target=target_m / m_scaling.scaling,  # Make sure your target fits the scaling
            key="mass",
        )

    # --- Options --- #
    bio_model = TorqueBiorbdModel(biorbd_model_path, parameters=parameters)
    n_tau = bio_model.nb_tau

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10)

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
    tau_min, tau_max, tau_init = -300, 300, 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        use_sx=use_sx,
    )


def prepare_ocp_custom_objectives(
    biorbd_model_path, ode_solver=OdeSolver.RK4(), phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolverBase
        The type of ode solver used
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = TorqueBiorbdModel(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", list_index=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, list_index=2)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=2, list_index=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=3, list_index=4)
    objective_functions.add(
        custom_func_track_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.START,
        quadratic=True,
        first_marker="m0",
        second_marker="m1",
        weight=1000,
        list_index=5,
    )
    objective_functions.add(
        custom_func_track_markers,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        quadratic=True,
        first_marker="m0",
        second_marker="m2",
        weight=1000,
        list_index=6,
    )
    target = np.array([[1, 2, 3], [4, 5, 6]])[:, :, np.newaxis]
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_MARKERS, list_index=7, index=[1, 2], target=target)

    # Dynamics
    dynamics = DynamicsOptions(ode_solver=ode_solver, expand_dynamics=True, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][1:, [0, -1]] = 0
    x_bounds["q"][2, -1] = 1.57
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("with_mayer", [True, False])
@pytest.mark.parametrize("with_lagrange", [True, False])
@pytest.mark.parametrize("with_constraints", [True, False])
def test_phase_transitions(with_mayer, with_lagrange, with_constraints, phase_dynamics):
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/models/cube.bioMod"
    ocp = prepare_ocp_phase_transitions(
        model_path,
        with_mayer=with_mayer,
        with_lagrange=with_lagrange,
        with_constraints=with_constraints,
        phase_dynamics=phase_dynamics,
    )
    if with_lagrange and with_mayer is not False:
        ocp.nlp[0].J[0].quadratic = False
        ocp.nlp[0].J[0].target = np.array([[2]])
    ocp.print(to_console=True, to_graph=False)  # False so it does not attack the programmer with lot of graphs!
    OcpToGraph(ocp)._prepare_print()


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_parameters(phase_dynamics):
    optim_gravity = True
    optim_mass = True
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/models/pendulum.bioMod"
    target_g = np.zeros((3, 1))
    target_g[2] = -9.81
    ocp = prepare_ocp_parameters(
        biorbd_model_path=model_path,
        final_time=3,
        n_shooting=100,
        optim_gravity=optim_gravity,
        optim_mass=optim_mass,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=target_g,
        target_m=20,
        phase_dynamics=phase_dynamics,
    )
    ocp.print(to_console=True, to_graph=False)  # False so it does not attack the programmer with lot of graphs!
    OcpToGraph(ocp)._prepare_print()


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("quadratic", [True, False])
def test_objectives_target(quadratic, phase_dynamics):
    bioptim_folder = TestUtils.bioptim_folder()
    model_path = bioptim_folder + "/examples/getting_started/models/cube.bioMod"
    ocp = prepare_ocp_custom_objectives(biorbd_model_path=model_path, phase_dynamics=phase_dynamics)
    ocp.nlp[0].J[1].quadratic = quadratic
    ocp.nlp[0].J[1].target = np.repeat([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], ocp.nlp[0].ns, axis=0).T
    ocp.print(to_graph=False)  # False so it does not attack the programmer with lot of graphs!
    OcpToGraph(ocp)._prepare_print()
