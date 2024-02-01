import casadi as cas
import numpy as np

from bioptim import (
    BiMappingList,
    BiorbdModel,
    BoundsList,
    ConstraintFcn,
    ConstraintList,
    DynamicsFcn,
    DynamicsList,
    InitialGuessList,
    InterpolationType,
    MultinodeConstraintFcn,
    MultinodeConstraintList,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OptimalControlProgram,
    PenaltyController,
    Solver,
)


def custom_trampoline_bed_in_peripheral_vision(controller: PenaltyController) -> cas.MX:
    """
    This function aims to encourage the avatar to keep the trampoline bed in his peripheral vision.
    It is done by discretizing the vision cone into vectors and determining if the vector projection of the gaze are
    inside the trampoline bed.
    """

    a = 1.07  # Trampoline with/2
    b = 2.14  # Trampoline length/2
    n = 6  # order of the polynomial for the trampoline bed rectangle equation

    # Get the gaze vector
    eyes_vect_start_marker_idx = controller.model.marker_index(f"eyes_vect_start")
    eyes_vect_end_marker_idx = controller.model.marker_index(f"eyes_vect_end")
    gaze_vector = (
        controller.model.markers(controller.states["q"].mx)[eyes_vect_end_marker_idx]
        - controller.model.markers(controller.states["q"].mx)[eyes_vect_start_marker_idx]
    )

    point_in_the_plane = np.array([1, 2, -0.83])
    vector_normal_to_the_plane = np.array([0, 0, 1])
    obj = 0
    for i_r in range(11):
        for i_th in range(10):
            marker_idx = controller.model.marker_index(f"cone_approx_{i_r}_{i_th}")
            vector_origin = controller.model.markers(controller.states["q"].mx)[eyes_vect_start_marker_idx]
            vector_end = controller.model.markers(controller.states["q"].mx)[marker_idx]
            vector = vector_end - vector_origin

            t = (
                cas.dot(point_in_the_plane, vector_normal_to_the_plane)
                - cas.dot(vector_normal_to_the_plane, vector_origin)
            ) / cas.dot(vector, vector_normal_to_the_plane)
            point_projection = vector_origin + vector * cas.fabs(t)

            obj += cas.tanh(((point_projection[0] / a) ** n + (point_projection[1] / b) ** n) - 1) + 1

    val = cas.if_else(
        gaze_vector[2] > -0.01,
        2 * 10 * 11,
        cas.if_else(
            cas.fabs(gaze_vector[0] / gaze_vector[2]) > np.tan(3 * np.pi / 8),
            2 * 10 * 11,
            cas.if_else(cas.fabs(gaze_vector[1] / gaze_vector[2]) > np.tan(3 * np.pi / 8), 2 * 10 * 11, obj),
        ),
    )
    out = controller.mx_to_cx("peripheral_vision", val, controller.states["q"])

    return out


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    use_sx: bool = True,
) -> OptimalControlProgram:
    """
    This function build an optimal control program and instantiate it.
    It can be seen as a factory for the OptimalControlProgram class.

    Parameters
    ----------
    # TODO fill this section

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Declaration of generic elements
    n_shooting = (40, 40, 40, 40, 40)
    phase_time = (0.36, 0.36, 0.36, 0.36, 0.36)
    nb_phases = 5

    bio_model = [BiorbdModel(biorbd_model_path) for _ in range(nb_phases)]

    # Declaration of the constraints and objectives of the ocp
    constraints = ConstraintList()
    objective_functions = ObjectiveList()

    for i in range(nb_phases):
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            weight=1.0,
            key="tau",
            node=Node.ALL_SHOOTING,
            quadratic=True,
            phase=i,
        )

        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            weight=1.0,
            key="tau",
            node=Node.ALL_SHOOTING,
            quadratic=True,
            derivative=True,
            phase=i,
        )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_TIME,
        weight=100000.0,
        min_bound=0.1,
        max_bound=2.0,
        node=Node.END,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        weight=1.0,
        first_marker="MiddleRightHand",
        second_marker="PikeTargetRightHand",
        node=Node.END,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        weight=1.0,
        first_marker="MiddleLeftHand",
        second_marker="PikeTargetLeftHand",
        node=Node.END,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[12, 13, 16, 17],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY,
        weight=10.0,
        segment="Head",
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=1.0,
        key="qdot",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=10.0,
        key="q",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=100.0,
        key="q",
        index=[6, 7],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        custom_trampoline_bed_in_peripheral_vision,
        custom_type=ObjectiveFcn.Lagrange,
        weight=100.0,
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_TIME,
        weight=-100.0,
        min_bound=0.1,
        max_bound=2.0,
        node=Node.END,
        quadratic=True,
        phase=1,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[10, 11, 14, 15],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=1,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_STATE,
        weight=100.0,
        key="q",
        index=[4],
        node=Node.ALL,
        quadratic=True,
        phase=1,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=1.0,
        key="qdot",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=1,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=10.0,
        key="q",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=1,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=100.0,
        key="q",
        index=[6, 7],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=1,
    )

    constraints.add(
        constraint=ConstraintFcn.SUPERIMPOSE_MARKERS,
        min_bound=-0.05,
        max_bound=0.05,
        first_marker="MiddleRightHand",
        second_marker="PikeTargetRightHand",
        node=Node.ALL_SHOOTING,
        quadratic=False,
        phase=1,
    )

    constraints.add(
        constraint=ConstraintFcn.SUPERIMPOSE_MARKERS,
        min_bound=-0.05,
        max_bound=0.05,
        first_marker="MiddleLeftHand",
        second_marker="PikeTargetLeftHand",
        node=Node.ALL_SHOOTING,
        quadratic=False,
        phase=1,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_TIME,
        weight=100000.0,
        min_bound=0.1,
        max_bound=2.0,
        node=Node.END,
        quadratic=True,
        phase=2,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[18],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=2,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[12, 13, 16, 17],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=2,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=1.0,
        key="qdot",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=2,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=10.0,
        key="q",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=2,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=100.0,
        key="q",
        index=[6, 7],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=2,
    )

    objective_functions.add(
        custom_trampoline_bed_in_peripheral_vision,
        custom_type=ObjectiveFcn.Lagrange,
        weight=100.0,
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=2,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_TIME,
        weight=-0.01,
        min_bound=0.1,
        max_bound=2.0,
        node=Node.END,
        quadratic=True,
        phase=3,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[12, 13, 16, 17],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=3,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=1.0,
        key="qdot",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=3,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=10.0,
        key="q",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=3,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=100.0,
        key="q",
        index=[6, 7],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=3,
    )

    objective_functions.add(
        custom_trampoline_bed_in_peripheral_vision,
        custom_type=ObjectiveFcn.Lagrange,
        weight=100.0,
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=3,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_TIME,
        weight=-0.01,
        min_bound=0.1,
        max_bound=2.0,
        node=Node.END,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[12, 13, 16, 17],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=50000.0,
        key="q",
        index=[18],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Mayer.MINIMIZE_STATE,
        weight=1000.0,
        key="q",
        index=[4],
        node=Node.END,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY,
        weight=10.0,
        segment="Head",
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=1.0,
        key="qdot",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=10.0,
        key="q",
        index=[8, 9],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        weight=100.0,
        key="q",
        index=[6, 7],
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        custom_trampoline_bed_in_peripheral_vision,
        custom_type=ObjectiveFcn.Lagrange,
        weight=100.0,
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=4,
    )

    objective_functions.add(
        objective=ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
        weight=1.0,
        vector_0_marker_0="eyes_vect_start",
        vector_0_marker_1="eyes_vect_end",
        vector_1_marker_0="eyes_vect_start",
        vector_1_marker_1="fixation_front",
        node=Node.ALL_SHOOTING,
        quadratic=True,
        phase=4,
    )

    # Declaration of the dynamics function used during integration
    dynamics = DynamicsList()

    for i in range(nb_phases):
        dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN,
            phase=i,
        )

    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(
        MultinodeConstraintFcn.TRACK_TOTAL_TIME,
        nodes_phase=(0, 1, 2, 3, 4),
        nodes=(Node.END, Node.END, Node.END, Node.END, Node.END),
        min_bound=final_time - 0.1,
        max_bound=final_time + 0.1,
    )

    # Declaration of optimization variables bounds and initial guesses
    # Path constraint
    x_bounds = BoundsList()
    x_initial_guesses = InitialGuessList()

    u_bounds = BoundsList()
    u_initial_guesses = InitialGuessList()

    x_bounds.add(
        "q",
        min_bound=[
            [-0.0, -1.0, -1.0],
            [-0.0, -1.0, -1.0],
            [-0.0, -0.1, -0.1],
            [-0.0, -2.36, -2.36],
            [0.0, -0.79, -0.79],
            [0.0, -0.2, -0.2],
            [-0.1, -1.05, -1.05],
            [-0.1, -1.22, -1.22],
            [-0.1, -0.39, -0.39],
            [0.29, -0.52, -0.52],
            [0.0, -0.65, -0.65],
            [2.9, -0.05, -0.05],
            [0.0, -1.8, -1.8],
            [0.0, -2.65, -2.65],
            [0.0, -2.0, -2.0],
            [-2.9, -3.0, -3.0],
            [0.0, -1.1, -1.1],
            [0.0, -2.65, -2.65],
            [0.0, -2.6, -2.6],
            [0.0, -0.1, -0.1],
        ],
        max_bound=[
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 15.0, 15.0],
            [-0.0, 0.1, -0.69],
            [0.0, 0.79, 0.79],
            [0.0, 0.2, 0.2],
            [0.1, 1.05, 1.05],
            [0.1, 0.39, 0.39],
            [0.1, 0.39, 0.39],
            [0.49, 0.52, 0.52],
            [0.0, 2.0, 2.0],
            [2.9, 3.0, 3.0],
            [0.0, 1.1, 1.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.65, 0.65],
            [-2.9, 0.05, 0.05],
            [0.0, 1.8, 1.8],
            [0.0, 0.0, 0.0],
            [0.0, 0.2, -2.2],
            [0.0, 0.1, 0.1],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=0,
    )

    x_bounds.add(
        "q",
        min_bound=[
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-0.1, -0.1, -0.1],
            [-2.36, -9.62, -9.62],
            [-0.39, -0.39, -0.39],
            [-0.2, -0.2, -0.2],
            [-1.05, -1.05, -1.05],
            [-1.22, -1.22, -1.22],
            [-0.39, -0.39, -0.39],
            [-0.52, -0.52, -0.52],
            [-0.65, -0.65, -0.65],
            [-0.05, -0.05, -0.05],
            [-1.8, -1.8, -1.8],
            [-2.65, -2.65, -2.65],
            [-2.0, -2.0, -2.0],
            [-3.0, -3.0, -3.0],
            [-1.1, -1.1, -1.1],
            [-2.65, -2.65, -2.65],
            [-2.6, -2.6, -2.6],
            [-0.1, -0.1, -0.1],
        ],
        max_bound=[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [15.0, 15.0, 15.0],
            [-0.69, -0.69, -6.08],
            [0.39, 0.39, 0.39],
            [0.2, 0.2, 0.2],
            [1.05, 1.05, 1.05],
            [0.39, 0.39, 0.39],
            [0.39, 0.39, 0.39],
            [0.52, 0.52, 0.52],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [1.1, 1.1, 1.1],
            [0.0, 0.0, 0.0],
            [0.65, 0.65, 0.65],
            [0.05, 0.05, 0.05],
            [1.8, 1.8, 1.8],
            [0.0, 0.0, 0.0],
            [-2.2, -2.2, -2.2],
            [0.1, 0.1, 0.1],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=1,
    )

    x_bounds.add(
        "q",
        min_bound=[
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-0.1, -0.1, -0.1],
            [-9.62, -9.62, -9.62],
            [-0.79, -0.79, -0.79],
            [-0.2, -0.2, -0.2],
            [-1.05, -1.05, -1.05],
            [-1.22, -1.22, -1.22],
            [-0.39, -0.39, -0.39],
            [-0.52, -0.52, -0.52],
            [-0.65, -0.65, -0.65],
            [-0.05, -0.05, -0.05],
            [-1.8, -1.8, -1.8],
            [-2.65, -2.65, -2.65],
            [-2.0, -2.0, -2.0],
            [-3.0, -3.0, -3.0],
            [-1.1, -1.1, -1.1],
            [-2.65, -2.65, -2.65],
            [-2.61, -2.61, -0.35],
            [-0.1, -0.1, -0.1],
        ],
        max_bound=[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [15.0, 15.0, 15.0],
            [-6.08, -6.08, -6.08],
            [0.79, 0.79, 0.79],
            [0.2, 3.34, 3.34],
            [1.05, 1.05, 1.05],
            [0.39, 0.39, 0.39],
            [0.39, 0.39, 0.39],
            [0.52, 0.52, 0.52],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [1.1, 1.1, 1.1],
            [0.0, 0.0, 0.0],
            [0.65, 0.65, 0.65],
            [0.05, 0.05, 0.05],
            [1.8, 1.8, 1.8],
            [0.0, 0.0, 0.0],
            [-2.19, 0.35, 0.35],
            [0.1, 0.1, 0.1],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=2,
    )

    x_bounds.add(
        "q",
        min_bound=[
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-0.1, -0.1, -0.1],
            [-9.62, -11.2, -11.2],
            [-0.39, -0.39, -0.39],
            [-0.2, -0.2, 6.08],
            [-1.05, -1.05, -1.05],
            [-1.22, -1.22, -1.22],
            [-0.39, -0.39, -0.39],
            [-0.52, -0.52, -0.52],
            [-0.65, -0.65, -0.65],
            [-0.05, -0.05, 0.0],
            [-1.8, -1.8, -0.1],
            [-2.65, -2.65, -0.1],
            [-2.0, -2.0, -2.0],
            [-3.0, -3.0, -0.39],
            [-1.1, -1.1, -0.1],
            [-2.65, -2.65, -0.1],
            [-0.35, -0.35, -0.35],
            [-0.1, -0.1, -0.1],
        ],
        max_bound=[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [15.0, 15.0, 15.0],
            [-6.08, -6.08, -10.8],
            [0.39, 0.39, 0.39],
            [3.34, 6.48, 6.48],
            [1.05, 1.05, 1.05],
            [0.39, 0.39, 0.39],
            [0.39, 0.39, 0.39],
            [0.52, 0.52, 0.52],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 0.39],
            [1.1, 1.1, 0.1],
            [0.0, 0.0, 0.1],
            [0.65, 0.65, 0.65],
            [0.05, 0.05, 0.0],
            [1.8, 1.8, 0.1],
            [0.0, 0.0, 0.1],
            [0.35, 0.35, 0.35],
            [0.1, 0.1, 0.1],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=3,
    )

    x_bounds.add(
        "q",
        min_bound=[
            [-1.0, -1.0, -0.01],
            [-1.0, -1.0, -0.01],
            [-0.1, -0.1, 0.0],
            [-11.2, -12.67, -12.12],
            [-0.2, -0.2, -0.2],
            [6.18, 6.18, 6.18],
            [-1.05, -1.05, -1.05],
            [-1.22, -1.22, -1.22],
            [-0.39, -0.39, -0.39],
            [-0.52, -0.52, -0.52],
            [-0.65, -0.65, -0.1],
            [0.0, -0.05, 2.8],
            [-0.1, -1.8, -0.1],
            [-0.1, -2.65, -0.1],
            [-2.0, -2.0, -0.1],
            [-1.57, -3.0, -3.0],
            [-0.1, -1.1, -0.1],
            [-0.1, -2.65, -0.1],
            [0.0, -0.6, -0.55],
            [-0.1, -0.1, -0.1],
        ],
        max_bound=[
            [1.0, 1.0, 0.01],
            [1.0, 1.0, 0.01],
            [15.0, 15.0, 0.01],
            [-10.8, -10.9, -12.02],
            [0.2, 0.2, 0.2],
            [6.38, 6.38, 6.38],
            [1.05, 1.05, 1.05],
            [0.39, 0.39, 0.39],
            [0.39, 0.39, 0.39],
            [0.52, 0.52, 0.52],
            [2.0, 2.0, 0.1],
            [1.57, 3.0, 3.0],
            [0.1, 1.1, 0.1],
            [0.1, 0.0, 0.1],
            [0.65, 0.65, 0.1],
            [0.0, 0.05, -2.8],
            [0.1, 1.8, 0.1],
            [0.1, 0.0, 0.1],
            [0.35, 0.35, -0.45],
            [0.1, 0.1, 0.1],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=4,
    )

    x_bounds.add(
        "qdot",
        min_bound=[
            [-0.5, -10.0, -10.0],
            [-0.5, -10.0, -10.0],
            [6.83, -100.0, -100.0],
            [-200.0, -200.0, -200.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
        ],
        max_bound=[
            [0.5, 10.0, 10.0],
            [0.5, 10.0, 10.0],
            [10.83, 100.0, 100.0],
            [-0.5, -0.5, -0.5],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=0,
    )

    x_bounds.add(
        "qdot",
        min_bound=[
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0],
            [-100.0, -100.0, -100.0],
            [-200.0, -200.0, -200.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ],
        max_bound=[
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [-0.5, -0.5, -0.5],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=1,
    )

    x_bounds.add(
        "qdot",
        min_bound=[
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0],
            [-100.0, -100.0, -100.0],
            [-200.0, -200.0, -200.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ],
        max_bound=[
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [-0.5, -0.5, -0.5],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=2,
    )

    x_bounds.add(
        "qdot",
        min_bound=[
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0],
            [-100.0, -100.0, -100.0],
            [-200.0, -200.0, -200.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ],
        max_bound=[
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [-0.5, -0.5, -0.5],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=3,
    )

    x_bounds.add(
        "qdot",
        min_bound=[
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0],
            [-100.0, -100.0, -100.0],
            [-200.0, -200.0, -200.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ],
        max_bound=[
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0],
            [-0.5, -0.5, -0.5],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        phase=4,
    )

    for i in range(nb_phases):
        u_bounds.add(
            "tau",
            min_bound=[
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
                [-500.0],
            ],
            max_bound=[
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
                [500.0],
            ],
            interpolation=InterpolationType.CONSTANT,
            phase=i,
        )

    x_initial_guesses.add(
        "q",
        initial_guess=[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 7.45],
            [-0.0, -1.52],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, -0.41],
            [0.0, 0.0],
            [0.39, 0.0],
            [0.0, 0.68],
            [2.9, 1.48],
            [0.0, -0.35],
            [0.0, -1.32],
            [0.0, -0.68],
            [-2.9, -1.48],
            [0.0, 0.35],
            [0.0, -1.32],
            [0.0, -2.4],
            [0.0, 0.0],
        ],
        interpolation=InterpolationType.LINEAR,
        phase=0,
    )

    x_initial_guesses.add(
        "q",
        initial_guess=[
            [0.0, 0.0],
            [0.0, 0.0],
            [7.45, 7.45],
            [-1.52, -7.85],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-0.41, -0.41],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.68, 0.68],
            [1.48, 1.48],
            [-0.35, -0.35],
            [-1.32, -1.32],
            [-0.68, -0.68],
            [-1.48, -1.48],
            [0.35, 0.35],
            [-1.32, -1.32],
            [-2.4, -2.4],
            [0.0, 0.0],
        ],
        interpolation=InterpolationType.LINEAR,
        phase=1,
    )

    x_initial_guesses.add(
        "q",
        initial_guess=[
            [0.0, 0.0],
            [0.0, 0.0],
            [7.45, 7.45],
            [-7.85, -7.85],
            [0.0, 0.0],
            [0.0, 1.57],
            [0.0, 0.0],
            [-0.41, -0.41],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.68, 0.68],
            [1.48, 1.48],
            [-0.35, -0.35],
            [-1.32, -1.32],
            [-0.68, -0.68],
            [-1.48, -1.48],
            [0.35, 0.35],
            [-1.32, -1.32],
            [-2.4, 0.0],
            [0.0, 0.0],
        ],
        interpolation=InterpolationType.LINEAR,
        phase=2,
    )

    x_initial_guesses.add(
        "q",
        initial_guess=[
            [0.0, 0.0],
            [0.0, 0.0],
            [7.45, 7.45],
            [-7.85, -11.0],
            [0.0, 0.0],
            [1.57, 6.28],
            [0.0, 0.0],
            [-0.41, -0.41],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.68, 0.68],
            [1.48, 0.2],
            [-0.35, 0.0],
            [-1.32, 0.0],
            [-0.68, -0.68],
            [-1.48, -0.2],
            [0.35, 0.0],
            [-1.32, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        interpolation=InterpolationType.LINEAR,
        phase=3,
    )

    x_initial_guesses.add(
        "q",
        initial_guess=[
            [0.0, 0.0],
            [0.0, 0.0],
            [7.45, 0.0],
            [-11.0, -12.07],
            [0.0, 0.0],
            [6.28, 6.28],
            [0.0, 0.0],
            [-0.41, -0.41],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.68, 0.0],
            [0.79, 2.9],
            [0.0, 0.0],
            [0.0, 0.0],
            [-0.68, 0.0],
            [-0.79, -2.9],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.18, -0.5],
            [0.0, 0.0],
        ],
        interpolation=InterpolationType.LINEAR,
        phase=4,
    )

    x_initial_guesses.add(
        "qdot",
        initial_guess=[
            [0.0],
            [0.0],
            [7.06],
            [-12.57],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        interpolation=InterpolationType.CONSTANT,
        phase=0,
    )

    x_initial_guesses.add(
        "qdot",
        initial_guess=[
            [0.0],
            [0.0],
            [5.3],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        interpolation=InterpolationType.CONSTANT,
        phase=1,
    )

    x_initial_guesses.add(
        "qdot",
        initial_guess=[
            [0.0],
            [0.0],
            [5.3],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        interpolation=InterpolationType.CONSTANT,
        phase=2,
    )

    x_initial_guesses.add(
        "qdot",
        initial_guess=[
            [0.0],
            [0.0],
            [5.3],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        interpolation=InterpolationType.CONSTANT,
        phase=3,
    )

    x_initial_guesses.add(
        "qdot",
        initial_guess=[
            [0.0],
            [0.0],
            [5.3],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        interpolation=InterpolationType.CONSTANT,
        phase=4,
    )

    for i in range(nb_phases):
        u_initial_guesses.add(
            "tau",
            initial_guess=[
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ],
            interpolation=InterpolationType.CONSTANT,
            phase=i,
        )

    mapping = BiMappingList()
    mapping.add(
        "tau",
        to_second=[None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        to_first=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    )

    # Construct and return the optimal control program (OCP)
    return OptimalControlProgram(
        bio_model=bio_model,
        n_shooting=n_shooting,
        phase_time=phase_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_initial_guesses,
        u_init=u_initial_guesses,
        objective_functions=objective_functions,
        variable_mappings=mapping,
        use_sx=use_sx,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
    )


def main():
    # --- Prepare the multi-start and run it --- #
    ocp = prepare_ocp(
        biorbd_model_path="models/pike_double_full_with_vision.bioMod",
        final_time=1.8,
        use_sx=False,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(3000)
    solver.set_convergence_tolerance(1e-6)

    sol = ocp.solve(solver=solver)

    # --- Show results --- #
    sol.print_cost()
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
