from pathlib import Path

import numpy as np
import pytest

import biorbd
import casadi

from bioptim import (
    Bounds,
    BoundsList,
    Constraint,
    ConstraintList,
    Data,
    DynamicsType,
    DynamicsTypeList,
    InitialGuess,
    InitialGuessList,
    Node,
    InterpolationType,
    Objective,
    ObjectiveList,
    ObjectiveOption,
    OdeSolver,
    OptimalControlProgram,
    ParameterList,
    QAndQDotBounds,
    ShowResult,
)


def prepare_ocp(phase_time_constraint, use_parameter):
    # --- Inputs --- #
    final_time = (2, 5, 4)
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    ns = (20, 30, 20)
    PROJECT_FOLDER = Path(__file__).parent / ".."
    biorbd_model_path = str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/cube.bioMod"
    ode_solver = OdeSolver.RK

    # --- Options --- #
    nb_phases = len(ns)

    # Model path
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=0)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=1)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsType.TORQUE_DRIVEN, phase=2)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=1, phase=0)
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2, phase=0)
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=1, phase=1)
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2, phase=2)

    constraints.add(
        Constraint.TIME_CONSTRAINT,
        node=Node.END,
        minimum=time_min[0],
        maximum=time_max[0],
        phase=phase_time_constraint,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model[0]))  # Phase 0
    x_bounds.add(QAndQDotBounds(biorbd_model[0]))  # Phase 1
    x_bounds.add(QAndQDotBounds(biorbd_model[0]))  # Phase 2

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds[i, [0, -1]] = 0
    x_bounds[0][2, 0] = 0.0
    x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()])
    u_bounds.add([[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()])
    u_bounds.add([[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()])

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    parameters = ParameterList()
    if use_parameter:

        def my_target_function(ocp, value, target_value):
            return value - target_value

        def my_parameter_function(biorbd_model, value, extra_value):
            biorbd_model.setGravity(biorbd.Vector3d(0, 0, 2))

        min_g = -10
        max_g = -6
        target_g = -8
        bound_gravity = Bounds(min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)
        initial_gravity = InitialGuess((min_g + max_g) / 2)
        parameter_objective_functions = ObjectiveOption(
            my_target_function, weight=10, quadratic=True, custom_type=Objective.Parameter, target_value=target_g
        )
        parameters.add(
            "gravity_z",
            my_parameter_function,
            initial_gravity,
            bound_gravity,
            size=1,
            penalty_list=parameter_objective_functions,
            extra_value=1,
        )

    # ------------- #

    return OptimalControlProgram(
        biorbd_model[:nb_phases],
        dynamics,
        ns,
        final_time[:nb_phases],
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        parameters=parameters,
    )


@pytest.mark.parametrize("phase_time_constraint", [0, 1, 2])
@pytest.mark.parametrize("use_parameter", [False, True])
def test_variable_time(phase_time_constraint, use_parameter):
    ocp = prepare_ocp(phase_time_constraint, use_parameter)

    # --- Solve the program --- #
    np.random.seed(42)
    sol = np.random.random((649 + use_parameter, 1))

    # --- Show results --- #
    param = Data.get_data(ocp, sol)

    if use_parameter:
        if phase_time_constraint == 0:
            np.testing.assert_almost_equal(
                param[0]["q"][0][0:8],
                np.array([0.7319939, 0.9699099, 0.6118529, 0.0464504, 0.684233, 0.520068, 0.0884925, 0.5426961]),
            )
            np.testing.assert_almost_equal(
                param[0]["q"][1][0:8],
                np.array([0.5986585, 0.8324426, 0.1394939, 0.6075449, 0.4401525, 0.5467103, 0.1959829, 0.1409242]),
            )
        if phase_time_constraint == 1:
            np.testing.assert_almost_equal(
                param[0]["q"][1][8:16],
                np.array([0.7290072, 0.3109823, 0.5612772, 0.314356, 0.1612213, 0.8074402, 0.5107473, 0.3636296]),
            )
            np.testing.assert_almost_equal(
                param[0]["q"][2][8:16],
                np.array([0.7712703, 0.3251833, 0.7709672, 0.5085707, 0.9296977, 0.8960913, 0.417411, 0.9717821]),
            )
        if phase_time_constraint == 2:
            np.testing.assert_almost_equal(
                param[0]["q"][1][16:24],
                np.array([0.502679, 0.6721355, 0.8353025, 0.6451728, 0.2418523, 0.8870864, 0.6635018, 0.3253997]),
            )
            np.testing.assert_almost_equal(
                param[0]["q"][2][16:24],
                np.array([0.0514788, 0.7616196, 0.3207801, 0.1743664, 0.0931028, 0.7798755, 0.0050616, 0.7464914]),
            )
    else:
        if phase_time_constraint == 0:
            np.testing.assert_almost_equal(
                param[0]["q"][0][0:8],
                np.array([0.9507143, 0.0205845, 0.2912291, 0.5924146, 0.0976721, 0.3117111, 0.9218742, 0.2809345]),
            )
            np.testing.assert_almost_equal(
                param[0]["q"][1][0:8],
                np.array([0.7319939, 0.9699099, 0.6118529, 0.0464504, 0.684233, 0.520068, 0.0884925, 0.5426961]),
            )
        if phase_time_constraint == 1:
            np.testing.assert_almost_equal(
                param[0]["q"][1][8:16],
                np.array([0.7068573, 0.0635584, 0.760785, 0.6364104, 0.2897515, 0.5393422, 0.0069521, 0.703019]),
            )
            np.testing.assert_almost_equal(
                param[0]["q"][2][8:16],
                np.array([0.7290072, 0.3109823, 0.5612772, 0.314356, 0.1612213, 0.8074402, 0.5107473, 0.3636296]),
            )
        if phase_time_constraint == 2:
            np.testing.assert_almost_equal(
                param[0]["q"][1][16:24],
                np.array([0.6095643, 0.2420553, 0.0902898, 0.2264958, 0.5296506, 0.8971103, 0.1014715, 0.2372491]),
            )
            np.testing.assert_almost_equal(
                param[0]["q"][2][16:24],
                np.array([0.502679, 0.6721355, 0.8353025, 0.6451728, 0.2418523, 0.8870864, 0.6635018, 0.3253997]),
            )
