import biorbd
import pickle
from time import time

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Objective,
    InterpolationType,
    Data,
)


def my_parameter_function(biorbd_model, value, target_value_via_custom_param):
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, value))


def my_custom_parameter_target_function(ocp, value, target_value_via_custom_param):
    return value - target_value_via_custom_param


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, min_g, max_g):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -30, 30, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 10})

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[:, [0, -1]] = 0
    X_bounds.max[:, [0, -1]] = 0
    X_bounds.min[1, -1] = 3.14
    X_bounds.max[1, -1] = 3.14

    # Initial guess
    X_init = InitialConditions([0] * (n_q + n_qdot))

    # Define control path constraint
    U_bounds = Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)
    U_bounds.min[1, :] = 0
    U_bounds.max[1, :] = 0

    U_init = InitialConditions([torque_init] * n_tau)

    # ------------- #
    bound_length = Bounds(min_bound=min_g, max_bound=max_g, interpolation_type=InterpolationType.CONSTANT)
    initial_length = InitialConditions(7)
    parameters = (
        {"name": "gravity_z", "type": Objective.Mayer, "function": my_parameter_function,
         "bounds": bound_length, "initial_guess": initial_length, "size": 1,
         # "target_function": my_custom_parameter_target_function, "weight": 1000, "quadratic": True,
         "target_value_via_custom_param": -9.81,
         }
    )

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        parameters=parameters,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, number_shooting_points=100, min_g=-10, max_g=-8)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    length = params["gravity_z"][0, 0]
    print(length)

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=200)
