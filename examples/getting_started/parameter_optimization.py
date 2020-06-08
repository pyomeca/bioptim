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
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which decribed by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, value))


def my_target_function(ocp, value, target_value_via_custom_param):
    # The target function is a penalty function.
    # `ocp` and `value` are mandatory. The rest is defined in the
    # parameter by the user
    return value - target_value_via_custom_param


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, min_g, max_g, target_g):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -30, 30, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 10}

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

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    bound_length = Bounds(min_bound=min_g, max_bound=max_g, interpolation_type=InterpolationType.CONSTANT)
    # and an initial condition
    initial_length = InitialConditions(7)
    parameters = {
        "name": "gravity_z",  # The name of the parameter
        "function": my_parameter_function,  # The function that modifies the biorbd model
        "bounds": bound_length,  # The bounds
        "initial_guess": initial_length,  # The initial guess
        "size": 1,  # The number of elements this particular parameter vector has
        "type": Objective.Mayer,  # The type objective or constraint function (if there is any)
        "target_function": my_target_function,  # The penalty function (if there is any)
        "weight": 10,  # The weight of the objective function
        "quadratic": True,  # If the objective function is quadratic
        "target_value_via_custom_param": target_g,  # Supplementary element defined by the user
    }

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
    ocp = prepare_ocp(
        biorbd_model_path="pendulum.bioMod", final_time=3, number_shooting_points=100, min_g=-10, max_g=-6, target_g=-8,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)
    length = params["gravity_z"][0, 0]
    print(length)

    # --- Show results --- #
    ShowResult(ocp, sol).animate(nb_frames=200)
