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
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, nb_threads, use_SX=False):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE}

    # Dynamics
    problem_type = {"type": ProblemType.TORQUE_DRIVEN}

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
    U_bounds.min[n_tau - 1, :] = 0
    U_bounds.max[n_tau - 1, :] = 0

    U_init = InitialConditions([torque_init] * n_tau)

    # ------------- #

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
        nb_threads=nb_threads,
        use_SX=use_SX,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, number_shooting_points=100, nb_threads=4)

    # --- Solve the program --- #
    tic = time()
    sol, sol_iterations = ocp.solve(show_online_optim=True, return_iterations=True)
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Access to all iterations  --- #
    nb_iter = len(sol_iterations)
    third_iteration = sol_iterations[2]

    # --- Save result of get_data --- #
    ocp.save_get_data(sol, "pendulum.bob", sol_iterations)  # you don't have to specify the extension ".bob"

    # --- Load result of get_data --- #
    with open("pendulum.bob", "rb") as file:
        data = pickle.load(file)

    # --- Save the optimal control program and the solution --- #
    ocp.save(sol, "pendulum.bo")  # you don't have to specify the extension ".bo"

    # --- Load the optimal control program and the solution --- #
    ocp_load, sol_load = OptimalControlProgram.load("pendulum.bo")

    # --- Show results --- #
    result = ShowResult(ocp_load, sol_load)
    result.animate()
