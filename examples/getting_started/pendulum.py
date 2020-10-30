import pickle
from time import time

import numpy as np
import biorbd

from bioptim import (
    OptimalControlProgram,
    DynamicsType,
    DynamicsTypeOption,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    ShowResult,
    Data,
    Simulate,
    Objective,
    ObjectiveOption,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, nb_threads, use_SX=False):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = ObjectiveOption(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE)

    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    x_init = InitialGuessOption([0] * (n_q + n_qdot))

    # Define control path constraint
    u_bounds = BoundsOption([[tau_min] * n_tau, [tau_max] * n_tau])
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuessOption([tau_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        nb_threads=nb_threads,
        use_SX=use_SX,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, number_shooting_points=100, nb_threads=4)

    # --- Solve the program --- #
    tic = time()
    sol, sol_iterations, sol_obj = ocp.solve(show_online_optim=True, return_iterations=True, return_objectives=True)
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Simulation --- #
    # It is not an optimal control, it only apply a Runge Kutta at each nodes
    Simulate.from_solve(ocp, sol, single_shoot=True)
    Simulate.from_data(ocp, Data.get_data(ocp, sol), single_shoot=False)

    # --- Access to all iterations  --- #
    if sol_iterations:  # If the processor is too fast, this will be empty since it is attached to the update function
        nb_iter = len(sol_iterations)
        third_iteration = sol_iterations[2]

    # --- Print objective cost  --- #
    print(f"Final objective value : {np.nansum(sol_obj)} \n")
    analyse = Objective.Printer(ocp, sol_obj)
    analyse.by_function()
    analyse.by_nodes()

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
