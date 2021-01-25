"""
This is a clone of the getting_started/pendulum.py example. It is designed to show how to create and solve a problem,
and afterward, save it to the hard drive and reload it. It shows an example of both *.bo and *.bob method
"""

import pickle
from time import time

import numpy as np
import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ShowResult,
    ObjectiveFcn,
    Objective,
    ObjectivePrinter,
    OdeSolver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    number_shooting_points: int,
    nb_threads: int,
    use_sx: bool = False,
    ode_solver: OdeSolver = OdeSolver.RK4,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    number_shooting_points: int
        The number of shooting points
    nb_threads: int
        The number of threads to use while using multithreading
    ode_solver: OdeSolver
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)

    Returns
    -------
    The ocp ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

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
        use_sx=use_sx,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    """
    Create and solve a program. Then it saves it using the .bob and .bo method
    """

    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, number_shooting_points=100, nb_threads=4)

    # --- Solve the program --- #
    tic = time()
    sol, sol_iterations, sol_obj = ocp.solve(show_online_optim=True, return_iterations=True, return_objectives=True)
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Access to all iterations  --- #
    if sol_iterations:  # If the processor is too fast, this will be empty since it is attached to the update function
        nb_iter = len(sol_iterations)
        third_iteration = sol_iterations[2]

    # --- Print objective cost  --- #
    print(f"Final objective value : {np.nansum(sol_obj)} \n")
    analyse = ObjectivePrinter(ocp, sol_obj)
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
