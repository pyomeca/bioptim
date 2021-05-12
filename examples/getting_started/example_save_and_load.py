"""
This is a clone of the getting_started/pendulum.py example.

It is designed to show how to create and solve a problem, and afterward, save it to the hard drive and reload it.

It shows an example of *.bo method and how to use the stand_alone boolean parameter of the save function. If set
to True, the variable dictionaries (states, controls and parameters) are saved instead of the full Solution class
itself. This allows to load the saved file into a setting where bioptim is not installed using the pickle package, but
prevents from using the class methods Solution offers after loading the file.
"""

import pickle
from time import time

import numpy as np
from casadi import MX
import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    PlotType,
)


def custom_plot_callback(x: MX, q_to_plot: list) -> MX:
    """
    Create a used defined plot function with extra_parameters

    Parameters
    ----------
    x: MX
        The current states of the optimization
    q_to_plot: list
        The slice indices to plot

    Returns
    -------
    The value to plot
    """

    return x[q_to_plot, :]


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    n_threads: int,
    use_sx: bool = False,
    ode_solver: OdeSolver = OdeSolver.RK4(),
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
    n_threads: int
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
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        n_threads=n_threads,
        use_sx=use_sx,
        ode_solver=ode_solver,
    )


def main():
    """
    Create and solve a program. Then it saves it using the .bo method, and then using te stand_alone option.
    """

    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, n_shooting=100, n_threads=4)

    # --- Solve the program --- #
    tic = time()
    sol = ocp.solve(show_online_optim=True)
    toc = time() - tic
    print(f"Time to solve : {toc}sec")

    # --- Print objective cost  --- #
    print(f"Final objective value : {np.nansum(sol.cost)} \n")
    sol.print()

    # --- Save the optimal control program and the solution with stand_alone = False --- #
    ocp.save(sol, "pendulum.bo")  # you don't have to specify the extension ".bo"

    # --- Load the optimal control program and the solution --- #
    ocp_load, sol_load = OptimalControlProgram.load("pendulum.bo")

    # --- Show results --- #
    sol_load.animate()
    sol_load.graphs()

    # --- Save the optimal control program and the solution with stand_alone = True --- #
    ocp.save(sol, f"pendulum_sa.bo", stand_alone=True)

    # --- Load the solution saved with stand_alone = True --- #
    with open(f"pendulum_sa.bo", "rb") as file:
        states, controls, parameters = pickle.load(file)


if __name__ == "__main__":
    main()
