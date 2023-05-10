"""
This is a clone of the getting_started/pendulum.py example.

It is designed to show how to create and solve a problem, and afterward, save it to the hard drive and reload it.

It shows an example of *.bo method and how to use the stand_alone boolean parameter of the save function. If set
to True, the variable dictionaries (states, controls and parameters) are saved instead of the full Solution class
itself. This allows to load the saved file into a setting where bioptim is not installed using the pickle package, but
prevents from using the class methods Solution offers after loading the file.
"""

import pickle

import numpy as np
from casadi import MX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
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
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    assume_phase_dynamics: bool = True,
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
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The ocp ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True)

    # Dynamics
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        bio_model,
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
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Create and solve a program. Then it saves it using the .bo method, and then using te stand_alone option.
    """

    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1, n_shooting=100, n_threads=4)

    # --- Solve the program --- #
    sol = ocp.solve()
    print(f"Time to solve : {sol.real_time_to_optimize}sec")

    # --- Print objective cost  --- #
    print(f"Final objective value : {np.nansum(sol.cost)} \n")
    sol.print_cost()

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
