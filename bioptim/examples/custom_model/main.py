"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""
import numpy as np

from bioptim import (
    OptimalControlProgram,
    Bounds,
    InterpolationType,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    CostType,
    Solver,
    Model,
    DynamicsList,
    ObjectiveList,
)

# import the custom model
from my_model import MyModel

# import the custom dynamics and configuration
from custom_dynamics import custom_dynamics, custom_configure_my_dynamics


def prepare_ocp(
    model: Model,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=5),
    use_sx: bool = True,
    n_threads: int = 1,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    model: Model
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    # dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
    # Declare the dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure_my_dynamics, dynamic_function=custom_dynamics)

    # Path constraint
    x_bounds = Bounds(
        min_bound=np.array([[0, -6.28, 3.14], [0, -20, 0]]),
        max_bound=np.array([[0, 6.28, 3.14], [0, 20, 0]]),
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    # x_bounds[:, [0, -1]] = 0
    # x_bounds[0, -1] = 3.14
    # x_bounds[0, 0] = 1
    # x_bounds[0, -1] = 1

    # Initial guess
    n_q = model.nbQ()
    n_qdot = model.nbQdot()
    x_init = InitialGuess([20] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -20, 20, 10
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)

    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(model=MyModel(), final_time=1, n_shooting=30)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=True, to_graph=False)

    # --- Solve the ocp --- #
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(100)

    sol = ocp.solve(solver=solver)
    sol.graphs(show_bounds=True)

    # --- Show results --- #
    sol.detailed_cost_values()
    sol.print_cost()
    # sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
