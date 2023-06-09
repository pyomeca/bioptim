"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""
import numpy as np

# import the custom model
from bioptim.examples.custom_model.custom_package import MyModel

from bioptim import (
    OptimalControlProgram,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    DynamicsList,
)


def prepare_ocp(
    model: MyModel,
    final_time: float,
    n_shooting: int,
    configure_dynamics: callable = None,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    model: BioModel
        The object of the custom model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    configure_dynamics: callable
        The function to configure the dynamics
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(configure_dynamics, dynamic_function=dynamics)

    # Path constraint
    # the pendulum is constrained to point down with zero velocity at the beginning
    # the pendulum has to be in vertical position pointing up
    # at the end of the movement (3.14 rad) with zero speed (0 rad/s)
    x_bounds = BoundsList()
    x_bounds["q"] = np.array([[0, -6.28, 3.14]]), np.array([[0, 6.28, 3.14]])
    x_bounds["qdot"] = np.array([[0, -20, 0]]), np.array([[0, 20, 0]])

    # Initial guess
    n_q = model.nb_q
    n_qdot = model.nb_qdot
    x_init = InitialGuessList()
    x_init["q"] = [20] * n_q
    x_init["qdot"] = [20] * n_qdot

    # Define control path constraint
    n_tau = model.nb_tau
    tau_min, tau_max, tau_init = -20, 20, 10
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau

    u_init = InitialGuessList()
    u_init["tau"] = [tau_init] * n_tau

    return OptimalControlProgram(
        model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=2,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    If main.py is run as a script, it will perform the optimization
    """

    # import the custom dynamics and configuration
    from bioptim.examples.custom_model.custom_package.custom_dynamics import custom_configure_my_dynamics

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        model=MyModel(),
        final_time=1,
        n_shooting=30,
        configure_dynamics=custom_configure_my_dynamics,
    )

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=True, to_graph=False)

    # --- Solve the ocp --- #
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(100)
    sol = ocp.solve(solver=solver)

    # --- Show results --- #
    sol.graphs(show_bounds=True)
    sol.detailed_cost_values()
    sol.print_cost()

    # --- Animation --- #
    # not implemented yet
    # sol.animate()


if __name__ == "__main__":
    main()
