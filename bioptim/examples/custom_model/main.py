"""
This script doesn't use biorbd
This an example of how to use bioptim to solve a simple pendulum problem
"""
import numpy as np

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
    PhaseDynamics,
)

# import the custom model
from bioptim.examples.custom_model.custom_package import MyModel


def prepare_ocp(
    model: MyModel,
    final_time: float,
    n_shooting: int,
    configure_dynamics: callable = None,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    n_threads: int = 2,
    expand_dynamics: bool = True,
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
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    n_threads: int
        Number of threads to use
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_dynamics,
        dynamic_function=dynamics,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

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
        n_threads=n_threads,
    )


def main():
    """
    If main.py is run as a script, it will perform the optimization
    """

    # import the custom dynamics and configuration
    from bioptim.examples.custom_model.custom_package.custom_dynamics import (
        custom_configure_my_dynamics,
    )

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
    # sol.graphs(show_bounds=True)
    sol.print_cost()

    # --- Animation --- #
    # not implemented yet
    # sol.animate()


if __name__ == "__main__":
    main()
