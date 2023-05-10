"""
This is a very simple example aiming to show how vairable scaling can be used.
Variable scaling is important for the conditioning of the problem thus it might improve for convergence.
This example is copied from the getting_started/pendulum.py example.

One scaling should be declared for each phase for the states and controls. The scaling of the parameters should be
declared in the parameter declaration like in the example getting_started/custom_parameters.py.
"""

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
    CostType,
    Solver,
    VariableScalingList,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
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
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = biorbd_model.bounds_from_ranges(["q", "qdot"])
    x_bounds.min[2:4, :] = -3.14 * 100
    x_bounds.max[2:4, :] = 3.14 * 100
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = biorbd_model.nb_q
    n_qdot = biorbd_model.nb_q
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = biorbd_model.nb_tau
    tau_min, tau_max, tau_init = -1000, 1000, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

    # Variable scaling
    x_scaling = VariableScalingList()
    x_scaling.add("q", scaling=[1, 3])  # declare keys in order, so that they are concatenated in the right order
    x_scaling.add("qdot", scaling=[85, 85])

    u_scaling = VariableScalingList()
    u_scaling.add("tau", scaling=[900, 1])

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_scaling=x_scaling,
        u_scaling=u_scaling,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1 / 10, n_shooting=30)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    sol.graphs()

    # --- Show the results in a bioviz animation --- #
    sol.detailed_cost_values()
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
