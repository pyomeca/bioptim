"""
An optimal control program consisting in a pendulum starting downward and ending upward while requiring
the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

There is a catch however: there are regions in which the weight of the pendulum cannot go.
The problem is solved in two passes. In the first pass, continuity is an objective rather than a constraint.
The goal of the first pass is to find quickly find a good initial guess. This initial guess is then given
to the second pass in which continuity is a constraint to find the optimal solution.

During the optimization process, the graphs are updated real-time. Finally, once it finished optimizing, it animates
the model using the optimal solution.

User might want to start reading the script by the `main` function to get a better feel.
"""

from casadi import sqrt
import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Node,
    DynamicsFcn,
    Dynamics,
    Bounds,
    InterpolationType,
    NoisedInitialGuess,
    InitialGuess,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintFcn,
    ConstraintList,
    OdeSolver,
    CostType,
    Solver,
    Solution,
)


def out_of_sphere(all_pn, y, z):
    q = all_pn.nlp.states[0]["q"].mx    # TODO: [0] to [node_index]
    marker_q = all_pn.nlp.model.markers(q)[1]

    distance = sqrt((y - marker_q[1]) ** 2 + (z - marker_q[2]) ** 2)

    return all_pn.nlp.mx_to_cx("out_of_sphere", distance, all_pn.nlp.states[0]["q"])    # TODO: [0] to [node_index]


def prepare_ocp_first_pass(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    state_continuity_weight: float,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
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
    state_continuity_weight: float
        The weight on the continuity objective.
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

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=1, key="tau")
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, 0] = 0

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = NoisedInitialGuess([0] * (n_q + n_qdot), bounds=x_bounds, magnitude=0.001, n_shooting=n_shooting + 1)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -300, 300, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    u_init = NoisedInitialGuess([tau_init] * n_tau, bounds=u_bounds, magnitude=0.01, n_shooting=n_shooting)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="marker_2", second_marker="target_2")
    constraints.add(out_of_sphere, y=-0.45, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=0.05, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    # for another good example, comment out this line below here and in second pass (see HERE)
    constraints.add(out_of_sphere, y=0.55, z=-0.85, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=0.75, z=0.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=1.4, z=0.5, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=2, z=1.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        state_continuity_weight=state_continuity_weight,
    )


def prepare_ocp_second_pass(
    biorbd_model_path: str,
    solution: Solution,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
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

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=1, key="tau")
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, 0] = 0

    # Initial guess
    x_init = np.vstack((solution.states[0]["q"], solution.states[0]["qdot"]))
    x_init = InitialGuess(x_init, interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -300, 300, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuess(solution.controls[0]["tau"][:, :-1], interpolation=InterpolationType.EACH_FRAME)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="marker_2", second_marker="target_2")
    constraints.add(out_of_sphere, y=-0.45, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=0.05, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    # HERE (referenced in first pass)
    constraints.add(out_of_sphere, y=0.55, z=-0.85, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=0.75, z=0.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=1.4, z=0.5, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=2, z=1.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        solution.ns,
        solution.phase_time[-1],
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- First pass --- #
    # --- Prepare the ocp --- #
    np.random.seed(123456)
    ocp_first = prepare_ocp_first_pass(
        biorbd_model_path="models/pendulum_maze.bioMod",
        final_time=5,
        n_shooting=500,
        # change the weight to observe the impact on the continuity of the solution
        # or comment to see how the constrained program would fare
        state_continuity_weight=1_000_000,
        n_threads=3,
    )
    # ocp_first.print(to_console=True)

    solver_first = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    # change maximum iterations to affect the initial solution
    # it doesn't mather if it exits before the optimal solution, only that there is an initial guess
    solver_first.set_maximum_iterations(500)

    # Custom plots
    ocp_first.add_plot_penalty(CostType.OBJECTIVES)

    # --- Solve the ocp --- #
    sol_first = ocp_first.solve(solver_first)
    # sol_first.graphs()

    # # --- Second pass ---#
    # # --- Prepare the ocp --- #
    solver_second = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver_second.set_maximum_iterations(10000)

    ocp_second = prepare_ocp_second_pass(
        biorbd_model_path="models/pendulum_maze.bioMod", solution=sol_first, n_threads=3
    )

    # Custom plots
    ocp_second.add_plot_penalty(CostType.CONSTRAINTS)

    # --- Solve the ocp --- #
    sol_second = ocp_second.solve(solver_second)
    # sol_second.graphs()

    # --- Show the results in a bioviz animation --- #
    sol_first.detailed_cost_values()
    sol_first.print_cost()
    sol_first.animate(n_frames=100)

    sol_second.detailed_cost_values()
    sol_second.print_cost()
    sol_second.animate(n_frames=100)


if __name__ == "__main__":
    main()
