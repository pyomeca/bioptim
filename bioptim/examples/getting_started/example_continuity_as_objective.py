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

import platform

import numpy as np
from casadi import sqrt

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Node,
    DynamicsFcn,
    Dynamics,
    BoundsList,
    InterpolationType,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintFcn,
    ConstraintList,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    Solution,
    PenaltyController,
    PhaseDynamics,
    SolutionMerge,
)


def out_of_sphere(controller: PenaltyController, y, z):
    q = controller.states["q"].cx
    marker_q = controller.model.marker(1)(q, controller.parameters.cx)

    distance = sqrt((y - marker_q[1]) ** 2 + (z - marker_q[2]) ** 2)

    return distance


def prepare_ocp_first_pass(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    state_continuity_weight: float,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    minimize_time: bool = True,
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
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    minimize_time: bool
        If the time should be minimized

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=1, key="tau")
    if minimize_time:
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100 / n_shooting)

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        state_continuity_weight=state_continuity_weight,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=ode_solver,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuessList()
    x_init["q"] = [0] * n_q
    x_init["qdot"] = [0] * n_qdot
    x_init.add_noise(bounds=x_bounds, magnitude=0.001, n_shooting=n_shooting + 1)

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -300, 300, 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuessList()
    u_init["tau"] = [tau_init] * n_tau
    u_init.add_noise(bounds=u_bounds, magnitude=0.01, n_shooting=n_shooting)

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
        use_sx=use_sx,
        n_threads=n_threads,
    )


def prepare_ocp_second_pass(
    biorbd_model_path: str,
    n_shooting: int,
    solution: Solution,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    minimize_time: bool = True,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    solution: Solution
        The first pass solution
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    minimize_time: bool
        If the time should be minimized

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, weight=1, key="tau")
    if minimize_time:
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100 / n_shooting)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, ode_solver=ode_solver)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(
        "q", solution.decision_states(to_merge=SolutionMerge.NODES)["q"], interpolation=InterpolationType.EACH_FRAME
    )
    x_init.add(
        "qdot",
        solution.decision_states(to_merge=SolutionMerge.NODES)["qdot"],
        interpolation=InterpolationType.EACH_FRAME,
    )

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -300, 300, 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuessList()
    u_init.add(
        "tau",
        solution.decision_controls(to_merge=SolutionMerge.NODES)["tau"],
        interpolation=InterpolationType.EACH_FRAME,
    )

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="marker_2", second_marker="target_2")
    constraints.add(out_of_sphere, y=-0.45, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=0.05, z=0, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=0.55, z=-0.85, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=0.75, z=0.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=1.4, z=0.5, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)
    constraints.add(out_of_sphere, y=2, z=1.2, min_bound=0.35, max_bound=np.inf, node=Node.ALL_SHOOTING)

    final_time = float(solution.decision_time(to_merge=SolutionMerge.NODES)[-1, 0])

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

    solver_first = Solver.IPOPT(
        show_online_optim=platform.system() == "Linux",
        show_options=dict(show_bounds=True),
    )
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
    solver_second = Solver.IPOPT(
        show_online_optim=platform.system() == "Linux",
        show_options=dict(show_bounds=True),
    )
    solver_second.set_maximum_iterations(10000)

    ocp_second = prepare_ocp_second_pass(
        biorbd_model_path="models/pendulum_maze.bioMod", n_shooting=500, solution=sol_first, n_threads=3
    )

    # Custom plots
    ocp_second.add_plot_penalty(CostType.CONSTRAINTS)

    # --- Solve the ocp --- #
    sol_second = ocp_second.solve(solver_second)
    # sol_second.graphs()

    # --- Show the results in a bioviz animation --- #
    sol_first.print_cost()
    sol_first.animate(n_frames=100)

    sol_second.print_cost()
    sol_second.animate(n_frames=100)


if __name__ == "__main__":
    main()
