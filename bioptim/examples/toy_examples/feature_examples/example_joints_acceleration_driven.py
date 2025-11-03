"""
A simple optimal control program consisting of a double pendulum starting downward and ending upward while requiring the
 minimum of generalized forces. The solver is only allowed to apply an angular acceleration the joint linking the second
 pendulum to the first one.

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

from bioptim import (
    JointAccelerationBiorbdModel,
    OptimalControlProgram,
    DynamicsOptions,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    JointAccelerationBiorbdModel,
    OnlineOptim,
)
from bioptim.examples.utils import ExampleUtils


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
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
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = JointAccelerationBiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints")

    # DynamicsOptions
    dynamics = DynamicsOptions(ode_solver=ode_solver)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][0, -1] = 3.14
    x_bounds["q"][1, -1] = 0

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuessList()
    x_init.add("q", initial_guess=[0] * n_q)
    x_init.add("qdot", initial_guess=[0] * n_qdot)

    # Define control path constraint
    n_qddot_joints = bio_model.nb_qddot - bio_model.nb_root  # 2 - 1 = 1 in this example
    qddot_joints_min, qddot_joints_max, qddot_joints_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds.add(
        "qddot_joints", min_bound=[qddot_joints_min] * n_qddot_joints, max_bound=[qddot_joints_max] * n_qddot_joints
    )

    u_init = InitialGuessList()
    u_init.add("qddot_joints", initial_guess=[qddot_joints_init] * n_qddot_joints)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    biorbd_model_path = ExampleUtils.folder + "/models/double_pendulum.bioMod"
    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path, final_time=10, n_shooting=100)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(online_optim=OnlineOptim.DEFAULT))
    # sol.graphs()

    # --- Show the results in a bioviz animation --- #
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
