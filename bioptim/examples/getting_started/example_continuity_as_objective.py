"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

There is however a catch: there are no hard continuity constraint. The continuity is instead added to the objective
function. The idea behind this is to allow the solver to better find the solution more easily without hitting walls.

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

from casadi import logic_and
import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    Node,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    ObjectiveList,
    ConstraintFcn,
    ConstraintList,
    OdeSolver,
    CostType,
    Solver,
    BiorbdInterface,
)


def out_of_rect(all_pn, y, z, length, height, **extra):
    top = z
    left = y
    right = y + length
    bottom = z - height

    q = all_pn.nlp.states["q"].mx
    marker_q = all_pn.nlp.model.markers(q)[1].to_mx()
    y = marker_q[1]
    z = marker_q[2]

    lt_top = z < top
    gt_left = y > left
    lt_right = y < right
    gt_bottom = z > bottom

    in_rect = logic_and(lt_top, logic_and(gt_left, logic_and(lt_right, gt_bottom)))

    return BiorbdInterface.mx_to_cx("out_of_rect", in_rect, all_pn.nlp.states["q"])


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
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

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    # objective_functions = ObjectiveList()  # BUG: causes continuity objectives to disappear
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, max_bound=final_time, weight=1)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    Ytrans = 0
    Xrot = 1
    START = 0
    END = -1

    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, START] = 0
    # x_bounds.min[Ytrans, END] = -0.1  # Give a little slack on the end position
    # x_bounds.max[Ytrans, END] = 0.1
    # x_bounds.min[Xrot, END] = 3.14 - 0.1
    # x_bounds.max[Xrot, END] = 3.14 + 0.1
    x_bounds[Xrot, END] = 3.14

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuess([tau_init] * n_tau)

    constraints = ConstraintList()
    # constraints.add(out_of_rect, node=Node.ALL_SHOOTING, y=-0.85, z=0.5, length=1, height=0.5)
    constraints.add(out_of_rect, node=Node.ALL_SHOOTING, y=.65, z=-.5, length=.25, height=.5)
    # constraints.add(out_of_rect, node=Node.ALL_SHOOTING, y=2.25, z=1, length=1, height=2)
    # constraints.add(out_of_rect, node=Node.ALL_SHOOTING, y=-2.75, z=1, length=1, height=2)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, max_bound=final_time)

    return OptimalControlProgram(
        biorbd_model,
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
        # state_continuity_weight=1000000,  # change the weight to observe the impact on the continuity of the solution
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="models/pendulum_maze.bioMod", final_time=1, n_shooting=30, n_threads=3)

    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(10000)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(solver)
    # sol.graphs()

    # --- Show the results in a bioviz animation --- #
    sol.detailed_cost_values()
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
