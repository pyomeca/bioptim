"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""
import platform

import numpy as np

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    ConstraintList,
    ConstraintFcn,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    BiorbdModel,
    ControlType,
    PhaseDynamics,
    Node,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
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
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Path state constraints
    constraints = ConstraintList()
    # TODO: @pariterre: why no warning message when nodes= instead of node=?
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=[0, 0])
    constraints.add(
        ConstraintFcn.BOUND_STATE,
        key="q",
        node=Node.MID,
        min_bound=np.array([-1, -2 * np.pi]),
        max_bound=np.array([5, 2 * np.pi]),
    )
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=0, node=Node.END, target=0)
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", index=1, node=Node.END, target=3.14)

    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=[0, 0])
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.END, target=np.array([0, 0]))

    # Define control path constraints
    constraints.add(
        ConstraintFcn.BOUND_CONTROL,
        key="tau",
        index=0,
        node=Node.ALL_SHOOTING,
        min_bound=-100,
        max_bound=100,
    )
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="tau", index=1, node=Node.ALL_SHOOTING, target=0)

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    n_tau = bio_model.nb_tau
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        constraints=constraints,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=1, n_shooting=30, n_threads=2)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- If one is interested in checking the conditioning of the problem, they can uncomment the following line --- #
    # ocp.check_conditioning()

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    # sol.graphs(show_bounds=True)

    # --- Show the results in a bioviz animation --- #
    sol.print_cost()
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
