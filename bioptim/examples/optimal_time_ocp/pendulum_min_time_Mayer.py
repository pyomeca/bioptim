"""
This is a clone of the example/getting_started/pendulum.py where a pendulum must be balance. The difference is that
the time to perform the task is now free and minimized by the solver. This example shows how to define such an optimal
control program with a Mayer criteria (value of final_time)

The difference between Mayer and Lagrange minimization time is that the former can define bounds to
the values, while the latter is the most common way to define optimal time
"""

import platform

import numpy as np
from bioptim import (
    TorqueBiorbdModel,
    OptimalControlProgram,
    Dynamics,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    ControlType,
    PhaseDynamics,
    SolutionMerge,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    weight: float = 1,
    min_time=0,
    max_time=np.inf,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
) -> OptimalControlProgram:
    """
    Prepare the optimal control program

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The initial guess for the final time
    n_shooting: int
        The number of shooting points
    ode_solver: OdeSolverBase
        The ode solver to use
    weight: float
        The weighting of the minimize time objective function
    min_time: float
        The minimum time allowed for the final node
    max_time: float
        The maximum time allowed for the final node
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    bio_model = TorqueBiorbdModel(biorbd_model_path)
    tau_min, tau_max = -100, 100
    n_tau = bio_model.nb_tau

    # Add objective functions
    objective_functions = ObjectiveList()
    # A weight of -1 will maximize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight, min_bound=min_time, max_bound=max_time)

    # Dynamics
    dynamics = Dynamics(ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][-1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"][-1, :] = 0

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        control_type=control_type,
    )


def main():
    """
    Prepare, solve and animate a time minimizer ocp using a Mayer criteria
    """

    ocp = prepare_ocp(
        biorbd_model_path="models/pendulum.bioMod", final_time=2, n_shooting=50, ode_solver=OdeSolver.RK4()
    )

    # Let's show the objectives
    ocp.add_plot_penalty(CostType.OBJECTIVES)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    times = float(sol.decision_time(to_merge=SolutionMerge.NODES)[-1, 0])
    print(f"The optimized phase time is: {times}, good job Mayer!")
    sol.animate()


if __name__ == "__main__":
    main()
