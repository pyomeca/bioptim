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
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    weight: float = 1,
    min_time=0,
    max_time=np.inf,
    assume_phase_dynamics: bool = True,
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
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # --- Options --- #
    bio_model = BiorbdModel(biorbd_model_path)
    tau_min, tau_max = -100, 100
    n_tau = bio_model.nb_tau

    # Add objective functions
    objective_functions = ObjectiveList()
    # A weight of -1 will maximize time
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight, min_bound=min_time, max_bound=max_time)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

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
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
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
    print(f"The optimized phase time is: {sol.parameters['time'][0, 0]}, good job Mayer!")
    sol.animate()


if __name__ == "__main__":
    main()
