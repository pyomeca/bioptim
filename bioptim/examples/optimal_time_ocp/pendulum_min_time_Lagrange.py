"""
This is a clone of the example/getting_started/pendulum.py where a pendulum must be balance. The difference is that
the time to perform the task is now free and minimized by the solver. This example shows how to define such an optimal
control program with a Lagrange criteria (integral of dt)

The difference between Mayer and Lagrange minimization time is that the former can define bounds to
the values, while the latter is the most common way to define optimal time
"""

import platform

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
    CostType,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    weight: float = 1,
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
    ode_solver: OdeSolver
        The ode solver to use
    weight: float
        The weighting of the minimize time objective function

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    # A weight of -1 will maximize time
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=weight)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, [0, -1]] = 0
    x_bounds[0][n_q - 1, -1] = 3.14

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[0][n_tau - 1, :] = 0

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
        assume_phase_dynamics=True,
    )


def main():
    """
    Prepare, solve and animate a time minimizer ocp using a Lagrange criteria
    """

    ocp = prepare_ocp(biorbd_model_path="models/pendulum.bioMod", final_time=2, n_shooting=50)

    # Let's show the objectives
    ocp.add_plot_penalty(CostType.OBJECTIVES)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    print(f"The optimized phase time is: {sol.parameters['time'][0, 0]}, good job Lagrange!")

    sol.animate()


if __name__ == "__main__":
    main()
