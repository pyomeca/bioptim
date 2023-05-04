"""
This is a clone of the example/getting_started/pendulum.py where a pendulum must be balance. The difference is that
the time to perform the task is now free for the solver to change. This example shows how to define such an optimal
control program
"""

import platform

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    Constraint,
    ConstraintFcn,
    Bounds,
    InitialGuess,
    Node,
    OdeSolver,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    time_min: float,
    time_max: float,
    ode_solver: OdeSolver = OdeSolver.RK4(),
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
    time_min: float
        The minimal time the phase can have
    time_max: float
        The maximal time the phase can have
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = Constraint(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min, max_bound=time_max)

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[n_q - 1, -1] = 3.14

    # Initial guess
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

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
        constraints,
        ode_solver=ode_solver,
        assume_phase_dynamics=True,
    )


def main():
    """
    Prepare, solve and animate a free time ocp
    """

    time_min = 0.6
    time_max = 1
    ocp = prepare_ocp(
        biorbd_model_path="models/pendulum.bioMod",
        final_time=2,
        n_shooting=50,
        time_min=time_min,
        time_max=time_max,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    print(f"The optimized phase time is: {sol.parameters['time'][0, 0]}")

    sol.animate()


if __name__ == "__main__":
    main()
