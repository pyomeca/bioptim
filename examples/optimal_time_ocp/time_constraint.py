"""
This is a clone of the example/getting_started/pendulum.py where a pendulum must be balance. The difference is that
the time to perform the task is now free for the solver to change. This example shows how to define such an optimal
control program
"""

import biorbd
from bioptim import (
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    Constraint,
    ConstraintFcn,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    Node,
    ShowResult,
    Data,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path: str, final_time: float, number_shooting_points: int, time_min: float, time_max: float, ode_solver: OdeSolver = OdeSolver.RK4) -> OptimalControlProgram:
    """
    Prepare the optimal control program

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    final_time: float
        The initial guess for the final time
    number_shooting_points: int
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

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = Constraint(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min, max_bound=time_max)

    # Path constraint
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[n_q - 1, -1] = 3.14

    # Initial guess
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    """
    Prepare, solve and animate a free time ocp
    """

    time_min = 0.6
    time_max = 1
    ocp = prepare_ocp(
        biorbd_model_path="pendulum.bioMod",
        final_time=2,
        number_shooting_points=50,
        time_min=time_min,
        time_max=time_max,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    print(f"The optimized phase time is: {param['time'][0, 0]}")

    result = ShowResult(ocp, sol)
    result.animate()
