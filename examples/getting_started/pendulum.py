import biorbd

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ShowResult,
    ObjectiveFcn,
    Objective,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points):
    # Load a biorbd model
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE)  # Minimize the generalized forces

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0  # Initial and final position and velocities are 0 for all the degrees of freedom
    x_bounds[1, -1] = 3.14  # Except for the final position of the rotation that is half turn rotated

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

    # Prepare the ocp
    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
    )


if __name__ == "__main__":
    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, number_shooting_points=100)

    # --- Solve the ocp --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show the results in a bioviz animation --- #
    result = ShowResult(ocp, sol)
    result.animate()
