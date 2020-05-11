import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, show_online_optim=False):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = ()

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.first_node_min = [0] * (n_q + n_qdot)
    X_bounds.first_node_max = [0] * (n_q + n_qdot)
    X_bounds.last_node_min = [0] * (n_q + n_qdot)
    X_bounds.last_node_max = [0] * (n_q + n_qdot)
    X_bounds.last_node_min[n_q - 1] = 3.14
    X_bounds.last_node_max[n_q - 1] = 3.14

    # Initial guess
    X_init = InitialConditions([0] * (n_q + n_qdot))

    # Define control path constraint
    U_bounds = Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)
    U_bounds.first_node_min[n_tau - 1] = 0
    U_bounds.first_node_max[n_tau - 1] = 0
    U_bounds.min[n_tau - 1] = 0
    U_bounds.max[n_tau - 1] = 0
    U_bounds.last_node_min[n_tau - 1] = 0
    U_bounds.last_node_max[n_tau - 1] = 0

    U_init = InitialConditions([torque_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="pendulum.bioMod", final_time=2, number_shooting_points=50, show_online_optim=False,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
