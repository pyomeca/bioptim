import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -100, 100, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = {"type": Objective.Mayer.MINIMIZE_TIME}

    # Dynamics
    problem_type = {"type": ProblemType.TORQUE_DRIVEN}

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[:, [0, -1]] = 0
    X_bounds.max[:, [0, -1]] = 0
    X_bounds.min[n_q - 1, -1] = 3.14
    X_bounds.max[n_q - 1, -1] = 3.14

    # Initial guess
    X_init = InitialConditions([0] * (n_q + n_qdot))

    # Define control path constraint
    U_bounds = Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)
    U_bounds.min[n_tau - 1, :] = 0
    U_bounds.max[n_tau - 1, :] = 0

    U_init = InitialConditions([torque_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=2, number_shooting_points=50,)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    param = Data.get_data(ocp, sol["x"], get_states=False, get_controls=False, get_parameters=True)
    print(f"The optimized phase time is: {param['time'][0, 0]}, good job Mayer!")

    result = ShowResult(ocp, sol)
    result.animate()
