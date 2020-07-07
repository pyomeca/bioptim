import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
)


def prepare_ocp(
    biorbd_model_path,
    final_time,
    number_shooting_points,
    marker_velocity_or_displacement,
    marker_in_first_coordinates_system,
):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    biorbd_model.markerNames()
    # Problem parameters
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    if marker_in_first_coordinates_system:
        coordinates_system_idx = 0
    else:
        coordinates_system_idx = -1

    if marker_velocity_or_displacement == "disp":
        objective_functions = (
            {
                "type": Objective.Lagrange.MINIMIZE_MARKERS_DISPLACEMENT,
                "coordinates_system_idx": coordinates_system_idx,
                "markers_idx": 6,
                "weight": 1000,
            },
            {"type": Objective.Lagrange.MINIMIZE_STATE, "states_idx": 6, "weight": -1},
            {"type": Objective.Lagrange.MINIMIZE_STATE, "states_idx": 7, "weight": -1},
        )
    elif marker_velocity_or_displacement == "velo":
        objective_functions = (
            {"type": Objective.Lagrange.MINIMIZE_MARKERS_VELOCITY, "markers_idx": 6, "weight": 1000},
            {"type": Objective.Lagrange.MINIMIZE_STATE, "states_idx": 6, "weight": -1},
            {"type": Objective.Lagrange.MINIMIZE_STATE, "states_idx": 7, "weight": -1},
        )
    else:
        raise RuntimeError(
            "Wrong choice of marker_velocity_or_displacement, actual value is "
            "{marker_velocity_or_displacement}, should be 'velo' or 'disp'."
        )

    # Dynamics
    problem_type = {"type": ProblemType.TORQUE_DRIVEN}

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    for i in range(nq, 2 * nq):
        X_bounds.min[i, :] = -10
        X_bounds.max[i, :] = 10

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))
    for i in range(2):
        X_init.init[i] = 1.5
    for i in range(4, 6):
        X_init.init[i] = 0.7
    for i in range(6, 8):
        X_init.init[i] = 0.6

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque(), [torque_max] * biorbd_model.nbGeneralizedTorque(),
    )
    U_init = InitialConditions([torque_init] * biorbd_model.nbGeneralizedTorque())

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
        nb_integration_steps=5,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="cube_and_line.bioMod",
        number_shooting_points=30,
        final_time=2,
        marker_velocity_or_displacement="disp",  # "velo"
        marker_in_first_coordinates_system=True,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
