import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    Objective,
    ProblemType,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -1, 1, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1},
        {"type": Objective.Mayer.ALIGN_MARKERS, "first_marker_idx": 0, "second_marker_idx": 5, "weight": 1,},
    )

    # Dynamics
    problem_type = {"type": ProblemType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN}

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Set the initial position
    X_bounds.min[:, 0] = (0.07, 1.4, 0, 0)
    X_bounds.max[:, 0] = (0.07, 1.4, 0, 0)

    # Initial guess
    X_init = InitialConditions([1.57] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    U_init = InitialConditions(
        [torque_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal()
    )
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
    ocp = prepare_ocp(biorbd_model_path="arm26.bioMod", final_time=2, number_shooting_points=20)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
