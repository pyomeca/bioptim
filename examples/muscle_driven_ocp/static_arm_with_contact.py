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


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, show_online_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -1, 1, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1},
        {"type": Objective.Mayer.ALIGN_MARKERS, "first_marker": 0, "second_marker": 5, "weight": 1,},
    )

    # Dynamics
    problem_type = ProblemType.muscles_and_torque_driven_with_contact

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Set the initial position
    X_bounds.first_node_min = (0, 0.07, 1.4, 0, 0, 0)
    X_bounds.first_node_max = (0, 0.07, 1.4, 0, 0, 0)

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
        biorbd_model_path="arm26_with_contact.bioMod", final_time=2, number_shooting_points=20, show_online_optim=True
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(show_meshes=False)
    result.graphs()
