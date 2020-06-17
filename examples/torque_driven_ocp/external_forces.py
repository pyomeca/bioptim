import numpy as np

import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path="cube_with_forces.bioMod", ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100}

    # Dynamics
    problem_type = {"type": ProblemType.TORQUE_DRIVEN}

    # Constraints
    constraints = (
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker_idx": 0, "second_marker_idx": 1,},
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},
    )

    # External forces
    external_forces = [
        np.repeat(
            np.array([[0, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 5]]).T[:, :, np.newaxis], number_shooting_points, axis=2
        )
    ]

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[3:6, [0, -1]] = 0
    X_bounds.max[3:6, [0, -1]] = 0

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

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
        objective_functions=objective_functions,
        constraints=constraints,
        external_forces=external_forces,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
