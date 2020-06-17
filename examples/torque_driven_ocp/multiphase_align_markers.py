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


def prepare_ocp(biorbd_model_path="cube.bioMod", ode_solver=OdeSolver.RK, long_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    if long_optim:
        number_shooting_points = (100, 300, 100)
    else:
        number_shooting_points = (20, 30, 20)
    final_time = (2, 5, 4)
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = (
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
    )

    # Dynamics
    variable_type = (
        {"type": ProblemType.TORQUE_DRIVEN},
        {"type": ProblemType.TORQUE_DRIVEN},
        {"type": ProblemType.TORQUE_DRIVEN},
    )

    # Constraints
    constraints = (
        (
            {
                "type": Constraint.ALIGN_MARKERS,
                "instant": Instant.START,
                "first_marker_idx": 0,
                "second_marker_idx": 1,
            },
            {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},
        ),
        ({"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 1,},),
        ({"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},),
    )

    # Path constraint
    X_bounds = [QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0])]

    for bounds in X_bounds:
        for i in [1, 3, 4, 5]:
            bounds.min[i, [0, -1]] = 0
            bounds.max[i, [0, -1]] = 0
    X_bounds[0].min[2, 0] = 0.0
    X_bounds[0].max[2, 0] = 0.0
    X_bounds[2].min[2, [0, -1]] = [0.0, 1.57]
    X_bounds[2].max[2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    U_bounds = [
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
    ]
    U_init = InitialConditions([torque_init] * biorbd_model[0].nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        number_shooting_points,
        final_time,
        (X_init, X_init, X_init),
        (U_init, U_init, U_init),
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(long_optim=False)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
