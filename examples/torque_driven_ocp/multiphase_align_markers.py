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


def prepare_ocp(biorbd_model_path="cube.bioMod", show_online_optim=False, ode_solver=OdeSolver.RK, long_optim=False):
    # --- Options --- #
    # Model path
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    if long_optim:
        number_shooting_points = (100, 1000)
    else:
        number_shooting_points = (20, 30)
    final_time = (2, 5)
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = (
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
    )

    # Dynamics
    variable_type = (ProblemType.torque_driven, ProblemType.torque_driven)

    # Constraints
    constraints = (
        (
            {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker": 0, "second_marker": 1,},
            {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker": 0, "second_marker": 2,},
        ),
        ({"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker": 0, "second_marker": 1,},),
    )

    # Path constraint
    X_bounds = [QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0])]

    for bounds in X_bounds:
        for i in range(6):
            if i != 0 and i != 2:
                bounds.first_node_min[i] = 0
                bounds.last_node_min[i] = 0
                bounds.first_node_max[i] = 0
                bounds.last_node_max[i] = 0
    X_bounds[0].first_node_min[2] = 0.0
    X_bounds[0].first_node_max[2] = 0.0
    X_bounds[1].first_node_min[2] = 0.0
    X_bounds[1].first_node_max[2] = 0.0
    X_bounds[1].last_node_min[2] = 1.57
    X_bounds[1].last_node_max[2] = 1.57

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
    ]
    U_init = InitialConditions([torque_init] * biorbd_model[0].nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        number_shooting_points,
        final_time,
        objective_functions,
        (X_init, X_init),
        (U_init, U_init),
        X_bounds,
        U_bounds,
        constraints,
        ode_solver=ode_solver,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=True, long_optim=True)

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
