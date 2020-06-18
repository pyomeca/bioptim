import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ProblemType,
    BidirectionalMapping,
    Mapping,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path="cubeSym.bioMod", ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    torque_min, torque_max, torque_init = -100, 100, 0
    all_generalized_mapping = BidirectionalMapping(Mapping([0, 1, 2, 2], [3]), Mapping([0, 1, 2]))

    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100}

    # Dynamics
    variable_type = {"type": ProblemType.TORQUE_DRIVEN}

    # Constraints
    constraints = (
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker_idx": 0, "second_marker_idx": 1,},
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 2,},
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model, all_generalized_mapping)
    X_bounds.min[3:6, [0, -1]] = 0
    X_bounds.max[3:6, [0, -1]] = 0

    # Initial guess
    X_init = InitialConditions([0] * all_generalized_mapping.reduce.len * 2)

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * all_generalized_mapping.reduce.len, [torque_max] * all_generalized_mapping.reduce.len,
    )
    U_init = InitialConditions([torque_init] * all_generalized_mapping.reduce.len)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        all_generalized_mapping=all_generalized_mapping,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
