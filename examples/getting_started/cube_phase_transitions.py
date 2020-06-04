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
    PhaseTransition,
    ShowResult,
    OdeSolver,
)


def custom_phase_transition(state_pre, state_post, idx_1, idx_2):
    return state_pre[idx_1:idx_2] - state_post[idx_1:idx_2]


def prepare_ocp(biorbd_model_path="cube.bioMod", ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path),)

    # Problem parameters
    number_shooting_points = (20, 20, 20, 20)
    final_time = (2, 5, 4, 2)
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = (
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
        ({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},),
    )

    # Dynamics
    variable_type = (ProblemType.torque_driven, ProblemType.torque_driven, ProblemType.torque_driven, ProblemType.torque_driven)

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
        ({"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 1, },),
    )

    # Path constraint
    X_bounds = [QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0])]

    X_bounds[0].min[[1, 3, 4, 5], 0] = 0
    X_bounds[0].max[[1, 3, 4, 5], 0] = 0
    X_bounds[-1].min[[1, 3, 4, 5], -1] = 0
    X_bounds[-1].max[[1, 3, 4, 5], -1] = 0

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
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
    ]
    U_init = InitialConditions([torque_init] * biorbd_model[0].nbGeneralizedTorque())

    phase_transitions = (
        {"type": PhaseTransition.IMPACT, "phase_pre_idx": 1, },
        {"type": PhaseTransition.CUSTOM, "phase_pre_idx": 2, "function": custom_phase_transition, "idx_1": 1, "idx_2": 3, },
        {"type": PhaseTransition.CONTINUOUS, "phase_pre_idx": 3, },
    )

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        number_shooting_points,
        final_time,
        (X_init, X_init, X_init, X_init),
        (U_init, U_init, U_init, U_init),
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        phase_transitions=phase_transitions,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
