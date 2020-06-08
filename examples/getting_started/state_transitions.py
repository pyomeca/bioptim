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
    StateTransition,
    ShowResult,
    OdeSolver,
)


def custom_state_transition(state_pre, state_post, idx_1, idx_2):
    """
    Custom function returning the value to be added in the constraint or objective vector (if there is a weight higher
    than zero) and whose value we want to be 0.
    In this example of custom function for state transition, this custom function ensures continuity between states
    whose index is between idx_1 and idx_2 (idx_2 not included).
    """
    return state_pre[idx_1:idx_2] - state_post[idx_1:idx_2]


def prepare_ocp(biorbd_model_path="cube.bioMod", ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = (
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
    )

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
    variable_type = (
        ProblemType.torque_driven,
        ProblemType.torque_driven,
        ProblemType.torque_driven,
        ProblemType.torque_driven,
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
        ({"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 1,},),
    )

    # Path constraint
    X_bounds = [
        QAndQDotBounds(biorbd_model[0]),
        QAndQDotBounds(biorbd_model[0]),
        QAndQDotBounds(biorbd_model[0]),
        QAndQDotBounds(biorbd_model[0]),
    ]

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

    """
    By default, all state transitions (here phase 0 to phase 1, phase 1 to phase 2 and phase 2 to phase 3)
    are continuous. In the event that one (or more) state transition(s) is desired to be discontinuous, 
    as for example IMPACT or CUSTOM can be used as below. 
    "phase_pre_idx" corresponds to the index of the phase preceding the transition.
    IMPACT will cause an impact related discontinuity when defining one or more contact points in the model.
    CUSTOM will allow to call the custom function previously presented in order to have its own state transition.
    Finally, if you want a state transition (continuous or not) between the last and the first phase (cyclicity) 
    you can use the dedicated StateTransition.Cyclic or use a continuous set at the lase phase_pre_idx.
    
    If for some reason, you don't want the state transition to be hard constraint, you can specify a weight higher than
    zero. It will thereafter be treated as a Mayer objective function with the specified weight. 
    """

    state_transitions = (
        {"type": StateTransition.IMPACT, "phase_pre_idx": 1,},
        {
            "type": StateTransition.CUSTOM,
            "phase_pre_idx": 2,
            "function": custom_state_transition,
            "idx_1": 1,
            "idx_2": 3,
        },
        {"type": StateTransition.CYCLIC},
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
        state_transitions=state_transitions,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
