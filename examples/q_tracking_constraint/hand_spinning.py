import numpy as np
import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    Constraint,
    Objective,
    ProblemType,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    StateTransition,
)


def state_transition_function(state_pre, state_post):
    return state_pre[1:] - state_post[1:]


def prepare_ocp(biorbd_model_path="HandSpinner.bioMod"):
    end_crank_idx = 0
    hand_marker_idx = 18

    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Problem parameters
    number_shooting_points = 30
    final_time = 1.0

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_MARKERS_DISPLACEMENT, "weight": 1, "markers_idx": hand_marker_idx},
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 1},
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
    )

    # Dynamics
    problem_type = {"type": ProblemType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN}

    # Constraints
    constraints = (
        {
            "type": Constraint.ALIGN_MARKERS,
            "first_marker_idx": hand_marker_idx,
            "second_marker_idx": end_crank_idx,
            "instant": Instant.ALL,
        },
        {
            "type": Constraint.TRACK_STATE,
            "instant": Instant.ALL,
            "states_idx": 0,
            "data_to_track": np.linspace(0, 2 * np.pi, number_shooting_points + 1),
        },
    )

    state_transitions = ({"type": StateTransition.CUSTOM, "function": state_transition_function, "phase_pre_idx": 0,},)

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Initial guess
    X_init = InitialConditions([0, -0.9, 1.7, 0.9, 2.0, -1.3] + [0] * biorbd_model.nbQdot())

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
        state_transitions=state_transitions,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
