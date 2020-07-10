import numpy as np
import biorbd

from biorbd_optim import (
    Instant,
    OptimalControlProgram,
    ConstraintList,
    Constraint,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    ShowResult,
    StateTransitionList,
)


def state_transition_function(state_pre, state_post):
    return state_pre[1:] - state_post[1:]


def prepare_ocp(biorbd_model_path="HandSpinner.bioMod"):
    end_crank_idx = 0
    hand_marker_idx = 18

    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Problem parameters
    number_shooting_points = 30
    final_time = 1.0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_MARKERS_DISPLACEMENT, markers_idx=hand_marker_idx)
    objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        Constraint.ALIGN_MARKERS, first_marker_idx=hand_marker_idx, second_marker_idx=end_crank_idx, instant=Instant.ALL
    )
    constraints.add(
        Constraint.TRACK_STATE,
        instant=Instant.ALL,
        states_idx=0,
        target=np.linspace(0, 2 * np.pi, number_shooting_points + 1),
    )

    state_transitions = StateTransitionList()
    state_transitions.add(state_transition_function, phase_pre_idx=0)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))

    # Initial guess
    x_init = InitialConditionsList()
    x_init.add([0, -0.9, 1.7, 0.9, 2.0, -1.3] + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [
            [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
            [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
        ]
    )

    u_init = InitialConditionsList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
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
