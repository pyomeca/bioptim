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
    StateTransition,
)


def prepare_ocp(biorbd_model_path, number_shooting_points, final_time, loop_from_constraint, ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = [{"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100}]

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.MID, "first_marker_idx": 0, "second_marker_idx": 2,},
        {"type": Constraint.TRACK_STATE, "instant": Instant.MID, "states_idx": 2,},
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.END, "first_marker_idx": 0, "second_marker_idx": 1,},
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[2:6, -1] = [1.57, 0, 0, 0]
    X_bounds.max[2:6, -1] = [1.57, 0, 0, 0]

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque(), [torque_max] * biorbd_model.nbGeneralizedTorque(),
    )
    U_init = InitialConditions([torque_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #
    # A state transition loop constraint is treated as
    # hard penalty (constraint) if weight is <= 0 [or if no weight is provided], or
    # as a soft penalty (objective) otherwise
    if loop_from_constraint:
        state_transitions = ({"type": StateTransition.CYCLIC, "weight": 0},)
    else:
        state_transitions = ({"type": StateTransition.CYCLIC, "weight": 10000},)

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
        ode_solver=ode_solver,
        state_transitions=state_transitions,
    )


if __name__ == "__main__":
    # First node is free but mid and last are constrained to be exactly at a certain point.
    # The cyclic penalty ensures that the first node and the last node are the same.
    ocp = prepare_ocp("cube.bioMod", number_shooting_points=30, final_time=2, loop_from_constraint=True)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
