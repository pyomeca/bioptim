import biorbd

from bioptim import (
    Node,
    OptimalControlProgram,
    DynamicsTypeOption,
    DynamicsType,
    ObjectiveOption,
    Objective,
    ConstraintList,
    Constraint,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    ShowResult,
    OdeSolver,
    StateTransitionList,
    StateTransition,
)


def prepare_ocp(biorbd_model_path, number_shooting_points, final_time, loop_from_constraint, ode_solver=OdeSolver.RK):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveOption(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.MID, first_marker_idx=0, second_marker_idx=2)
    constraints.add(Constraint.TRACK_STATE, node=Node.MID, index=2)
    constraints.add(Constraint.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=1)

    # Path constraint
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[2:6, -1] = [1.57, 0, 0, 0]

    # Initial guess
    x_init = InitialGuessOption([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsOption(
        [[tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque()]
    )

    u_init = InitialGuessOption([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #
    # A state transition loop constraint is treated as
    # hard penalty (constraint) if weight is <= 0 [or if no weight is provided], or
    # as a soft penalty (objective) otherwise
    state_transitions = StateTransitionList()
    if loop_from_constraint:
        state_transitions.add(StateTransition.CYCLIC, weight=0)
    else:
        state_transitions.add(StateTransition.CYCLIC, weight=10000)

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
