"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end. Moreover, the movement must be cyclic, meaning
that the states at the end and at the beginning are equal. It is designed to provide a comprehensible example of the way
to declare a cyclic constraint or objective function

A phase transition loop constraint is treated as hard penalty (constraint)
if weight is <= 0 [or if no weight is provided], or as a soft penalty (objective) otherwise
"""


import biorbd
from bioptim import (
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    OdeSolver,
    PhaseTransitionList,
    PhaseTransitionFcn,
)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    loop_from_constraint: bool,
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    loop_from_constraint: bool
        If the looping cost should be a constraint [True] or an objective [False]
    ode_solver: OdeSolver
        The type of ode solver used

    Returns
    -------
    The ocp ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, name="tau", weight=100)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="m0", second_marker="m2")
    constraints.add(ConstraintFcn.TRACK_STATE, node=Node.MID, index=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1")

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    # First node is free but mid and last are constrained to be exactly at a certain point.
    # The cyclic penalty ensures that the first node and the last node are the same.
    x_bounds[2:6, -1] = [1.57, 0, 0, 0]

    # Initial guess
    x_init = InitialGuess([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque())

    u_init = InitialGuess([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #
    # A phase transition loop constraint is treated as
    # hard penalty (constraint) if weight is <= 0 [or if no weight is provided], or
    # as a soft penalty (objective) otherwise
    phase_transitions = PhaseTransitionList()
    if loop_from_constraint:
        phase_transitions.add(PhaseTransitionFcn.CYCLIC, weight=0)
    else:
        phase_transitions.add(PhaseTransitionFcn.CYCLIC, weight=10000)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        phase_transitions=phase_transitions,
    )


def main():
    """
    Runs and animate the program
    """

    ocp = prepare_ocp("cube.bioMod", n_shooting=30, final_time=2, loop_from_constraint=True)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
