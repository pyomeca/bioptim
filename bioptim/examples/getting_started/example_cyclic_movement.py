"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end. Moreover, the movement must be cyclic, meaning
that the states at the end and at the beginning are equal. It is designed to provide a comprehensible example of the way
to declare a cyclic constraint or objective function

A phase transition loop constraint is treated as hard penalty (constraint)
if weight is <= 0 [or if no weight is provided], or as a soft penalty (objective) otherwise
"""

import platform

from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Bounds,
    InitialGuess,
    OdeSolver,
    OdeSolverBase,
    PhaseTransitionList,
    PhaseTransitionFcn,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    loop_from_constraint: bool,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    assume_phase_dynamics: bool = True,
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
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The ocp ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="m0", second_marker="m2")
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.MID, index=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1")

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    # First node is free but mid and last are constrained to be exactly at a certain point.
    # The cyclic penalty ensures that the first node and the last node are the same.
    x_bounds[2:6, -1] = [1.57, 0, 0, 0]

    # Initial guess
    x_init = InitialGuess([0] * (bio_model.nb_q + bio_model.nb_qdot))

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)

    u_init = InitialGuess([tau_init] * bio_model.nb_tau)

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
        bio_model,
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
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Runs and animate the program
    """

    ocp = prepare_ocp("models/cube.bioMod", n_shooting=30, final_time=2, loop_from_constraint=True)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
