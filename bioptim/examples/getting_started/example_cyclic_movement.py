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
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    PhaseTransitionList,
    PhaseTransitionFcn,
    Solver,
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    loop_from_constraint: bool,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
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
    ode_solver: OdeSolverBase
        The type of ode solver used
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN, ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="m0", second_marker="m2")
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.MID, index=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1")

    # Path constraint
    # First node is free but mid and last are constrained to be exactly at a certain point.
    # The cyclic penalty ensures that the first node and the last node are the same.
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][2, -1] = 1.57  # end with cube 90 degrees rotated
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, -1] = 0  # Start and end without any velocity

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau

    # ------------- #
    # A phase transition loop constraint is treated as
    # hard penalty (constraint) if weight is <= 0 [or if no weight is provided], or
    # as a soft penalty (objective) otherwise
    phase_transitions = PhaseTransitionList()
    if loop_from_constraint:
        phase_transitions.add(PhaseTransitionFcn.CYCLIC)
    else:
        phase_transitions.add(PhaseTransitionFcn.CYCLIC, weight=10000)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transitions,
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
