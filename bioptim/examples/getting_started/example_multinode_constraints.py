"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement and
a the at different marker at the end of each phase. Moreover a constraint on the rotation is imposed on the cube.
Extra constraints are defined between specific nodes of phases.
It is designed to show how one can define a multinode constraints and objectives in a multiphase optimal control program
"""

from casadi import MX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    MultinodeObjectiveList,
    MultinodeObjectiveFcn,
    PenaltyController,
    BiMapping,
    PhaseDynamics,
)


def custom_multinode_constraint(
    controllers: list[PenaltyController], coef: float, states_mapping: BiMapping = None
) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

    Parameters
    ----------
    controllers: list[PenaltyController]
        All the controller for the penalties
    coef: float
        The coefficient of the phase transition (makes no physical sens)
    states_mapping: BiMapping
        The mapping between states of the two nodes (if for instance they are not aligned)

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # states_mapping can be defined as an argument (such as coef). For this particular example, one could simply
    # ignore the mapping stuff (it is merely for the sake of example how to use the mappings)
    if states_mapping is None:
        states_mapping = BiMapping(range(controllers[0].states.cx.shape[0]), range(controllers[1].states.cx.shape[0]))
    states_pre = states_mapping.to_second.map(controllers[0].states.cx)
    states_post = states_mapping.to_first.map(controllers[1].states.cx)
    return states_pre * coef - states_post


def prepare_ocp(
    biorbd_model_path: str,
    n_shootings: tuple,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    with_too_much_constraints: bool = False,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    n_shootings: tuple
        The number of shooting points
    ode_solver: OdeSolverBase
        The ode solve to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.

    with_too_much_constraints: bool
        This is to show what happens in the case too many constraints are declared in the multinode constraints (that
        is more than three in the same phase). It will raise ValueError if phase_dynamics is
        PhaseDynamics.SHARED_DURING_THE_PHASE since maximum three nodes are created by phase.
        If is it set to PhaseDynamics.ONE_PER_NODE, it will work just fine
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    final_time = (2, 5, 4)
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics, ode_solver=ode_solver
    )
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics, ode_solver=ode_solver
    )
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics, ode_solver=ode_solver
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=2)

    # Constraints
    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    multinode_constraints.add(
        MultinodeConstraintFcn.STATES_EQUALITY,
        nodes_phase=(0, 2, 2),
        nodes=(Node.START, Node.START, Node.MID),
        key="all",
    )
    # Objectives with the weight as an argument
    multinode_objectives = MultinodeObjectiveList()
    multinode_objectives.add(
        MultinodeObjectiveFcn.STATES_EQUALITY, nodes_phase=(0, 2), nodes=(2, Node.MID), weight=2, key="all"
    )
    # Objectives with the weight as an argument
    multinode_objectives.add(
        MultinodeObjectiveFcn.STATES_EQUALITY, nodes_phase=(0, 1), nodes=(Node.MID, Node.END), weight=0.1, key="all"
    )
    # Objectives with the weight as an argument
    multinode_objectives.add(
        custom_multinode_constraint, nodes_phase=(0, 1), nodes=(Node.MID, Node.PENULTIMATE), weight=0.1, coef=2
    )

    # This is a useless constraint (as it already does that anyway) to show how to add three constraints on the same
    # phase. More than 3 will only work with phase_dynamics to PhaseDynamics.ONE_PER_NODE
    multinode_constraints.add(
        MultinodeConstraintFcn.CONTROLS_EQUALITY,
        nodes_phase=(1, 1, 1),
        nodes=(Node.START, Node.MID, Node.PENULTIMATE),
        index=2,
    )
    # This constraint is for documentation purposes. Up to 3 nodes, it will work, but it won't for more than 3 if
    # phase_dynamics is set to PhaseDynamics.SHARED_DURING_THE_PHASE
    if with_too_much_constraints:
        multinode_constraints.add(
            MultinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 0, 0, 0), nodes=(0, 1, 2, 3), key="all"
        )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)

    for bounds in x_bounds:
        bounds["q"][1, [0, -1]] = 0
        bounds["qdot"][:, [0, -1]] = 0
    x_bounds[0]["q"][2, 0] = 0.0
    x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=1)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=2)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shootings,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        multinode_objectives=multinode_objectives,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """

    ocp = prepare_ocp(biorbd_model_path="models/cube.bioMod", n_shootings=(100, 300, 100))

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    sol.print_cost()
    sol.graphs()
    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
