"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement and
a the at different marker at the end of each phase. Moreover a constraint on the rotation is imposed on the cube.
Extra constraints are defined between specific nodes of phases.
It is designed to show how one can define a binode constraints and objectives in a multiphase optimal control program
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
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    BinodeConstraintList,
    BinodeConstraintFcn,
    BinodeConstraint,
    PenaltyController
)


def custom_binode_constraint(
    binode_constraint: BinodeConstraint, controllers: list[PenaltyController, ...], coef: float
) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

    Parameters
    ----------
    binode_constraint: BinodeConstraint
        The placeholder for the binode_constraint
    controllers: list[PenaltyController, ...]
        All the controller for the penalties
    coef: float
        The coefficient of the phase transition (makes no physical sens)

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # states_mapping can be defined in PhaseTransitionList. For this particular example, one could simply ignore the
    # mapping stuff (it is merely for the sake of example how to use the mappings)
    states_pre = binode_constraint.states_mapping.to_second.map(controllers[0].states.cx)
    states_post = binode_constraint.states_mapping.to_first.map(controllers[1].states.cx)
    return states_pre * coef - states_post


def prepare_ocp(
    biorbd_model_path: str,
    n_shootings: tuple,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    assume_phase_dynamics: bool = True,
    with_too_much_constraints: bool = False,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolverBase
        The ode solve to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node
    with_too_much_constraints: bool
        This is to show what happens in the case too many constraints are declared in the multinode constraints (that
        is more than three in the same phase). It will raise ValueError if assume_phase_dynamics is True since maximum
        three nodes are created by phase. If is it set too False, it will work just fine

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    final_time = (2, 5, 4)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=2)

    # Constraints
    binode_constraints = BinodeConstraintList()
    # hard constraint
    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 2, 2), nodes=(Node.START, Node.START, Node.MID), key="all"
    )
    # Objectives with the weight as an argument
    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 2), nodes=(2, Node.MID), weight=2, key="all"
    )
    # Objectives with the weight as an argument
    binode_constraints.add(
        BinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 1), nodes=(Node.MID, Node.END), weight=0.1, key="all"
    )
    # Objectives with the weight as an argument
    binode_constraints.add(
        custom_binode_constraint, nodes_phase=(0, 1), nodes=(Node.MID, Node.PENULTIMATE), weight=0.1, coef=2
    )

    # This is a useless constraint (as it already does that anyway) to show how to add three constraints on the same
    # phase. More than 3 will only work with assume_phase_dynamics to False
    binode_constraints.add(
        BinodeConstraintFcn.CONTROLS_EQUALITY,
        nodes_phase=(1, 1, 1),
        nodes=(Node.START, Node.MID, Node.PENULTIMATE),
        index=2,
    )
    # This constraint is for documentation purposes. Up to 3 nodes, it will work, but it won't for more than 3 if
    # assume_phase_dynamics is set to True
    if with_too_much_constraints:
        binode_constraints.add(
            BinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 0, 0, 0), nodes=(0, 1, 2, 3), key="all"
        )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds[i, [0, -1]] = 0
    x_bounds[0][2, 0] = 0.0
    x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)
    u_bounds.add([tau_min] * bio_model[0].nb_tau, [tau_max] * bio_model[0].nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[0].nb_tau)
    u_init.add([tau_init] * bio_model[0].nb_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shootings,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        binode_constraints=binode_constraints,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
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
