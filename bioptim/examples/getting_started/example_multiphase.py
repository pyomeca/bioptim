"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement and
a the at different marker at the end of each phase. Moreover a constraint on the rotation is imposed on the cube.
Finally, an objective for the transition continuity on the control is added. Please note that the "last" control
of the previous phase is the last shooting node (and not the node arrival).
It is designed to show how one can define a multiphase optimal control program
"""

import platform

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    PenaltyController,
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
    CostType,
    MultinodeObjectiveList,
    PhaseDynamics,
    ControlType,
    QuadratureRule,
)


def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls.cx - post.controls.cx


def prepare_ocp(
    biorbd_model_path: str = "models/cube.bioMod",
    ode_solver: list[OdeSolverBase] = [OdeSolver.RK4(), OdeSolver.RK4(), OdeSolver.RK4()],
    long_optim: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
    quadrature_rule: QuadratureRule = QuadratureRule.RECTANGLE_LEFT,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolverBase
        The ode solve to use
    long_optim: bool
        If the solver should solve the precise optimization (500 shooting points) or the approximate (50 points)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls
    quadrature_rule: QuadratureRule
        The quadrature method to use to integrate the objective functions

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    if long_optim:
        n_shooting = (100, 300, 100)
    else:
        n_shooting = (20, 30, 20)
    final_time = (2, 5, 4)
    tau_min, tau_max = -100, 100

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=100,
        phase=0,
        integration_rule=quadrature_rule,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=100,
        phase=1,
        integration_rule=quadrature_rule,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="tau",
        weight=100,
        phase=2,
        integration_rule=quadrature_rule,
    )

    multinode_objective = MultinodeObjectiveList()
    multinode_objective.add(
        minimize_difference,
        weight=100,
        nodes_phase=(1, 2),
        nodes=(Node.PENULTIMATE, Node.START),
        quadratic=True,
    )

    # Dynamics
    if not isinstance(ode_solver):
        ode_solver = [ode_solver] * 3
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=ode_solver[0],
    )
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=ode_solver[1],
    )
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=ode_solver[2],
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=2)

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
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_objectives=multinode_objective,
        control_type=control_type,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """

    ocp = prepare_ocp(long_optim=False)
    ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    sol.graphs(show_bounds=True)

    # --- Show results --- #
    sol.print_cost()
    sol.animate()


if __name__ == "__main__":
    main()
