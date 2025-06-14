"""
This example is a trivial multiphase box that must superimpose different markers at beginning and end of each
phase with one of its corner
It is designed to show how one can define its phase transition constraints if the provided ones are not sufficient.

More specifically, this example mimics the behaviour of the most common PhaseTransitionFcn.CONTINUOUS
"""

import platform

from casadi import MX
from bioptim import (
    TorqueBiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsOptionsList,
    DynamicsOptions,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintFcn,
    ConstraintList,
    BoundsList,
    InitialGuessList,
    PhaseTransitionFcn,
    PhaseTransitionList,
    OdeSolver,
    OdeSolverBase,
    BiMapping,
    Solver,
    PenaltyController,
    PhaseDynamics,
)


def custom_phase_transition(
    controllers: list[PenaltyController, PenaltyController], coef: float, states_mapping: BiMapping = None
) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty
    coef: float
        The coefficient of the phase transition (makes no physical sens)

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
    biorbd_model_path: str = "models/cube.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
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

    # BioModel path
    bio_model = (
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
        TorqueBiorbdModel(biorbd_model_path),
    )

    # Problem parameters
    n_shooting = (20, 20, 20, 20)
    final_time = (2, 5, 4, 2)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=3)

    # DynamicsOptions
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics))

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=3)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds.add("q", bounds=bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=bio_model[3].bounds_from_ranges("qdot"), phase=3)

    x_bounds[0]["q"][1, 0] = 0
    x_bounds[0]["qdot"][:, 0] = 0
    x_bounds[-1]["q"][1, -1] = 0
    x_bounds[-1]["qdot"][:, -1] = 0

    x_bounds[0]["q"][2, 0] = 0.0
    x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]

    # Initial guess (Still optional)
    x_init = InitialGuessList()
    x_init.add("q", [0] * bio_model[0].nb_q, phase=0)
    x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=0)
    x_init.add("q", [0] * bio_model[0].nb_q, phase=1)
    x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=1)
    x_init.add("q", [0] * bio_model[0].nb_q, phase=2)
    x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=2)
    x_init.add("q", [0] * bio_model[0].nb_q, phase=3)
    x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=3)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=1)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=2)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=3)

    # Initial guess (Still optional)
    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=0)
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=1)
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=2)
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=3)

    """
    By default, all phase transitions (here phase 0 to phase 1, phase 1 to phase 2 and phase 2 to phase 3)
    are continuous. In the event that one (or more) phase transition(s) is desired to be discontinuous,
    as for example IMPACT or CUSTOM can be used as below.
    "phase_pre_idx" corresponds to the index of the phase preceding the transition.
    IMPACT will cause an impact related discontinuity when defining one or more contact points in the model.
    CUSTOM will allow to call the custom function previously presented in order to have its own phase transition.
    Finally, if you want a phase transition (continuous or not) between the last and the first phase (cyclicity)
    you can use the dedicated PhaseTransitionFcn.Cyclic or use a continuous set at the last phase_pre_idx.

    If for some reason, you don't want the phase transition to be hard constraint, you can specify a weight higher than
    zero. It will thereafter be treated as a Mayer objective function with the specified weight.
    """
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0, states_mapping=BiMapping(range(3), range(3)))
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)
    phase_transitions.add(custom_phase_transition, phase_pre_idx=2, coef=0.5)
    phase_transitions.add(PhaseTransitionFcn.CYCLIC)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transitions,
    )


def main():
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.print_cost()
    sol.animate()


if __name__ == "__main__":
    main()
