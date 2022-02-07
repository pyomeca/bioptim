"""
This example is a trivial multiphase box that must superimpose different markers at beginning and end of each
phase with one of its corner
It is designed to show how one can define its phase transition constraints if the provided ones are not sufficient.

More specifically, this example mimics the behaviour of the most common PhaseTransitionFcn.CONTINUOUS
"""

from casadi import MX
import biorbd_casadi as biorbd
from bioptim import (
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsList,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintFcn,
    ConstraintList,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    PhaseTransitionFcn,
    PhaseTransitionList,
    OdeSolver,
    OptimizationVariableList,
    PhaseTransition,
    BiMapping,
    Solver,
    NonLinearProgram,
)


def custom_phase_transition(
    transition: PhaseTransition, nlp_pre: NonLinearProgram, nlp_post: NonLinearProgram, coef: float
) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

    Parameters
    ----------
    transition: PhaseTransition
        The placeholder for the transition
    nlp_pre: NonLinearProgram
        The nonlinear program of the pre phase
    nlp_post: NonLinearProgram
        The nonlinear program of the post phase
    coef: float
        The coefficient of the phase transition (makes no physical sens)

    Returns
    -------
    The constraint such that: c(x) = 0
    """
    state_pre = nlp_pre.states
    state_post = nlp_post.states
    # states_mapping can be defined in PhaseTransitionList. For this particular example, one could simply ignore the
    # mapping stuff (it is merely for the sake of example how to use the mappings)
    states_pre = transition.states_mapping.to_second.map(state_pre.cx_end)
    states_post = transition.states_mapping.to_first.map(state_post.cx)

    return states_pre * coef - states_post


def prepare_ocp(
    biorbd_model_path: str = "models/cube.bioMod", ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The type of ode solver used

    Returns
    -------
    The ocp ready to be solved
    """

    # Model path
    biorbd_model = (
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
        biorbd.Model(biorbd_model_path),
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

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=3)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][[1, 3, 4, 5], 0] = 0
    x_bounds[-1][[1, 3, 4, 5], -1] = 0

    x_bounds[0][2, 0] = 0.0
    x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

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
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
