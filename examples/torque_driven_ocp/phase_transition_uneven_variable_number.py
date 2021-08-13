import biorbd_casadi as biorbd
from casadi import MX
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    Node,
    QAndQDotBounds,
    ObjectiveFcn,
    BiMappingList,
    Axis,
    PhaseTransitionList,
)

def custom_dof_matcher(state_pre: MX, state_post: MX) -> MX:
    transition_penalty = state_pre - state_post
    return transition_penalty


def prepare_ocp(
        biorbd_model_path: str = "double_pendulum.bioMod",
        biorbd_model_path_withTranslations: str = "double_pendulum_WithTranslations.bioMod"
        ) -> OptimalControlProgram:

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path_withTranslations))

    # Problem parameters
    n_shooting = (20, 30)
    final_time = (2, 3)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0], [1], phase=0)
    tau_mappings.add("tau", [None, None, None, 0], [3], phase=1)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-1000, axes=Axis.Z, phase=1)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.01, max_bound=3, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.01, max_bound=3, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))

    # Phase 0
    x_bounds[0][1, 0] = 0
    x_bounds[0][0, 0] = 3.14

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([1] * (biorbd_model[1].nbQ() + biorbd_model[1].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(tau_mappings[0]["tau"].to_first), [tau_max] * len(tau_mappings[0]["tau"].to_first))
    u_bounds.add([tau_min] * len(tau_mappings[1]["tau"].to_first), [tau_max] * len(tau_mappings[1]["tau"].to_first))

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * len(tau_mappings[0]["tau"].to_first))
    u_init.add([tau_init] * len(tau_mappings[1]["tau"].to_first))

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_dof_matcher, phase_pre_idx=0, states_pre_idx=[0, 1, 2, 3], states_post_idx=[2, 3, 6, 7])

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=tau_mappings,
        phase_transitions=phase_transitions,
    )

def main():

    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    sol = ocp.solve(show_online_optim=False)

    sol.print()
    sol.animate()

if __name__ == "__main__":
    main()


