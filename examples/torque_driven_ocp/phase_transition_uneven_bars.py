import biorbd_casadi as biorbd
from casadi import MX
from bioptim import (
    PenaltyNode,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    Dynamics,
    ConstraintList,
    ConstraintFcn,
    Bounds,
    BoundsList,
    InitialGuessList,
    Node,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    BiMappingList,
    Axis,
    PhaseTransitionList,
    PhaseTransitionFcn,
)

def custom_dof_matcher(state_pre: MX, state_post: MX) -> MX:

    # transition_penalty = state_pre.cx_end[idx_1:idx_2, :] - state_post.cx[idx_1:idx_2, :]
    transition_penalty = state_pre.cx_end[[0, 1], :] - state_post.cx[[2, 3], :]

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
    tau_mappings.add("tau_0", [None, 0], [1], phase=0)
    tau_mappings.add("tau_1", [None, None, None, 0], [3], phase=1)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-1000, axes=Axis.Z, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))

    # Phase 0
    x_bounds[0][:, 0] = 0
    x_bounds[0][[0, 1], 0] = 3.14

    # Phase 1
    x_bounds[1][[2, 3], -1] = -3.14

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([1] * (biorbd_model[1].nbQ() + biorbd_model[1].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(tau_mappings[0]["tau_0"].to_first), [tau_max] * len(tau_mappings[0]["tau_0"].to_first))
    u_bounds.add([tau_min] * len(tau_mappings[1]["tau_1"].to_first), [tau_max] * len(tau_mappings[1]["tau_1"].to_first))

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * len(tau_mappings[0]["tau_0"].to_first))
    u_init.add([tau_init] * len(tau_mappings[1]["tau_1"].to_first))

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_dof_matcher, phase_pre_idx=0)

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
        variable_mappings=tau_mappings,
        phase_transitions=phase_transitions,
    )

def main():

    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    sol = ocp.solve(show_online_optim=True)  # , solver_options={'îpopt.max_iter':1}

    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()

