import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    Node,
    ObjectiveFcn,
    BiMappingList,
    Axis,
    PhaseTransitionList,
    PhaseTransitionFcn,
    BiMapping,
)


def prepare_ocp(
    biorbd_model_path: str = "models/double_pendulum.bioMod",
    biorbd_model_path_withTranslations: str = "models/double_pendulum_with_translations.bioMod",
    n_shooting: tuple = (40, 40),
) -> OptimalControlProgram:
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path_withTranslations))

    # Problem parameters
    final_time = (1.5, 2.5)
    tau_min, tau_max, tau_init = -200, 200, 0

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0], [1], phase=0)
    tau_mappings.add("tau", [None, None, None, 0], [3], phase=1)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=0.01, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=0.01, phase=1)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-1000, axes=Axis.Z, phase=1, quadratic=False
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=2, node=Node.END, weight=-100, phase=1, quadratic=False
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.3, max_bound=3, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.3, max_bound=3, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"]))

    # Phase 0
    x_bounds[0][1, 0] = 0
    x_bounds[0][0, 0] = 3.14
    x_bounds[0].min[0, -1] = 2 * 3.14

    # Phase 1
    x_bounds[1][[0, 1, 4, 5], 0] = 0
    x_bounds[1].min[2, -1] = 3 * 3.14

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([1] * (bio_model[1].nb_q + bio_model[1].nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(tau_mappings[0]["tau"].to_first), [tau_max] * len(tau_mappings[0]["tau"].to_first))
    u_bounds.add([tau_min] * len(tau_mappings[1]["tau"].to_first), [tau_max] * len(tau_mappings[1]["tau"].to_first))

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * len(tau_mappings[0]["tau"].to_first))
    u_init.add([tau_init] * len(tau_mappings[1]["tau"].to_first))

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(
        PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0, states_mapping=BiMapping([0, 1, 2, 3], [2, 3, 6, 7])
    )

    return OptimalControlProgram(
        bio_model,
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
        assume_phase_dynamics=True,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
