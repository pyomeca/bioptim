from bioptim import (
    TorqueBiorbdModel,
    OptimalControlProgram,
    DynamicsOptionsList,
    DynamicsOptions,
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
    PhaseDynamics,
)


def prepare_ocp(
    biorbd_model_path: str = "models/double_pendulum.bioMod",
    biorbd_model_path_with_translations: str = "models/double_pendulum_with_translations.bioMod",
    n_shooting: tuple = (40, 40),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:

    bio_model = (TorqueBiorbdModel(biorbd_model_path), TorqueBiorbdModel(biorbd_model_path_with_translations))

    # Problem parameters
    final_time = (0.5, 0.5)
    tau_min, tau_max = -200, 200

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", to_second=[None, 0], to_first=[1], phase=0)
    tau_mappings.add("tau", to_second=[None, None, None, 0], to_first=[3], phase=1)

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
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics))
    dynamics.add(DynamicsOptions(expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics))

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bio_model[1].bounds_from_ranges("qdot"), phase=1)

    # Phase 0
    x_bounds[0]["q"][0, 0] = 3.14
    x_bounds[0]["q"].min[0, -1] = 2 * 3.14
    x_bounds[0]["q"][1, 0] = 0

    # Phase 1
    x_bounds[1]["q"][[0, 1], 0] = 0
    x_bounds[1]["q"].min[2, -1] = 3 * 3.14
    x_bounds[1]["qdot"][[0, 1], 0] = 0

    # Initial guess
    x_init = InitialGuessList()
    # Phase 0 is initialized at 0
    x_init.add("q", [1] * bio_model[1].nb_q, phase=1)
    x_init.add("qdot", [1] * bio_model[1].nb_qdot, phase=1)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        "tau",
        min_bound=[tau_min] * len(tau_mappings[0]["tau"].to_first),
        max_bound=[tau_max] * len(tau_mappings[0]["tau"].to_first),
        phase=0,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min] * len(tau_mappings[1]["tau"].to_first),
        max_bound=[tau_max] * len(tau_mappings[1]["tau"].to_first),
        phase=1,
    )

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(
        PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0, states_mapping=BiMapping([0, 1, 2, 3], [2, 3, 6, 7])
    )

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
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

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
