import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    BiMappingList,
    PhaseTransitionList,
    PhaseTransitionFcn,
    NodeMappingList,
)


def prepare_ocp(
    biorbd_model_path: str = "models/double_pendulum.bioMod",
    biorbd_model_path_modified_inertia: str = "models/double_pendulum_modified_inertia.bioMod",
    n_shooting: tuple = (40, 40),
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path_modified_inertia))

    # Problem parameters
    final_time = (1.5, 1.5)
    tau_min, tau_max, tau_init = -200, 200, 0

    # Variable Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0], [1], phase=0)
    tau_mappings.add("tau", [None, 0], [1], phase=1)

    # Parameters mapping
    parameter_mappings = BiMappingList()
    parameter_mappings.add("time", [0, 0], [0])

    # Phase mapping
    node_mappings = NodeMappingList()
    node_mappings.add("tau", map_controls=True, phase_pre=0, phase_post=1)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-6, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-6, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=bio_model[1].bounds_from_ranges(["q", "qdot"]))

    # Phase 0
    x_bounds[0][0, 0] = -np.pi
    x_bounds[0][1, 0] = 0
    x_bounds[0].min[0, 2] = np.pi - 0.1
    x_bounds[0].max[0, 2] = np.pi + 0.1
    x_bounds[0][1, 2] = 0

    # Phase 1
    x_bounds[1][0, 0] = -np.pi
    x_bounds[1][1, 0] = 0
    x_bounds[1].min[0, 2] = np.pi - 0.1
    x_bounds[1].max[0, 2] = np.pi + 0.1
    x_bounds[1][1, 2] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model[0].nb_q + bio_model[0].nb_qdot))
    x_init.add([0] * (bio_model[1].nb_q + bio_model[1].nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(tau_mappings[0]["tau"].to_first), [tau_max] * len(tau_mappings[0]["tau"].to_first))
    u_bounds.add()

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * len(tau_mappings[0]["tau"].to_first))
    u_init.add()

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(
        PhaseTransitionFcn.DISCONTINUOUS,
        phase_pre_idx=0,
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
        variable_mappings=tau_mappings,
        node_mappings=node_mappings,
        phase_transitions=phase_transitions,
        parameter_mappings=parameter_mappings,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    # Please note that this example is currently broken and will therefore raise a NotImplementedError

    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve()

    # sol.graphs()

    # --- Show results --- #
    show_solution_animation = False
    if show_solution_animation:
        q_both = np.vstack((sol.states[0]["q"], sol.states[1]["q"]))
        import bioviz

        b = bioviz.Viz("models/double_pendulum_both_inertia.bioMod")
        b.load_movement(q_both)
        b.exec()


if __name__ == "__main__":
    main()
