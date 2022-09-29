import biorbd_casadi as biorbd
import numpy as np
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    Node,
    QAndQDotBounds,
    ObjectiveFcn,
    BiMappingList,
    PhaseTransitionList,
    PhaseTransitionFcn,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    NodeMappingList,
)


def prepare_ocp(
    biorbd_model_path: str = "models/double_pendulum.bioMod",
    biorbd_model_path_modified_inertia: str = "models/double_pendulum_modified_inertia.bioMod",
) -> OptimalControlProgram:

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path_modified_inertia))

    # Problem parameters
    n_shooting = (40, 40)
    final_time = (1.5, 1.5)
    tau_min, tau_max, tau_init = -200, 200, 0

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0], [1], phase=0)
    tau_mappings.add("tau", [None, 0], [1], phase=1)

    # Phase mapping
    node_mappings = NodeMappingList()
    node_mappings.add("tau", phase_pre=0, phase_post=1, nodes_pre=Node.ALL, nodes_post=Node.ALL)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-6, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-6, phase=1)

    # Multi-node constraints
    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(MultinodeConstraintFcn.TIME_CONSTRAINT, phase_first_idx=0, phase_second_idx=1,
                              first_node=Node.END, second_node=Node.END)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))

    # Phase 0
    x_bounds[0][0, 0] = - np.pi
    x_bounds[0][1, 0] = 0
    x_bounds[0].min[0, 2] = np.pi - 0.1
    x_bounds[0].max[0, 2] = np.pi + 0.1
    x_bounds[0][1, 2] = 0

    # Phase 1
    x_bounds[1][0, 0] = - np.pi
    x_bounds[1][1, 0] = 0
    x_bounds[1].min[0, 2] = np.pi - 0.1
    x_bounds[1].max[0, 2] = np.pi + 0.1
    x_bounds[1][1, 2] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[1].nbQ() + biorbd_model[1].nbQdot()))

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
        PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=0,
    )

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
        node_mappings=node_mappings,
        phase_transitions=phase_transitions,
        multinode_constraints=multinode_constraints,
    )


def main():

    # --- Prepare the ocp --- #
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    sol.animate(n_frames=80)


if __name__ == "__main__":
    main()
