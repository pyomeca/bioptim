"""
This example executes on full rotation of two triple pendulums with different inertia.
The first DoF of each model is not actuated, the second DoF is actuated with the same torque for each model and the last DoF are independently actuated for the two models.
"""

import numpy as np
from bioptim import (
    MultiBiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    BiMappingList,
)


def prepare_ocp(
    biorbd_model_path: str = "models/triple_pendulum.bioMod",
    biorbd_model_path_modified_inertia: str = "models/triple_pendulum_modified_inertia.bioMod",
    n_shooting: int = 40,
) -> OptimalControlProgram:
    # Adding the models to the same phase
    bio_models = MultiBiorbdModel((biorbd_model_path, biorbd_model_path_modified_inertia))

    # Problem parameters
    final_time = 1.5
    tau_min, tau_max, tau_init = -200, 200, 0

    # Variable Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0, 1, None, 0, 2], [1, 2, 5])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-6)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_models.bounds_from_ranges(["q", "qdot"]))

    x_bounds[0][[0, 3], 0] = -np.pi
    x_bounds[0][[1, 4], 0] = 0
    x_bounds[0][[2, 5], 0] = 0
    x_bounds[0].min[[0, 3], 2] = np.pi - 0.1
    x_bounds[0].max[[0, 3], 2] = np.pi + 0.1
    x_bounds[0][[1, 4], 2] = 0
    x_bounds[0][[2, 5], 2] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_models.nb_q + bio_models.nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(tau_mappings[0]["tau"].to_first), [tau_max] * len(tau_mappings[0]["tau"].to_first))

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * len(tau_mappings[0]["tau"].to_first))

    return OptimalControlProgram(
        bio_models,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        variable_mappings=tau_mappings,
        assume_phase_dynamics=True,
    )


def main():
    # --- Prepare the ocp --- #
    ocp = prepare_ocp()
    ocp.add_plot_penalty()

    # --- Solve the program --- #
    sol = ocp.solve()

    sol.graphs(show_bounds=True)

    # --- Show results --- #
    show_solution_animation = False
    if show_solution_animation:
        q = sol.states["q"]
        import bioviz

        b = bioviz.Viz("models/triple_pendulum_both_inertia.bioMod")
        b.load_movement(q)
        b.exec()


if __name__ == "__main__":
    main()
