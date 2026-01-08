"""
This example executes on full rotation of two triple pendulums with different inertia.
The first DoF of each model is not actuated, the second DoF is actuated with the same torque for each model and the last DoF are independently actuated for the two models.
"""

from bioptim import (
    MultiTorqueBiorbdModel,
    OptimalControlProgram,
    DynamicsOptions,
    ObjectiveList,
    BoundsList,
    ObjectiveFcn,
    BiMappingList,
    PhaseDynamics,
)
from bioptim.examples.utils import ExampleUtils
import numpy as np


def prepare_ocp(
    biorbd_model_path,
    biorbd_model_path_modified_inertia,
    n_shooting: int = 40,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:

    # Adding the models to the same phase
    bio_models = MultiTorqueBiorbdModel((biorbd_model_path, biorbd_model_path_modified_inertia))

    # Problem parameters
    final_time = 1.5
    tau_min, tau_max = -200, 200

    # Variable Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", to_second=[None, 0, 1, None, 0, 2], to_first=[1, 2, 5])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1e-6)

    # Dynamics
    dynamics = DynamicsOptions(expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_models.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_models.bounds_from_ranges("qdot")

    x_bounds["q"][[0, 3], 0] = -np.pi
    x_bounds["q"][[1, 4], 0] = 0
    x_bounds["q"][[2, 5], 0] = 0
    x_bounds["q"].min[[0, 3], 2] = np.pi - 0.1
    x_bounds["q"].max[[0, 3], 2] = np.pi + 0.1
    x_bounds["q"][[1, 4], 2] = 0
    x_bounds["q"][[2, 5], 2] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * len(tau_mappings[0]["tau"].to_first), [tau_max] * len(tau_mappings[0]["tau"].to_first)

    return OptimalControlProgram(
        bio_models,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        variable_mappings=tau_mappings,
    )


def main():
    example_folder = ExampleUtils.folder
    biorbd_model_path = example_folder + "/models/triple_pendulum.bioMod"
    biorbd_model_path_modified_inertia = example_folder + "/models/triple_pendulum_modified_inertia.bioMod"

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path, biorbd_model_path_modified_inertia=biorbd_model_path_modified_inertia
    )
    ocp.add_plot_penalty()

    # --- Solve the program --- #
    sol = ocp.solve()

    sol.graphs(show_bounds=True)

    # --- Animate results with bioviz --- #
    # show_solution_animation = False
    # if show_solution_animation:
    #     states = sol.decision_states(to_merge=SolutionMerge.NODES)
    #     q = states["q"]
    #     import bioviz
    #
    #     b = bioviz.Viz(ExampleUtils.folder + "/models/triple_pendulum_both_inertia.bioMod")
    #     b.load_movement(q)
    #     b.exec()

    # --- Animate results with pyorerun --- #
    sol.animate(viewer="pyorerun")


if __name__ == "__main__":
    main()
