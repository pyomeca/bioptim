"""
All the examples in muscle_driven_with_contact are merely to show some dynamics and prepare some OCP for the tests.
It is not really relevant and will be removed when unitary tests for the dynamics will be implemented
"""

import importlib.util
from pathlib import Path

from bioptim import (
    MusclesBiorbdModel,
    OptimalControlProgram,
    DynamicsOptions,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
    SolutionMerge,
    Node,
    ContactType,
    OnlineOptim,
)
from bioptim.examples.utils import ExampleUtils
import numpy as np


# Load track_segment_on_rt
spec = importlib.util.spec_from_file_location(
    "data_to_track", str(Path(__file__).parent) + "/contact_forces_inequality_constraint_muscle.py"
)
data_to_track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_to_track)


def prepare_ocp(
    biorbd_model_path, phase_time, n_shooting, muscle_activations_ref, contact_forces_ref, ode_solver=OdeSolver.RK4()
):
    # BioModel path
    bio_model = MusclesBiorbdModel(
        biorbd_model_path, with_residual_torque=True, contact_types=[ContactType.RIGID_EXPLICIT]
    )
    tau_min, tau_max, tau_init = -500.0, 500.0, 0.0
    activation_min, activation_max, activation_init = 0.0, 1.0, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_CONTROL, key="muscles", target=muscle_activations_ref, node=Node.ALL_SHOOTING
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_EXPLICIT_RIGID_CONTACT_FORCES, target=contact_forces_ref, node=Node.ALL_SHOOTING
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=0.001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=0.001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=0.001)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="torque", weight=0.001)

    # Dynamics
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
    )

    # Path constraint
    q_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = q_at_first_node

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = q_at_first_node

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = ([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)
    u_bounds["muscles"] = ([activation_min] * bio_model.nb_muscles, [activation_max] * bio_model.nb_muscles)

    u_init = InitialGuessList()
    u_init["tau"] = [tau_init] * bio_model.nb_tau
    u_init["muscles"] = [activation_init] * bio_model.nb_muscles

    # ------------- #

    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


def main():
    # Define the problem
    biorbd_model_path = ExampleUtils.folder + "/models/2segments_4dof_2contacts_1muscle.bioMod"
    final_time = 0.7
    ns = 20

    # Generate data using another optimization that will be feedback in as tracking data
    ocp_to_track = data_to_track.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        phase_time=final_time,
        n_shooting=ns,
        min_bound=50,
        max_bound=np.inf,
    )
    sol = ocp_to_track.solve()

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau, mus = states["q"], states["qdot"], controls["tau"], controls["muscles"]

    x = np.concatenate((q, qdot), axis=0)
    u = np.concatenate((tau, mus), axis=0)
    contact_forces_ref = (
        np.array([ocp_to_track.nlp[0].contact_forces_func([], x[:, i], u[:, i], [], [], []) for i in range(ns)])
        .squeeze()
        .T
    )
    muscle_activations_ref = mus

    # Track these data
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        phase_time=final_time,
        n_shooting=ns,
        muscle_activations_ref=muscle_activations_ref,
        contact_forces_ref=contact_forces_ref,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(online_optim=OnlineOptim.DEFAULT))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
