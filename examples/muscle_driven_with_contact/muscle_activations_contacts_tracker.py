import importlib.util
from pathlib import Path

import numpy as np
import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    Data,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)

# Load align_segment_on_rt
spec = importlib.util.spec_from_file_location(
    "data_to_track", str(Path(__file__).parent) + "/contact_forces_inequality_constraint_muscle.py"
)
data_to_track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_to_track)


def prepare_ocp(
    model_path, phase_time, number_shooting_points, muscle_activations_ref, contact_forces_ref,
):
    # Model path
    biorbd_model = biorbd.Model(model_path)
    torque_min, torque_max, torque_init = -500, 500, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": muscle_activations_ref},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 1, "data_to_track": contact_forces_ref},
    )

    # Dynamics
    problem_type = {"type": ProblemType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT}

    # Constraints
    constraints = ()

    # Path constraint
    nb_q = biorbd_model.nbQ()
    nb_qdot = nb_q
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize X_bounds
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[:, 0] = pose_at_first_node + [0] * nb_qdot
    X_bounds.max[:, 0] = pose_at_first_node + [0] * nb_qdot

    # Initial guess
    X_init = [InitialConditions(pose_at_first_node + [0] * nb_qdot)]

    # Define control path constraint
    U_bounds = [
        Bounds(
            min_bound=[torque_min] * biorbd_model.nbGeneralizedTorque()
            + [activation_min] * biorbd_model.nbMuscleTotal(),
            max_bound=[torque_max] * biorbd_model.nbGeneralizedTorque()
            + [activation_max] * biorbd_model.nbMuscleTotal(),
        )
    ]
    U_init = [
        InitialConditions(
            [torque_init] * biorbd_model.nbGeneralizedTorque() + [activation_init] * biorbd_model.nbMuscleTotal()
        )
    ]

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        phase_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
    )


if __name__ == "__main__":
    # Define the problem
    model_path = "2segments_4dof_2contacts_1muscle.bioMod"
    final_time = 0.7
    ns = 20

    # Generate data using another optimization that will be feedback in as tracking data
    ocp_to_track = data_to_track.prepare_ocp(
        model_path=model_path, phase_time=final_time, number_shooting_points=ns, direction="GREATER_THAN", boundary=50,
    )
    sol_to_track = ocp_to_track.solve()
    states, controls = Data.get_data(ocp_to_track, sol_to_track)
    q, q_dot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]
    x = np.concatenate((q, q_dot))
    u = np.concatenate((tau, mus))
    contact_forces_ref = np.array(ocp_to_track.nlp[0]["contact_forces_func"](x[:, :-1], u[:, :-1]))
    muscle_activations_ref = mus

    # Track these data
    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=final_time,
        number_shooting_points=ns,
        muscle_activations_ref=muscle_activations_ref[:, :-1].T,
        contact_forces_ref=contact_forces_ref.T,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
