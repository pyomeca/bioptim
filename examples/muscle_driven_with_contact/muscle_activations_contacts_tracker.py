from scipy.integrate import solve_ivp
import numpy as np
import biorbd
from casadi import MX, Function
from matplotlib import pyplot as plt

from biorbd_optim import (
    OptimalControlProgram,
    BidirectionalMapping,
    Mapping,
    Dynamics,
    Data,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)

def prepare_ocp(
    model_path,
    phase_time,
    number_shooting_points,
    muscle_activations_ref,
    contact_forces_ref,
    show_online_optim,
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
    problem_type = ProblemType.muscles_and_torque_driven_with_contact

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
        Bounds(min_bound=[torque_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(), max_bound=[torque_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal())
    ]
    U_init = [
        InitialConditions([torque_init] * biorbd_model.nbGeneralizedTorque() + [activation_init] * biorbd_model.nbMuscleTotal())]

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        phase_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("2segments_4dof_2contacts_1muscle.bioMod")
    final_time = 0.3
    ns = 10

    # Load data to fit from previous exemple
    contact_forces_ref = np.load("contact_forces.npy")
    muscle_activations_ref = np.load("muscle_activations.npy")

    # Track these data
    model_path = "2segments_4dof_2contacts_1muscle.bioMod"
    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=final_time,
        number_shooting_points=ns,
        muscle_activations_ref=muscle_activations_ref[:, :-1].T,
        contact_forces_ref=contact_forces_ref.T,
        show_online_optim=False,
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show the results --- #
    nlp = ocp.nlp[0]
    nlp["model"] = biorbd.Model(model_path)

    states, controls = Data.get_data(ocp, sol["x"])
    q, q_dot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]

    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]
    n_contact = ocp.nlp[0]["model"].nbContacts()

    x = np.concatenate((q, q_dot))
    u = np.concatenate((tau, mus))
    contact_forces = np.array(nlp["contact_forces_func"](x[:, :-1], u[:, :-1]))

    t = np.linspace(0, final_time, ns + 1)
    plt.figure("Muscle activations")
    plt.step(t, muscle_activations_ref.T, "k", where="post")
    plt.step(t, mus.T, "r--", where="post")

    plt.figure("Contact forces")
    plt.plot(t[:-1], contact_forces_ref.T, "k.-")
    plt.plot(t[:-1], contact_forces.T, "r.--")

    plt.figure("Residual forces")
    plt.step(t, tau.T, where="post")

    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
