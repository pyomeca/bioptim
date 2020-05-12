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

def generate_data(biorbd_model, final_time, nb_shooting):
    # Aliases
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_markers = biorbd_model.nbMarkers()
    nb_contacts = biorbd_model.nbContacts()
    dt = final_time / nb_shooting

    # Casadi related stuff
    symbolic_states = MX.sym("x", nb_q + nb_qdot, 1)
    symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
    nlp = {
        "model": biorbd_model,
        "nbTau": nb_tau,
        "nbMuscle": nb_mus,
        "q_mapping": BidirectionalMapping(Mapping(range(nb_q)), Mapping(range(nb_q))),
        "q_dot_mapping": BidirectionalMapping(Mapping(range(nb_qdot)), Mapping(range(nb_qdot))),
        "tau_mapping": BidirectionalMapping(Mapping(range(nb_tau)), Mapping(range(nb_tau))),
    }
    markers_func = []
    for i in range(nb_markers):
        markers_func.append(
            Function(
                "ForwardKin",
                [symbolic_states],
                [biorbd_model.marker(symbolic_states[:nb_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )
    dynamics_func = Function(
        "ForwardDyn",
        [symbolic_states, symbolic_controls],
        [Dynamics.forward_dynamics_torque_muscle_driven_with_contact(symbolic_states, symbolic_controls, nlp)],
        ["x", "u"],
        ["xdot"],
    ).expand()

    compute_contact = Function(
        "computeContactForces",
        [symbolic_states, symbolic_controls],
        [Dynamics.forces_from_forward_dynamics_torque_muscle_driven_with_contact(symbolic_states, symbolic_controls, nlp)],
        ["x", "u"],
        ["contact_forces"],
    ).expand()

    def dyn_interface(t, x, u):
        u = np.concatenate([np.zeros(nb_tau), u])
        return np.array(dynamics_func(x, u)).squeeze()

    # Generate some muscle activation
    U = np.random.rand(nb_shooting, nb_mus)

    # Integrate and collect the position of the markers accordingly
    X = np.ndarray((nb_q + nb_qdot, nb_shooting + 1))
    markers = np.ndarray((3, biorbd_model.nbMarkers(), nb_shooting + 1))
    CF = np.ndarray((nb_contacts, nb_shooting + 1))

    def add_to_data(i, q):
        X[:, i] = q
        for j, mark_func in enumerate(markers_func):
            markers[:, j, i] = np.array(mark_func(q)).squeeze()

    x_init = np.array([0] * nb_q + [0] * nb_qdot)
    add_to_data(0, x_init)
    for i, u in enumerate(U):
        CF[:, i] = np.array(compute_contact(x_init, np.concatenate([np.zeros(nb_tau), u]))).squeeze()
        sol = solve_ivp(dyn_interface, (0, dt), x_init, method="RK45", args=(u,))
        x_init = sol["y"][:, -1]
        add_to_data(i + 1, x_init)

    time_interp = np.linspace(0, final_time, nb_shooting + 1)
    return time_interp, markers, X, U, CF


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
    #
    #
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": muscle_activations_ref},
        {"type": Objective.Lagrange.TRACK_CONTACT_FORCES, "weight": 100, "data_to_track": contact_forces_ref},
    )

    # Dynamics
    problem_type = ProblemType.muscles_and_torque_driven_with_contact

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    X_bounds.min[[0, 1], :] = -2 * np.pi
    X_bounds.max[[0, 1], :] = 2 * np.pi

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

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
    final_time = 0.5
    ns = 10

    # Generate random data to fit
    t, markers_ref, x_ref, muscle_activations_ref, contact_forces_ref = generate_data(biorbd_model, final_time, ns)

    # Track these data
    model_path = "2segments_4dof_2contacts_1muscle.bioMod"
    ocp = prepare_ocp(
        model_path=model_path,
        phase_time=final_time,
        number_shooting_points=ns,
        muscle_activations_ref=muscle_activations_ref,
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

    # markers = np.ndarray((3, n_mark, q.shape[1]))
    # symbolic_states = MX.sym("x", n_q, 1)
    # markers_func = Function(
    #     "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"],
    # ).expand()
    # for i in range(n_frames):
    #     markers[:, :, i] = markers_func(q[:, i])

    # plt.figure("Markers")
    # for i in range(markers.shape[1]):
    #     plt.plot(np.linspace(0, final_time, ns + 1), markers[:, i, :].T, "r--")
    #
    # plt.figure("Q")
    # plt.plot(np.linspace(0, 2, ns + 1), q.T, "r--")
    #
    # plt.figure("Tau")
    # plt.step(np.linspace(0, final_time, ns + 1), tau.T, where="post")

    muscle_activations_ref = np.append(muscle_activations_ref, muscle_activations_ref[-1:, :], axis=0)
    plt.figure("Muscle activations")
    plt.step(np.linspace(0, final_time, ns + 1), muscle_activations_ref, "k", where="post")
    plt.step(np.linspace(0, final_time, ns + 1), mus.T, "r--", where="post")

    plt.figure("Contact forces")
    plt.plot(np.linspace(0, final_time, ns + 1)[:-1], contact_forces_ref.T, "k.-")
    plt.plot(np.linspace(0, final_time, ns + 1)[:-1], contact_forces.T, "r.--")

    plt.show()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate()
