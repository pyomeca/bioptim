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
)


def generate_data(biorbd_model, final_time, nb_shooting, residual_torques=True):
    # Aliases
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_markers = biorbd_model.nbMarkers()
    dt = final_time / nb_shooting

    # Casadi related stuff
    symbolic_states = MX.sym("x", nb_q + nb_qdot, 1)

    if residual_torques:
        symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
    else:
        symbolic_controls = MX.sym("u", nb_mus, 1)

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

    if residual_torques:
        nlp = {
            "model": biorbd_model,
            "nbTau": nb_tau,
            "nbMuscle": nb_mus,
            "q_mapping": BidirectionalMapping(Mapping(range(nb_q)), Mapping(range(nb_q))),
            "q_dot_mapping": BidirectionalMapping(Mapping(range(nb_qdot)), Mapping(range(nb_qdot))),
            "tau_mapping": BidirectionalMapping(Mapping(range(nb_tau)), Mapping(range(nb_tau))),
        }
        dynamics_func = Function(
            "ForwardDyn",
            [symbolic_states, symbolic_controls],
            [Dynamics.forward_dynamics_torque_muscle_driven(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()
    else:
        nlp = {
            "model": biorbd_model,
            "nbMuscle": nb_mus,
            "q_mapping": BidirectionalMapping(Mapping(range(nb_q)), Mapping(range(nb_q))),
            "q_dot_mapping": BidirectionalMapping(Mapping(range(nb_qdot)), Mapping(range(nb_qdot))),
        }
        dynamics_func = Function(
            "ForwardDyn",
            [symbolic_states, symbolic_controls],
            [Dynamics.forward_dynamics_muscle_activations_driven(symbolic_states, symbolic_controls, nlp)],
            ["x", "u"],
            ["xdot"],
        ).expand()

    def dyn_interface_true(t, x, u):
        u = np.concatenate([np.zeros(nb_tau), u])
        return np.array(dynamics_func(x, u)).squeeze()

    def dyn_interface_false(t, x, u):
        return np.array(dynamics_func(x, u)).squeeze()

    # Generate some muscle activation
    U = np.random.rand(nb_shooting, nb_mus)

    # Integrate and collect the position of the markers accordingly
    X = np.ndarray((nb_q + nb_qdot, nb_shooting + 1))
    markers = np.ndarray((3, biorbd_model.nbMarkers(), nb_shooting + 1))

    def add_to_data(i, q):
        X[:, i] = q
        for j, mark_func in enumerate(markers_func):
            markers[:, j, i] = np.array(mark_func(q)).squeeze()

    x_init = np.array([0] * nb_q + [0] * nb_qdot)
    add_to_data(0, x_init)
    for i, u in enumerate(U):
        if residual_torques:
            sol = solve_ivp(dyn_interface_true, (0, dt), x_init, method="RK45", args=(u,))
        else:
            sol = solve_ivp(dyn_interface_false, (0, dt), x_init, method="RK45", args=(u,))

        x_init = sol["y"][:, -1]
        add_to_data(i + 1, x_init)

    time_interp = np.linspace(0, final_time, nb_shooting + 1)
    return time_interp, markers, X, U


def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    activations_ref,
    q_ref,
    kin_data_to_track="markers",
    residual_torques=True,
):
    # Problem parameters
    torque_min, torque_max, torque_init = -100, 100, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    nq = biorbd_model.nbQ()

    # Add objective functions
    objective_functions = [
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": activations_ref},
    ]

    if residual_torques:
        objective_functions.append({"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1})

    if kin_data_to_track == "markers":
        objective_functions.append(
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
        )
    elif kin_data_to_track == "q":
        objective_functions.append(
            {
                "type": Objective.Lagrange.TRACK_STATE,
                "weight": 100,
                "data_to_track": q_ref,
                "states_idx": range(biorbd_model.nbQ()),
            },
        )
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    if residual_torques:
        variable_type = ProblemType.muscle_activations_and_torque_driven
    else:
        variable_type = ProblemType.muscle_activations_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    X_bounds.min[:nq, :] = -2 * np.pi
    X_bounds.max[:nq, :] = 2 * np.pi

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    if residual_torques:
        U_bounds = Bounds(
            [torque_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(),
            [torque_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal(),
        )
        U_init = InitialConditions(
            [torque_init] * biorbd_model.nbGeneralizedTorque() + [activation_init] * biorbd_model.nbMuscleTotal()
        )
    else:
        U_bounds = Bounds(
            [activation_min] * biorbd_model.nbMuscleTotal(), [activation_max] * biorbd_model.nbMuscleTotal(),
        )
        U_init = InitialConditions([activation_init] * biorbd_model.nbMuscleTotal())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        nb_shooting,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("arm26.bioMod")
    final_time = 2
    n_shooting_points = 29

    residual_torques = False

    # Generate random data to fit
    t, markers_ref, x_ref, muscle_activations_ref = generate_data(
        biorbd_model, final_time, n_shooting_points, residual_torques
    )

    # Track these data
    biorbd_model = biorbd.Model("arm26.bioMod")  # To allow for non free variable, the model must be reloaded
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        kin_data_to_track="q",
        residual_torques=residual_torques,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show the results --- #
    muscle_activations_ref = np.append(muscle_activations_ref, muscle_activations_ref[-1:, :], axis=0)

    states, controls = Data.get_data(ocp, sol["x"])
    q = states["q"]
    qdot = states["q_dot"]
    mus = controls["muscles"]

    if residual_torques:
        tau = controls["tau"]

    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = Function(
        "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"],
    ).expand()
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    plt.figure("Markers")
    for i in range(markers.shape[1]):
        plt.plot(np.linspace(0, 2, n_shooting_points + 1), markers_ref[:, i, :].T, "k")
        plt.plot(np.linspace(0, 2, n_shooting_points + 1), markers[:, i, :].T, "r--")

    # --- Plot --- #
    plt.show()
