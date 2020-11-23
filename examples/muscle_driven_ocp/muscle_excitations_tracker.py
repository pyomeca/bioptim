from scipy.integrate import solve_ivp
import numpy as np
import biorbd
from casadi import MX, Function
from matplotlib import pyplot as plt

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    BidirectionalMapping,
    Mapping,
    DynamicsTypeList,
    DynamicsType,
    DynamicsFunctions,
    Data,
    ObjectiveList,
    Objective,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
)


def generate_data(biorbd_model, final_time, nb_shooting):
    # Aliases
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_markers = biorbd_model.nbMarkers()
    dt = final_time / nb_shooting

    # Casadi related stuff
    symbolic_states = MX.sym("x", nb_q + nb_qdot + nb_mus, 1)
    symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
    symbolic_parameters = MX.sym("u", 0, 0)
    nlp = NonLinearProgram(
        model=biorbd_model,
        shape={
            "q": nb_q,
            "q_dot": nb_qdot,
            "tau": nb_tau,
            "muscle": nb_mus,
        },
        mapping={
            "q": BidirectionalMapping(Mapping(range(nb_q)), Mapping(range(nb_q))),
            "q_dot": BidirectionalMapping(Mapping(range(nb_qdot)), Mapping(range(nb_qdot))),
            "tau": BidirectionalMapping(Mapping(range(nb_tau)), Mapping(range(nb_tau))),
        },
    )
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
        [symbolic_states, symbolic_controls, symbolic_parameters],
        [
            DynamicsFunctions.forward_dynamics_muscle_excitations_and_torque_driven(
                symbolic_states, symbolic_controls, symbolic_parameters, nlp
            )
        ],
        ["x", "u", "p"],
        ["xdot"],
    ).expand()

    def dyn_interface(t, x, u):
        u = np.concatenate([np.zeros(nb_tau), u])
        return np.array(dynamics_func(x, u, np.empty((0, 0)))).squeeze()

    # Generate some muscle excitations
    U = np.random.rand(nb_shooting, nb_mus).T

    # Integrate and collect the position of the markers accordingly
    X = np.ndarray((nb_q + nb_qdot + nb_mus, nb_shooting + 1))
    markers = np.ndarray((3, biorbd_model.nbMarkers(), nb_shooting + 1))

    def add_to_data(i, q):
        X[:, i] = q
        for j, mark_func in enumerate(markers_func):
            markers[:, j, i] = np.array(mark_func(q)).squeeze()

    x_init = np.array([0] * nb_q + [0] * nb_qdot + [0.5] * nb_mus)
    add_to_data(0, x_init)
    for i, u in enumerate(U.T):
        sol = solve_ivp(dyn_interface, (0, dt), x_init, method="RK45", args=(u,))
        x_init = sol["y"][:, -1]
        add_to_data(i + 1, x_init)

    time_interp = np.linspace(0, final_time, nb_shooting + 1)
    return time_interp, markers, X, U


def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    excitations_ref,
    q_ref,
    use_residual_torque,
    kin_data_to_track="markers",
):
    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    excitation_min, excitation_max, excitation_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, target=excitations_ref)
    if use_residual_torque:
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE)
    if kin_data_to_track == "markers":
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=100, target=markers_ref)
    elif kin_data_to_track == "q":
        objective_functions.add(
            Objective.Lagrange.TRACK_STATE, weight=100, target=q_ref, states_idx=range(biorbd_model.nbQ())
        )
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_residual_torque:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    else:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    x_bounds[0].min[[0, 1], :] = -2 * np.pi
    x_bounds[0].max[[0, 1], :] = 2 * np.pi

    # Add muscle to the bounds
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0] * biorbd_model.nbMuscles())

    # Define control path constraint
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    if use_residual_torque:
        u_bounds.add(
            [
                [tau_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscles(),
                [tau_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscles(),
            ]
        )
        u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [excitation_init] * biorbd_model.nbMuscles())
    else:
        u_bounds.add([[excitation_min] * biorbd_model.nbMuscles(), [excitation_max] * biorbd_model.nbMuscles()])
        u_init.add([excitation_init] * biorbd_model.nbMuscles())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        nb_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("arm26.bioMod")
    final_time = 1.5
    n_shooting_points = 29
    use_residual_torque = True

    # Generate random data to fit
    t, markers_ref, x_ref, muscle_excitations_ref = generate_data(biorbd_model, final_time, n_shooting_points)
    muscle_activations_ref = x_ref[biorbd_model.nbQ() + biorbd_model.nbQdot() :, :].T

    # Track these data
    biorbd_model = biorbd.Model("arm26.bioMod")  # To allow for non free variable, the model must be reloaded
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show the results --- #
    muscle_excitations_ref = np.append(muscle_excitations_ref, muscle_excitations_ref[-1:, :], axis=0)

    states_sol, controls_sol = Data.get_data(ocp, sol["x"])
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    if use_residual_torque:
        tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

    n_q = ocp.nlp[0].model.nbQ()
    n_qdot = ocp.nlp[0].model.nbQdot()
    n_mark = ocp.nlp[0].model.nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = Function(
        "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
    ).expand()
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    plt.figure("Markers")
    for i in range(markers.shape[1]):
        plt.plot(np.linspace(0, 2, n_shooting_points + 1), markers_ref[:, i, :].T, "k")
        plt.plot(np.linspace(0, 2, n_shooting_points + 1), markers[:, i, :].T, "r--")
    plt.xlabel("Time")
    plt.ylabel("Markers Position")

    # --- Plot --- #
    plt.show()
