from scipy.integrate import solve_ivp
import numpy as np
import biorbd
from casadi import MX, Function
from matplotlib import pyplot as plt

from biorbd_optim import (
    OptimalControlProgram,
    BidirectionalMapping,
    Mapping,
    Data,
    DynamicsTypeList,
    DynamicsType,
    DynamicsFunctions,
    ObjectiveList,
    Objective,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
)


def generate_data(biorbd_model, final_time, nb_shooting, use_residual_torque=True):
    # Aliases
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_markers = biorbd_model.nbMarkers()
    dt = final_time / nb_shooting

    if use_residual_torque:
        nu = nb_tau + nb_mus
    else:
        nu = nb_mus

    # Casadi related stuff
    symbolic_states = MX.sym("x", nb_q + nb_qdot, 1)
    symbolic_controls = MX.sym("u", nu, 1)
    symbolic_parameters = MX.sym("params", 0, 0)

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

    nlp = {
        "model": biorbd_model,
        "nbMuscle": nb_mus,
        "q_mapping": BidirectionalMapping(Mapping(range(nb_q)), Mapping(range(nb_q))),
        "q_dot_mapping": BidirectionalMapping(Mapping(range(nb_qdot)), Mapping(range(nb_qdot))),
        "parameters_to_optimize": {},
    }
    if use_residual_torque:
        nlp["nbTau"] = nb_tau
        nlp["tau_mapping"] = BidirectionalMapping(Mapping(range(nb_tau)), Mapping(range(nb_tau)))
        dyn_func = DynamicsFunctions.forward_dynamics_torque_muscle_driven
    else:
        dyn_func = DynamicsFunctions.forward_dynamics_muscle_activations_driven

    dynamics_func = Function(
        "ForwardDyn",
        [symbolic_states, symbolic_controls, symbolic_parameters],
        [dyn_func(symbolic_states, symbolic_controls, symbolic_parameters, nlp)],
        ["x", "u", "p"],
        ["xdot"],
    ).expand()

    def dyn_interface(t, x, u):
        if use_residual_torque:
            u = np.concatenate([np.zeros(nb_tau), u])
        return np.array(dynamics_func(x, u, np.empty((0, 0)))).squeeze()

    # Generate some muscle activation
    U = np.random.rand(nb_shooting, nb_mus).T

    # Integrate and collect the position of the markers accordingly
    X = np.ndarray((nb_q + nb_qdot, nb_shooting + 1))
    markers = np.ndarray((3, biorbd_model.nbMarkers(), nb_shooting + 1))

    def add_to_data(i, q):
        X[:, i] = q
        for j, mark_func in enumerate(markers_func):
            markers[:, j, i] = np.array(mark_func(q)).squeeze()

    x_init = np.array([0] * nb_q + [0] * nb_qdot)
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
    activations_ref,
    q_ref,
    kin_data_to_track="markers",
    use_residual_torque=True,
):
    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    nq = biorbd_model.nbQ()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, target=activations_ref)

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
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
    else:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    x_bounds[0].min[:nq, :] = -2 * np.pi
    x_bounds[0].max[:nq, :] = 2 * np.pi

    # Initial guess
    x_init = InitialConditionsList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_init = InitialConditionsList()
    if use_residual_torque:
        u_bounds.add(
            [
                [tau_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(),
                [tau_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal(),
            ]
        )
        u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [activation_init] * biorbd_model.nbMuscleTotal())
    else:
        u_bounds.add([[activation_min] * biorbd_model.nbMuscleTotal(), [activation_max] * biorbd_model.nbMuscleTotal()])
        u_init.add([activation_init] * biorbd_model.nbMuscleTotal())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model, dynamics, nb_shooting, final_time, x_init, u_init, x_bounds, u_bounds, objective_functions,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("arm26.bioMod")
    final_time = 2
    n_shooting_points = 29
    use_residual_torque = True

    # Generate random data to fit
    t, markers_ref, x_ref, muscle_activations_ref = generate_data(
        biorbd_model, final_time, n_shooting_points, use_residual_torque=use_residual_torque
    )

    # Track these data
    biorbd_model = biorbd.Model("arm26.bioMod")  # To allow for non free variable, the model must be reloaded
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        kin_data_to_track="q",
        use_residual_torque=use_residual_torque,
    )

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show the results --- #
    muscle_activations_ref = np.append(muscle_activations_ref, muscle_activations_ref[-1:, :], axis=0)

    states, controls = Data.get_data(ocp, sol["x"])
    q = states["q"]
    qdot = states["q_dot"]
    mus = controls["muscles"]

    if use_residual_torque:
        tau = controls["tau"]

    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
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

    # --- Plot --- #
    plt.show()
