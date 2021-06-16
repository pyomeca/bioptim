"""
This is an example of muscle excitation(EMG)/skin marker OR state tracking.
Random data are created by generating a random set of EMG and then by generating the kinematics associated with these
data. The solution is trivial since no noise is applied to the data. Still, it is a relevant example to show how to
track data using a musculoskeletal model. In real situation, the EMG and kinematics would indeed be acquired via
data acquisition devices

The difference between muscle activation and excitation is that the latter is the derivative of the former
"""

from scipy.integrate import solve_ivp
import numpy as np
import biorbd
from casadi import MX, vertcat
from matplotlib import pyplot as plt
from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    BiMapping,
    DynamicsList,
    DynamicsFcn,
    DynamicsFunctions,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
)


def generate_data(
    biorbd_model: biorbd.Model, final_time: float, n_shooting: int, use_residual_torque: bool = True
) -> tuple:
    """
    Generate random data. If np.random.seed is defined before, it will always return the same results

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The loaded biorbd model
    final_time: float
        The time at final node
    n_shooting: int
        The number of shooting points
    use_residual_torque: bool
        If residual torque are present or not in the dynamics

    Returns
    -------
    The time, marker, states and controls of the program. The ocp will try to track these
    """

    # Aliases
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()
    n_mus = biorbd_model.nbMuscleTotal()
    n_markers = biorbd_model.nbMarkers()
    dt = final_time / n_shooting

    # Casadi related stuff
    symbolic_q = MX.sym("q", n_q, 1)
    symbolic_qdot = MX.sym("qdot", n_qdot, 1)
    symbolic_mus_states = MX.sym("mus", n_mus, 1)

    symbolic_tau = MX.sym("tau", n_tau, 1)
    symbolic_mus_controls = MX.sym("mus", n_mus, 1)

    symbolic_states = vertcat(*(symbolic_q, symbolic_qdot, symbolic_mus_states))
    symbolic_controls = vertcat(*(symbolic_tau, symbolic_mus_controls))

    symbolic_parameters = MX.sym("u", 0, 0)
    nlp = NonLinearProgram()
    nlp.model = biorbd_model
    nlp.variable_mappings = {
        "q": BiMapping(range(n_q), range(n_q)),
        "qdot": BiMapping(range(n_qdot), range(n_qdot)),
        "tau": BiMapping(range(n_tau), range(n_tau)),
        "muscles": BiMapping(range(n_mus), range(n_mus)),
    }
    markers_func = biorbd.to_casadi_func("ForwardKin", biorbd_model.markers, symbolic_q)

    nlp.states.cx = MX()
    nlp.controls.cx = MX()
    nlp.states.append("q", symbolic_q, symbolic_q, nlp.variable_mappings["q"])
    nlp.states.append("qdot", symbolic_qdot, symbolic_qdot, nlp.variable_mappings["qdot"])
    nlp.states.append("muscles", symbolic_mus_states, symbolic_mus_states, nlp.variable_mappings["muscles"])

    nlp.controls.append("tau", symbolic_tau, symbolic_tau, nlp.variable_mappings["tau"])
    nlp.controls.append("muscles", symbolic_mus_controls, symbolic_mus_controls, nlp.variable_mappings["muscles"])

    dynamics_func = biorbd.to_casadi_func(
        "ForwardDyn",
        DynamicsFunctions.muscles_driven, # DynamicsFunctions.forward_dynamics_muscle_excitations_and_torque_driven,
        symbolic_states,
        symbolic_controls,
        symbolic_parameters,
        nlp,
        False,
    )

    def dyn_interface(t, x, u):
        u = np.concatenate([np.zeros(n_tau), u])
        return np.array(dynamics_func(x, u, np.empty((0, 0)))).squeeze()

    # Generate some muscle excitations
    U = np.random.rand(n_shooting, n_mus).T

    # Integrate and collect the position of the markers accordingly
    X = np.ndarray((n_q + n_qdot + n_mus, n_shooting + 1))
    markers = np.ndarray((3, biorbd_model.nbMarkers(), n_shooting + 1))

    def add_to_data(i, q):
        X[:, i] = q
        markers[:, :, i] = markers_func(q[:n_q])

    x_init = np.array([0] * n_q + [0] * n_qdot + [0.5] * n_mus)
    add_to_data(0, x_init)
    for i, u in enumerate(U.T):
        sol = solve_ivp(dyn_interface, (0, dt), x_init, method="RK45", args=(u,))
        x_init = sol["y"][:, -1]
        add_to_data(i + 1, x_init)

    time_interp = np.linspace(0, final_time, n_shooting + 1)
    return time_interp, markers, X, U


def prepare_ocp(
    biorbd_model: biorbd.Model,
    final_time: float,
    n_shooting: int,
    markers_ref: np.ndarray,
    excitations_ref: np.ndarray,
    q_ref: np.ndarray,
    use_residual_torque: bool,
    kin_data_to_track: str = "markers",
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:
    """
    Prepare the ocp to solve

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The loaded biorbd model
    final_time: float
        The time at final node
    n_shooting: int
        The number of shooting points
    markers_ref: np.ndarray
        The marker to track if 'markers' is chosen in kin_data_to_track
    excitations_ref: np.ndarray
        The muscle excitation (EMG) to track
    q_ref: np.ndarray
        The state to track if 'q' is chosen in kin_data_to_track
    kin_data_to_track: str
        The type of kin data to track ('markers' or 'q')
    use_residual_torque: bool
        If residual torque are present or not in the dynamics
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to solve
    """

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL, target=excitations_ref)
    if use_residual_torque:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE)
    if kin_data_to_track == "markers":
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100, target=markers_ref)
    elif kin_data_to_track == "q":
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_STATE, weight=100, target=q_ref, index=range(biorbd_model.nbQ())
        )
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_excitations=True, with_residual_torque=use_residual_torque)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    x_bounds[0].min[[0, 1], :] = -2 * np.pi
    x_bounds[0].max[[0, 1], :] = 2 * np.pi

    # Add muscle to the bounds
    activation_min, activation_max, activation_init = 0, 1, 0.5
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )
    x_bounds[0][(biorbd_model.nbQ() + biorbd_model.nbQdot()) :, 0] = excitations_ref[:, 0]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0] * biorbd_model.nbMuscles())

    # Define control path constraint
    excitation_min, excitation_max, excitation_init = 0, 1, 0.5
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    if use_residual_torque:
        tau_min, tau_max, tau_init = -100, 100, 0
        u_bounds.add(
            [tau_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscles(),
            [tau_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscles(),
        )
        u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [excitation_init] * biorbd_model.nbMuscles())
    else:
        u_bounds.add([excitation_min] * biorbd_model.nbMuscles(), [excitation_max] * biorbd_model.nbMuscles())
        u_init.add([excitation_init] * biorbd_model.nbMuscles())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
    )


def main():
    """
    Generate random data, then create a tracking problem, and finally solve it and plot some relevant information
    """

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

    q = sol.states["q"]
    qdot = sol.states["qdot"]
    activations = sol.states["muscles"]
    if use_residual_torque:
        tau = sol.controls["tau"]
    excitations = sol.controls["muscles"]

    n_q = ocp.nlp[0].model.nbQ()
    n_qdot = ocp.nlp[0].model.nbQdot()
    n_mark = ocp.nlp[0].model.nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = biorbd.to_casadi_func("ForwardKin", biorbd_model.markers, symbolic_states)
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


if __name__ == "__main__":
    main()
