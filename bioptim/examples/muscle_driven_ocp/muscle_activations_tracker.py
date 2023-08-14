"""
This is an example of muscle activation/skin marker OR state tracking.
Random data are created by generating a random set of muscle activations and then by generating the kinematics
associated with these data. The solution is trivial since no noise is applied to the data. Still, it is a relevant
example to show how to track data using a musculoskeletal model. In real situation, the muscle activation
and kinematics would indeed be acquired via data acquisition devices

The difference between muscle activation and excitation is that the latter is the derivative of the former
"""

import platform

from scipy.integrate import solve_ivp
import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, vertcat
from matplotlib import pyplot as plt
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    NonLinearProgram,
    BiMapping,
    DynamicsList,
    DynamicsFcn,
    DynamicsFunctions,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    RigidBodyDynamics,
)

from bioptim.optimization.optimization_variable import OptimizationVariableContainer


def generate_data(
    bio_model: BiorbdModel,
    final_time: float,
    n_shooting: int,
    use_residual_torque: bool = True,
    assume_phase_dynamics: bool = True,
) -> tuple:
    """
    Generate random data. If np.random.seed is defined before, it will always return the same results

    Parameters
    ----------
    bio_model: BiorbdModel
        The loaded biorbd model
    final_time: float
        The time at final node
    n_shooting: int
        The number of shooting points
    use_residual_torque: bool
        If residual torque are present or not in the dynamics
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The time, marker, states and controls of the program. The ocp will try to track these
    """

    # Aliases
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_qddot = bio_model.nb_qddot
    n_tau = bio_model.nb_tau
    n_mus = bio_model.nb_muscles
    dt = final_time / n_shooting

    nlp = NonLinearProgram(assume_phase_dynamics=assume_phase_dynamics)
    nlp.model = bio_model
    nlp.variable_mappings = {
        "q": BiMapping(range(n_q), range(n_q)),
        "qdot": BiMapping(range(n_qdot), range(n_qdot)),
        "tau": BiMapping(range(n_tau), range(n_tau)),
        "muscles": BiMapping(range(n_mus), range(n_mus)),
        "qddot": BiMapping(range(n_qddot), range(n_qddot)),
    }

    # Casadi related stuff
    symbolic_time = MX.sym("time", 0, 0)
    symbolic_q = MX.sym("q", n_q, 1)
    symbolic_qdot = MX.sym("qdot", n_qdot, 1)
    symbolic_qddot = MX.sym("qddot", n_qddot, 1)
    symbolic_tau = MX.sym("tau", n_tau, 1)
    symbolic_mus = MX.sym("muscles", n_mus, 1)
    symbolic_parameters = MX.sym("params", 0, 0)
    markers_func = biorbd.to_casadi_func("ForwardKin", bio_model.markers, symbolic_q)

    nlp.states = OptimizationVariableContainer(assume_phase_dynamics=assume_phase_dynamics)
    nlp.states_dot = OptimizationVariableContainer(assume_phase_dynamics=assume_phase_dynamics)
    nlp.controls = OptimizationVariableContainer(assume_phase_dynamics=assume_phase_dynamics)
    nlp.states.initialize_from_shooting(n_shooting, MX)
    nlp.states_dot.initialize_from_shooting(n_shooting, MX)
    nlp.controls.initialize_from_shooting(n_shooting, MX)

    for node_index in range(n_shooting):
        nlp.states.append(
            "q",
            [symbolic_q, symbolic_q, symbolic_q],
            [symbolic_q, symbolic_q, symbolic_q],
            symbolic_q,
            nlp.variable_mappings["q"],
            node_index,
        )
        nlp.states.append(
            "qdot",
            [symbolic_qdot, symbolic_qdot, symbolic_qdot],
            [symbolic_qdot, symbolic_qdot, symbolic_qdot],
            symbolic_qdot,
            nlp.variable_mappings["qdot"],
            node_index,
        )

        nlp.states_dot.append(
            "qdot",
            [symbolic_qdot, symbolic_qdot, symbolic_qdot],
            [symbolic_qdot, symbolic_qdot, symbolic_qdot],
            symbolic_qdot,
            nlp.variable_mappings["qdot"],
            node_index,
        )
        nlp.states_dot.append(
            "qddot",
            [symbolic_qddot, symbolic_qddot, symbolic_qddot],
            [symbolic_qddot, symbolic_qddot, symbolic_qddot],
            symbolic_qddot,
            nlp.variable_mappings["qddot"],
            node_index,
        )

        if use_residual_torque:
            nlp.controls.append(
                "tau",
                [symbolic_tau, symbolic_tau, symbolic_tau],
                [symbolic_tau, symbolic_tau, symbolic_tau],
                symbolic_tau,
                nlp.variable_mappings["tau"],
                node_index,
            )
        nlp.controls.append(
            "muscles",
            [symbolic_mus, symbolic_mus, symbolic_mus],
            [symbolic_mus, symbolic_mus, symbolic_mus],
            symbolic_mus,
            nlp.variable_mappings["muscles"],
            node_index,
        )

    if use_residual_torque:
        nlp.variable_mappings["tau"] = BiMapping(range(n_tau), range(n_tau))
    dyn_func = DynamicsFunctions.muscles_driven

    symbolic_states = vertcat(*(symbolic_q, symbolic_qdot))
    symbolic_controls = vertcat(*(symbolic_tau, symbolic_mus)) if use_residual_torque else vertcat(symbolic_mus)

    dynamics_func = biorbd.to_casadi_func(
        "ForwardDyn",
        dyn_func(
            time=symbolic_time,
            states=symbolic_states,
            controls=symbolic_controls,
            parameters=symbolic_parameters,
            stochastic_variables=MX(),
            nlp=nlp,
            with_contact=False,
            rigidbody_dynamics=RigidBodyDynamics.ODE,
        ).dxdt,
        symbolic_states,
        symbolic_controls,
        symbolic_parameters,
        nlp,
        False,
    )

    def dyn_interface(t, x, u):
        if use_residual_torque:
            u = np.concatenate([np.zeros(n_tau), u])
        return np.array(dynamics_func(x, u, [])[:, 0]).squeeze()

    # Generate some muscle activation
    U = np.random.rand(n_shooting, n_mus).T

    # Integrate and collect the position of the markers accordingly
    X = np.ndarray((n_q + n_qdot, n_shooting + 1))
    markers = np.ndarray((3, bio_model.nb_markers, n_shooting + 1))

    def add_to_data(i, q):
        X[:, i] = q
        markers[:, :, i] = markers_func(q[0:n_q])

    x_init = np.array([0] * n_q + [0] * n_qdot)
    add_to_data(0, x_init)
    for i, u in enumerate(U.T):
        sol = solve_ivp(dyn_interface, (0, dt), x_init, method="RK45", args=(u,))

        x_init = sol["y"][:, -1]
        add_to_data(i + 1, x_init)

    time_interp = np.linspace(0, final_time, n_shooting + 1)
    return time_interp, markers, X, U


def prepare_ocp(
    bio_model: BiorbdModel,
    final_time: float,
    n_shooting: int,
    markers_ref: np.ndarray,
    activations_ref: np.ndarray,
    q_ref: np.ndarray,
    kin_data_to_track: str = "markers",
    use_residual_torque: bool = True,
    ode_solver: OdeSolverBase = OdeSolver.COLLOCATION(),
    n_threads: int = 1,
    assume_phase_dynamics: bool = True,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp to solve

    Parameters
    ----------
    bio_model: BiorbdModel
        The loaded biorbd model
    final_time: float
        The time at final node
    n_shooting: int
        The number of shooting points
    markers_ref: np.ndarray
        The marker to track if 'markers' is chosen in kin_data_to_track
    activations_ref: np.ndarray
        The muscle activation to track
    q_ref: np.ndarray
        The state to track if 'q' is chosen in kin_data_to_track
    kin_data_to_track: str
        The type of kin data to track ('markers' or 'q')
    use_residual_torque: bool
        If residual torque are present or not in the dynamics
    ode_solver: OdeSolverBase
        The ode solver to use
    n_threads: int
        The number of threads
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OptimalControlProgram ready to solve
    """

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key="muscles", target=activations_ref)

    if use_residual_torque:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    if kin_data_to_track == "markers":
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100, target=markers_ref[:, :, :-1])
    elif kin_data_to_track == "q":
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=100, target=q_ref, node=Node.ALL)
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=use_residual_torque, expand=expand_dynamics)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = [-2 * np.pi] * bio_model.nb_q, [2 * np.pi] * bio_model.nb_q
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    # Define control path constraint
    activation_min, activation_max, activation_init = 0.0, 1.0, 0.5
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    if use_residual_torque:
        tau_min, tau_max, tau_init = -100.0, 100.0, 0.0
        u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau
    u_bounds["muscles"] = [activation_min] * bio_model.nb_muscles, [activation_max] * bio_model.nb_muscles
    u_init["muscles"] = [activation_init] * bio_model.nb_muscles
    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        u_init=u_init,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        n_threads=n_threads,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Generate random data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    bio_model = BiorbdModel("models/arm26.bioMod")
    final_time = 0.5
    n_shooting_points = 50
    use_residual_torque = True

    # Generate random data to fit
    t, markers_ref, x_ref, muscle_activations_ref = generate_data(
        bio_model,
        final_time,
        n_shooting_points,
        use_residual_torque=use_residual_torque,
    )

    # Track these data
    bio_model = BiorbdModel("models/arm26.bioMod")  # To allow for non free variable, the model must be reloaded
    ocp = prepare_ocp(
        bio_model,
        final_time,
        n_shooting_points,
        markers_ref,
        muscle_activations_ref,
        x_ref[: bio_model.nb_q, :],
        kin_data_to_track="q",
        use_residual_torque=use_residual_torque,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show the results --- #
    q = sol.states["q"]
    n_q = ocp.nlp[0].model.nb_q
    n_mark = ocp.nlp[0].model.nb_markers
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = biorbd.to_casadi_func("ForwardKin", bio_model.markers, symbolic_states)

    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    plt.figure("Markers")
    n_steps_ode = ocp.nlp[0].ode_solver.steps + 1 if ocp.nlp[0].ode_solver.is_direct_collocation else 1
    for i in range(markers.shape[1]):
        plt.plot(
            np.linspace(0, final_time, n_shooting_points + 1),
            markers_ref[:, i, :].T,
            "k",
        )
        plt.plot(
            np.linspace(0, final_time, n_shooting_points * n_steps_ode + 1),
            markers[:, i, :].T,
            "r--",
        )

    # --- Plot --- #
    plt.show()


if __name__ == "__main__":
    main()
