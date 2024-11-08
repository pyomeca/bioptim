"""
This is an example of muscle excitation(EMG)/skin marker OR state tracking.
Random data are created by generating a random set of EMG and then by generating the kinematics associated with these
data. The solution is trivial since no noise is applied to the data. Still, it is a relevant example to show how to
track data using a musculoskeletal model. In real situation, the EMG and kinematics would indeed be acquired via
data acquisition devices

The difference between muscle activation and excitation is that the latter is the derivative of the former
"""

import platform

import numpy as np
from casadi import MX, SX, vertcat, horzcat, Function
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

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
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    PhaseDynamics,
    SolutionMerge,
)
from bioptim.optimization.optimization_variable import OptimizationVariableContainer


def generate_data(
    bio_model: BiorbdModel,
    final_time: float,
    n_shooting: int,
    use_residual_torque: bool = True,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    use_sx: bool = False,
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
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node

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

    # Casadi related stuff
    symbolic_q = MX.sym("q", n_q, 1)
    symbolic_qdot = MX.sym("qdot", n_qdot, 1)
    symbolic_qddot = MX.sym("qddot", n_qddot, 1)
    symbolic_mus_states = MX.sym("mus", n_mus, 1)

    symbolic_tau = MX.sym("tau", n_tau, 1)
    symbolic_mus_controls = MX.sym("mus", n_mus, 1)

    symbolic_time = MX.sym("t", 0, 0)
    symbolic_states = vertcat(*(symbolic_q, symbolic_qdot, symbolic_mus_states))
    symbolic_controls = vertcat(*(symbolic_tau, symbolic_mus_controls))

    symbolic_parameters = MX.sym("u", 0, 0)
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=use_sx)
    nlp.cx = SX if use_sx else MX
    nlp.model = bio_model
    nlp.variable_mappings = {
        "q": BiMapping(range(n_q), range(n_q)),
        "qdot": BiMapping(range(n_qdot), range(n_qdot)),
        "qddot": BiMapping(range(n_qddot), range(n_qddot)),
        "tau": BiMapping(range(n_tau), range(n_tau)),
        "muscles": BiMapping(range(n_mus), range(n_mus)),
    }
    nlp.states = OptimizationVariableContainer(phase_dynamics=phase_dynamics)
    nlp.states_dot = OptimizationVariableContainer(phase_dynamics=phase_dynamics)
    nlp.controls = OptimizationVariableContainer(phase_dynamics=phase_dynamics)
    nlp.algebraic_states = OptimizationVariableContainer(phase_dynamics=phase_dynamics)
    nlp.states.initialize_from_shooting(n_shooting, MX)
    nlp.states_dot.initialize_from_shooting(n_shooting, MX)
    nlp.controls.initialize_from_shooting(n_shooting, MX)
    nlp.algebraic_states.initialize_from_shooting(n_shooting, MX)

    for node_index in range(n_shooting):
        nlp.states.append(
            name="q",
            cx=[symbolic_q, symbolic_q, symbolic_q],
            cx_scaled=[symbolic_q, symbolic_q, symbolic_q],
            mapping=nlp.variable_mappings["q"],
            node_index=node_index,
        )
        nlp.states.append(
            name="qdot",
            cx=[symbolic_qdot, symbolic_qdot, symbolic_qdot],
            cx_scaled=[symbolic_qdot, symbolic_qdot, symbolic_qdot],
            mapping=nlp.variable_mappings["qdot"],
            node_index=node_index,
        )
        nlp.states.append(
            name="muscles",
            cx=[symbolic_mus_states, symbolic_mus_states, symbolic_mus_states],
            cx_scaled=[symbolic_mus_states, symbolic_mus_states, symbolic_mus_states],
            mapping=nlp.variable_mappings["muscles"],
            node_index=node_index,
        )

        nlp.controls.append(
            name="tau",
            cx=[symbolic_tau, symbolic_tau, symbolic_tau],
            cx_scaled=[symbolic_tau, symbolic_tau, symbolic_tau],
            mapping=nlp.variable_mappings["tau"],
            node_index=node_index,
        )
        nlp.controls.append(
            name="muscles",
            cx=[symbolic_mus_controls, symbolic_mus_controls, symbolic_mus_controls],
            cx_scaled=[symbolic_mus_controls, symbolic_mus_controls, symbolic_mus_controls],
            mapping=nlp.variable_mappings["muscles"],
            node_index=node_index,
        )
        nlp.states_dot.append(
            name="qdot",
            cx=[symbolic_qdot, symbolic_qdot, symbolic_qdot],
            cx_scaled=[symbolic_qdot, symbolic_qdot, symbolic_qdot],
            mapping=nlp.variable_mappings["qdot"],
            node_index=node_index,
        )
        nlp.states_dot.append(
            name="qddot",
            cx=[symbolic_qddot, symbolic_qddot, symbolic_qddot],
            cx_scaled=[symbolic_qddot, symbolic_qddot, symbolic_qddot],
            mapping=nlp.variable_mappings["qddot"],
            node_index=node_index,
        )

    dynamics_func = Function(
        "ForwardDyn",
        [symbolic_time, symbolic_states, symbolic_controls, symbolic_parameters],
        [
            DynamicsFunctions.muscles_driven(
                time=symbolic_time,
                states=symbolic_states,
                controls=symbolic_controls,
                parameters=symbolic_parameters,
                algebraic_states=MX(),
                numerical_timeseries=MX(),
                nlp=nlp,
                with_contact=False,
            ).dxdt
        ],
    )

    def dyn_interface(t, x, u):
        u = np.concatenate([np.zeros(n_tau), u])
        return np.array(dynamics_func(t, x, u, [])[:, 0]).squeeze()

    # Generate some muscle excitations
    U = np.random.rand(n_shooting, n_mus).T

    # Integrate and collect the position of the markers accordingly
    X = np.ndarray((n_q + n_qdot + n_mus, n_shooting + 1))
    markers = np.ndarray((3, bio_model.nb_markers, n_shooting + 1))

    def add_to_data(i, q):
        X[:, i] = q
        markers[:, :, i] = bio_model.markers()(q[:n_q], [])

    x_init = np.array([0.0] * n_q + [0.0] * n_qdot + [0.5] * n_mus)
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
    excitations_ref: np.ndarray,
    q_ref: np.ndarray,
    use_residual_torque: bool,
    kin_data_to_track: str = "markers",
    ode_solver: OdeSolverBase = OdeSolver.COLLOCATION(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
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
    excitations_ref: np.ndarray
        The muscle excitation (EMG) to track
    q_ref: np.ndarray
        The state to track if 'q' is chosen in kin_data_to_track
    kin_data_to_track: str
        The type of kin data to track ('markers' or 'q')
    use_residual_torque: bool
        If residual torque are present or not in the dynamics
    ode_solver: OdeSolverBase
        The ode solver to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
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
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key="muscles", target=excitations_ref)
    if use_residual_torque:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    if kin_data_to_track == "markers":
        node = Node.ALL_SHOOTING if type(ode_solver) == OdeSolver.COLLOCATION else Node.ALL
        ref = markers_ref[:, :, :-1] if type(ode_solver) == OdeSolver.COLLOCATION else markers_ref
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, node=node, weight=100, target=ref)
    elif kin_data_to_track == "q":
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            key="q",
            weight=100,
            node=Node.ALL,
            target=q_ref,
            index=range(bio_model.nb_q),
        )
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.MUSCLE_DRIVEN,
        with_excitations=True,
        with_residual_torque=use_residual_torque,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    x_bounds["q"].min[:, :] = -2 * np.pi
    x_bounds["q"].max[:, :] = 2 * np.pi

    # Add muscle to the bounds
    activation_min, activation_max, activation_init = 0, 1, 0.5
    x_bounds["muscles"] = [activation_min] * bio_model.nb_muscles, [activation_max] * bio_model.nb_muscles
    x_bounds["muscles"][:, 0] = excitations_ref[:, 0]

    # Define control path constraint
    excitation_min, excitation_max, excitation_init = 0, 1, 0.5
    u_bounds = BoundsList()
    if use_residual_torque:
        tau_min, tau_max = -100.0, 100.0
        u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    u_bounds["muscles"] = [excitation_min] * bio_model.nb_muscles, [excitation_max] * bio_model.nb_muscles
    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
    )


def main():
    """
    Generate random data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    bio_model = BiorbdModel("models/arm26.bioMod")
    final_time = 0.5
    n_shooting_points = 30
    use_residual_torque = True
    phase_dynamics = PhaseDynamics.SHARED_DURING_THE_PHASE

    # Generate random data to fit
    t, markers_ref, x_ref, muscle_excitations_ref = generate_data(
        bio_model, final_time, n_shooting_points, use_residual_torque, phase_dynamics
    )

    # Track these data
    bio_model = BiorbdModel("models/arm26.bioMod")  # To allow for non free variable, the model must be reloaded
    ocp = prepare_ocp(
        bio_model,
        final_time,
        n_shooting_points,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: bio_model.nb_q, :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show the results --- #
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]

    n_q = ocp.nlp[0].model.nb_q
    n_mark = ocp.nlp[0].model.nb_markers
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    for i in range(n_frames):
        markers[:, :, i] = horzcat(*bio_model.markers()(q[:, i]))

    plt.figure("Markers")
    n_steps_ode = ocp.nlp[0].ode_solver.steps + 1 if ocp.nlp[0].ode_solver.is_direct_collocation else 1
    for i in range(markers.shape[1]):
        plt.plot(np.linspace(0, 2, n_shooting_points + 1), markers_ref[:, i, :].T, "k")
        plt.plot(np.linspace(0, 2, n_shooting_points * n_steps_ode + 1), markers[:, i, :].T, "r--")
    plt.xlabel("Time")
    plt.ylabel("Markers Position")

    # --- Plot --- #
    plt.show()


if __name__ == "__main__":
    main()
