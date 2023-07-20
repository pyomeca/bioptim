"""
This example is adapted from arm_reaching_muscle_driven.py to make it torque driven.
The states dynamics is explicit, while the stochastic variables dynamics is implicit.
This formulation allow to decouple the covariance matrix with the previous states reducing the comlexity of resolution,
but increases largely the number of variables to optimize.
"""

import platform

import pickle
import biorbd_casadi as biorbd
import matplotlib.pyplot as plt
import casadi as cas
import numpy as np
import scipy.io as sio

from bioptim import (
    OptimalControlProgram,
    StochasticOptimalControlProgram,
    ObjectiveFcn,
    Solver,
    BiorbdModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    DynamicsList,
    BoundsList,
    InterpolationType,
    SocpType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
    MultinodeConstraintList,
    MultinodeObjectiveList,
    InitialGuessList,
)


def get_force_field(q, force_field_magnitude):
    """
    Get the effect of the force field.

    Parameters
    ----------
    q: MX.sym
        The generalized coordinates
    force_field_magnitude: float
        The magnitude of the force field
    """
    l1 = 0.3
    l2 = 0.33
    f_force_field = force_field_magnitude * (l1 * cas.cos(q[0]) + l2 * cas.cos(q[0] + q[1]))
    hand_pos = cas.MX(2, 1)
    hand_pos[0] = l2 * cas.sin(q[0] + q[1]) + l1 * cas.sin(q[0])
    hand_pos[1] = l2 * cas.sin(q[0] + q[1])
    tau_force_field = -f_force_field @ hand_pos
    return tau_force_field


def get_excitation_with_feedback(k, hand_pos_velo, ref, sensory_noise):
    """
    Get the effect of the feedback.

    Parameters
    ----------
    k: MX.sym
        The feedback gains
    hand_pos_velo: MX.sym
        The position and velocity of the hand
    ref: MX.sym
        The reference position and velocity of the hand
    sensory_noise: MX.sym
        The sensory noise
    """
    return k @ ((hand_pos_velo - ref) + sensory_noise)


def stochastic_forward_dynamics(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    stochastic_variables: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    motor_noise,
    sensory_noise,
    force_field_magnitude,
    with_gains,
) -> DynamicsEvaluation:
    """
    The dynamic function of the states including feedback gains.

    Parameters
    ----------
    states: MX.sym
        The states
    controls: MX.sym
        The controls
    parameters: MX.sym
        The parameters
    stochastic_variables: MX.sym
        The stochastic variables
    nlp: NonLinearProgram
        The current non-linear program
    motor_noise: MX.sym
        The motor noise
    sensory_noise: MX.sym
        The sensory noise
    force_field_magnitude: float
        The magnitude of the force field
    with_gains: bool
        If the feedback gains are included or not to the torques
    """
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    n_q = q.shape[0]
    n_qdot = qdot.shape[0]
    n_tau = tau.shape[0]

    tau_fb = tau
    if with_gains:
        ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        k_matrix = cas.MX(n_q + n_qdot, n_tau)
        for s0 in range(n_q + n_qdot):
            for s1 in range(n_tau):
                k_matrix[s0, s1] = k[s0 * n_tau + s1]
        k_matrix = k_matrix.T

        hand_pos = nlp.model.markers(q)[2][:2]
        hand_vel = nlp.model.marker_velocities(q, qdot)[2][:2]
        ee = cas.vertcat(hand_pos, hand_vel)

        tau_fb += get_excitation_with_feedback(k_matrix, ee, ref, sensory_noise)

    tau_force_field = get_force_field(q, force_field_magnitude)

    torques_computed = tau_fb + motor_noise + tau_force_field

    friction = np.array([[0.05, 0.025], [0.025, 0.05]])

    mass_matrix = nlp.model.mass_matrix(q)
    non_linear_effects = nlp.model.non_linear_effects(q, qdot)

    dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - non_linear_effects - friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(qdot, dqdot_computed))


def configure_stochastic_optimal_control_problem(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, motor_noise, sensory_noise
):
    """
    Configure the stochastic optimal control problem.
    """
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_tau(ocp, nlp, False, True)

    # Stochastic variables
    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=2, n_feedbacks=4)
    ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=4)
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=4)
    ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=4)
    ConfigureProblem.configure_stochastic_a(ocp, nlp, n_noised_states=4)
    ConfigureProblem.configure_stochastic_c(ocp, nlp, n_feedbacks=4, n_noise=6)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise: nlp.dynamics_type.dynamic_function(
            states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise, with_gains=False
        ),
        motor_noise=motor_noise,
        sensory_noise=sensory_noise,
        expand=False,
    )
    return


def get_ref(controller: PenaltyController, q, qdot) -> cas.MX:
    """
    Get the reference had position and velocity.

    Parameters
    ----------
    controller: PenaltyController
        The controller.
    q: cas.MX
        The current joint position.
    qdot: cas.MX
        The current joint velocity.
    """
    hand_pos = controller.model.markers(q)[2][:2]
    hand_vel = controller.model.marker_velocities(q, qdot)[2][:2]
    ee = cas.vertcat(hand_pos, hand_vel)
    return ee


def hand_equals_ref(controller: PenaltyController) -> cas.MX:
    """
    Get the error between the hand position and the reference.
    """
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ref = controller.stochastic_variables["ref"].cx_start
    ee = get_ref(controller, q, qdot)
    return ee - ref


def reach_target_consistantly(controllers: list[PenaltyController]) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """

    nx = controllers[-1].states["q"].cx_start.shape[0]

    q_sym = cas.MX.sym("q_sym", nx)
    qdot_sym = cas.MX.sym("qdot_sym", nx)
    cov_sym = cas.MX.sym("cov", controllers[0].stochastic_variables["cov"].cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = (
        controllers[0]
        .stochastic_variables["cov"]
        .reshape_to_matrix(
            cov_sym_dict,
            controllers[0].states.cx_start.shape[0],
            controllers[0].states.cx_start.shape[0],
            Node.START,
            "cov",
        )
    )

    hand_pos = controllers[0].model.markers(q_sym)[2][:2]
    hand_vel = controllers[0].model.marker_velocities(q_sym, qdot_sym)[2][:2]

    jac_marker_q = cas.jacobian(hand_pos, q_sym)
    jac_marker_qdot = cas.jacobian(hand_vel, cas.vertcat(q_sym, qdot_sym))

    cov_matrix_q = cov_matrix[:2, :2]
    cov_matrix_qdot = cov_matrix[:4, :4]

    pos_constraint = jac_marker_q @ cov_matrix_q @ jac_marker_q.T
    vel_constraint = jac_marker_qdot @ cov_matrix_qdot @ jac_marker_qdot.T

    out = cas.vertcat(pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1])

    fun = cas.Function("reach_target_consistantly", [q_sym, qdot_sym, cov_sym], [out])
    val = fun(
        controllers[-1].states["q"].cx_start,
        controllers[-1].states["qdot"].cx_start,
        controllers[-1].stochastic_variables["cov"].cx_start,
    )
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val


def expected_feedback_effort(controller: PenaltyController, sensory_noise_magnitude: cas.DM) -> cas.MX:
    """
    This function computes the expected effort due to the motor command and feedback gains for a given sensory noise
    magnitude.
    It is computed as Jacobian(effort, states) @ cov @ Jacobian(effort, states) +
                        Jacobian(efforst, motor_noise) @ sigma_w @ Jacobian(efforst, motor_noise)

    Parameters
    ----------
    controller : PenaltyController
        Controller to be used to compute the expected effort.
    sensory_noise_magnitude : cas.DM
        Magnitude of the sensory noise.
    """
    n_tau = controller.controls["tau"].cx_start.shape[0]
    n_q = controller.states["q"].cx_start.shape[0]
    n_qdot = controller.states["qdot"].cx_start.shape[0]

    sensory_noise_matrix = sensory_noise_magnitude * cas.MX_eye(4)

    # create the casadi function to be evaluated
    # Get the symbolic variables
    ref = controller.stochastic_variables["ref"].cx_start
    cov_matrix = controller.stochastic_variables["cov"].reshape_to_matrix(
        controller.stochastic_variables,
        controller.states.cx_start.shape[0],
        controller.states.cx_start.shape[0],
        Node.START,
        "cov",
    )

    k = controller.stochastic_variables["k"].cx_start
    k_matrix = cas.MX(n_q + n_qdot, n_tau)
    for s0 in range(n_q + n_qdot):
        for s1 in range(n_tau):
            k_matrix[s0, s1] = k[s0 * n_tau + s1]
    k_matrix = k_matrix.T

    # Compute the expected effort
    hand_pos = controller.model.markers(controller.states["q"].cx_start)[2][:2]
    hand_vel = controller.model.marker_velocities(controller.states["q"].cx_start, controller.states["qdot"].cx_start)[
        2
    ][:2]
    trace_k_sensor_k = cas.trace(k_matrix @ sensory_noise_matrix @ k_matrix.T)
    ee = cas.vertcat(hand_pos, hand_vel)
    e_fb = k_matrix @ ((ee - ref) + sensory_noise_magnitude)
    jac_e_fb_x = cas.jacobian(e_fb, controller.states.cx_start)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ cov_matrix @ jac_e_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function(
        "f_expectedEffort_fb",
        [controller.states.cx_start, controller.stochastic_variables.cx_start],
        [expectedEffort_fb_mx],
    )

    out = func(controller.states.cx_start, controller.stochastic_variables.cx_start)

    return out


def track_final_marker(controller: PenaltyController) -> cas.MX:
    """
    Track the hand position.
    """
    q = controller.states["q"].cx_start
    ee_pos = controller.model.markers(q)[2][:2]
    return ee_pos


def trapezoidal_integration_continuity_constraint(
    controllers: list[PenaltyController], force_field_magnitude
) -> cas.MX:
    """
    This function computes the continuity constraint for the trapezoidal integration scheme.
    It is computed as:
        x_i_plus - x_i - dt/2 * (f(x_i, u_i) + f(x_i_plus, u_i_plus)) = 0
    """
    n_q = controllers[0].model.nb_q
    n_qdot = controllers[0].model.nb_qdot
    n_tau = controllers[0].model.nb_tau

    motor_noise = np.zeros((n_tau, 1))
    sensory_noise = np.zeros((n_q + n_qdot, 1))
    dt = controllers[0].tf / controllers[0].ns

    dyn = stochastic_forward_dynamics(
        controllers[0].states.cx_start,
        controllers[0].controls.cx_start,
        controllers[0].parameters.cx_start,
        controllers[0].stochastic_variables.cx_start,
        controllers[0].get_nlp,
        motor_noise,
        sensory_noise,
        force_field_magnitude=force_field_magnitude,
        with_gains=False,
    )
    dx_i = dyn.dxdt

    dx_i_plus = stochastic_forward_dynamics(
        controllers[1].states.cx_start,
        controllers[1].controls.cx_start,
        controllers[1].parameters.cx_start,
        controllers[1].stochastic_variables.cx_start,
        controllers[1].get_nlp,
        motor_noise,
        sensory_noise,
        force_field_magnitude=force_field_magnitude,
        with_gains=False,
    ).dxdt

    continuity = controllers[1].states.cx_start - (controllers[0].states.cx_start + (dx_i + dx_i_plus) / 2 * dt)
    continuity *= 1e3

    return continuity


def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ee_final_position: np.ndarray,
    motor_noise_magnitude: cas.DM,
    sensory_noise_magnitude: cas.DM,
    force_field_magnitude: float = 0,
    problem_type: str = "CIRCLE",
) -> StochasticOptimalControlProgram:
    """
    The initialization of an ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ee_final_position: np.ndarray
        The final position of the end effector
    motor_noise_magnitude: cas.DM
        The magnitude of the motor noise
    sensory_noise_magnitude: cas.DM
        The magnitude of the sensory noise
    force_field_magnitude: float
        The magnitude of the force field
    problem_type: str
        The type of problem to solve (CIRCLE or BAR)
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    n_tau = bio_model.nb_tau
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_states = n_q * 2

    shoulder_pos_initial = 0.349065850398866
    shoulder_pos_final = 0.959931088596881
    elbow_pos_initial = 2.245867726451909  # Optimized in Tom's version
    elbow_pos_final = 1.159394851847144  # Optimized in Tom's version

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="tau", weight=1e3 / 2, quadratic=True
    )

    objective_functions.add(
        expected_feedback_effort,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING,
        sensory_noise_magnitude=sensory_noise_magnitude,
        weight=1e3 / 2,
        quadratic=False,
        phase=0,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(hand_equals_ref, node=Node.ALL_SHOOTING)
    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=np.array([shoulder_pos_initial, elbow_pos_initial])
    )
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=np.array([0, 0]))
    constraints.add(track_final_marker, node=Node.PENULTIMATE, target=ee_final_position)
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.PENULTIMATE, target=np.array([0, 0]))
    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", node=Node.ALL, min_bound=0, max_bound=180
    )  # This is a bug, it should be in radians

    if problem_type == "BAR":
        max_bounds_lateral_variation = cas.inf
    elif problem_type == "CIRCLE":
        max_bounds_lateral_variation = 0.004
    else:
        raise NotImplementedError("Wrong problem type")

    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(
        reach_target_consistantly,
        nodes_phase=[0 for _ in range(n_shooting)],
        nodes=[i for i in range(n_shooting)],
        min_bound=np.array([-cas.inf, -cas.inf, -cas.inf, -cas.inf]),
        max_bound=np.array([max_bounds_lateral_variation**2, 0.004**2, 0.05**2, 0.05**2]),
    )
    for i in range(n_shooting - 1):
        multinode_constraints.add(
            trapezoidal_integration_continuity_constraint,
            nodes_phase=[0, 0],
            nodes=[i, i + 1],
            force_field_magnitude=force_field_magnitude,
        )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_stochastic_optimal_control_problem,
        dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, motor_noise, sensory_noise, with_gains: stochastic_forward_dynamics(
            states,
            controls,
            parameters,
            stochastic_variables,
            nlp,
            motor_noise,
            sensory_noise,
            force_field_magnitude=force_field_magnitude,
            with_gains=with_gains,
        ),
        motor_noise=np.zeros((n_tau, 1)),
        sensory_noise=np.zeros((n_q + n_qdot, 1)),
    )  # expand=False

    states_min = np.ones((n_states, n_shooting + 1)) * -cas.inf
    states_max = np.ones((n_states, n_shooting + 1)) * cas.inf

    x_bounds = BoundsList()
    x_bounds.add(
        "q", min_bound=states_min[:n_q, :], max_bound=states_max[:n_q, :], interpolation=InterpolationType.EACH_FRAME
    )
    x_bounds.add(
        "qdot",
        min_bound=states_min[n_q : n_q + n_qdot, :],
        max_bound=states_max[n_q : n_q + n_qdot, :],
        interpolation=InterpolationType.EACH_FRAME,
    )

    controls_min = np.ones((n_tau, 3)) * -cas.inf
    controls_max = np.ones((n_tau, 3)) * cas.inf

    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=controls_min, max_bound=controls_max)

    # Initial guesses
    states_init = np.zeros((n_states, n_shooting + 1))
    states_init[0, :-1] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting)
    states_init[0, -1] = shoulder_pos_final
    states_init[1, :-1] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting)
    states_init[1, -1] = elbow_pos_final
    states_init[n_states:, :] = 0.01

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=states_init[:n_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=states_init[n_q : n_q + n_qdot, :], interpolation=InterpolationType.EACH_FRAME)

    controls_init = np.ones((n_tau, n_shooting)) * 0.01

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=controls_init, interpolation=InterpolationType.EACH_FRAME)

    s_init = InitialGuessList()
    s_bounds = BoundsList()
    n_stochastic = n_tau * (n_q + n_qdot) + n_q + n_qdot + n_states * n_states  # K(2x4) + ref(4x1) + M(6x6)
    stochastic_init = np.zeros((n_stochastic, n_shooting + 1))
    stochastic_min = np.ones((n_stochastic, 3)) * -cas.inf
    stochastic_max = np.ones((n_stochastic, 3)) * cas.inf
    curent_index = 0
    stochastic_init[: n_tau * (n_q + n_qdot), :] = 0.01  # K
    s_init.add(
        "k", initial_guess=stochastic_init[: n_tau * (n_q + n_qdot), :], interpolation=InterpolationType.EACH_FRAME
    )
    s_bounds.add(
        "k",
        min_bound=stochastic_min[: n_tau * (n_q + n_qdot), :],
        max_bound=stochastic_max[: n_tau * (n_q + n_qdot), :],
    )
    curent_index += n_tau * (n_q + n_qdot)
    stochastic_init[curent_index : curent_index + n_q + n_qdot, :] = 0.01  # ref
    s_init.add(
        "ref",
        initial_guess=stochastic_init[curent_index : curent_index + n_q + n_qdot, :],
        interpolation=InterpolationType.EACH_FRAME,
    )
    s_bounds.add(
        "ref",
        min_bound=stochastic_min[curent_index : curent_index + n_q + n_qdot, :],
        max_bound=stochastic_max[curent_index : curent_index + n_q + n_qdot, :],
    )
    curent_index += n_q + n_qdot
    stochastic_init[curent_index : curent_index + n_states * n_states, :] = 0.01  # M
    s_init.add(
        "m",
        initial_guess=stochastic_init[curent_index : curent_index + n_states * n_states, :],
        interpolation=InterpolationType.EACH_FRAME,
    )
    s_bounds.add(
        "m",
        min_bound=stochastic_min[curent_index : curent_index + n_states * n_states, :],
        max_bound=stochastic_max[curent_index : curent_index + n_states * n_states, :],
    )

    return StochasticOptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        s_init=s_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        s_bounds=s_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        ode_solver=None,
        skip_continuity=True,
        n_threads=1,
        assume_phase_dynamics=False,
        problem_type=SocpType.SOCP_IMPLICIT(motor_noise_magnitude, sensory_noise_magnitude),
    )


def main():
    # --- Options --- #
    vizualize_sol_flag = True

    biorbd_model_path = "models/LeuvenArmModel.bioMod"

    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])  # Directly from Tom's version

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.8
    n_shooting = int(final_time / dt) + 1
    final_time += dt

    # --- Noise constants --- #
    motor_noise_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024

    motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt, motor_noise_std**2 / dt]))
    wPq_magnitude = cas.DM(np.array([wPq_std**2 / dt, wPq_std**2 / dt]))
    wPqdot_magnitude = cas.DM(np.array([wPqdot_std**2 / dt, wPqdot_std**2 / dt]))
    sensory_noise_magnitude = cas.vertcat(wPq_magnitude, wPqdot_magnitude)

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver("mumps")
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(10000)
    solver.set_hessian_approximation("limited-memory")
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_nlp_scaling_method("none")

    problem_type = "CIRCLE"
    force_field_magnitude = 0
    socp = prepare_socp(
        biorbd_model_path=biorbd_model_path,
        final_time=final_time,
        n_shooting=n_shooting,
        ee_final_position=ee_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        problem_type=problem_type,
        force_field_magnitude=force_field_magnitude,
    )

    sol_socp = socp.solve(solver)
    # sol_socp.graphs()

    q_sol = sol_socp.states["q"]
    qdot_sol = sol_socp.states["qdot"]
    tau_sol = sol_socp.controls["tau"]
    k_sol = sol_socp.stochastic_variables["k"]
    ref_sol = sol_socp.stochastic_variables["ref"]
    m_sol = sol_socp.stochastic_variables["m"]
    cov_sol = sol_socp.stochastic_variables["cov"]
    a_sol = sol_socp.stochastic_variables["a"]
    c_sol = sol_socp.stochastic_variables["c"]
    stochastic_variables_sol = np.vstack((k_sol, ref_sol, m_sol, cov_sol, a_sol, c_sol))
    data = {
        "q_sol": q_sol,
        "qdot_sol": qdot_sol,
        "tau_sol": tau_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
        "a_sol": a_sol,
        "c_sol": c_sol,
        "stochastic_variables_sol": stochastic_variables_sol,
    }

    # --- Save the results --- #
    with open(f"leuvenarm_torque_driven_socp_{problem_type}_forcefield{force_field_magnitude}.pkl", "wb") as file:
        pickle.dump(data, file)

    # --- Visualize the results --- #
    if vizualize_sol_flag:
        import bioviz

        b = bioviz.Viz(model_path=biorbd_model_path)
        b.load_movement(q_sol[:, :-1])
        b.exec()


if __name__ == "__main__":
    main()
