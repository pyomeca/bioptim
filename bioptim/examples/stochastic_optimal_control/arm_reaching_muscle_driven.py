"""
This example replicates the results from "An approximate stochastic optimal control framework to simulate nonlinear
neuro-musculoskeletal models in the presence of noise"(https://doi.org/10.1371/journal.pcbi.1009338).
The task is to unfold the arm to reach a target further from the trunk.
Noise is added on the motor execution (motor_noise) and on the feedback (wEE=wP and wEE_dot=wPdot).
The expected joint angles (x_mean) are optimized like in a deterministic OCP, but the covariance matrix is minimized to
reduce uncertainty. This covariance matrix is computed from the expected states.
"""

import platform

from typing import Callable
import pickle
import biorbd_casadi as biorbd
import matplotlib.pyplot as plt
import casadi as cas
import numpy as np
import scipy.io as sio

from bioptim import (
    OptimalControlProgram,
    StochasticOptimalControlProgram,
    InitialGuessList,
    ObjectiveFcn,
    Solver,
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
)

from bioptim.examples.stochastic_optimal_control.leuven_arm_model import LeuvenArmModel


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
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    mus_activations = DynamicsFunctions.get(nlp.states["muscles"], states)
    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)

    mus_excitations_fb = mus_excitations
    if with_gains:
        ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        k_matrix = cas.MX(4, 6)
        for s0 in range(4):
            for s1 in range(6):
                k_matrix[s0, s1] = k[s0 * 6 + s1]
        k_matrix = k_matrix.T

        hand_pos = nlp.model.end_effector_position(q)
        hand_vel = nlp.model.end_effector_velocity(q, qdot)
        ee = cas.vertcat(hand_pos, hand_vel)

        mus_excitations_fb += nlp.model.get_excitation_with_feedback(k_matrix, ee, ref, sensory_noise)

    muscles_tau = nlp.model.get_muscle_torque(q, qdot, mus_activations)

    tau_force_field = nlp.model.get_force_field(q, force_field_magnitude)

    torques_computed = muscles_tau + motor_noise + tau_force_field
    dq_computed = qdot
    dactivations_computed = (mus_excitations_fb - mus_activations) / nlp.model.tau_coef

    a1 = nlp.model.I1 + nlp.model.I2 + nlp.model.m2 * nlp.model.l1**2
    a2 = nlp.model.m2 * nlp.model.l1 * nlp.model.lc2
    a3 = nlp.model.I2

    theta_elbow = q[1]
    dtheta_shoulder = qdot[0]
    dtheta_elbow = qdot[1]

    mass_matrix = cas.MX(2, 2)
    mass_matrix[0, 0] = a1 + 2 * a2 * cas.cos(theta_elbow)
    mass_matrix[0, 1] = a3 + a2 * cas.cos(theta_elbow)
    mass_matrix[1, 0] = a3 + a2 * cas.cos(theta_elbow)
    mass_matrix[1, 1] = a3

    nleffects = cas.MX(2, 1)
    nleffects[0] = a2 * cas.sin(theta_elbow) * (-dtheta_elbow * (2 * dtheta_shoulder + dtheta_elbow))
    nleffects[1] = a2 * cas.sin(theta_elbow) * dtheta_shoulder**2

    dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - nleffects - nlp.model.friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(dq_computed, dqdot_computed, dactivations_computed), defects=None)


def configure_stochastic_optimal_control_problem(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, motor_noise, sensory_noise
):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_muscles(
        ocp, nlp, True, True
    )  # Muscles activations as states + muscles excitations as controls

    # Stochastic variables
    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=6, n_feedbacks=4)
    ConfigureProblem.configure_stochastic_ref(ocp, nlp, n_references=4)
    ConfigureProblem.configure_stochastic_m(ocp, nlp, n_noised_states=10)
    mat_p_init = cas.DM_eye(10) * np.array([1e-4, 1e-4, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])  # P
    ConfigureProblem.configure_stochastic_cov_explicit(ocp, nlp, n_noised_states=10, initial_matrix=mat_p_init)
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


def minimize_uncertainty(controllers: list[PenaltyController], key: str) -> cas.MX:
    """
    Minimize the uncertainty (covariance matrix) of the states.
    """
    dt = controllers[0].tf / controllers[0].ns
    out = 0
    for i, ctrl in enumerate(controllers):
        cov_matrix = ctrl.integrated_values["cov"].reshape_to_matrix(
            ctrl.integrated_values, ctrl.states.cx.shape[0], ctrl.states.cx.shape[0], Node.START, "cov"
        )
        p_partial = cov_matrix[ctrl.states[key].index, ctrl.states[key].index]
        out += cas.trace(p_partial) * dt
    return out


def hand_equals_ref(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ref = controller.stochastic_variables["ref"].cx_start
    ee = controller.model.end_effector_pos_velo(q, qdot)
    return ee - ref


def get_cov_mat(nlp, node_index, force_field_magnitude, motor_noise_magnitude, sensory_noise_magnitude):
    dt = nlp.tf / nlp.ns

    nlp.states.node_index = node_index - 1
    nlp.controls.node_index = node_index - 1
    nlp.stochastic_variables.node_index = node_index - 1
    nlp.integrated_values.node_index = node_index - 1

    nx = nlp.states.cx_start.shape[0]
    m_matrix = nlp.stochastic_variables["m"].reshape_to_matrix(nlp.stochastic_variables, nx, nx, Node.START, "m")

    motor_noise = cas.MX.sym("motor_noise", nlp.states["q"].cx_start.shape[0])
    sensory_noise = cas.MX.sym("sensory_noise", nlp.states["q"].cx_start.shape[0] * 2)
    sigma_w = cas.vertcat(sensory_noise, motor_noise) * cas.MX_eye(6)
    cov_sym = cas.MX.sym("cov", nlp.integrated_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = nlp.integrated_values.reshape_to_matrix(cov_sym_dict, nx, nx, Node.START, "cov")

    dx = stochastic_forward_dynamics(
        nlp.states.cx_start,
        nlp.controls.cx_start,
        nlp.parameters,
        nlp.stochastic_variables.cx_start,
        nlp,
        motor_noise,
        sensory_noise,
        force_field_magnitude=force_field_magnitude,
        with_gains=True,
    )

    ddx_dwm = cas.jacobian(dx.dxdt, cas.vertcat(sensory_noise, motor_noise))
    dg_dw = -ddx_dwm * dt
    ddx_dx = cas.jacobian(dx.dxdt, nlp.states.cx_start)
    dg_dx = -(ddx_dx * dt / 2 + cas.MX_eye(ddx_dx.shape[0]))

    p_next = m_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) @ m_matrix.T
    func_eval = cas.Function(
        "p_next",
        [
            nlp.states.cx_start,
            nlp.controls.cx_start,
            nlp.parameters,
            nlp.stochastic_variables.cx_start,
            cov_sym,
            motor_noise,
            sensory_noise,
        ],
        [p_next],
    )(
        nlp.states.cx_start,
        nlp.controls.cx_start,
        nlp.parameters,
        nlp.stochastic_variables.cx_start,
        nlp.integrated_values["cov"].cx_start,
        motor_noise_magnitude,
        sensory_noise_magnitude,
    )
    p_vector = nlp.integrated_values.reshape_to_vector(func_eval)
    return p_vector


def reach_target_consistantly(controllers: list[PenaltyController]) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """

    q_sym = cas.MX.sym("q_sym", controllers[-1].states["q"].cx_start.shape[0])
    qdot_sym = cas.MX.sym("qdot_sym", controllers[-1].states["qdot"].cx_start.shape[0])
    cov_sym = cas.MX.sym("cov", controllers[-1].integrated_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = (
        controllers[-1]
        .integrated_values["cov"]
        .reshape_to_matrix(
            cov_sym_dict,
            controllers[-1].states.cx_start.shape[0],
            controllers[-1].states.cx_start.shape[0],
            Node.START,
            "cov",
        )
    )

    hand_pos = controllers[0].model.end_effector_position(q_sym)
    hand_vel = controllers[0].model.end_effector_velocity(q_sym, qdot_sym)

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
        controllers[-1].integrated_values.cx_start,
    )
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val


def expected_feedback_effort(controllers: list[PenaltyController], sensory_noise_magnitude: cas.DM) -> cas.MX:
    """
    This function computes the expected effort due to the motor command and feedback gains for a given sensory noise
    magnitude.
    It is computed as Jacobian(effort, states) @ cov @ Jacobian(effort, states) +
                        Jacobian(effort, motor_noise) @ sigma_w @ Jacobian(effort, motor_noise)

    Parameters
    ----------
    controllers : list[PenaltyController]
        List of controllers to be used to compute the expected effort.
    sensory_noise_magnitude : cas.DM
        Magnitude of the sensory noise.
    """
    dt = controllers[0].tf / controllers[0].ns
    sensory_noise_matrix = sensory_noise_magnitude * cas.MX_eye(4)

    # create the casadi function to be evaluated
    # Get the symbolic variables
    ref = controllers[0].stochastic_variables["ref"].cx_start
    cov_sym = cas.MX.sym("cov", controllers[0].integrated_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = (
        controllers[0]
        .integrated_values["cov"]
        .reshape_to_matrix(
            cov_sym_dict,
            controllers[0].states.cx_start.shape[0],
            controllers[0].states.cx_start.shape[0],
            Node.START,
            "cov",
        )
    )

    k = controllers[0].stochastic_variables["k"].cx_start
    k_matrix = cas.MX(4, 6)
    for s0 in range(4):
        for s1 in range(6):
            k_matrix[s0, s1] = k[s0 * 6 + s1]
    k_matrix = k_matrix.T

    # Compute the expected effort
    hand_pos = controllers[0].model.end_effector_position(controllers[0].states["q"].cx_start)
    hand_vel = controllers[0].model.end_effector_velocity(
        controllers[0].states["q"].cx_start, controllers[0].states["qdot"].cx_start
    )
    trace_k_sensor_k = cas.trace(k_matrix @ sensory_noise_matrix @ k_matrix.T)
    ee = cas.vertcat(hand_pos, hand_vel)
    e_fb = k_matrix @ ((ee - ref) + sensory_noise_magnitude)
    jac_e_fb_x = cas.jacobian(e_fb, controllers[0].states.cx_start)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ cov_matrix @ jac_e_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function(
        "f_expectedEffort_fb",
        [controllers[0].states.cx_start, controllers[0].stochastic_variables.cx_start, cov_sym],
        [expectedEffort_fb_mx],
    )

    f_expectedEffort_fb = 0
    for i, ctrl in enumerate(controllers):
        P_vector = ctrl.integrated_values.cx_start
        out = func(ctrl.states.cx_start, ctrl.stochastic_variables.cx_start, P_vector)
        f_expectedEffort_fb += out * dt

    return f_expectedEffort_fb


def zero_acceleration(
    controller: PenaltyController, motor_noise: np.ndarray, sensory_noise: np.ndarray, force_field_magnitude: float
) -> cas.MX:
    """
    No acceleration of the joints at the beginning and end of the movement.
    """
    dx = stochastic_forward_dynamics(
        controller.states.cx_start,
        controller.controls.cx_start,
        controller.parameters.cx_start,
        controller.stochastic_variables.cx_start,
        controller.get_nlp,
        motor_noise,
        sensory_noise,
        force_field_magnitude=force_field_magnitude,
        with_gains=False,
    )
    return dx.dxdt[2:4]


def track_final_marker(controller: PenaltyController) -> cas.MX:
    """
    Track the hand position.
    """
    q = controller.states["q"].cx_start
    ee_pos = controller.model.end_effector_position(q)
    return ee_pos


def trapezoidal_integration_continuity_constraint(
    controllers: list[PenaltyController], force_field_magnitude
) -> cas.MX:
    """
    This function computes the continuity constraint for the trapezoidal integration scheme.
    It is computed as:
        x_i_plus - x_i - dt/2 * (f(x_i, u_i) + f(x_i_plus, u_i_plus)) = 0
    """
    motor_noise = np.zeros((2, 1))
    sensory_noise = np.zeros((4, 1))
    dt = controllers[0].tf / controllers[0].ns

    dx_i = stochastic_forward_dynamics(
        controllers[0].states.cx_start,
        controllers[0].controls.cx_start,
        controllers[0].parameters.cx_start,
        controllers[0].stochastic_variables.cx_start,
        controllers[0].get_nlp,
        motor_noise,
        sensory_noise,
        force_field_magnitude=force_field_magnitude,
        with_gains=False,
    ).dxdt
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

    out = controllers[1].states.cx_start - (controllers[0].states.cx_start + (dx_i + dx_i_plus) / 2 * dt)

    return out * 1e3


def prepare_socp(
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

    bio_model = LeuvenArmModel()

    shoulder_pos_initial = 0.349065850398866
    shoulder_pos_final = 0.959931088596881
    elbow_pos_initial = 2.245867726451909  # Optimized in Tom's version
    elbow_pos_final = 1.159394851847144  # Optimized in Tom's version

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="muscles", weight=1e3 / 2, quadratic=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, node=Node.ALL_SHOOTING, key="muscles", weight=1e3 / 2, quadratic=True
    )

    multinode_objectives = MultinodeObjectiveList()
    multinode_objectives.add(
        minimize_uncertainty,
        nodes_phase=[0 for _ in range(n_shooting)],
        nodes=[i for i in range(n_shooting)],
        key="muscles",
        weight=1e3 / 2,
        quadratic=False,
    )
    multinode_objectives.add(
        expected_feedback_effort,
        nodes_phase=[0 for _ in range(n_shooting)],
        nodes=[i for i in range(n_shooting)],
        sensory_noise_magnitude=sensory_noise_magnitude,
        weight=1e3 / 2,
        quadratic=False,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(hand_equals_ref, node=Node.ALL_SHOOTING)
    constraints.add(
        ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=np.array([shoulder_pos_initial, elbow_pos_initial])
    )
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=np.array([0, 0]))
    constraints.add(
        zero_acceleration,
        node=Node.START,
        motor_noise=np.zeros((2, 1)),
        sensory_noise=np.zeros((4, 1)),
        force_field_magnitude=force_field_magnitude,
    )
    constraints.add(track_final_marker, node=Node.PENULTIMATE, target=ee_final_position)
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.PENULTIMATE, target=np.array([0, 0]))
    constraints.add(
        zero_acceleration,
        node=Node.PENULTIMATE,
        motor_noise=np.zeros((2, 1)),
        sensory_noise=np.zeros((4, 1)),
        force_field_magnitude=force_field_magnitude,
    )  # Not possible sice the control on the last node is NaN
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="muscles", node=Node.ALL_SHOOTING, min_bound=0.001, max_bound=1)
    constraints.add(ConstraintFcn.TRACK_STATE, key="muscles", node=Node.ALL, min_bound=0.001, max_bound=1)
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
        motor_noise=np.zeros((2, 1)),
        sensory_noise=np.zeros((4, 1)),
    )

    n_muscles = 6
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_states = n_q + n_qdot + n_muscles

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
    x_bounds.add(
        "muscles",
        min_bound=states_min[n_q + n_qdot :, :],
        max_bound=states_max[n_q + n_qdot :, :],
        interpolation=InterpolationType.EACH_FRAME,
    )

    u_bounds = BoundsList()
    controls_min = np.ones((n_muscles, 3)) * -cas.inf
    controls_max = np.ones((n_muscles, 3)) * cas.inf
    u_bounds.add("muscles", min_bound=controls_min, max_bound=controls_max)

    # Initial guesses
    states_init = np.zeros((n_states, n_shooting + 1))
    states_init[0, :-1] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting)
    states_init[0, -1] = shoulder_pos_final
    states_init[1, :-1] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting)
    states_init[1, -1] = elbow_pos_final
    states_init[n_q + n_qdot :, :] = 0.01

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=states_init[:n_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=states_init[n_q : n_q + n_qdot, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("muscles", initial_guess=states_init[n_q + n_qdot :, :], interpolation=InterpolationType.EACH_FRAME)

    controls_init = np.ones((n_muscles, n_shooting)) * 0.01

    u_init = InitialGuessList()
    u_init.add("muscles", initial_guess=controls_init, interpolation=InterpolationType.EACH_FRAME)

    n_stochastic = n_muscles * (n_q + n_qdot) + n_q + n_qdot + n_states * n_states  # K(6x4) + ref(4x1) + M(10x10)
    s_init = InitialGuessList()
    s_bounds = BoundsList()
    stochastic_min = np.ones((n_stochastic, 3)) * -cas.inf
    stochastic_max = np.ones((n_stochastic, 3)) * cas.inf

    stochastic_init = np.zeros((n_stochastic, n_shooting + 1))
    curent_index = 0
    stochastic_init[: n_muscles * (n_q + n_qdot), :] = 0.01  # K
    s_init.add(
        "k",
        initial_guess=stochastic_init[: n_muscles * (n_q + n_qdot), :],
        interpolation=InterpolationType.EACH_FRAME,
    )
    s_bounds.add(
        "k",
        min_bound=stochastic_min[: n_muscles * (n_q + n_qdot), :],
        max_bound=stochastic_max[: n_muscles * (n_q + n_qdot), :],
    )
    curent_index += n_muscles * (n_q + n_qdot)
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

    integrated_value_functions = {
        "cov": lambda nlp, node_index: get_cov_mat(
            nlp,
            node_index,
            force_field_magnitude=force_field_magnitude,
            motor_noise_magnitude=motor_noise_magnitude,
            sensory_noise_magnitude=sensory_noise_magnitude,
        )
    }

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
        multinode_objectives=multinode_objectives,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        ode_solver=None,
        skip_continuity=True,
        n_threads=1,
        assume_phase_dynamics=False,
        problem_type=SocpType.SOCP_EXPLICIT(motor_noise_magnitude, sensory_noise_magnitude),
        integrated_value_functions=integrated_value_functions,
    )


def main():
    # --- Options --- #
    plot_sol_flag = False  # True
    vizualise_sol_flag = False  # True

    biorbd_model_path = "models/LeuvenArmModel.bioMod"

    ee_initial_position = np.array([0.0, 0.2742])  # Directly from Tom's version
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
        final_time=final_time,
        n_shooting=n_shooting,
        ee_final_position=ee_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        problem_type=problem_type,
        force_field_magnitude=force_field_magnitude,
    )

    sol_socp = socp.solve(solver)

    q_sol = sol_socp.states["q"]
    qdot_sol = sol_socp.states["qdot"]
    activations_sol = sol_socp.states["muscles"]
    excitations_sol = sol_socp.controls["muscles"]
    k_sol = sol_socp.stochastic_variables["k"]
    ref_sol = sol_socp.stochastic_variables["ref"]
    m_sol = sol_socp.stochastic_variables["m"]
    cov_sol_vect = sol_socp.integrated_values["cov"]
    cov_sol = np.zeros((10, 10, n_shooting))
    for i in range(n_shooting):
        for j in range(10):
            for k in range(10):
                cov_sol[j, k, i] = cov_sol_vect[j * 10 + k, i]
    stochastic_variables_sol = np.vstack((k_sol, ref_sol, m_sol))
    data = {
        "q_sol": q_sol,
        "qdot_sol": qdot_sol,
        "activations_sol": activations_sol,
        "excitations_sol": excitations_sol,
        "k_sol": k_sol,
        "ref_sol": ref_sol,
        "m_sol": m_sol,
        "cov_sol": cov_sol,
        "stochastic_variables_sol": stochastic_variables_sol,
    }

    # --- Save the results --- #
    with open(f"leuvenarm_muscle_driven_socp_{problem_type}_forcefield{force_field_magnitude}.pkl", "wb") as file:
        pickle.dump(data, file)

    # --- Visualize the solution --- #
    if vizualise_sol_flag:
        import bioviz

        b = bioviz.Viz(model_path=biorbd_model_path)
        b.load_movement(q_sol[:, :-1])
        b.exec()

    # --- Plot the results --- #
    if plot_sol_flag:
        model = LeuvenArmModel()
        q_sym = cas.MX.sym("q_sym", 2, 1)
        qdot_sym = cas.MX.sym("qdot_sym", 2, 1)
        hand_pos_fcn = cas.Function("hand_pos", [q_sym], [model.end_effector_position(q_sym)])
        hand_vel_fcn = cas.Function("hand_vel", [q_sym, qdot_sym], [model.end_effector_velocity(q_sym, qdot_sym)])

        states = socp.nlp[0].states.cx_start
        controls = socp.nlp[0].controls.cx_start
        parameters = socp.nlp[0].parameters.cx_start
        stochastic_variables = socp.nlp[0].stochastic_variables.cx_start
        nlp = socp.nlp[0]
        motor_noise_sym = cas.MX.sym("motor_noise", 2, 1)
        sensory_noise_sym = cas.MX.sym("sensory_noise", 4, 1)
        out = stochastic_forward_dynamics(
            states,
            controls,
            parameters,
            stochastic_variables,
            nlp,
            motor_noise_sym,
            sensory_noise_sym,
            force_field_magnitude=force_field_magnitude,
            with_gains=True,
        )
        dyn_fun = cas.Function(
            "dyn_fun",
            [states, controls, parameters, stochastic_variables, motor_noise_sym, sensory_noise_sym],
            [out.dxdt],
        )

        fig, axs = plt.subplots(3, 2)
        n_simulations = 30
        q_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        qdot_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        mus_activation_simulated = np.zeros((n_simulations, 6, n_shooting + 1))
        hand_pos_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        hand_vel_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        for i_simulation in range(n_simulations):
            motor_noise = np.random.normal(0, motor_noise_std, (2, n_shooting + 1))
            wPq = np.random.normal(0, wPq_std, (2, n_shooting + 1))
            wPqdot = np.random.normal(0, wPqdot_std, (2, n_shooting + 1))
            sensory_noise = cas.vertcat(wPq, wPqdot)
            q_simulated[i_simulation, :, 0] = q_sol[:, 0]
            qdot_simulated[i_simulation, :, 0] = qdot_sol[:, 0]
            mus_activation_simulated[i_simulation, :, 0] = activations_sol[:, 0]
            for i_node in range(n_shooting):
                x_prev = cas.vertcat(
                    q_simulated[i_simulation, :, i_node],
                    qdot_simulated[i_simulation, :, i_node],
                    mus_activation_simulated[i_simulation, :, i_node],
                )
                hand_pos_simulated[i_simulation, :, i_node] = np.reshape(hand_pos_fcn(x_prev[:2])[:2], (2,))
                hand_vel_simulated[i_simulation, :, i_node] = np.reshape(
                    hand_vel_fcn(x_prev[:2], x_prev[2:4])[:2], (2,)
                )
                u = excitations_sol[:, i_node]
                s = stochastic_variables_sol[:, i_node]
                k1 = dyn_fun(x_prev, u, [], s, motor_noise[:, i_node], sensory_noise[:, i_node])
                x_next = x_prev + dt * dyn_fun(
                    x_prev + dt / 2 * k1, u, [], s, motor_noise[:, i_node], sensory_noise[:, i_node]
                )
                q_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[:2], (2,))
                qdot_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[2:4], (2,))
                mus_activation_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[4:], (6,))
            hand_pos_simulated[i_simulation, :, i_node + 1] = np.reshape(hand_pos_fcn(x_next[:2])[:2], (2,))
            hand_vel_simulated[i_simulation, :, i_node + 1] = np.reshape(
                hand_vel_fcn(x_next[:2], x_next[2:4])[:2], (2,)
            )
            axs[0, 0].plot(
                hand_pos_simulated[i_simulation, 0, :], hand_pos_simulated[i_simulation, 1, :], color="tab:red"
            )
            axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 0, :], color="k")
            axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 1, :], color="k")
            axs[0, 1].plot(
                hand_vel_simulated[i_simulation, 0, :], hand_vel_simulated[i_simulation, 1, :], color="tab:red"
            )
            axs[1, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_simulated[i_simulation, 0, :], color="k")
            axs[2, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_simulated[i_simulation, 1, :], color="k")
        hand_pos_without_noise = np.zeros((2, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            hand_pos_without_noise[:, i_node] = np.reshape(hand_pos_fcn(q_sol[:, i_node])[:2], (2,))
        axs[0, 0].plot(hand_pos_without_noise[0, :], hand_pos_without_noise[1, :], color="k")
        axs[0, 0].plot(ee_initial_position[0], ee_initial_position[1], color="tab:green", marker="o")
        axs[0, 0].plot(ee_final_position[0], ee_final_position[1], color="tab:red", marker="o")
        axs[0, 0].set_xlabel("X [m]")
        axs[0, 0].set_ylabel("Y [m]")
        axs[0, 0].set_title("Hand position simulated")
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 0].set_ylabel("Shoulder angle [rad]")
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_ylabel("Elbow angle [rad]")
        axs[0, 1].set_xlabel("X velocity [m/s]")
        axs[0, 1].set_ylabel("Y velocity [m/s]")
        axs[0, 1].set_title("Hand velocity simulated")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].set_ylabel("Shoulder velocity [rad/s]")
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 1].set_ylabel("Elbow velocity [rad/s]")
        axs[0, 0].axis("equal")
        plt.tight_layout()
        plt.savefig("simulated_results.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
