"""
This example replicates the results from "An approximate stochastic optimal control framework to simulate nonlinear
neuro-musculoskeletal models in the presence of noise"(https://doi.org/10.1371/journal.pcbi.1009338).

The task is to unfold the arm to reach a target further from the trunk.
Noise is added on the motor execution (wM) and on the feedback (wEE=wP and wEE_dot=wPdot).
The expected joint angles (x_mean) are optimized like in a deterministic OCP, but the covariance matrix is minimized to
reduce uncertainty. This covariance matrix is computed from the expected states.
"""

import platform

import pickle
import biorbd_casadi as biorbd
import matplotlib.pyplot as plt
import casadi as cas
import numpy as np
from IPython import embed
import scipy.io as sio

import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    OptimalControlProgram,
    Bounds,
    InitialGuess,
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
    OcpType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
    MultinodeConstraintList,
    MultinodeObjectiveList,
)



def get_muscle_force(q, qdot):
    """
    Fa: active muscle force [N]
    Fp: passive muscle force [N]
    lMtilde: normalized fiber lenght [-]
    vMtilde: optimal fiber lenghts per second at which muscle is lengthening or shortening [-]
    FMltilde: force-length multiplier [-]
    FMvtilde: force-velocity multiplier [-]
    Fce: Active muscle force [N]
    Fpe: Passive elastic force [N]
    Fm: Passive viscous force [N]
    """

    dM_coefficients = np.array([[0, 0, 0.0100, 0.0300, -0.0110, 1.9000],
                                [0, 0, 0.0100, -0.0190, 0, 0.0100],
                                [0.0400, -0.0080, 1.9000, 0, 0, 0.0100],
                                [-0.0420, 0, 0.0100, 0, 0, 0.0100],
                                [0.0300, -0.0110, 1.9000, 0.0320, -0.0100, 1.9000],
                                [-0.0390, 0, 0.0100, -0.0220, 0, 0.0100]])

    LMT_coefficients = np.array([[1.1000, -5.206336195535022],
                                 [0.8000, -7.538918356984516],
                                 [1.2000, -3.938098437958920],
                                 [0.7000, -3.031522725559912],
                                 [1.1000, -2.522778221157014],
                                 [0.8500, -1.826368199415192]])

    vMtilde_max = np.ones((6, 1)) * 10
    Fiso = np.array([572.4000, 445.2000, 699.6000, 381.6000, 159.0000, 318.0000])
    Faparam = np.array([0.814483478343008, 1.055033428970575, 0.162384573599574, 0.063303448465465, 0.433004984392647,
                        0.716775413397760, -0.029947116970696, 0.200356847296188])
    Fvparam = np.array([-0.318323436899127, -8.149156043475250, -0.374121508647863, 0.885644059915004])
    Fpparam = np.array([-0.995172050006169, 53.598150033144236])
    muscleDampingCoefficient = np.ones((6, 1)) * 0.01

    a_shoulder = dM_coefficients[:, 0]
    b_shoulder = dM_coefficients[:, 1]
    c_shoulder = dM_coefficients[:, 2]
    a_elbow = dM_coefficients[:, 3]
    b_elbow = dM_coefficients[:, 4]
    c_elbow = dM_coefficients[:, 5]
    l_base = LMT_coefficients[:, 0]
    l_multiplier = LMT_coefficients[:, 1]
    theta_shoulder = q[0]
    theta_elbow = q[1]
    dtheta_shoulder = qdot[0]
    dtheta_elbow = qdot[1]

    # Normalized muscle fiber length (without tendon)
    l_full = a_shoulder * theta_shoulder + b_shoulder * cas.sin(
        c_shoulder * theta_shoulder) / c_shoulder + a_elbow * theta_elbow + b_elbow * cas.sin(
        c_elbow * theta_elbow) / c_elbow
    lMtilde = l_full * l_multiplier + l_base

    # fiber velocity normalized by the optimal fiber length
    nCoeff = a_shoulder.shape[0]
    v_full = a_shoulder * dtheta_shoulder + b_shoulder * cas.cos(c_shoulder * theta_shoulder) * cas.repmat(
        dtheta_shoulder, nCoeff, 1) + a_elbow * dtheta_elbow + b_elbow * cas.cos(c_elbow * theta_elbow) * cas.repmat(
        dtheta_elbow, nCoeff, 1)
    vMtilde = l_multiplier * v_full

    vMtilde_normalizedToMaxVelocity = vMtilde / vMtilde_max

    # Active muscle force-length characteristic
    b11 = Faparam[0]
    b21 = Faparam[1]
    b31 = Faparam[2]
    b41 = Faparam[3]
    b12 = Faparam[4]
    b22 = Faparam[5]
    b32 = Faparam[6]
    b42 = Faparam[7]

    b13 = 0.1
    b23 = 1
    b33 = 0.5 * cas.sqrt(0.5)
    b43 = 0
    num3 = lMtilde - b23
    den3 = b33 + b43 * lMtilde
    FMtilde3 = b13 * cas.exp(-0.5 * num3 ** 2 / den3 ** 2)

    num1 = lMtilde - b21
    den1 = b31 + b41 * lMtilde
    FMtilde1 = b11 * cas.exp(-0.5 * num1 ** 2 / den1 ** 2)

    num2 = lMtilde - b22
    den2 = b32 + b42 * lMtilde
    FMtilde2 = b12 * cas.exp(-0.5 * num2 ** 2 / den2 ** 2)

    FMltilde = FMtilde1 + FMtilde2 + FMtilde3

    e1 = Fvparam[0]
    e2 = Fvparam[1]
    e3 = Fvparam[2]
    e4 = Fvparam[3]

    FMvtilde = e1 * cas.log(
        (e2 @ vMtilde_normalizedToMaxVelocity + e3) + cas.sqrt(
            (e2 @ vMtilde_normalizedToMaxVelocity + e3) ** 2 + 1)) + e4

    # Active muscle force
    Fce = FMltilde * FMvtilde

    # Passive muscle force - length characteristic
    e0 = 0.6
    kpe = 4
    t5 = cas.exp(kpe * (lMtilde - 0.10e1) / e0)
    Fpe = ((t5 - 0.10e1) - Fpparam[0]) / Fpparam[1]

    # Muscle force + damping
    Fpv = muscleDampingCoefficient * vMtilde_normalizedToMaxVelocity
    Fa = Fiso * Fce
    Fp = Fiso * (Fpe + Fpv)

    return Fa, Fp


def torque_force_relationship(Fm, q):

    dM_coefficients = np.array([[0, 0, 0.0100, 0.0300, -0.0110, 1.9000],
                                [0, 0, 0.0100, -0.0190, 0, 0.0100],
                                [0.0400, -0.0080, 1.9000, 0, 0, 0.0100],
                                [-0.0420, 0, 0.0100, 0, 0, 0.0100],
                                [0.0300, -0.0110, 1.9000, 0.0320, -0.0100, 1.9000],
                                [-0.0390, 0, 0.0100, -0.0220, 0, 0.0100]])

    a_shoulder = dM_coefficients[:, 0]
    b_shoulder = dM_coefficients[:, 1]
    c_shoulder = dM_coefficients[:, 2]
    a_elbow = dM_coefficients[:, 3]
    b_elbow = dM_coefficients[:, 4]
    c_elbow = dM_coefficients[:, 5]
    theta_shoulder = q[0]
    theta_elbow = q[1]

    dM_matrix = cas.horzcat(a_shoulder + b_shoulder * cas.cos(c_shoulder @ theta_shoulder),
                            a_elbow + b_elbow * cas.cos(c_elbow @ theta_elbow)).T
    tau = dM_matrix @ Fm
    return tau

def get_muscle_torque(q, qdot, mus_activations):
    Fa, Fp = get_muscle_force(q, qdot)
    Fm = mus_activations * Fa + Fp
    muscles_tau = torque_force_relationship(Fm, q)
    return muscles_tau

def get_force_field(q, force_field_magnitude):
    l1 = 0.3
    l2 = 0.33
    F_forceField = force_field_magnitude * (l1 * cas.cos(q[0]) + l2 * cas.cos(q[0] + q[1]))
    hand_pos = cas.MX(2, 1)
    hand_pos[0] = l2 * cas.sin(q[0] + q[1]) + l1 * cas.sin(q[0])
    hand_pos[1] = l2 * cas.sin(q[0] + q[1])
    tau_force_field = -F_forceField @ hand_pos
    return tau_force_field

def get_excitation_with_feedback(K, EE, EE_ref, wPq, wPqdot):
    return K @ ((EE - EE_ref) + cas.vertcat(wPq, wPqdot))

def stochastic_forward_dynamics(
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    stochastic_variables: cas.MX | cas.SX,
    nlp: NonLinearProgram,
    wM,
    wPq,
    wPqdot,
    force_field_magnitude,
    with_gains=True,
) -> DynamicsEvaluation:

    tau_coef = 0.1500

    m2 = 1
    l1 = 0.3
    lc2 = 0.16
    I1 = 0.025
    I2 = 0.045

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    mus_activations = DynamicsFunctions.get(nlp.states["muscles"], states)
    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)

    mus_excitations_fb = mus_excitations
    if with_gains:
        ee_ref = DynamicsFunctions.get(nlp.stochastic_variables["ee_ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        K_matrix = cas.MX(4, 6)
        for s0 in range(4):
            for s1 in range(6):
                K_matrix[s0, s1] = k[s0*6 + s1]
        K_matrix = K_matrix.T

        hand_pos = end_effector_position(q)
        hand_vel = end_effector_velocity(q, qdot)
        ee = cas.vertcat(hand_pos, hand_vel)

        mus_excitations_fb += get_excitation_with_feedback(K_matrix, ee, ee_ref, wPq, wPqdot)

    muscles_tau = get_muscle_torque(q, qdot, mus_activations)

    tau_force_field = get_force_field(q, force_field_magnitude)

    torques_computed = muscles_tau + wM + tau_force_field  # + residual_tau
    dq_computed = qdot  ### Do not use "DynamicsFunctions.compute_qdot(nlp, q, qdot)" it introduces errors!!
    dactivations_computed = (mus_excitations_fb - mus_activations) / tau_coef

    friction = np.array([[0.05, 0.025], [0.025, 0.05]])

    a1 = I1 + I2 + m2 * l1 ** 2
    a2 = m2 * l1 * lc2
    a3 = I2

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
    nleffects[1] = a2 * cas.sin(theta_elbow) * dtheta_shoulder ** 2

    dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(dq_computed, dqdot_computed, dactivations_computed), defects=None)

def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram, dynamic_function: callable, wM, wPq, wPqdot):

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    ConfigureProblem.configure_muscles(ocp, nlp, True, True)  # Muscles activations as states + muscles excitations as controls

    # Stochastic variables
    ConfigureProblem.configure_k(ocp, nlp, n_controls=6, n_feedbacks=4)
    ConfigureProblem.configure_ee_ref(ocp, nlp, n_references=4)
    ConfigureProblem.configure_m(ocp, nlp)
    ConfigureProblem.configure_cov(ocp, nlp)
    ConfigureProblem.configure_dynamics_function(ocp, nlp,
                                                 dyn_func=lambda states, controls, parameters,
                                                                stochastic_variables, nlp, wM, wPq, wPqdot: dynamic_function(states,
                                                                                            controls,
                                                                                            parameters,
                                                                                            stochastic_variables,
                                                                                            nlp,
                                                                                            wM,
                                                                                            wPq,
                                                                                            wPqdot,
                                                                                            with_gains=False),
                                                 wM=wM, wPq=wPq, wPqdot=wPqdot, expand=False)

def minimize_uncertainty(controllers: list[PenaltyController], key: str) -> cas.MX:
    """
    Minimize the uncertainty (covariance matrix) of the states.
    """
    dt = controllers[0].tf / controllers[0].ns
    out = 0
    for i, ctrl in enumerate(controllers):
        P_matrix = ctrl.restore_matrix_from_vector(ctrl.update_values, ctrl.states.cx.shape[0],
                                                         ctrl.states.cx.shape[0], Node.START, "cov")
        P_partial = P_matrix[ctrl.states[key].index, ctrl.states[key].index]
        out += cas.trace(P_partial) * dt
    return out

def get_ee(controller: PenaltyController, q, qdot) -> cas.MX:
    hand_pos = end_effector_position(q)
    hand_vel = end_effector_velocity(q, qdot)
    ee = cas.vertcat(hand_pos, hand_vel)
    return ee

def ee_equals_ee_ref(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    qdot = controller.states["qdot"].cx_start
    ee_ref = controller.stochastic_variables["ee_ref"].cx_start
    ee = get_ee(controller, q, qdot)
    return ee - ee_ref


def get_p_mat(nlp, node_index, force_field_magnitude):
    wM_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024
    dt = nlp.tf / nlp.ns
    wM_numerical = np.array([wM_std ** 2 / dt, wM_std ** 2 / dt])
    wPq_numerical = np.array([wPq_std ** 2 / dt, wPq_std ** 2 / dt])
    wPqdot_numerical = np.array([wPqdot_std ** 2 / dt, wPqdot_std ** 2 / dt])

    nlp.states.node_index = node_index - 1
    nlp.controls.node_index = node_index - 1
    nlp.stochastic_variables.node_index = node_index - 1
    nlp.update_values.node_index = node_index - 1

    nx = nlp.states.cx_start.shape[0]
    M_matrix = nlp.restore_matrix_from_vector(nlp.stochastic_variables, nx, nx, Node.START, "m")

    wM = cas.MX.sym("wM", nlp.states['q'].cx_start.shape[0])
    wP = cas.MX.sym("wP", nlp.states['q'].cx_start.shape[0])
    wPdot = cas.MX.sym("wPdot", nlp.states['q'].cx_start.shape[0])
    sigma_w = cas.vertcat(wP, wPdot, wM) * cas.MX_eye(6)
    cov_sym = cas.MX.sym("cov", nlp.update_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = nlp.restore_matrix_from_vector(cov_sym_dict, nx, nx, Node.START, "cov")

    dx = stochastic_forward_dynamics(nlp.states.cx_start, nlp.controls.cx_start,
                                     nlp.parameters, nlp.stochastic_variables.cx_start,
                                     nlp, wM, wP, wPdot, force_field_magnitude=force_field_magnitude, with_gains=True)

    ddx_dwM = cas.jacobian(dx.dxdt, cas.vertcat(wP, wPdot, wM))
    dg_dw = - ddx_dwM * dt
    ddx_dx = cas.jacobian(dx.dxdt, nlp.states.cx_start)
    dg_dx = - (ddx_dx * dt / 2 + cas.MX_eye(ddx_dx.shape[0]))

    p_next = M_matrix @ (dg_dx @ cov_matrix @ dg_dx.T + dg_dw @ sigma_w @ dg_dw.T) @ M_matrix.T
    func_eval = cas.Function("p_next", [nlp.states.cx_start, nlp.controls.cx_start,
                                          nlp.parameters, nlp.stochastic_variables.cx_start, cov_sym,
                                          wM, wP, wPdot], [p_next])(nlp.states.cx_start,
                                                                          nlp.controls.cx_start,
                                                                          nlp.parameters,
                                                                          nlp.stochastic_variables.cx_start,
                                                                          nlp.update_values.cx_start,  # Should be the right shape to work
                                                                          wM_numerical,
                                                                          wPq_numerical,
                                                                          wPqdot_numerical)
    p_vector = nlp.restore_vector_from_matrix(func_eval)
    return p_vector

def reach_target_consistantly(controllers: list[PenaltyController]) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    This is a multi-node constraint because the covariance matrix depends on all the precedent nodes, but it only
    applies at the END node.
    """

    Q = cas.MX.sym("q_sym", controllers[-1].states["q"].cx_start.shape[0])
    Qdot = cas.MX.sym("qdot_sym", controllers[-1].states["qdot"].cx_start.shape[0])
    cov_sym = cas.MX.sym("cov", controllers[-1].update_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = controllers[-1].restore_matrix_from_vector(cov_sym_dict, controllers[-1].states.cx_start.shape[0], controllers[-1].states.cx_start.shape[0], Node.START, "cov")

    hand_pos = end_effector_position(Q)
    hand_vel = end_effector_velocity(Q, Qdot)

    jac_marker_q = cas.jacobian(hand_pos, Q)
    jac_marker_qdot = cas.jacobian(hand_vel, cas.vertcat(Q, Qdot))

    P_matrix_q = cov_matrix[:2, :2]
    P_matrix_qdot = cov_matrix[:4, :4]

    pos_constraint = jac_marker_q @ P_matrix_q @ jac_marker_q.T
    vel_constraint = jac_marker_qdot @ P_matrix_qdot @ jac_marker_qdot.T

    out = cas.vertcat(pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1])

    fun = cas.Function("reach_target_consistantly", [Q, Qdot, cov_sym], [out])
    val = fun(controllers[-1].states["q"].cx_start, controllers[-1].states["qdot"].cx_start, controllers[-1].update_values.cx_start)
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val

def end_effector_position(q):
    l1 = 0.3
    l2 = 0.33

    theta_shoulder = q[0]
    theta_elbow = q[1]
    ee_pos = cas.vertcat(cas.cos(theta_shoulder)*l1 + cas.cos(theta_shoulder + theta_elbow)*l2,
                        cas.sin(theta_shoulder)*l1 + cas.sin(theta_shoulder + theta_elbow)*l2)
    return ee_pos


def end_effector_velocity(q, qdot):
    l1 = 0.3
    l2 = 0.33

    theta_shoulder = q[0]
    theta_elbow = q[1]
    a = theta_shoulder + theta_elbow
    dtheta_shoulder = qdot[0]
    dtheta_elbow = qdot[1]
    da = dtheta_shoulder + dtheta_elbow

    ee_vel = cas.vertcat(dtheta_shoulder * cas.sin(theta_shoulder)*l1 + da*cas.sin(a)*l2,
            -dtheta_shoulder * cas.cos(theta_shoulder)*l1 - da * cas.cos(a)*l2)
    return ee_vel

def expected_feedback_effort(controllers: list[PenaltyController]) -> cas.MX:
    """
    ...
    """
    # Constants TODO: remove fom here
    # TODO: How do we choose?
    wM_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024
    dt = controllers[0].tf / controllers[0].ns
    wM_numerical = np.array([wM_std ** 2 / dt, wM_std ** 2 / dt])
    wPq_numerical = np.array([wPq_std ** 2 / dt, wPq_std ** 2 / dt])
    wPqdot_numerical = np.array([wPqdot_std ** 2 / dt, wPqdot_std ** 2 / dt])

    sensory_noise = cas.vertcat(wPq_numerical, wPqdot_numerical)
    sensory_noise_matrix = sensory_noise * cas.MX_eye(4)

    # create the casadi function to be evaluated
    # Get the symbolic variables
    ee_ref = controllers[0].stochastic_variables["ee_ref"].cx_start
    cov_sym = cas.MX.sym("cov", controllers[0].update_values.cx_start.shape[0])
    cov_sym_dict = {"cov": cov_sym}
    cov_sym_dict["cov"].cx_start = cov_sym
    cov_matrix = controllers[0].restore_matrix_from_vector(cov_sym_dict, controllers[0].states.cx_start.shape[0], controllers[0].states.cx_start.shape[0], Node.START, "cov")

    k = controllers[0].stochastic_variables["k"].cx_start
    K_matrix = cas.MX(4, 6)
    for s0 in range(4):
        for s1 in range(6):
            K_matrix[s0, s1] = k[s0 * 6 + s1]
    K_matrix = K_matrix.T

    # Compute the expected effort
    hand_pos = end_effector_position(controllers[0].states["q"].cx_start)
    hand_vel = end_effector_velocity(controllers[0].states["q"].cx_start, controllers[0].states["qdot"].cx_start)
    trace_k_sensor_k = cas.trace(K_matrix @ sensory_noise_matrix @ K_matrix.T)
    ee = cas.vertcat(hand_pos, hand_vel)
    e_fb = K_matrix @ ((ee - ee_ref) + sensory_noise)
    jac_e_fb_x = cas.jacobian(e_fb, controllers[0].states.cx_start)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ cov_matrix @ jac_e_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    func = cas.Function('f_expectedEffort_fb',
                                       [controllers[0].states.cx_start, controllers[0].stochastic_variables.cx_start, cov_sym],
                                       [expectedEffort_fb_mx])

    f_expectedEffort_fb = 0
    for i, ctrl in enumerate(controllers):
        P_vector = ctrl.update_values.cx_start
        out = func(ctrl.states.cx_start, ctrl.stochastic_variables.cx_start, P_vector)
        f_expectedEffort_fb += out * dt

    return f_expectedEffort_fb


def zero_acceleration(controller: PenaltyController, wM: np.ndarray, wPq: np.ndarray, wPqdot: np.ndarray, force_field_magnitude:float) -> cas.MX:
    dx = stochastic_forward_dynamics(controller.states.cx_start, controller.controls.cx_start,
                                     controller.parameters.cx_start, controller.stochastic_variables.cx_start,
                                     controller.get_nlp, wM, wPq, wPqdot, force_field_magnitude=force_field_magnitude)
    return dx.dxdt[2:4]

def track_final_marker(controller: PenaltyController) -> cas.MX:
    q = controller.states["q"].cx_start
    ee_pos = end_effector_position(q)
    return ee_pos

def leuven_trapezoidal(controllers: list[PenaltyController], force_field_magnitude) -> cas.MX:

    wM = np.zeros((2, 1))
    wPq = np.zeros((2, 1))
    wPqdot = np.zeros((2, 1))
    dt = controllers[0].tf / controllers[0].ns

    dX_i = stochastic_forward_dynamics(controllers[0].states.cx_start, controllers[0].controls.cx_start,
                                        controllers[0].parameters.cx_start, controllers[0].stochastic_variables.cx_start,
                                        controllers[0].get_nlp, wM, wPq, wPqdot, force_field_magnitude=force_field_magnitude, with_gains=False).dxdt
    dX_i_plus = stochastic_forward_dynamics(controllers[1].states.cx_start, controllers[1].controls.cx_start,
                                        controllers[1].parameters.cx_start, controllers[1].stochastic_variables.cx_start,
                                        controllers[1].get_nlp, wM, wPq, wPqdot, force_field_magnitude=force_field_magnitude, with_gains=False).dxdt

    out = controllers[1].states.cx_start - (controllers[0].states.cx_start + (dX_i + dX_i_plus) / 2 * dt)

    return out * 1e3

def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ee_final_position: np.ndarray,
    force_field_magnitude: float = 0,
    problem_type: str = "CIRCLE",
) -> OptimalControlProgram:
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
    ee_initial_position: np.ndarray
        The initial position of the end effector
    force_field_magnitude: float
        The magnitude of the force field
    problem_type: str
        The type of problem to solve (CIRCLE or BAR)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    shoulder_pos_initial = 0.349065850398866
    shoulder_pos_final = 0.959931088596881
    elbow_pos_initial = 2.245867726451909  # Optimized in Tom's version
    elbow_pos_final = 1.159394851847144  # Optimized in Tom's version

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="muscles", weight=1e3/2, quadratic=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, node=Node.ALL_SHOOTING, key="muscles", weight=1e3/2, quadratic=True)

    multinode_objectives = MultinodeObjectiveList()
    multinode_objectives.add(minimize_uncertainty,
                                nodes_phase=[0 for _ in range(n_shooting)],
                                nodes=[i for i in range(n_shooting)],
                                key="muscles",
                                weight=1e3 / 2,
                                quadratic=False)
    multinode_objectives.add(expected_feedback_effort,
                             nodes_phase=[0 for _ in range(n_shooting)],
                             nodes=[i for i in range(n_shooting)],
                             weight=1e3 / 2,
                             quadratic=False)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ee_equals_ee_ref, node=Node.ALL_SHOOTING)
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.START, target=np.array([shoulder_pos_initial, elbow_pos_initial]))
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.START, target=np.array([0, 0]))
    constraints.add(zero_acceleration, node=Node.START, wM=np.zeros((2, 1)), wPq=np.zeros((2, 1)), wPqdot=np.zeros((2, 1)), force_field_magnitude=force_field_magnitude)
    constraints.add(track_final_marker, node=Node.PENULTIMATE, target=ee_final_position)
    constraints.add(ConstraintFcn.TRACK_STATE, key="qdot", node=Node.PENULTIMATE, target=np.array([0, 0]))
    constraints.add(zero_acceleration, node=Node.PENULTIMATE, wM=np.zeros((2, 1)), wPq=np.zeros((2, 1)), wPqdot=np.zeros((2, 1)), force_field_magnitude=force_field_magnitude)  # Not possible sice the control on the last node is NaN
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="muscles", node=Node.ALL_SHOOTING, min_bound=0.001, max_bound=1)
    constraints.add(ConstraintFcn.TRACK_STATE, key="muscles", node=Node.ALL, min_bound=0.001, max_bound=1)
    constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.ALL, min_bound=0, max_bound=180)  # This is a bug, it should be in radians

    if problem_type == "BAR":
        max_bounds_lateral_variation = cas.inf
    elif problem_type == "CIRCLE":
        max_bounds_lateral_variation = 0.004
    else:
        raise NotImplementedError("Wrong problem type")

    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(reach_target_consistantly,
                              nodes_phase=[0 for _ in range(n_shooting)],
                              nodes=[i for i in range(n_shooting)],
                              min_bound=np.array([-cas.inf, -cas.inf, -cas.inf, -cas.inf]),
                              max_bound=np.array([max_bounds_lateral_variation**2, 0.004**2, 0.05**2, 0.05**2]))
    for i in range(n_shooting-1):
        multinode_constraints.add(leuven_trapezoidal,
                                  nodes_phase=[0, 0],
                                  nodes=[i, i+1],
                                  force_field_magnitude=force_field_magnitude)

    # Dynamics
    dynamics = DynamicsList()
    # dynamics.add(configure_stochastic_optimal_control_problem, dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, wM, wPq, wPqdot: stochastic_forward_dynamics(states, controls, parameters, stochastic_variables, nlp, wM, wPq, wPqdot, force_field_magnitude=force_field_magnitude, with_gains=False), wM=np.zeros((2, 1)), wPq=np.zeros((2, 1)), wPqdot=np.zeros((2, 1)))
    # dynamic_function = stochastic_forward_dynamics,
    dynamics.add(configure_stochastic_optimal_control_problem,
                 dynamic_function=lambda states, controls, parameters, stochastic_variables, nlp, wM, wPq,
                                         wPqdot, with_gains: stochastic_forward_dynamics(states, controls, parameters,
                                                                             stochastic_variables, nlp, wM, wPq, wPqdot,
                                                                             force_field_magnitude=force_field_magnitude,
                                                                             with_gains=with_gains),
                 wM=np.zeros((2, 1)), wPq=np.zeros((2, 1)), wPqdot=np.zeros((2, 1)))

    n_muscles = 6
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_states = n_q + n_qdot + n_muscles

    states_min = np.ones((n_states, n_shooting+1)) * -cas.inf
    states_max = np.ones((n_states, n_shooting+1)) * cas.inf

    x_bounds = BoundsList()
    x_bounds.add(bounds=Bounds(states_min, states_max, interpolation=InterpolationType.EACH_FRAME))

    u_bounds = BoundsList()
    controls_min = np.ones((n_muscles, 3)) * -cas.inf
    controls_max = np.ones((n_muscles, 3)) * cas.inf
    u_bounds.add(bounds=Bounds(controls_min, controls_max))

    input_sol_FLAG = False  # True
    if input_sol_FLAG:
        #load pickle
        with open(f"leuvenarm_muscle_driven_socp_{problem_type}_forcefield{force_field_magnitude}.pkl", 'rb') as f:
            data = pickle.load(f)
            q_sol = data["q_sol"]
            qdot_sol = data["qdot_sol"]
            activations_sol = data["activations_sol"]
            excitations_sol = data["excitations_sol"]
            k_sol = data["k_sol"]
            ee_ref_sol = data["ee_ref_sol"]
            m_sol = data["m_sol"]
            # cov_sol = data["cov_sol"]
            stochastic_variables_sol = data["stochastic_variables_sol"]

    # Initial guesses
    if not input_sol_FLAG:
        states_init = np.zeros((n_states, n_shooting + 1))
        states_init[0, :-1] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting)
        states_init[0, -1] = shoulder_pos_final
        states_init[1, :-1] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting)
        states_init[1, -1] = elbow_pos_final
        states_init[n_q + n_qdot:, :] = 0.01
    else:
        states_init = cas.vertcat(q_sol, qdot_sol, activations_sol)
    x_init = InitialGuess(states_init, interpolation=InterpolationType.EACH_FRAME)

    if not input_sol_FLAG:
        controls_init = np.ones((n_muscles, n_shooting)) * 0.01
    else:
        controls_init = excitations_sol[:, :-1]
    u_init = InitialGuess(controls_init, interpolation=InterpolationType.EACH_FRAME)

    # TODO: This should probably be done automatically, not defined by the user
    n_stochastic = n_muscles*(n_q + n_qdot) + n_q+n_qdot + n_states*n_states  # K(6x4) + ee_ref(4x1) + M(10x10)
    if not input_sol_FLAG:
        stochastic_init = np.zeros((n_stochastic, n_shooting + 1))
        curent_index = 0
        stochastic_init[:n_muscles * (n_q + n_qdot), :] = 0.01  # K
        curent_index += n_muscles * (n_q + n_qdot)
        stochastic_init[curent_index: curent_index + n_q + n_qdot, :] = 0.01  # ee_ref
        # stochastic_init[curent_index : curent_index + n_q+n_qdot, 0] = np.array([ee_initial_position[0], ee_initial_position[1], 0, 0])  # ee_ref
        # stochastic_init[curent_index : curent_index + n_q+n_qdot, 1] = np.array([ee_final_position[0], ee_final_position[1], 0, 0])
        curent_index += n_q + n_qdot
        stochastic_init[curent_index: curent_index + n_states * n_states, :] = 0.01  # M
    else:
        stochastic_init = stochastic_variables_sol
    s_init = InitialGuess(stochastic_init, interpolation=InterpolationType.EACH_FRAME)

    s_bounds = BoundsList()
    stochastic_min = np.ones((n_stochastic, 3)) * -cas.inf
    stochastic_max = np.ones((n_stochastic, 3)) * cas.inf
    s_bounds.add(bounds=Bounds(stochastic_min, stochastic_max))
    # TODO: we should probably change the name stochastic_variables -> helper_variables ?

    return OptimalControlProgram(
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
        problem_type=OcpType.SOCP_EXPLICIT,  # TODO: seems weird for me to do StochasticOPtim... (comme mhe)
        update_value_function=lambda nlp, node_index: get_p_mat(nlp, node_index, force_field_magnitude=force_field_magnitude),
    )

def main():

    wM_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024

    RUN_OPTIM_FLAG = True  # False
    PLOT_SOL_FLAG = False  # True
    VIZUALIZE_SOL_FLAG = False  # True

    biorbd_model_path = "models/LeuvenArmModel.bioMod"

    ee_initial_position = np.array([0.0, 0.2742])  # Directly from Tom's version
    ee_final_position = np.array([9.359873986980460e-12, 0.527332023564034])  # Directly from Tom's version

    # --- Prepare the ocp --- #
    # dt = 0.01
    # final_time = 0.8
    # n_shooting = int(final_time/dt) + 1
    # final_time += dt
    n_shooting = 4
    final_time = 0.8

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver('mumps')
    # solver.set_linear_solver('ma57')
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    # solver.set_maximum_iterations(10000)
    solver.set_maximum_iterations(4)
    solver.set_hessian_approximation('limited-memory')
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_nlp_scaling_method('none')

    problem_type = "CIRCLE"
    force_field_magnitude = 0
    socp = prepare_socp(biorbd_model_path=biorbd_model_path,
                        final_time=final_time,
                        n_shooting=n_shooting,
                        ee_final_position=ee_final_position,
                        problem_type=problem_type,
                        force_field_magnitude=force_field_magnitude)

    if RUN_OPTIM_FLAG:
        sol_socp = socp.solve(solver)
        print('ici')
        # iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
        #    0  5.2443422e-01 2.05e+03 1.19e+00   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
        #    1  5.2686175e-01 1.85e+03 1.41e+02  -4.9 3.77e+01    -  4.34e-03 9.73e-02h  1
        #    2  2.1143195e+02 2.64e+03 1.07e+05  -0.7 2.45e+00    -  6.04e-02 1.00e+00h  1
        #    3  7.4999941e+01 1.38e+03 2.24e+04  -0.3 1.14e+00    -  9.93e-01 4.78e-01f  1
        #    4  4.4754216e+01 9.68e+02 5.42e+04   0.0 7.98e-01    -  1.00e+00 2.97e-01f  1

        # q_sol = sol_socp.states["q"]
        # qdot_sol = sol_socp.states["qdot"]
        # activations_sol = sol_socp.states["muscles"]
        # excitations_sol = sol_socp.controls["muscles"]
        # k_sol = sol_socp.stochastic_variables["k"]
        # ee_ref_sol = sol_socp.stochastic_variables["ee_ref"]
        # m_sol = sol_socp.stochastic_variables["m"]
        # # cov_sol = sol_socp.update_values["cov"]
        # stochastic_variables_sol = np.vstack((k_sol, ee_ref_sol, m_sol))  # , cov_sol))
        # data = {"q_sol": q_sol,
        #         "qdot_sol": qdot_sol,
        #         "activations_sol": activations_sol,
        #         "excitations_sol": excitations_sol,
        #         "k_sol": k_sol,
        #         "ee_ref_sol": ee_ref_sol,
        #         "m_sol": m_sol,
        #         # "cov_sol": cov_sol,
        #         "stochastic_variables_sol": stochastic_variables_sol}
        #
        # # --- Save the results --- #
        # with open(f"leuvenarm_muscle_driven_socp_{problem_type}_forcefield{force_field_magnitude}.pkl", "wb") as file:
        #     pickle.dump(data, file)
    else:
        with open(f"leuvenarm_muscle_driven_socp_{problem_type}_forcefield{force_field_magnitude}.pkl", "rb") as file:
            data = pickle.load(file)
        q_sol = data["q_sol"]
        qdot_sol = data["qdot_sol"]
        activations_sol = data["activations_sol"]
        excitations_sol = data["excitations_sol"]
        k_sol = data["k_sol"]
        ee_ref_sol = data["ee_ref_sol"]
        m_sol = data["m_sol"]
        # cov_sol = data["cov_sol"]
        stochastic_variables_sol = np.vstack((k_sol, ee_ref_sol, m_sol))

    # Save .mat files
    sio.savemat(f"leuvenarm_muscle_driven_socp_{problem_type}_forcefield{force_field_magnitude}.mat",
                        {"q_sol": q_sol,
                            "qdot_sol": qdot_sol,
                            "activations_sol": activations_sol,
                            "excitations_sol": excitations_sol,
                            "k_sol": k_sol,
                            "ee_ref_sol": ee_ref_sol,
                            "m_sol": m_sol,
                            "stochastic_variables_sol": stochastic_variables_sol})

    if VIZUALIZE_SOL_FLAG:
        import bioviz
        b = bioviz.Viz(model_path=biorbd_model_path)
        b.load_movement(q_sol[:, :-1])
        b.exec()


    # --- Plot the results --- #
    if PLOT_SOL_FLAG:
        Q_sym = cas.MX.sym('Q', 2, 1)
        Qdot_sym = cas.MX.sym('Qdot', 2, 1)
        hand_pos_fcn = cas.Function("hand_pos", [Q_sym], [end_effector_position(Q_sym)])
        hand_vel_fcn = cas.Function("hand_vel", [Q_sym, Qdot_sym], [end_effector_velocity(Q_sym, Qdot_sym)])

        states = socp.nlp[0].states.cx_start
        controls = socp.nlp[0].controls.cx_start
        parameters = socp.nlp[0].parameters.cx_start
        stochastic_variables = socp.nlp[0].stochastic_variables.cx_start
        nlp = socp.nlp[0]
        wM_sym = cas.MX.sym('wM', 2, 1)
        wPq_sym = cas.MX.sym('wPq', 2, 1)
        wPqdot_sym = cas.MX.sym('wPqdot', 2, 1)
        out = stochastic_forward_dynamics(states, controls, parameters, stochastic_variables, nlp, wM_sym, wPq_sym, wPqdot_sym, force_field_magnitude=force_field_magnitude, with_gains=True)
        dyn_fun = cas.Function("dyn_fun", [states, controls, parameters, stochastic_variables, wM_sym, wPq_sym, wPqdot_sym], [out.dxdt])

        fig, axs = plt.subplots(3, 2)
        n_simulations = 30
        q_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        qdot_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        mus_activation_simulated = np.zeros((n_simulations, 6, n_shooting + 1))
        hand_pos_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        hand_vel_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        for i_simulation in range(n_simulations):
            wM = np.random.normal(0, wM_std, (2, n_shooting + 1))
            wPq = np.random.normal(0, wPq_std, (2, n_shooting + 1))
            wPqdot = np.random.normal(0, wPqdot_std, (2, n_shooting + 1))
            q_simulated[i_simulation, :, 0] = q_sol[:, 0]
            qdot_simulated[i_simulation, :, 0] = qdot_sol[:, 0]
            mus_activation_simulated[i_simulation, :, 0] = activations_sol[:, 0]
            for i_node in range(n_shooting):
                x_prev = cas.vertcat(q_simulated[i_simulation, :, i_node], qdot_simulated[i_simulation, :, i_node], mus_activation_simulated[i_simulation, :, i_node])
                hand_pos_simulated[i_simulation, :, i_node] = np.reshape(hand_pos_fcn(x_prev[:2])[:2], (2,))
                hand_vel_simulated[i_simulation, :, i_node] = np.reshape(hand_vel_fcn(x_prev[:2], x_prev[2:4])[:2], (2,))
                u = excitations_sol[:, i_node]
                s = stochastic_variables_sol[:, i_node]
                k1 = dyn_fun(x_prev, u, [], s, wM[:, i_node], wPq[:, i_node], wPqdot[:, i_node])
                x_next = x_prev + dt * dyn_fun(x_prev + dt / 2 * k1, u, [], s, wM[:, i_node], wPq[:, i_node], wPqdot[:, i_node])
                q_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[:2], (2, ))
                qdot_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[2:4], (2, ))
                mus_activation_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[4:], (6, ))
            hand_pos_simulated[i_simulation, :, i_node + 1] = np.reshape(hand_pos_fcn(x_next[:2])[:2], (2,))
            hand_vel_simulated[i_simulation, :, i_node + 1] = np.reshape(hand_vel_fcn(x_next[:2], x_next[2:4])[:2], (2, ))
            axs[0, 0].plot(hand_pos_simulated[i_simulation, 0, :], hand_pos_simulated[i_simulation, 1, :], color="tab:red")
            axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 0, :], color="k")
            axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 1, :], color="k")
            axs[0, 1].plot(hand_vel_simulated[i_simulation, 0, :], hand_vel_simulated[i_simulation, 1, :], color="tab:red")
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

    # TODO: integrate to see the error they commit with the trapezoidal


# --- Define constants to specify the model  --- #
# (To simplify the problem, relationships are used instead of "real" muscles)
# dM_coefficients = np.array([[0, 0, 0.0100, 0.0300, -0.0110, 1.9000],
#                             [0, 0, 0.0100, -0.0190, 0, 0.0100],
#                             [0.0400, -0.0080, 1.9000, 0, 0, 0.0100],
#                             [-0.0420, 0, 0.0100, 0, 0, 0.0100],
#                             [0.0300, -0.0110, 1.9000, 0.0320, -0.0100, 1.9000],
#                             [-0.0390, 0, 0.0100, -0.0220, 0, 0.0100]])

# LMT_coefficients = np.array([[1.1000, -5.206336195535022],
#                              [0.8000, -7.538918356984516],
#                              [1.2000, -3.938098437958920],
#                              [0.7000, -3.031522725559912],
#                              [1.1000, -2.522778221157014],
#                              [0.8500, -1.826368199415192]])
# vMtilde_max = np.ones((6, 1)) * 10
# Fiso = np.array([572.4000, 445.2000, 699.6000, 381.6000, 159.0000, 318.0000])
# Faparam = np.array([0.814483478343008, 1.055033428970575, 0.162384573599574, 0.063303448465465, 0.433004984392647, 0.716775413397760, -0.029947116970696, 0.200356847296188])
# Fvparam = np.array([-0.318323436899127, -8.149156043475250, -0.374121508647863, 0.885644059915004])
# Fpparam = np.array([-0.995172050006169, 53.598150033144236])
# muscleDampingCoefficient = np.ones((6, 1)) * 0.01
# tau_coef = 0.1500

# m1 = 1.4
# m2 = 1
# l1 = 0.3
# l2 = 0.33
# lc1 = 0.11
# lc2 = 0.16
# I1 = 0.025
# I2 = 0.045

# TODO: How do we choose the values?
# wM_std = 0.05
# wPq_std = 3e-4
# wPqdot_std = 0.0024

# dt = 0.01
# wM_numerical = np.array([wM_std ** 2 / dt, wM_std ** 2 / dt])
# wPq_numerical = np.array([wPq_std ** 2 / dt, wPq_std ** 2 / dt])
# wPqdot_numerical = np.array([wPqdot_std ** 2 / dt, wPqdot_std ** 2 / dt])

if __name__ == "__main__":
    main()

