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

import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    OptimalControlProgram,
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    OdeSolver,
    Solver,
    BiorbdModel,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    DynamicsList,
    VariableScalingList,
    BoundsList,
    InterpolationType,
    OcpType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
    DefectType,
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

    global dM_coefficients, LMT_coefficients, vMtilde_max, Fiso, Faparam, Fvparam, Fpparam, muscleDampingCoefficient, tau_coef

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
    FMtilde1 = b11 * cas.exp(-0.5 * num1 ** 2. / den1 ** 2)

    num2 = lMtilde - b22
    den2 = b32 + b42 * lMtilde
    FMtilde2 = b12 * cas.exp(-0.5 * num2 ** 2 / den2 ** 2)

    FMltilde = FMtilde1 + FMtilde2 + FMtilde3

    e1 = Fvparam[0]
    e2 = Fvparam[1]
    e3 = Fvparam[2]
    e4 = Fvparam[3]

    FMvtilde = e1 * cas.log(
        (e2 * vMtilde_normalizedToMaxVelocity + e3) + cas.sqrt(
            (e2 * vMtilde_normalizedToMaxVelocity + e3) ** 2 + 1)) + e4

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
    global dM_coefficients, LMT_coefficients, vMtilde_max, Fiso, Faparam, Fvparam, Fpparam, muscleDampingCoefficient, tau_coef

    a_shoulder = dM_coefficients[:, 0]
    b_shoulder = dM_coefficients[:, 1]
    c_shoulder = dM_coefficients[:, 2]
    a_elbow = dM_coefficients[:, 3]
    b_elbow = dM_coefficients[:, 4]
    c_elbow = dM_coefficients[:, 5]
    theta_shoulder = q[0]
    theta_elbow = q[1]

    dM_matrix = cas.horzcat(a_shoulder + b_shoulder * cas.cos(c_shoulder * theta_shoulder),
                            a_elbow + b_elbow * cas.cos(c_elbow * theta_elbow)).T
    tau = dM_matrix @ Fm
    return tau

def get_muscle_torque(q, qdot, mus_activations):
    Fa, Fp = get_muscle_force(q, qdot)
    Fm = mus_activations * Fa + Fp
    muscles_tau = torque_force_relationship(Fm, q)
    return muscles_tau

def get_force_field(q):
    force_field_magnitude = 200
    l1 = 0.3
    l2 = 0.33
    F_forceField = force_field_magnitude * (l1 * cas.cos(q[0]) + l2 * cas.cos(q[0] + q[1]))
    # Position of the hand marker ?
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
) -> DynamicsEvaluation:

    global tau_coef, m1, m2, l1, l2, lc1, lc2, I1, I2

    import biorbd_casadi as biorbd  # Pariterre: using controller.model.forward_dynamics gives free variables error ?
    model = biorbd.Model(
        "/home/charbie/Documents/Programmation/BiorbdOptim/bioptim/examples/stochastic_optimal_control/models/LeuvenArmModel.bioMod")  # controller.get_nlp.model.model

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    mus_activations = DynamicsFunctions.get(nlp.states["muscles"], states)
    # residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)

    ee_ref = DynamicsFunctions.get(nlp.stochastic_variables["ee_ref"], stochastic_variables)
    k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
    K_matrix = cas.MX(6, 4)
    i = 0
    for s0 in range(6):
        for s1 in range(4):
            K_matrix[s0, s1] = k[i]
            i += 1

    hand_pos = model.marker(q, 2).to_mx()[:2]
    hand_vel = model.markerVelocity(q, qdot, 2).to_mx()[:2]
    ee = cas.vertcat(hand_pos, hand_vel)
    mus_excitations_fb = mus_excitations + get_excitation_with_feedback(K_matrix, ee, ee_ref, wPq, wPqdot)

    muscles_tau = get_muscle_torque(q, qdot, mus_activations)

    tau_force_field = get_force_field(q)


    torques_computed = muscles_tau + tau_force_field + wM  # + residual_tau
    dq_computed = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    dactivations_computed = (mus_excitations_fb - mus_activations) / tau_coef

    friction = np.array([[0.05, 0.025], [0.025, 0.05]])
    # mass_matrix = model.massMatrix(q).to_mx()
    # nleffects = model.NonLinearEffect(q, qdot).to_mx()

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

    dqdot_derivative = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot)

    return DynamicsEvaluation(dxdt=cas.vertcat(dq_computed, dqdot_derivative, dactivations_computed), defects=None)

def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram, wM, wPq, wPqdot):

    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)
    # ConfigureProblem.configure_tau(ocp, nlp, False, True)  # Residual tau
    # TODO: eux ils n'utilsent pas de torques résiduels, si les muscles sont plannaire, ce devrit être good
    ConfigureProblem.configure_muscles(ocp, nlp, True, True)  # Muscles activation (states) + excitation (control)

    # Stochastic variables
    ConfigureProblem.configure_k(ocp, nlp)
    ConfigureProblem.configure_ee_ref(ocp, nlp)
    ConfigureProblem.configure_m(ocp, nlp)
    ConfigureProblem.configure_cov(ocp, nlp)
    # ?
    # ConfigureProblem.configure_c(ocp, nlp)
    # ConfigureProblem.configure_a(ocp, nlp)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, stochastic_forward_dynamics, wM=wM, wPq=wPq, wPqdot=wPqdot, expand=False)

def minimize_uncertainty(controller: PenaltyController, key: str) -> cas.MX:
    """
    Minimize the uncertainty (covariance matrix) of the states.
    """
    P_matrix = controller.restore_matrix_form_from_vector(controller.stochastic_variables, controller.states.cx.shape[0], controller.states.cx.shape[0], Node.START, "cov")
    P_partial = P_matrix[controller.states[key].index, controller.states[key].index]
    return cas.trace(P_partial)

def ee_equals_ee_ref(controller: PenaltyController) -> cas.MX:
    """
    # TODO: why?
    """

    import biorbd_casadi as biorbd  # Pariterre: using controller.model.forward_dynamics gives free variables error ?
    model = biorbd.Model(
        "/home/charbie/Documents/Programmation/BiorbdOptim/bioptim/examples/stochastic_optimal_control/models/LeuvenArmModel.bioMod")  # controller.get_nlp.model.model

    q = controller.states["q"].cx
    qdot = controller.states["qdot"].cx
    ee_ref = controller.stochastic_variables["ee_ref"].cx

    hand_pos = model.marker(q, 2).to_mx()[:2]
    hand_vel = model.markerVelocity(q, qdot, 2).to_mx()[:2]
    ee = cas.vertcat(hand_pos, hand_vel)

    # val = controller.mx_to_cx("ee_equals_ee_ref", ee - ee_ref, q, qdot, ee_ref)
    return ee - ee_ref

def reach_target_consistantly(controller: PenaltyController) -> cas.MX:
    """
    Constraint the hand to reach the target consistently.
    """

    import biorbd_casadi as biorbd  # Pariterre: using controller.model.forward_dynamics gives free variables error ?
    model = biorbd.Model(
        "/home/charbie/Documents/Programmation/BiorbdOptim/bioptim/examples/stochastic_optimal_control/models/LeuvenArmModel.bioMod")  # controller.get_nlp.model.model

    Q = cas.MX.sym("q_sym", controller.states["q"].cx.shape[0])
    Qdot = cas.MX.sym("qdot_sym", controller.states["qdot"].cx.shape[0])
    P_matrix = controller.restore_matrix_form_from_vector(controller.stochastic_variables, controller.states.cx.shape[0], controller.states.cx.shape[0], Node.START, "cov")

    hand_pos = model.marker(Q, 2).to_mx()[:2]
    hand_vel = model.markerVelocity(Qdot, Qdot, 2).to_mx()[:2]

    jac_marker_q = cas.jacobian(hand_pos, Q)
    jac_marker_qdot = cas.jacobian(hand_vel, cas.vertcat(Q, Qdot))

    P_matrix_q = P_matrix[:2, :2]
    P_matrix_qdot = P_matrix[:4, :4]

    pos_constraint = jac_marker_q @ P_matrix_q @ jac_marker_q.T
    vel_constraint = jac_marker_qdot @ P_matrix_qdot @ jac_marker_qdot.T

    out = cas.vertcat(pos_constraint[0, 0], pos_constraint[1, 1], vel_constraint[0, 0], vel_constraint[1, 1])

    fun = cas.Function("reach_target_consistantly", [Q, Qdot, controller.stochastic_variables.cx], [out])
    val = fun(controller.states["q"].cx_start, controller.states["qdot"].cx_start, controller.stochastic_variables.cx_start)
    # Since the stochastic variables are defined with ns+1, the cx_start actually refers to the last node (when using node=Node.END)

    return val

def expected_feedback_effort(controller: PenaltyController, final_time: float) -> cas.MX:

    # Constants TODO: remove fom here
    # TODO: How do we choose?
    global wM_std, wPq_std, wPqdot_std

    dt = final_time / controller.ns
    wM_magnitude = cas.DM(np.array([wM_std ** 2 / dt, wM_std ** 2 / dt]))
    wPq_magnitude = cas.DM(np.array([wPq_std ** 2 / dt, wPq_std ** 2 / dt]))
    wPqdot_magnitude = cas.DM(np.array([wPqdot_std ** 2 / dt, wPqdot_std ** 2 / dt]))
    sensory_noise = cas.vertcat(wPq_magnitude, wPqdot_magnitude)
    sensory_noise_matrix = sensory_noise * cas.MX_eye(4)

    # Get the symbolic variables
    ee_ref = controller.stochastic_variables["ee_ref"].cx_start
    P_matrix = controller.restore_matrix_form_from_vector(controller.stochastic_variables,
                                                          controller.states.cx.shape[0],
                                                          controller.states.cx.shape[0],
                                                          Node.START,
                                                          "cov")
    K_matrix = controller.restore_matrix_form_from_vector(controller.stochastic_variables,
                                                          controller.states["muscles"].cx.shape[0],
                                                          controller.states["q"].cx.shape[0] + controller.states["qdot"].cx.shape[0],
                                                          Node.START,
                                                          "k")

    # Compute the expected effort
    hand_pos = controller.model.marker(controller.states["q"].cx_start, 2)[:2]
    hand_vel = controller.model.marker_velocities(controller.states["q"].cx_start, controller.states["qdot"].cx_start, 2)[:2]
    trace_k_sensor_k = cas.trace(K_matrix @ sensory_noise_matrix @ K_matrix.T)
    ee = cas.vertcat(hand_pos, hand_vel)
    e_fb = K_matrix @ ((ee - ee_ref) + sensory_noise)
    jac_e_fb_x = cas.jacobian(e_fb, controller.states.cx_start)
    trace_jac_p_jack = cas.trace(jac_e_fb_x @ P_matrix @ jac_e_fb_x.T)
    expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k
    f_expectedEffort_fb = cas.Function('f_expectedEffort_fb', [controller.states.cx_start, controller.stochastic_variables.cx_start], [expectedEffort_fb_mx])(controller.states.cx_start, controller.stochastic_variables.cx_start)
    return f_expectedEffort_fb

def prepare_socp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    n_threads: int = 12,
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
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    shoulder_pos_init = 0.3491
    shoulder_pos_final = 0.9599
    elbow_pos_init = 2.2459  # Optimized in Tom's version
    elbow_pos_final = 1.1594  # Optimized in Tom's version
    ee_final_position = np.array([0.55190516, -0.017223, 0])  # Computed from the final hand position

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1e3/2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles", weight=1e3/2)
    objective_functions.add(minimize_uncertainty,  custom_type=ObjectiveFcn.Mayer, node=Node.ALL, key="muscles", weight=1e3/2)
    objective_functions.add(expected_feedback_effort, custom_type=ObjectiveFcn.Lagrange, weight=1e3, final_time=final_time)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ee_equals_ee_ref, node=Node.ALL)
    # No acceleration at the first and last nodes
    constraints.add(reach_target_consistantly, node=Node.END, min_bound=np.array([0, 0, 0, 0]), max_bound=np.array([1, 0.004**2, 0.05**2, 0.05**2]))
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.END, marker_index=2, target=ee_final_position)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(configure_stochastic_optimal_control_problem, dynamic_function=stochastic_forward_dynamics, wM=np.zeros((2, 1)), wPq=np.zeros((2, 1)), wPqdot=np.zeros((2, 1)), expand=False)

    n_states = bio_model.nb_q + bio_model.nb_qdot + bio_model.nb_muscles

    q_qdot_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    states_min = np.zeros((n_states, 3))
    states_max = np.zeros((n_states, 3))
    states_min[:bio_model.nb_q + bio_model.nb_qdot, :] = q_qdot_bounds.min
    states_max[:bio_model.nb_q + bio_model.nb_qdot, :] = q_qdot_bounds.max
    states_min[bio_model.nb_q + bio_model.nb_qdot:, :] = 0.01  # Activations
    states_max[bio_model.nb_q + bio_model.nb_qdot:, :] = 1  # Activations
    states_min[bio_model.nb_q + bio_model.nb_qdot:, :] = 0  # Initial activations
    states_max[bio_model.nb_q + bio_model.nb_qdot:, :] = 0  # Initial activations
    states_min[0:4, 0] = [shoulder_pos_init, elbow_pos_init, 0, 0]  # Initial position + velocities
    states_max[0:4, 0] = [shoulder_pos_init, elbow_pos_init, 0, 0]  # Initial position + velocities
    states_min[0:2, 1:3] = [0, 0]
    states_max[0:2, 1:3] = [np.pi, np.pi]
    states_min[2:4, 2] = [0, 0]  # Final velocities
    states_max[2:4, 2] = [0, 0]  # Final velocities

    x_bounds = BoundsList()
    x_bounds.add(bounds=Bounds(states_min, states_max))

    n_muscles = bio_model.nb_muscles
    # n_tau = bio_model.nb_tau
    # u_bounds = Bounds([-100] * n_tau + [0] * n_muscles, [100] * n_tau + [1] * n_muscles)
    u_bounds = Bounds([0.01] * n_muscles, [1] * n_muscles)

    # Initial guesses
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot

    states_init = np.zeros((n_states, 2))
    states_init[0, :] = [shoulder_pos_init, shoulder_pos_final]
    states_init[1, :] = [elbow_pos_init, elbow_pos_final]
    states_init[n_q + n_qdot:, :] = 0.01
    x_init = InitialGuess(states_init, interpolation=InterpolationType.LINEAR)

    u_init = InitialGuess([0.01] * n_muscles)

    # TODO: This should probably be done automatically, not defined by the user

    n_stochastic = n_muscles*(n_q + n_qdot) + n_q+n_qdot + n_states*n_states + n_states*n_states  # K(6x4) + ee_ref(4x1) + M(10x10) + P(10x10)
    # 216 ou 24
    stochastic_min = np.zeros((n_stochastic, 3))
    stochastic_max = np.zeros((n_stochastic, 3))
    stochastic_init = np.zeros((n_stochastic, 1))

    curent_index = 0
    stochastic_init[:n_muscles*(n_q + n_qdot), 0] = 0.01  # K
    stochastic_min[:n_muscles*(n_q + n_qdot), :] = -10
    stochastic_max[:n_muscles*(n_q + n_qdot), :] = 10

    curent_index += n_muscles*(n_q + n_qdot)
    stochastic_init[curent_index : curent_index + n_q+n_qdot, 0] = 0.01  # ee_ref
    stochastic_min[curent_index : curent_index + n_q+n_qdot, :] = -10
    stochastic_max[curent_index : curent_index + n_q+n_qdot, :] = 10

    curent_index += n_q+n_qdot
    stochastic_init[curent_index : curent_index + n_states*n_states, 0] = 0.01  # M
    stochastic_min[curent_index : curent_index + n_states*n_states, :] = 10
    stochastic_max[curent_index : curent_index + n_states*n_states, :] = 10
    # M at node ns+1 should not exist (my hope is that by constraining it IPOPT treats it as a constant)
    stochastic_min[curent_index : curent_index + n_states*n_states, 2] = 0.01
    stochastic_max[curent_index : curent_index + n_states*n_states, 2] = 0.01

    curent_index += n_states*n_states
    mat_p_init = np.eye(10) * np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-7, 1e-7])
    stochastic_init[curent_index:, 0] = mat_p_init.flatten()  # P
    stochastic_min[curent_index:, :] = 0
    stochastic_max[curent_index:, :] = 1
    stochastic_min[curent_index:, 0] = mat_p_init.flatten()
    stochastic_max[curent_index:, 0] = mat_p_init.flatten()

    s_init = InitialGuess(stochastic_init)

    s_bounds = BoundsList()
    # Didi not find it in the original code
    s_bounds.add(bounds=Bounds(stochastic_min, stochastic_max))

    # Variable scaling
    x_scaling = VariableScalingList()
    x_scaling.add("q", scaling=[1, 1])
    x_scaling.add("qdot", scaling=[1, 1])

    u_scaling = VariableScalingList()
    u_scaling.add("tau", scaling=[1, 1])
    u_scaling.add("muscles", scaling=[1, 1])

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
        constraints=constraints,
        # ode_solver=OdeSolver.RK2(n_integration_steps=1),
        ode_solver=OdeSolver.COLLOCATION(polynomial_degree=5, defects_type=DefectType.EXPLICIT),
        n_threads=1,  # n_threads,
        assume_phase_dynamics=True,
        problem_type=OcpType.socp,  # TODO: seems weird for me to do StochasticOPtim... (comme mhe)
    )

def main():

    global wM_std, wPq_std, wPqdot_std

    biorbd_model_path = "models/LeuvenArmModel.bioMod"
    # import bioviz
    # b = bioviz.Viz("models/LeuvenArmModel.bioMod")
    # b.exec()

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.8
    n_shooting = int(final_time/dt)

    # Solver parameters
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver('mumps')
    # solver.set_linear_solver('ma57')
    solver.set_hessian_approximation('limited-memory')
    solver.set_tol(1e-3)
    solver.set_dual_inf_tol(3e-4)
    solver.set_constr_viol_tol(1e-7)
    solver.set_maximum_iterations(1000)

    socp = prepare_socp(biorbd_model_path=biorbd_model_path, final_time=final_time, n_shooting=n_shooting)
    sol_socp = socp.solve(solver)

    q_sol = sol_socp.states["q"]
    qdot_sol = sol_socp.states["qdot"]
    activations_sol = sol_socp.states["muscles"]
    excitations_sol = sol_socp.controls["muscles"]
    k_sol = sol_socp.stochastic_variables["k"]
    ee_ref_sol = sol_socp.stochastic_variables["ee_ref"]
    m_sol = sol_socp.stochastic_variables["m"]
    cov_sol = sol_socp.stochastic_variables["cov"]
    parameters_sol = np.vstack((k_sol, ee_ref_sol, m_sol, cov_sol))
    data = {"q_sol": q_sol,
            "qdot_sol": qdot_sol,
            "activations_sol": activations_sol,
            "excitations_sol": excitations_sol,
            "k_sol": k_sol,
            "ee_ref_sol": ee_ref_sol,
            "m_sol": m_sol,
            "cov_sol": cov_sol}

    # --- Save the results --- #
    with open("leuvenarm_muscle_driven_socp.pkl", "wb") as file:
        pickle.dump(data, file)

    # --- Plot the results --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    Q_sym = cas.MX.sym('Q', 2, 1)
    Qdot_sym = cas.MX.sym('Qdot', 2, 1)
    hand_pos_fcn = cas.Function("hand_pos", [Q_sym], [biorbd_model.marker(Q_sym, 2).to_mx()])
    hand_vel_fcn = cas.Function("hand_vel", [Q_sym, Qdot_sym], [biorbd_model.markerVelocity(Q_sym, Qdot_sym, 2).to_mx()])

    states = socp.nlp[0].states.cx_start
    controls = socp.nlp[0].controls.cx_start
    parameters = socp.nlp[0].parameters.cx_start
    stochastic_variables = socp.nlp[0].stochastic_variables.cx_start
    nlp = socp.nlp[0]
    wM_sym = cas.MX.sym('wM', 2, 1)
    wPq_sym = cas.MX.sym('wPq', 2, 1)
    wPqdot_sym = cas.MX.sym('wPqdot', 2, 1)
    out = stochastic_forward_dynamics(states, controls, parameters, stochastic_variables, nlp, wM_sym, wPq_sym, wPqdot_sym)
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
            p = parameters_sol[:, i_node]
            k1 = dyn_fun(x_prev, u, [], p, wM[:, i_node], wPq[:, i_node], wPqdot[:, i_node])
            x_next = x_prev + dt * dyn_fun(x_prev + dt / 2 * k1, u, [], p, wM[:, i_node], wPq[:, i_node], wPqdot[:, i_node])
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
    axs[0, 0].set_xlabel("Y [m]")
    axs[0, 0].set_ylabel("X [m]")
    axs[0, 0].set_title("Hand position simulated")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Shoulder angle [rad]")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Elbow angle [rad]")
    axs[0, 1].set_xlabel("Y velocity [m/s]")
    axs[0, 1].set_ylabel("X velocity [m/s]")
    axs[0, 1].set_title("Hand velocity simulated")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Shoulder velocity [rad/s]")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Elbow velocity [rad/s]")
    plt.tight_layout()
    plt.savefig("simulated_results.png", dpi=300)
    plt.show()

    # TODO: integrate to see the error they commit with the trapezoidal

    # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)

    # --- If one is interested in checking the conditioning of the problem, they can uncomment the following line --- #
    # ocp.check_conditioning()

    # --- Solve the ocp --- #
    # sol.graphs()

    # --- Show the results in a bioviz animation --- #
    sol_socp.animate()


# --- Define constants to specify the model  --- #
# (To simplify the problem, relationships are used instead of "real" muscles)
dM_coefficients = np.array([[0, 0, 0.0100, 0.0300, -0.0110, 1.9000],
                            [0, 0, 0.0100, -0.0190, 0, 0.0100],
                            [0.0400, -0.0080, 1.9000, 0, 0, 0.0100],
                            [-0.0420, 0, 0.0100, 0, 0, 0.0100],
                            [0.0300, -0.0110, 1.9000, 0.0320, -0.0100, 1.9000],
                            [-0.0390, 0, 0.0100, -0.0220, 0, 0.0100]])

LMT_coefficients = np.array([[1.1000, -5.2063],
                             [0.8000, -7.5389],
                             [1.2000, -3.9381],
                             [0.7000, -3.0315],
                             [1.1000, -2.5228],
                             [0.8500, -1.8264]])
vMtilde_max = np.ones((6, 1)) * 10
Fiso = np.array([572.4000, 445.2000, 699.6000, 381.6000, 159.0000, 318.0000])
Faparam = np.array([0.8145, 1.0550, 0.1624, 0.0633, 0.4330, 0.7168, -0.0299, 0.2004])
Fvparam = np.array([-0.3183, -8.1492, -0.3741, 0.8856])
Fpparam = np.array([-0.9952, 53.5982])
muscleDampingCoefficient = np.ones((6, 1)) * 0.1
tau_coef = 0.1500

m1 = 1.4
m2 = 1
l1 = 0.3
l2 = 0.33
lc1 = 0.11
lc2 = 0.16
I1 = 0.025
I2 = 0.045

wM_std = 0.05
wPq_std = 3e-4
wPqdot_std = 0.0024

if __name__ == "__main__":
    main()

# TODO: Check expected feedback effort