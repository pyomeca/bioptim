"""
This file contains the functions that are common for multiple stochastic examples.
"""

import casadi as cas
import numpy as np
from bioptim import StochasticBioModel, DynamicsFunctions


def dynamics_torque_driven_with_feedbacks(states, controls, parameters, stochastic_variables, nlp, with_noise):
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    tau_feedback = 0
    motor_noise = 0
    if with_noise:
        ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        k_matrix = StochasticBioModel.reshape_sym_to_matrix(k, nlp.model.matrix_shape_k)

        motor_noise = nlp.model.motor_noise_sym
        sensory_noise = nlp.model.sensory_noise_sym
        end_effector = nlp.model.sensory_reference(states, controls, parameters, stochastic_variables, nlp)
        tau_feedback = get_excitation_with_feedback(k_matrix, end_effector, ref, sensory_noise)

    tau_force_field = get_force_field(q, nlp.model.force_field_magnitude)
    torques_computed = tau + tau_feedback + motor_noise + tau_force_field

    mass_matrix = nlp.model.mass_matrix(q)
    non_linear_effects = nlp.model.non_linear_effects(q, qdot)

    return cas.inv(mass_matrix) @ (torques_computed - non_linear_effects - nlp.model.friction_coefficients @ qdot)


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

def integrator(model, polynomial_degree, n_shooting, duration, states, controls, stochastic_variables):

    h = duration / n_shooting
    method = "legendre"

    # Coefficients of the collocation equation
    _c = type(model.motor_noise_sym).zeros((polynomial_degree + 1, polynomial_degree + 1))

    # Coefficients of the continuity equation
    _d = type(model.motor_noise_sym).zeros(polynomial_degree + 1)

    # Choose collocation points
    step_time = [0] + cas.collocation_points(polynomial_degree, method)

    # Dimensionless time inside one control interval
    time_control_interval = type(model.motor_noise_sym).sym("time_control_interval")

    # For all collocation points
    for j in range(polynomial_degree + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        _l = 1
        for r in range(polynomial_degree + 1):
            if r != j:
                _l *= (time_control_interval - step_time[r]) / (step_time[j] - step_time[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        if method == "radau":
            _d[j] = 1 if j == polynomial_degree else 0
        else:
            lfcn = cas.Function("lfcn", [time_control_interval], [_l])
            _d[j] = lfcn(1.0)

        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        _l = 1
        for r in range(polynomial_degree + 1):
            if r != j:
                _l *= (time_control_interval - step_time[r]) / (step_time[j] - step_time[r])

        # Evaluate the time derivative of the polynomial at all collocation points to get
        # the coefficients of the continuity equation
        tfcn = cas.Function("tfcn", [time_control_interval], [cas.tangent(_l, time_control_interval)])
        for r in range(polynomial_degree + 1):
            _c[j, r] = tfcn(step_time[r])

    # Total number of variables for one finite element
    states_end = _d[0] * states[:, 0]
    defects = []
    for j in range(1, polynomial_degree + 1):
        # Expression for the state derivative at the collocation point
        xp_j = 0
        for r in range(polynomial_degree + 1):
            xp_j += _c[r, j] * states[:, r]

        f_j = model.dynamics_numerical(
            states=states[:, j],
            controls=controls,  # Piecewise constant control
            stochastic_variables=stochastic_variables,
            with_noise=True,
        )

        defects.append(h * f_j - xp_j)

        # Add contribution to the end state
        states_end += _d[j] * states[:, j]

    # Concatenate constraints
    defects = cas.vertcat(*defects)

    return states_end, defects

def get_m_init(model,
               n_stochastic,
               n_shooting,
               duration,
               polynomial_degree,
               q_last,
               qdot_last,
               tau_last):
    """
    M = -dF_dz @ inv(dG_dz)
    """

    n_q = model.nb_q
    n_joints = model.nb_u

    x_q_joints = type(model.motor_noise_sym).sym("x_q_joints", n_joints, 1)
    x_qdot_joints = type(model.motor_noise_sym).sym("x_qdot_joints", n_joints, 1)
    z_q_joints = type(model.motor_noise_sym).sym("z_q_joints", n_joints, polynomial_degree)
    z_qdot_joints = type(model.motor_noise_sym).sym("z_qdot_joints", n_joints, polynomial_degree)
    controls_sym = type(model.motor_noise_sym).sym("controls", n_q, 1)
    stochastic_variables_sym = type(model.motor_noise_sym).sym("stochastic_variables", n_stochastic, 1)

    states_full = cas.vertcat(
        cas.horzcat(x_q_joints, z_q_joints),
        cas.horzcat(x_qdot_joints, z_qdot_joints),
    )

    states_end, defects = integrator(model, polynomial_degree, n_shooting, duration, states_full, controls_sym, stochastic_variables_sym)
    initial_polynomial_evaluation = cas.vertcat(x_q_joints, x_qdot_joints)
    defects = cas.vertcat(initial_polynomial_evaluation, defects)

    df_dz = cas.horzcat(
        cas.jacobian(states_end, x_q_joints),
        cas.jacobian(states_end, z_q_joints),
        cas.jacobian(states_end, x_qdot_joints),
        cas.jacobian(states_end, z_qdot_joints),
    )

    dg_dz = cas.horzcat(
        cas.jacobian(defects, x_q_joints),
        cas.jacobian(defects, z_q_joints),
        cas.jacobian(defects, x_qdot_joints),
        cas.jacobian(defects, z_qdot_joints),
    )

    df_dz_fun = cas.Function(
        "df_dz",
        [
            x_q_joints,
            x_qdot_joints,
            z_q_joints,
            z_qdot_joints,
            controls_sym,
            stochastic_variables_sym,
        ],
        [df_dz],
    )
    dg_dz_fun = cas.Function(
        "dg_dz",
        [
            x_q_joints,
            x_qdot_joints,
            z_q_joints,
            z_qdot_joints,
            controls_sym,
            stochastic_variables_sym,
        ],
        [dg_dz],
    )


    m_last = np.zeros((2 * n_joints * 2 * n_joints * (polynomial_degree+1), n_shooting + 1))
    for i in range(n_shooting+1):
        index_this_time = [i * polynomial_degree + j for j in range(polynomial_degree+1)]
        df_dz_evaluated = df_dz_fun(
            q_last[:, index_this_time[0]],
            qdot_last[:, index_this_time[0]],
            q_last[:, index_this_time[1:]],
            qdot_last[:, index_this_time[1:]],
            tau_last[:, i],
            np.vstack((np.zeros((2 * n_joints * 2 * n_joints * (polynomial_degree+1), 1)),  # M
                       np.zeros((2 * n_joints * 2 * n_joints, 1)))),  # cov
        )
        dg_dz_evaluated = dg_dz_fun(
            q_last[:, index_this_time[0]],
            qdot_last[:, index_this_time[0]],
            q_last[:, index_this_time[1:]],
            qdot_last[:, index_this_time[1:]],
            tau_last[:, i],
            np.vstack((np.zeros((2 * n_joints * 2 * n_joints * (polynomial_degree+1), 1)),
                       np.zeros((2 * n_joints * 2 * n_joints, 1)))),
        )

        m_this_time = -df_dz_evaluated @ np.linalg.inv(dg_dz_evaluated)

        shape_0, shape_1 = m_this_time.shape[0], m_this_time.shape[1]
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                m_last[shape_0 * s1 + s0, i] = m_this_time[s0, s1]

    return m_last

def get_cov_init(model,
                 n_shooting,
                 n_stochastic,
                 polynomial_degree,
                 duration,
                 q_last,
                 qdot_last,
                 tau_last,
                 m_last,
                 cov_init,
                 motor_noise_magnitude):
    """
    P_k+1 = M_k @ (dG_dx @ P_k @ dG_dx.T + dG_dw @ sigma_w @ dG_dw.T) @ M_k.T
    """

    n_q = model.nb_q
    n_joints = model.nb_u

    x_q_joints = type(model.motor_noise_sym).sym("x_q_joints", n_joints, 1)
    x_qdot_joints = type(model.motor_noise_sym).sym("x_qdot_joints", n_joints, 1)
    z_q_joints = type(model.motor_noise_sym).sym("z_q_joints", n_joints, polynomial_degree)
    z_qdot_joints = type(model.motor_noise_sym).sym("z_qdot_joints", n_joints, polynomial_degree)
    controls_sym = type(model.motor_noise_sym).sym("controls", n_q, 1)
    stochastic_variables_sym = type(model.motor_noise_sym).sym("stochastic_variables", n_stochastic, 1)

    states_full = cas.vertcat(
        cas.horzcat(x_q_joints, z_q_joints),
        cas.horzcat(x_qdot_joints, z_qdot_joints),
    )

    states_end, defects = integrator(model, polynomial_degree, n_shooting, duration, states_full, controls_sym,
                                     stochastic_variables_sym)
    initial_polynomial_evaluation = cas.vertcat(x_q_joints, x_qdot_joints)
    defects = cas.vertcat(initial_polynomial_evaluation, defects)

    dg_dx = cas.horzcat(
        cas.jacobian(defects, x_q_joints),
        cas.jacobian(defects, x_qdot_joints),
    )

    dg_dw = cas.jacobian(defects, model.motor_noise_sym)

    dg_dx_fun = cas.Function(
        "dg_dx",
        [
            x_q_joints,
            x_qdot_joints,
            z_q_joints,
            z_qdot_joints,
            controls_sym,
            stochastic_variables_sym,
            model.motor_noise_sym,
        ],
        [dg_dx],
    )
    dg_dw_fun = cas.Function(
        "dg_dw",
        [
            x_q_joints,
            x_qdot_joints,
            z_q_joints,
            z_qdot_joints,
            controls_sym,
            stochastic_variables_sym,
            model.motor_noise_sym,
        ],
        [dg_dw],
    )

    sigma_w_dm = motor_noise_magnitude * cas.DM_eye(motor_noise_magnitude.shape[0])
    cov_last = np.zeros((2 * n_joints * 2 * n_joints, n_shooting + 1))

    cov_last[:, 0] = cov_init[:, 0]
    for i in range(n_shooting):
        index_this_time = [i * polynomial_degree + j for j in range(polynomial_degree+1)]
        dg_dx_evaluated = dg_dx_fun(
            q_last[:, index_this_time[0]],
            qdot_last[:, index_this_time[0]],
            q_last[:, index_this_time[1:]],
            qdot_last[:, index_this_time[1:]],
            tau_last[:, i],
            np.vstack((m_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_joints * 2 * n_joints, 1)))),  # cov
            motor_noise_magnitude,
        )
        dg_dw_evaluated = dg_dw_fun(
            q_last[:, index_this_time[0]],
            qdot_last[:, index_this_time[0]],
            q_last[:, index_this_time[1:]],
            qdot_last[:, index_this_time[1:]],
            tau_last[:, i],
            np.vstack((m_last[:, i].reshape((-1, 1)),
                       np.zeros((2 * n_joints * 2 * n_joints, 1)))),
            motor_noise_magnitude,
        )

        m_matrix = np.zeros(model.matrix_shape_m)
        shape_0, shape_1 = model.matrix_shape_m
        for s0 in range(shape_1):
            for s1 in range(shape_0):
                m_matrix[s1, s0] = m_last[s0 * shape_0 + s1, i]

        cov_matrix = np.zeros((2*n_joints, 2*n_joints))
        for s0 in range(2*n_joints):
            for s1 in range(2*n_joints):
                cov_matrix[s1, s0] = cov_last[s0 * 2*n_joints + s1, i]

        cov_this_time = (
                m_matrix @ (dg_dx_evaluated @ cov_matrix @ dg_dx_evaluated.T + dg_dw_evaluated @ sigma_w_dm @ dg_dw_evaluated.T) @ m_matrix.T)
        for s0 in range(2*n_joints):
            for s1 in range(2*n_joints):
                cov_last[2*n_joints * s1 + s0, i+1] = cov_this_time[s0, s1]
    return cov_last


def test_matrix_semi_definite_positiveness(var):
    """
    This function tests if a matrix var is positive semi-definite.

    Parameters
    ----------
    var: np.ndarray | DM
        The matrix to test (in the form of a vector containing the elements of the matrix)
    """
    is_ok = True
    shape_0 = int(np.sqrt(var.shape[0]))
    matrix = np.zeros((shape_0, shape_0))
    for s0 in range(shape_0):
        for s1 in range(shape_0):
            matrix[s1, s0] = var[s0 * shape_0 + s1]

    # Symmetry
    symmetry_elements = matrix - matrix.T

    if np.sum(np.abs(symmetry_elements) > 1e-4) != 0:
        is_ok = False

    A = cas.SX.sym("A", shape_0, shape_0)
    D = cas.ldl(A)[0]  # Only guaranteed to work by casadi for positive definite matrix.
    func = cas.Function("diagonal_terms", [A], [D])

    matrix = StochasticBioModel.reshape_to_matrix(matrix, (shape_0, shape_0))
    diagonal_terms = func(matrix)

    if np.sum(diagonal_terms < 0) != 0:
        is_ok = False
        
    return is_ok


def test_robustified_constraint_value(model, q, qdot, cov_num):
    """
    This function tests if the robustified constraint contains NaNs.
    """
    is_ok = True
    p_x = cas.MX.sym("p_x", 1, 1)
    p_y = cas.MX.sym("p_x", 1, 1)
    v_x = cas.MX.sym("v_x", 1, 1)
    v_y = cas.MX.sym("v_x", 1, 1)
    cov = cas.MX.sym("cov", 4, 4)

    non_robust_constraint = cas.MX()
    robust_component = cas.MX()
    dh_dx_all = cas.MX()
    cov_all = cas.MX()
    sqrt_all = cas.MX()
    for i in range(2):
        h = (((p_x - model.super_ellipse_center_x[i])/ model.super_ellipse_a[i]) ** model.super_ellipse_n[i]
                + ((p_y - model.super_ellipse_center_y[i])/ model.super_ellipse_b[i]) ** model.super_ellipse_n[i]
                - 1
        )

        non_robust_constraint = cas.vertcat(non_robust_constraint, h)

        gamma = 1
        dh_dx = cas.jacobian(h, cas.vertcat(p_x, p_y, v_x, v_y))
        safe_guard = gamma * cas.sqrt(dh_dx @ cov @ dh_dx.T)
        robust_component = cas.vertcat(robust_component, safe_guard)

        dh_dx_all = cas.vertcat(dh_dx_all, dh_dx)
        cov_all = cas.vertcat(cov_all, cov)
        sqrt_all = cas.vertcat(sqrt_all, dh_dx @ cov @ dh_dx.T)

    func = cas.Function("out", [p_x, p_y, v_x, v_y, cov], [non_robust_constraint, robust_component, dh_dx_all, cov_all, sqrt_all])

    out_num = []
    for j in range(cov_num.shape[1]):
        p_x_value = q[0, j*(model.polynomial_degree + 1)]
        p_y_value = q[1, j*(model.polynomial_degree + 1)]
        v_x_value = qdot[0, j*(model.polynomial_degree + 1)]
        v_y_value = qdot[1, j*(model.polynomial_degree + 1)]
        cov_value = np.zeros((4, 4))
        for s0 in range(4):
            for s1 in range(4):
                cov_value[s1, s0] = cov_num[s0 * 4 + s1, j]
        out_this_time = func(p_x_value, p_y_value, v_x_value, v_y_value, cov_value)
        out_num += [out_this_time]

        if np.sum(np.isnan(out_this_time[1])) != 0:
            print("h : ", out_this_time[0])
            print("sqrt(dh_dx @ cov @ dh_dx) : ", out_this_time[1])
            print("dh_dx : ", out_this_time[2])
            print("cov : ", out_this_time[3])
            print("dh_dx @ cov @ dh_dx : ", out_this_time[4])
            is_ok = False
            break

    return is_ok

def test_eigen_values(var):
    is_ok = True
    shape_0 = int(np.sqrt(var.shape[0]))
    matrix = np.zeros((shape_0, shape_0))
    for s0 in range(shape_0):
        for s1 in range(shape_0):
            matrix[s1, s0] = var[s0 * shape_0 + s1]

    vals, vecs = np.linalg.eigh(matrix)
    if np.sum(vals < 0) != 0:
        is_ok = False
    return is_ok