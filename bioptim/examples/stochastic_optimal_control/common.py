"""
This file contains the functions that are common for multiple stochastic examples.
"""

import casadi as cas
import numpy as np
from bioptim import StochasticBioModel, DynamicsFunctions, SocpType


def dynamics_torque_driven_with_feedbacks(
    states, controls, parameters, stochastic_variables, nlp, with_noise
):
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    tau_feedback = 0
    motor_noise = 0
    if with_noise:
        ref = DynamicsFunctions.get(
            nlp.stochastic_variables["ref"], stochastic_variables
        )
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        k_matrix = StochasticBioModel.reshape_sym_to_matrix(k, nlp.model.matrix_shape_k)

        motor_noise = nlp.model.motor_noise_sym
        sensory_noise = nlp.model.sensory_noise_sym
        end_effector = nlp.model.sensory_reference(
            states, controls, parameters, stochastic_variables, nlp
        )
        tau_feedback = get_excitation_with_feedback(
            k_matrix, end_effector, ref, sensory_noise
        )

    tau_force_field = get_force_field(q, nlp.model.force_field_magnitude)
    torques_computed = tau + tau_feedback + motor_noise + tau_force_field

    mass_matrix = nlp.model.mass_matrix(q)
    non_linear_effects = nlp.model.non_linear_effects(q, qdot)

    return cas.inv(mass_matrix) @ (
        torques_computed - non_linear_effects - nlp.model.friction_coefficients @ qdot
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
    f_force_field = force_field_magnitude * (
        l1 * cas.cos(q[0]) + l2 * cas.cos(q[0] + q[1])
    )
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


def integrator_collocations(
    model,
    polynomial_degree,
    n_shooting,
    duration,
    states,
    controls,
    stochastic_variables,
):

    h = duration / n_shooting
    method = "legendre"

    # Coefficients of the collocation equation
    _c = type(model.motor_noise_sym).zeros(
        (polynomial_degree + 1, polynomial_degree + 1)
    )

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
                _l *= (time_control_interval - step_time[r]) / (
                    step_time[j] - step_time[r]
                )

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
                _l *= (time_control_interval - step_time[r]) / (
                    step_time[j] - step_time[r]
                )

        # Evaluate the time derivative of the polynomial at all collocation points to get
        # the coefficients of the continuity equation
        tfcn = cas.Function(
            "tfcn", [time_control_interval], [cas.tangent(_l, time_control_interval)]
        )
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

        # defects.append(h * f_j - xp_j)
        defects.append(f_j - xp_j / h)

        # Add contribution to the end state
        states_end += _d[j] * states[:, j]

    # Concatenate constraints
    defects = cas.vertcat(*defects)

    return states_end, defects, _d


def integrator_rk4(
    model, q, qdot, u, stochastic_variables, fun_cov, fun_states, n_shooting, duration
):
    step_time = duration / n_shooting
    n_step = 5
    h_norm = 1 / n_step
    h = step_time * h_norm

    nb_q = q.shape[0]
    x = np.zeros((2 * nb_q, n_step + 1))
    for j in range(nb_q):
        x[j, 0] = q[j]
    for j in range(nb_q):
        x[j + nb_q, 0] = qdot[j]

    s = np.zeros((stochastic_variables.shape[0], n_step + 1))
    s[:, 0] = stochastic_variables

    for i in range(1, n_step + 1):
        k1_states = fun_states(x[:nb_q, i - 1], x[nb_q:, i - 1], u, model.motor_noise_magnitude)
        k1_cov = fun_cov(
            x[:nb_q, i - 1], x[nb_q:, i - 1], s[:, i - 1], model.motor_noise_magnitude
        )
        k2_states = fun_states(
            x[:nb_q, i - 1] + h / 2 * k1_states[:nb_q],
            x[nb_q:, i - 1] + h / 2 * k1_states[nb_q:],
            u, model.motor_noise_magnitude
        )
        k2_cov = fun_cov(
            x[:nb_q, i - 1] + h / 2 * k1_states[:nb_q],
            x[nb_q:, i - 1] + h / 2 * k1_states[nb_q:],
            s[:, i - 1] + h / 2 * k1_cov,
            model.motor_noise_magnitude,
        )
        k3_states = fun_states(
            x[:nb_q, i - 1] + h / 2 * k2_states[:nb_q],
            x[nb_q:, i - 1] + h / 2 * k2_states[nb_q:],
            u, model.motor_noise_magnitude
        )
        k3_cov = fun_cov(
            x[:nb_q, i - 1] + h / 2 * k2_states[:nb_q],
            x[nb_q:, i - 1] + h / 2 * k2_states[nb_q:],
            s[:, i - 1] + h / 2 * k2_cov,
            model.motor_noise_magnitude,
        )
        k4_states = fun_states(
            x[:nb_q, i - 1] + h * k3_states[:nb_q],
            x[nb_q:, i - 1] + h * k3_states[nb_q:],
            u, model.motor_noise_magnitude
        )
        k4_cov = fun_cov(
            x[:nb_q, i - 1] + h * k3_states[:nb_q],
            x[nb_q:, i - 1] + h * k3_states[nb_q:],
            s[:, i - 1] + h * k3_cov,
            model.motor_noise_magnitude,
        )
        x[:, i] = np.reshape(
            x[:, i - 1]
            + h / 6 * (k1_states + 2 * k2_states + 2 * k3_states + k4_states),
            (-1,),
        )

        s[:, i] = np.reshape(
            s[:, i - 1] + h / 6 * (k1_cov + 2 * k2_cov + 2 * k3_cov + k4_cov), (-1,)
        )
        #todo: voir pour l'ordre

    return s[:, -1]


def get_m_init(
    model,
    n_stochastic,
    n_shooting,
    duration,
    polynomial_degree,
    q_last,
    qdot_last,
    u_last,
):
    """
    M = -dF_dz @ inv(dG_dz)
    """

    n_q = model.nb_q
    n_joints = model.nb_u

    x_q_joints = type(model.motor_noise_sym).sym("x_q_joints", n_joints, 1)
    x_qdot_joints = type(model.motor_noise_sym).sym("x_qdot_joints", n_joints, 1)
    z_q_joints = []
    z_qdot_joints = []
    for i in range(polynomial_degree):
        z_q_joints += [type(model.motor_noise_sym).sym("z_q_joints", n_joints, 1)]
        z_qdot_joints += [type(model.motor_noise_sym).sym("z_qdot_joints", n_joints, 1)]
    controls_sym = type(model.motor_noise_sym).sym("controls", n_q, 1)
    stochastic_variables_sym = type(model.motor_noise_sym).sym(
        "stochastic_variables", n_stochastic, 1
    )

    states_full = cas.vertcat(x_q_joints, x_qdot_joints)
    for i in range(polynomial_degree):
        states_full = cas.horzcat(
            states_full, cas.vertcat(z_q_joints[i], z_qdot_joints[i])
        )

    states_end, defects, _ = integrator_collocations(
        model,
        polynomial_degree,
        n_shooting,
        duration,
        states_full,
        controls_sym,
        stochastic_variables_sym,
    )
    initial_polynomial_evaluation = cas.vertcat(x_q_joints, x_qdot_joints)
    defects = cas.vertcat(initial_polynomial_evaluation, defects)

    df_dz = cas.jacobian(states_end, x_q_joints)
    df_dz = cas.horzcat(df_dz, cas.jacobian(states_end, x_qdot_joints))
    for i in range(polynomial_degree):
        df_dz = cas.horzcat(df_dz, cas.jacobian(states_end, z_q_joints[i]))
        df_dz = cas.horzcat(df_dz, cas.jacobian(states_end, z_qdot_joints[i]))

    dg_dz = cas.jacobian(defects, x_q_joints)
    dg_dz = cas.horzcat(dg_dz, cas.jacobian(defects, x_qdot_joints))
    for i in range(polynomial_degree):
        dg_dz = cas.horzcat(dg_dz, cas.jacobian(defects, z_q_joints[i]))
        dg_dz = cas.horzcat(dg_dz, cas.jacobian(defects, z_qdot_joints[i]))

    input_var_list = [x_q_joints, x_qdot_joints]
    for i in range(polynomial_degree):
        input_var_list += [z_q_joints[i], z_qdot_joints[i]]
    input_var_list += [controls_sym, stochastic_variables_sym]
    df_dz_fun = cas.Function(
        "df_dz",
        input_var_list,
        [df_dz],
    )
    dg_dz_fun = cas.Function(
        "dg_dz",
        input_var_list,
        [dg_dz],
    )

    m_last = np.zeros(
        (2 * n_joints * 2 * n_joints * (polynomial_degree + 1), n_shooting + 1)
    )
    for i in range(n_shooting + 1):
        index_this_time = [
            i * polynomial_degree + j for j in range(polynomial_degree + 1)
        ]
        input_num_list = [
            q_last[:, index_this_time[0]],
            qdot_last[:, index_this_time[0]],
        ]
        for j in range(polynomial_degree):
            input_num_list += [
                q_last[:, index_this_time[j + 1]],
                qdot_last[:, index_this_time[j + 1]],
            ]
        input_num_list += [
            u_last[:, i],
            np.vstack(
                (
                    np.zeros(
                        (2 * n_joints * 2 * n_joints * (polynomial_degree + 1), 1)
                    ),  # M
                    np.zeros((2 * n_joints * 2 * n_joints, 1)),
                )
            ),  # cov
        ]
        df_dz_evaluated = df_dz_fun(*input_num_list)
        dg_dz_evaluated = dg_dz_fun(*input_num_list)

        m_this_time = -df_dz_evaluated @ np.linalg.inv(
            dg_dz_evaluated
        )  # Does not varry
        # m_this_time = df_dz_evaluated @ np.linalg.inv(dg_dz_evaluated)  # Does not varry
        # m_this_time = np.linalg.inv(dg_dz_evaluated) @ df_dz_evaluated  # Does not varry
        # m_this_time = -np.linalg.inv(dg_dz_evaluated) @ df_dz_evaluated  # Does not varry

        shape_0, shape_1 = m_this_time.shape[0], m_this_time.shape[1]
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                m_last[shape_0 * s1 + s0, i] = m_this_time[s0, s1]

    return m_last


def get_cov_init_collocations(
    model,
    n_shooting,
    n_stochastic,
    polynomial_degree,
    duration,
    q_last,
    qdot_last,
    u_last,
    m_last,
    cov_init,
    motor_noise_magnitude,
):
    """
    P_k+1 = M_k @ (dG_dx @ P_k @ dG_dx.T + dG_dw @ sigma_w @ dG_dw.T) @ M_k.T
    """

    n_q = model.nb_q
    n_joints = model.nb_u

    x_q_joints = type(model.motor_noise_sym).sym("x_q_joints", n_joints, 1)
    x_qdot_joints = type(model.motor_noise_sym).sym("x_qdot_joints", n_joints, 1)
    z_q_joints = []
    z_qdot_joints = []
    for i in range(polynomial_degree):
        z_q_joints += [type(model.motor_noise_sym).sym("z_q_joints", n_joints, 1)]
        z_qdot_joints += [type(model.motor_noise_sym).sym("z_qdot_joints", n_joints, 1)]
    controls_sym = type(model.motor_noise_sym).sym("controls", n_q, 1)
    stochastic_variables_sym = type(model.motor_noise_sym).sym(
        "stochastic_variables", n_stochastic, 1
    )

    states_full = cas.vertcat(x_q_joints, x_qdot_joints)
    for i in range(polynomial_degree):
        states_full = cas.horzcat(
            states_full, cas.vertcat(z_q_joints[i], z_qdot_joints[i])
        )

    states_end, defects, _ = integrator_collocations(
        model,
        polynomial_degree,
        n_shooting,
        duration,
        states_full,
        controls_sym,
        stochastic_variables_sym,
    )
    initial_polynomial_evaluation = cas.vertcat(x_q_joints, x_qdot_joints)
    defects = cas.vertcat(initial_polynomial_evaluation, defects)

    dg_dx = cas.horzcat(
        cas.jacobian(defects, x_q_joints),
        cas.jacobian(defects, x_qdot_joints),
    )

    dg_dw = cas.jacobian(defects, model.motor_noise_sym)

    input_var_list = [x_q_joints, x_qdot_joints]
    for i in range(polynomial_degree):
        input_var_list += [z_q_joints[i], z_qdot_joints[i]]
    input_var_list += [controls_sym, stochastic_variables_sym, model.motor_noise_sym]
    dg_dx_fun = cas.Function(
        "dg_dx",
        input_var_list,
        [dg_dx],
    )
    dg_dw_fun = cas.Function(
        "dg_dw",
        input_var_list,
        [dg_dw],
    )

    # augmented_sigma_w_mx = cas.MX.zeros(2 * n_joints, 2 * n_joints)
    # for i in range(motor_noise_magnitude.shape[0]):
    #     augmented_sigma_w_mx[i, i] = model.motor_noise_sym[i]
    sigma_w_dm = cas.DM_eye(motor_noise_magnitude.shape[0]) * motor_noise_magnitude
    cov_last = np.zeros((2 * n_joints * 2 * n_joints, n_shooting + 1))

    # dx_dt = model.dynamics_numerical(
    #     cas.vertcat(x_q_joints, x_qdot_joints),
    #     controls_sym,
    #     stochastic_variables_sym,
    #     with_noise=True
    # )
    # cov_init_sym = augmented_sigma_w_mx + cas.jacobian(dx_dt, cas.vertcat(x_q_joints, x_qdot_joints)) + cas.jacobian(dx_dt, model.motor_noise_sym)
    # # cov_init_sym = cas.jacobian(defects[-4:], cas.vertcat(x_q_joints, x_qdot_joints)) + cas.jacobian(defects[-4:], model.motor_noise_sym)
    # cov_init_fcn = cas.Function("cov_init_fcn", [x_q_joints, x_qdot_joints, controls_sym, stochastic_variables_sym, model.motor_noise_sym], [cov_init_sym])
    #
    # cov_init_matrix = cov_init_fcn(q_last[:, 0], qdot_last[:, 0], u_last[:, 0], np.zeros((n_stochastic, 1)), motor_noise_magnitude)
    # cov_last[:, 0] = np.reshape(StochasticBioModel.reshape_to_vector(cov_init_matrix), (-1, ))
    cov_last[:, 0] = cov_init[:, 0]
    for i in range(n_shooting):
        index_this_time = [
            i * polynomial_degree + j for j in range(polynomial_degree + 1)
        ]
        input_num_list = [
            q_last[:, index_this_time[0]],
            qdot_last[:, index_this_time[0]],
        ]
        for j in range(polynomial_degree):
            input_num_list += [
                q_last[:, index_this_time[j + 1]],
                qdot_last[:, index_this_time[j + 1]],
            ]
        input_num_list += [
            u_last[:, i],
            np.vstack(
                (
                    m_last[:, i].reshape((-1, 1)),
                    np.zeros((2 * n_joints * 2 * n_joints, 1)),
                )
            ),  # cov
            motor_noise_magnitude,
        ]
        dg_dx_evaluated = dg_dx_fun(*input_num_list)
        dg_dw_evaluated = dg_dw_fun(*input_num_list)

        m_matrix = np.zeros(model.matrix_shape_m)
        shape_0, shape_1 = model.matrix_shape_m
        for s0 in range(shape_1):
            for s1 in range(shape_0):
                m_matrix[s1, s0] = m_last[s0 * shape_0 + s1, i]

        cov_matrix = np.zeros((2 * n_joints, 2 * n_joints))
        for s0 in range(2 * n_joints):
            for s1 in range(2 * n_joints):
                cov_matrix[s1, s0] = cov_last[s0 * 2 * n_joints + s1, i]

        cov_this_time = (
            m_matrix
            @ (
                dg_dx_evaluated @ cov_matrix @ dg_dx_evaluated.T
                + dg_dw_evaluated @ sigma_w_dm @ dg_dw_evaluated.T
            )
            @ m_matrix.T
        )
        for s0 in range(2 * n_joints):
            for s1 in range(2 * n_joints):
                cov_last[2 * n_joints * s1 + s0, i + 1] = cov_this_time[s0, s1]
    return cov_last


def get_cov_init_irk(
    model,
    n_shooting,
    n_stochastic,
    polynomial_degree,
    duration,
    q_last,
    qdot_last,
    u_last,
    cov_init,
    motor_noise_magnitude,
):

    n_q = model.nb_q
    n_joints = model.nb_u
    nx = n_joints * 2

    x_q_joints = type(model.motor_noise_sym).sym("x_q_joints", n_joints, 1)
    x_qdot_joints = type(model.motor_noise_sym).sym("x_qdot_joints", n_joints, 1)
    z_q_joints = []
    z_qdot_joints = []
    for i in range(polynomial_degree):
        z_q_joints += [type(model.motor_noise_sym).sym("z_q_joints", n_joints, 1)]
        z_qdot_joints += [type(model.motor_noise_sym).sym("z_qdot_joints", n_joints, 1)]
    controls_sym = type(model.motor_noise_sym).sym("controls", n_q, 1)
    stochastic_variables_sym = type(model.motor_noise_sym).sym(
        "stochastic_variables", n_stochastic, 1
    )

    x = cas.vertcat(x_q_joints, x_qdot_joints)
    z = []
    for i in range(polynomial_degree):
        z = cas.horzcat(
            z, cas.vertcat(z_q_joints[i], z_qdot_joints[i])
        )

    p = cas.vertcat( controls_sym, model.motor_noise_sym)

    states_end, defects, D = integrator_collocations(
        model,
        polynomial_degree,
        n_shooting,
        duration,
        cas.horzcat(x,z),
        controls_sym,
        stochastic_variables_sym,
    )

    # Root-finding function, implicitly defines x_collocation_points as a function of x0 and p
    vfcn = cas.Function(
        "vfcn",
        [z.reshape((-1, 1)), x, p],
        [defects],
    ).expand()

    # Create a implicit function instance to solve the system of equations
    ifcn = cas.rootfinder("ifcn", "newton", vfcn)
    x_irk_points = ifcn(cas.MX(), x, p)
    w = [x if r == 0 else x_irk_points[(r - 1) * nx: r * nx] for r in range(polynomial_degree + 1)]

    # Get an expression for the state at the end of the finite element
    xf = type(model.motor_noise_sym).zeros(nx, polynomial_degree + 1)  # 0 #
    for r in range(polynomial_degree + 1):
        xf[:, r] = xf[:, r - 1] + D[r] * w[r]


    # Fixed-step integrator
    irk_integrator = cas.Function('irk_integrator', {'x0': x, 'p': p, 'xf': xf[:, -1]},
                              cas.integrator_in(), cas.integrator_out())

    jac = irk_integrator.factory('jac_IRK', irk_integrator.name_in(), ['jac:xf:x0', 'jac:xf:p'])

    cov_last = np.zeros((2 * n_joints * 2 * n_joints, n_shooting + 1))
    cov_last[:, 0] = cov_init[:, 0]

    sigma_w_dm = cas.DM_eye(motor_noise_magnitude.shape[0]) * motor_noise_magnitude


    for i in range(n_shooting):
        # u = np.concatenate((u_last[:, i], np.zeros(motor_noise_magnitude.shape[0])))
        p = np.concatenate((u_last[:, i], motor_noise_magnitude))
        x0 = np.concatenate(
            (q_last[:, i],
             qdot_last[:, i]))

        J = jac(x0=x0, p=p)
        phi_x = J['jac_xf_x0']
        phi_w = J['jac_xf_p'][:, n_joints:]


        cov_matrix = np.zeros((nx, nx))
        for s0 in range(nx):
            for s1 in range(nx):
                cov_matrix[s1, s0] = cov_last[s0 * nx + s1, i]

        sink = phi_x @ cov_matrix @ phi_x.T
        source = phi_w @ sigma_w_dm @ phi_w.T


        cov_this_time = source #sink +
        if not (np.all(np.linalg.eigvals(cov_this_time.full()) >= 0)):
            print("not semi-positive definite")

        for s0 in range(nx):
            for s1 in range(nx):
                cov_last[nx * s1 + s0, i+1] = cov_this_time[s0, s1]


    return cov_last


def get_cov_init_dms(
    model,
    n_shooting,
    n_stochastic,
    duration,
    q_last,
    qdot_last,
    u_last,
    cov_init,
    motor_noise_magnitude,
):
    """
    RK4 integration of d/dt(P_k) = A @ P_k + P_k @ A.T + B @ sigma_w @ B.T
    """

    n_q = model.nb_q
    n_joints = model.nb_u

    x_q_joints = type(model.motor_noise_sym).sym("x_q_joints", n_joints, 1)
    x_qdot_joints = type(model.motor_noise_sym).sym("x_qdot_joints", n_joints, 1)
    controls_sym = type(model.motor_noise_sym).sym("controls", n_q, 1)
    stochastic_variables_sym = type(model.motor_noise_sym).sym(
        "stochastic_variables", n_stochastic, 1
    )

    states_full = cas.vertcat(x_q_joints, x_qdot_joints)

    dx_dt_without = model.dynamics_numerical(
        states_full, controls_sym, stochastic_variables_sym, with_noise=False
    )

    dx_dt_with = model.dynamics_numerical(
        states_full, controls_sym, stochastic_variables_sym, with_noise=True
    )

    A = cas.jacobian(dx_dt_with, states_full)
    B = cas.jacobian(dx_dt_with, model.motor_noise_sym)

    sigma_w = cas.MX.zeros(
        model.motor_noise_sym.shape[0], model.motor_noise_sym.shape[0]
    )
    for i in range(model.motor_noise_sym.shape[0]):
        sigma_w[i, i] = model.motor_noise_sym[i]

    sigma_w_dm = cas.DM_eye(motor_noise_magnitude.shape[0]) * motor_noise_magnitude # ADD: mick

    cov_matrix = StochasticBioModel.reshape_sym_to_matrix(
        stochastic_variables_sym, model.matrix_shape_cov
    )

    sink = A @ cov_matrix + cov_matrix @ A.T
    source = B @ sigma_w_dm @ B.T
    cov_derivative = source #+sink #+
    cov_derivative_vect = StochasticBioModel.reshape_to_vector(cov_derivative)
    cov_derivative_func = cas.Function(
        "cov_derivative_func",
        [x_q_joints, x_qdot_joints, stochastic_variables_sym, model.motor_noise_sym],
        [cov_derivative_vect],
    )
    states_derivative_func = cas.Function(
        "states_derivative_func",
        [x_q_joints, x_qdot_joints, controls_sym, model.motor_noise_sym],
        [dx_dt_with], #todo: MB: change dx_dt_without to with (add model.motor_noise_sym)
    )

    cov_last = np.zeros((2 * n_joints * 2 * n_joints, n_shooting + 1))
    cov_last[:, 0] = cov_init[:, 0]
    for i in range(n_shooting):
        cov_next_computed = integrator_rk4(
            model,
            q_last[:, i],
            qdot_last[:, i],
            u_last[:, i],
            cov_last[:, i],
            cov_derivative_func,
            states_derivative_func,
            n_shooting,
            duration,
        )
        cov_last[:, i + 1] = cov_next_computed
    return cov_last


def get_cov_init_slicot(
    model,
    n_shooting,
    n_stochastic,
    duration,
    q_last,
    qdot_last,
    u_last,
    cov_init,
    motor_noise_magnitude,
):
    """
    SLICOT integration of d/dt(P_k)
    """

    n_q = model.nb_q
    n_joints = model.nb_u

    x_q_joints = type(model.motor_noise_sym).sym("x_q_joints", n_joints, 1)
    x_qdot_joints = type(model.motor_noise_sym).sym("x_qdot_joints", n_joints, 1)
    controls_sym = type(model.motor_noise_sym).sym("controls", n_q, 1)
    stochastic_variables_sym = type(model.motor_noise_sym).sym(
        "stochastic_variables", n_stochastic, 1
    )

    states_full = cas.vertcat(x_q_joints, x_qdot_joints)

    dx_dt_without = model.dynamics_numerical(
        states_full, controls_sym, stochastic_variables_sym, with_noise=False
    )

    dx_dt_with = model.dynamics_numerical(
        states_full, controls_sym, stochastic_variables_sym, with_noise=True
    )

    A = cas.jacobian(dx_dt_with, states_full)
    B = cas.jacobian(dx_dt_with, model.motor_noise_sym)

    sigma_w = cas.MX.zeros(
        model.motor_noise_sym.shape[0], model.motor_noise_sym.shape[0]
    )
    for i in range(model.motor_noise_sym.shape[0]):
        sigma_w[i, i] = model.motor_noise_sym[i]

    cov_matrix = StochasticBioModel.reshape_sym_to_matrix(
        stochastic_variables_sym, model.matrix_shape_cov
    )

    sink = A @ cov_matrix + cov_matrix @ A.T
    source = B @ sigma_w @ B.T

    # SLICOT
    cas.dplesol(A, V, "slicot", dict())
    cov_derivative = sink + source
    cov_derivative_vect = StochasticBioModel.reshape_to_vector(cov_derivative)
    cov_derivative_func = cas.Function(
        "cov_derivative_func",
        [x_q_joints, x_qdot_joints, stochastic_variables_sym, model.motor_noise_sym],
        [cov_derivative_vect],
    )
    states_derivative_func = cas.Function(
        "states_derivative_func",
        [x_q_joints, x_qdot_joints, controls_sym],
        [dx_dt_without],
    )

    cov_last = np.zeros((2 * n_joints * 2 * n_joints, n_shooting + 1))
    cov_last[:, 0] = cov_init[:, 0]
    for i in range(n_shooting):
        cov_next_computed = integrator_rk4(
            model,
            q_last[:, i],
            qdot_last[:, i],
            u_last[:, i],
            cov_last[:, i],
            cov_derivative_func,
            states_derivative_func,
            n_shooting,
            duration,
        )
        cov_last[:, i + 1] = cov_next_computed
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

    matrix = np.zeros((shape_0, shape_0))
    for s0 in range(shape_0):
        for s1 in range(shape_0):
            matrix[s1, s0] = var[s0 * shape_0 + s1]

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
        h = (
            ((p_x - model.super_ellipse_center_x[i]) / model.super_ellipse_a[i])
            ** model.super_ellipse_n[i]
            + ((p_y - model.super_ellipse_center_y[i]) / model.super_ellipse_b[i])
            ** model.super_ellipse_n[i]
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

    func = cas.Function(
        "out",
        [p_x, p_y, v_x, v_y, cov],
        [non_robust_constraint, robust_component, dh_dx_all, cov_all, sqrt_all],
    )

    polynomial_degree = (
        model.polynomial_degree
        if isinstance(model.socp_type, SocpType.COLLOCATION)
        else 0
    )
    out_num = []
    for j in range(cov_num.shape[1]):
        p_x_value = q[0, j * (polynomial_degree + 1)]
        p_y_value = q[1, j * (polynomial_degree + 1)]
        v_x_value = qdot[0, j * (polynomial_degree + 1)]
        v_y_value = qdot[1, j * (polynomial_degree + 1)]
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
