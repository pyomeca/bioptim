"""
This file contains the functions that are common for multiple stochastic examples.
"""

import casadi as cas
import numpy as np
from bioptim import StochasticBioModel, DynamicsFunctions, SocpType
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


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


def get_m_cov_init(
    model,
    n_stochastic,
    n_shooting,
    duration,
    polynomial_degree,
    q_last,
    qdot_last,
    u_last,
    cov_init,
):
    """
    M = -dF_dz @ inv(dG_dz)
    """

    nx = model.nb_q + model.nb_qdot
    nu = model.nb_u

    m_last = np.zeros((nx * nx * (polynomial_degree + 1), n_shooting + 1))
    cov_last = np.zeros((nx * nx, n_shooting + 1))
    cov_last[:, 0] = cov_init
    p0 = np.array([duration, 0, 0])

    # F and G for test only
    F, G, _, Pf, Mf = collocation_fun_jac(model, polynomial_degree, duration / n_shooting)

    for i in range(n_shooting):
        idx = i * (polynomial_degree + 1)
        index_this_time = [i * polynomial_degree + j for j in range(polynomial_degree + 1)]

        x = np.concatenate(
            (
                q_last[:, index_this_time[0]],
                qdot_last[:, index_this_time[0]],
            )
        )

        z = np.concatenate((q_last[:, index_this_time], qdot_last[:, index_this_time]))
        z = z.reshape((-1,), order="F")

        u = u_last[:, i]

        m = Mf(x, z, u, p0)
        m_last[:, i] = StochasticBioModel.reshape_to_vector(m.full())

        cov = StochasticBioModel.reshape_to_matrix(cov_last[:, i], model.matrix_shape_cov)
        P_next = Pf(x, z, u, p0, cov, m)
        cov_last[:, i] = StochasticBioModel.reshape_to_vector(P_next.full())

    return m_last, cov_last


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
    nq = model.nb_q
    nx = model.nb_q + model.nb_qdot
    F, Pf = integration_collocations(model, polynomial_degree, duration / n_shooting)
    # Function(irk_integrator:(x0[4],z0[],p[5],u[],adj_xf[],adj_zf[],adj_qf[])->(xf[4],zf[],qf[],adj_x0[],adj_z0[],adj_p[],adj_u[]) MXFunction)
    # Function(P_irk:(i0[nx],i1[nu],i2[nw+1],i3[nx,nx])->(o0[nx,nx]) MXFunction)

    p = np.append(model.motor_noise_magnitude, duration)
    p0 = np.append(model.motor_noise_magnitude * 0, duration)
    cov_last = np.zeros((nx * nx, n_shooting + 1))
    cov_last[:, 0] = cov_init

    cov_last2 = np.zeros((nx * nx, n_shooting + 1))
    cov_last2[:, 0] = cov_init
    iter = 500
    X_next = np.zeros((nx, iter))

    fig, ax = plt.subplots(1, 1)
    for i in range(n_shooting):
        u = u_last[:, i]
        x = np.concatenate((q_last[:, i], qdot_last[:, i]))
        cov_matrix = reshape_to_matrix(cov_last[:, i], model.matrix_shape_cov)
        cov_next = Pf(x, u, p0, cov_matrix)
        draw_cov_ellipse(cov_next[:nq, :nq].full(), q_last[:, i + 1], ax, "r")

        noise = np.random.randn(model.nb_u, iter)
        for j in range(iter):
            X_next[:, j] = F(x0=x, u=u, p=np.append(noise[:, j], duration))["xf"].full().squeeze()
        plt.plot(X_next[0, :], X_next[1, :], ".")
        cov = np.cov(X_next)
        # confidence_ellipse(X_next[0,:], X_next[1,:], ax)
        draw_cov_ellipse(cov[:nq, :nq], np.mean(X_next[:nq, :], axis=1), ax, "b")
        cov_last2[:, i + 1] = StochasticBioModel.reshape_to_vector(cov)

        if not (np.all(np.linalg.eigvals(cov_next.full()) >= 0)):
            print("IRK cov not semi-positive definite")

        cov_last[:, i + 1] = StochasticBioModel.reshape_to_vector(cov_next.full())

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
    nq = model.nb_q
    nx = model.nb_q + model.nb_qdot
    q = cas.SX.sym("q", model.nb_q)
    qd = cas.SX.sym("qd", model.nb_qdot)
    x = cas.vertcat(q, qd)
    u = cas.SX.sym("u", model.nb_u)
    w = cas.SX.sym("w", model.nb_u)  # motor noise
    Sigma_ww = cas.SX.eye(model.nb_u)  # * w
    P = cas.SX.sym("P", nx, nx)

    # Continuous time dynamics
    xdot = model.dynamics_numerical(
        states=x,
        controls=u,  # Piecewise constant control
        motor_noise=w,
    )

    A = cas.jacobian(xdot, x)
    B = cas.jacobian(xdot, w)
    Pdot = A @ P + P @ A.T + B @ Sigma_ww @ B.T
    Pdot_vec = StochasticBioModel.reshape_to_vector(Pdot)
    dyn_fun = cas.Function("dyn_fun", [x, u, w], [xdot], ["x", "u", "w"], ["xdot"])  # .expand()
    Pdot_fun = cas.Function("Pdot", [x, u, w, P], [Pdot_vec], ["x", "u", "w", "P"], ["Pdot"])
    all_dot = cas.Function(
        "dyn_fun", [x, u, w, P], [cas.vertcat(xdot, Pdot_vec)], ["x", "u", "w", "P"], ["xPdot"]
    )  # .expand()

    p0 = model.motor_noise_magnitude * 0
    cov_last = np.zeros((nx * nx, n_shooting + 1))
    cov_last[:, 0] = cov_init.squeeze()
    for i in range(n_shooting):
        x0 = np.concatenate((q_last[:, i], qdot_last[:, i], cov_last[:, i]))

        def dyn(t, x):
            return (
                all_dot(
                    x[:nx],
                    u_last[:, i],
                    p0,
                    reshape_to_matrix(x[nx:], model.matrix_shape_cov),
                )
                .full()
                .squeeze()
            )

        sol = solve_ivp(fun=dyn, t_span=(0, duration / n_shooting), y0=x0, method="RK45", atol=1e-15)

        y = sol.y[:, -1]
        cov_last[:, i + 1] = y[nq * 2 :]

        cov_next_computed = integrator_rk4(
            model,
            q_last[:, i],
            qdot_last[:, i],
            u_last[:, i],
            cov_last[:, i],
            Pdot_fun,
            dyn_fun,
            n_shooting,
            duration,
        )
        # cov_last[:, i + 1] = cov_next_computed
        # print(y[n_q*2:] - cov_next_computed)

        cov = StochasticBioModel.reshape_to_matrix(cov_last[:, i + 1], model.matrix_shape_cov)

        if i == 0:
            print(cov)

        if not (np.all(np.linalg.eigvals(cov) >= 0)):
            print(f"DMS{i} cov not semi-positive definite")

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
    stochastic_variables_sym = type(model.motor_noise_sym).sym("stochastic_variables", n_stochastic, 1)

    states_full = cas.vertcat(x_q_joints, x_qdot_joints)

    dx_dt_without = model.dynamics_numerical(states_full, controls_sym, stochastic_variables_sym, with_noise=False)

    dx_dt_with = model.dynamics_numerical(states_full, controls_sym, stochastic_variables_sym, with_noise=True)

    A = cas.jacobian(dx_dt_with, states_full)
    B = cas.jacobian(dx_dt_with, model.motor_noise_sym)

    sigma_w = cas.MX.zeros(model.motor_noise_sym.shape[0], model.motor_noise_sym.shape[0])
    for i in range(model.motor_noise_sym.shape[0]):
        sigma_w[i, i] = model.motor_noise_sym[i]

    cov_matrix = StochasticBioModel.reshape_sym_to_matrix(stochastic_variables_sym, model.matrix_shape_cov)

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
            ((p_x - model.super_ellipse_center_x[i]) / model.super_ellipse_a[i]) ** model.super_ellipse_n[i]
            + ((p_y - model.super_ellipse_center_y[i]) / model.super_ellipse_b[i]) ** model.super_ellipse_n[i]
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

    polynomial_degree = model.polynomial_degree if isinstance(model.socp_type, SocpType.COLLOCATION) else 0
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


def confidence_ellipse(x, y, ax, n_std=1.0):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, color="r", alpha=0.1)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def draw_cov_ellipse(cov, pos, ax, color="b"):
    """
    Draw an ellipse representing the covariance at a given point.
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(vals)
    ellip = plt.matplotlib.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, color=color, alpha=0.1)

    ax.add_patch(ellip)
    return ellip


def reshape_to_matrix(var, shape):
    """
    Restore the matrix form of the variables
    """
    shape_0, shape_1 = shape
    if isinstance(var, np.ndarray):
        matrix = np.zeros((shape_0, shape_1))
    else:
        matrix = type(var).zeros(shape_0, shape_1)
    for s0 in range(shape_1):
        for s1 in range(shape_0):
            matrix[s1, s0] = var[s0 * shape_0 + s1]
    return matrix


def collocation_fun_jac(model, d, h):
    # Declare model variables

    # Declare model variables
    nx = model.nb_q + model.nb_qdot
    nu = model.nb_u
    q = cas.SX.sym("q", model.nb_q)
    qd = cas.SX.sym("qd", model.nb_qdot)
    x = cas.vertcat(q, qd)
    u = cas.SX.sym("u", model.nb_u)
    w = cas.SX.sym("w", model.nb_u)  # motor noise
    Sigma_ww = cas.SX.eye(model.nb_u)  # * w
    T = cas.SX.sym("T")  # Time horizon
    p = cas.vertcat(T, w)

    # Control discretization
    hsym = cas.SX.sym("h")

    P = cas.SX.sym("P", nx, nx)
    M = cas.SX.sym("M", nx, nx * (d + 1))

    # Continuous time dynamics
    xdot = model.dynamics_numerical(
        states=x,
        controls=u,  # Piecewise constant control
        motor_noise=w,
    )
    dyn_fun = cas.Function("dyn_fun", [x, u, w], [xdot], ["x", "u", "w"], ["xdot"])

    # Coefficients of the collocation equation (_c) and of the continuity equation (_d)
    _b, _c, _d = prepare_collocation("legendre", d)

    # The helper state sample used by collocation
    z = []
    for j in range(d + 1):
        zj = cas.SX.sym("z_" + str(j), nx)
        z.append(zj)

    # Loop over collocation points
    x0 = _d[0] * z[0]
    G_argout = [x0 + x]
    xf = x0
    for j in range(1, d + 1):
        # Expression for the state derivative at the collocation point
        xp = _c[0, j] * x
        for r in range(1, d + 1):
            xp = xp + _c[r, j] * z[r]

        # Append collocation equations
        fj = dyn_fun(z[j], u, w)
        G_argout.append(xp - h * fj)

        # Add contribution to the end state
        xf = xf + _d[j] * z[j]

    z_ = cas.vertcat(*z)
    # The function G in 0 = G(x_k,z_k,u_k,w_k)
    G = cas.Function("G", [z_, x, u, p], [cas.vertcat(*G_argout)], ["z", "x", "u", "p"], ["g"])

    # The function F in x_{k+1} = F(z_k)
    F = cas.Function("F", [z_], [xf], ["z"], ["xf"])

    Gdx = cas.jacobian(G(z_, x, u, p), x)
    Gdz = cas.jacobian(G(z_, x, u, p), z_)
    Gdw = cas.jacobian(G(z_, x, u, p), u)
    Fdz = cas.jacobian(F(z_), z_)

    # M expression to initialize only
    Mf = cas.Function("M", [x, z_, u, p], [-Fdz @ cas.inv(Gdz)])
    # Constraint Equality defining M
    Mc = cas.Function("M_cons", [x, z_, u, p, M], [Fdz.T - Gdz.T @ M.T])
    # Covariance propagation rule
    Pf = cas.Function("P_next", [x, z_, u, p, P, M], [M @ (Gdx @ P @ Gdx.T + Gdw @ Sigma_ww @ Gdw.T) @ M.T])

    return F, G, Mc, Pf, Mf


def prepare_collocation(method="legendre", d=5):
    # Get collocation points
    tau_root = np.append(0, cas.collocation_points(d, method))

    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))

    # Coefficients of the continuity equation
    D = np.zeros(d + 1)

    # Coefficients of the quadrature function
    B = np.zeros(d + 1)

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])

        # # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    return B, C, D


def integrator_rk4(model, q, qdot, u, stochastic_variables, fun_cov, fun_states, n_shooting, duration):
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
        k1_cov = fun_cov(x[:nb_q, i - 1], x[nb_q:, i - 1], s[:, i - 1], model.motor_noise_magnitude)
        k2_states = fun_states(
            x[:nb_q, i - 1] + h / 2 * k1_states[:nb_q],
            x[nb_q:, i - 1] + h / 2 * k1_states[nb_q:],
            u,
            model.motor_noise_magnitude,
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
            u,
            model.motor_noise_magnitude,
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
            u,
            model.motor_noise_magnitude,
        )
        k4_cov = fun_cov(
            x[:nb_q, i - 1] + h * k3_states[:nb_q],
            x[nb_q:, i - 1] + h * k3_states[nb_q:],
            s[:, i - 1] + h * k3_cov,
            model.motor_noise_magnitude,
        )
        x[:, i] = np.reshape(
            x[:, i - 1] + h / 6 * (k1_states + 2 * k2_states + 2 * k3_states + k4_states),
            (-1,),
        )

        s[:, i] = np.reshape(s[:, i - 1] + h / 6 * (k1_cov + 2 * k2_cov + 2 * k3_cov + k4_cov), (-1,))
        # todo: voir pour l'ordre order='F'

    return s[:, -1]
