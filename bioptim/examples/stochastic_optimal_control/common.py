"""
This file contains the functions that are common for multiple stochastic examples.
"""

import casadi as cas
import numpy as np
from bioptim import StochasticBioModel, DynamicsFunctions, SocpType
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
        k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

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
