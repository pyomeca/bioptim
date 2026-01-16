"""
This class contains different holonomic constraint function.
"""

from typing import Any, Callable

from casadi import (
    MX,
    Function,
    acos,
    dot,
    fabs,
    fmax,
    fmin,
    horzcat,
    if_else,
    jacobian,
    lt,
    norm_2,
    sin,
    trace,
    vertcat,
)

from ...misc.options import OptionDict
from .biomodel import BioModel


def skew_casadi(v):
    """CasADi version of the skew-symmetric matrix."""
    x, y, z = v[0], v[1], v[2]
    return vertcat(
        horzcat(0, -z, y),
        horzcat(z, 0, -x),
        horzcat(-y, x, 0),
    )


def axis_angle_from_R_casadi(R):
    """CasADi version of axis-angle extraction."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    c = (tr - 1) / 2  # cos(theta)
    c = fmax(fmin(c, 1), -1)  # clip for numerical safety
    theta = acos(c)

    # Handle small theta (avoid division by zero)
    eps = 1e-12
    sin_theta = sin(theta)
    small_theta = if_else(
        lt(fabs(sin_theta), eps),
        MX.ones(1),
        MX.zeros(1),
    )

    # Compute omega (axis)
    omega_x = (R[2, 1] - R[1, 2]) / (2 * sin_theta + eps)
    omega_y = (R[0, 2] - R[2, 0]) / (2 * sin_theta + eps)
    omega_z = (R[1, 0] - R[0, 1]) / (2 * sin_theta + eps)
    omega = vertcat(omega_x, omega_y, omega_z)

    # Normalize omega (avoid division by zero)
    omega_norm = norm_2(omega)
    omega = if_else(
        lt(omega_norm, eps),
        vertcat(1, 0, 0),  # default axis if omega_norm is small
        omega / omega_norm,
    )

    return theta, omega


class HolonomicConstraintsFcn:
    """
    This class contains different holonomic constraint.
    """

    @staticmethod
    def superimpose_markers(
        marker_1: str,
        marker_2: str = None,
        index: slice = slice(0, 3),
        local_frame_index: int = None,
        model: BioModel = None,
    ) -> tuple[Function, Function, Function]:
        """
        Generate the constraint functions to superimpose two markers.

        Parameters
        ----------
        model: BioModel
            The model.
        marker_1: str
            The name of the first marker.
        marker_2: str
            The name of the second marker. If None, the constraint will be to superimpose the first marker to the
            origin.
        index: slice
            The index of the markers to superimpose.
        local_frame_index: int
            The index of the frame in which the markers are expressed. If None, the markers are expressed in the global.


        Returns
        -------
        The constraint function, its jacobian and its double derivative.
        """

        # symbolic variables to create the functions
        q_sym = MX.sym("q", model.nb_q, 1)
        q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)
        q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)
        parameters = []

        # symbolic markers in global frame
        marker_1_sym = model.marker(index=model.marker_index(marker_1))(q_sym, parameters)
        if marker_2 is not None:
            marker_2_sym = model.marker(index=model.marker_index(marker_2))(q_sym, parameters)

        else:
            marker_2_sym = MX([0, 0, 0])

            # if local frame is provided, the markers are expressed in the same local frame
        if local_frame_index is not None:
            jcs_t = model.homogeneous_matrices_in_global(segment_index=local_frame_index, inverse=True)(
                q_sym, parameters
            )
            marker_1_sym = (jcs_t @ vertcat(marker_1_sym, 1))[:3]
            marker_2_sym = (jcs_t @ vertcat(marker_2_sym, 1))[:3]

        # the constraint is the distance between the two markers, set to zero
        constraint = (marker_1_sym - marker_2_sym)[index]
        # the jacobian of the constraint
        constraints_jacobian = jacobian(constraint, q_sym)

        constraints_func = Function(
            "holonomic_constraints",
            [q_sym],
            [constraint],
            ["q"],
            ["holonomic_constraint"],
        ).expand()

        constraints_jacobian_func = Function(
            "holonomic_constraints_jacobian",
            [q_sym],
            [constraints_jacobian],
            ["q"],
            ["holonomic_constraints_jacobian"],
        ).expand()

        # the double derivative of the constraint
        constraints_double_derivative = (
            constraints_jacobian_func(q_sym) @ q_ddot_sym + constraints_jacobian_func(q_dot_sym) @ q_dot_sym
        )

        constraints_double_derivative_func = Function(
            "holonomic_constraints_double_derivative",
            [q_sym, q_dot_sym, q_ddot_sym],
            [constraints_double_derivative],
            ["q", "q_dot", "q_ddot"],
            ["holonomic_constraints_double_derivative"],
        ).expand()

        return constraints_func, constraints_jacobian_func, constraints_double_derivative_func

    @staticmethod
    def align_frames(  # <-- new method
        model: BioModel = None,
        frame_1_idx: int = 0,  # index of the first segment / frame in the model
        frame_2_idx: int = 1,  # index of the second segment / frame
        local_frame_idx: int = None,
    ) -> tuple[Function, Function, Function]:
        """
        Generate the holonomic constraint that forces the orientation of two
        frames to be identical (i.e. the relative rotation between them is the
        identity).

        The constraint is expressed in the set of three independent
        equations using the relationships

        *   ``trace(R_rel) = 3``  (⇔ ``cosθ = 1`` ⇒ ``θ = 0``)
        *   ``R_rel_ij = R_rel_ji``  for the three off‑diagonal elements,
            i.e. ``R_rel – R_rel.T = 0``.

        Those four scalar equations are not independent (the trace condition
        already forces the three off‑diagonal elements to be zero when the
        matrix is orthogonal), so we drop one redundant equation and keep only
        three of them (the three independent components of the **skew‑symmetric**
        part of ``R_rel``).  The resulting constraint vector is exactly the one
        you would obtain from the axis‑angle formulation with ``θ = 0``.

        Parameters
        ----------
        model
            The :class:`~bioptim.BioModel` that contains the kinematic tree.
        frame_1_idx, frame_2_idx
            Indices of the two segments (or bodies) whose orientation must be
            aligned.
        local_frame_idx
            If given, the two frames are first expressed in the *local* frame
            identified by this index (the same behaviour as in
            ``superimpose_markers``).  When ``None`` the orientations are taken
            directly in the global reference.

        Returns
        -------
        (constraints_func,
         constraints_jacobian_func,
         constraints_double_derivative_func)

        each of type :class:`casadi.Function`.
        """
        # Symbolic variables
        q_sym = MX.sym("q", model.nb_q, 1)  # generalized coordinates
        q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)  # velocities
        q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)  # accelerations
        parameters = model.parameters  # optional model parameters

        # Homogeneous transformation matrices of the two frames
        # Global homogeneous matrices (4×4) of the two frames
        T1_glob = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)  # shape (4,4)
        T2_glob = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)  # shape (4,4)

        # If a *local* reference frame is requested we first bring the two frames
        # into that local frame (identical to the logic used for the marker
        # constraint).  The inverse transformation matrix ``T_loc`` maps global → local.
        if local_frame_idx is not None:
            T_loc = model.homogeneous_matrices_in_global(segment_index=local_frame_idx, inverse=True)(q_sym, parameters)
            T1_glob = T_loc @ T1_glob
            T2_glob = T_loc @ T2_glob

        # Extract only the 3×3 rotation part (the upper‑left block)
        R1 = T1_glob[:3, :3]  # shape (3,3)
        R2 = T2_glob[:3, :3]  # shape (3,3)

        # Relative rotation: R_rel = R1ᵀ·R2    (frame‑1 → frame‑2)
        R_rel = R1.T @ R2  # still a symbolic 3×3 matrix

        # Minimal set of scalar constraints (3 equations)
        # The skew‑symmetric part of a proper rotation is zero when the angle is zero:
        #   S = (R_rel - R_relᵀ) / 2  →  S = 0   ⇔   ω = 0, θ = 0
        # We vectorise the three independent components of S:
        #    S_21, S_31, S_32
        # (any consistent ordering works, we keep the same order used in the
        #  analytical derivation of the constraint in the OP.)
        S = (R_rel - R_rel.T) / 2.0  # still 3×3, skew‑symmetric
        constraint = vertcat(S[1, 0], S[2, 0], S[2, 1])  # r21 - r12  # r31 - r13  # r32 - r23
        # Note: you could also add ``trace(R_rel)-3`` as a fourth equation,
        # but it is redundant when the matrix stays orthogonal (the solver
        # already enforces orthonormality via the dynamics).

        # Jacobian and second derivative (CasADi)
        constraints_jacobian = jacobian(constraint, q_sym)

        # First derivative (velocity level of the holonomic constraint)
        velocity_constraint = constraints_jacobian @ q_dot_sym

        # Second derivative (acceleration level) – needed for OCP solvers that
        # treat holonomic constraints as second‑order (e.g. direct collocation)
        acceleration_constraint = (
            jacobian(velocity_constraint, q_sym) @ q_dot_sym + jacobian(constraint, q_sym) @ q_ddot_sym
        )

        constraints_func = Function(
            "align_frames_constraint",
            [q_sym],
            [constraint],
            ["q"],
            ["c_align"],
        ).expand()

        constraints_jacobian_func = Function(
            "align_frames_jacobian",
            [q_sym],
            [constraints_jacobian],
            ["q"],
            ["J_align"],
        ).expand()

        constraints_double_derivative_func = Function(
            "align_frames_ddot",
            [q_sym, q_dot_sym, q_ddot_sym],
            [acceleration_constraint],
            ["q", "q_dot", "q_ddot"],
            ["c_ddot_align"],
        ).expand()

        return (
            constraints_func,
            constraints_jacobian_func,
            constraints_double_derivative_func,
        )

    @staticmethod
    def align_frames_minimize_omega_cross_theta(
        model,
        frame_1_idx: int = 0,
        frame_2_idx: int = 1,
        local_frame_idx: int = None,
    ):
        # Symbolic variables
        q_sym = MX.sym("q", model.nb_q, 1)
        q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)
        q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)
        parameters = model.parameters

        # Get homogeneous matrices
        T1_glob = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)
        T2_glob = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)

        # Express in local frame if needed
        if local_frame_idx is not None:
            T_loc = model.homogeneous_matrices_in_global(segment_index=local_frame_idx, inverse=True)(q_sym, parameters)
            T1_glob = T_loc @ T1_glob
            T2_glob = T_loc @ T2_glob

        # Extract rotation matrices
        R1 = T1_glob[:3, :3]
        R2 = T2_glob[:3, :3]
        R_rel = R1.T @ R2

        # Extract theta and omega
        theta, omega = axis_angle_from_R_casadi(R_rel)

        # Objective: minimize ||omega x theta||^2
        # omega_cross_theta = skew_casadi(omega) @ vertcat(0, 0, theta)
        # constraint = dot(omega_cross_theta, omega_cross_theta)
        constraint = dot(theta, theta)

        # Create constraint function
        constraints_func = Function(
            "align_frames_cost",
            [q_sym],
            [constraint],
            ["q"],
            ["cost"],
        ).expand()

        # Jacobian of the constraint
        constraints_jacobian = jacobian(constraint, q_sym)

        constraints_jacobian_func = Function(
            "holonomic_constraints_jacobian",
            [q_sym],
            [constraints_jacobian],
            ["q"],
            ["holonomic_constraints_jacobian"],
        ).expand()

        # Hessian of the constraint
        constraints_hessian = jacobian(constraints_jacobian, q_sym)

        # Double derivative of the constraint
        constraints_double_derivative = constraints_hessian @ q_dot_sym + constraints_jacobian @ q_ddot_sym

        constraints_double_derivative_func = Function(
            "holonomic_constraints_double_derivative",
            [q_sym, q_dot_sym, q_ddot_sym],
            [constraints_double_derivative],
            ["q", "q_dot", "q_ddot"],
            ["holonomic_constraints_double_derivative"],
        ).expand()

        return constraints_func, constraints_jacobian_func, constraints_double_derivative_func

    @staticmethod
    def rigid_contacts(
        model: BioModel = None,
    ) -> tuple[Function, Function, Function]:
        """
        Generate the constraint functions to restrain contact movement.

        Parameters
        ----------
        model: BioModel
            The model.

        Returns
        -------
        The constraint function, its jacobian and its double derivative.
        """

        # symbolic variables to create the functions
        q_sym = MX.sym("q", model.nb_q, 1)
        q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)
        q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)
        parameters = model.parameters

        contact_position = MX()
        for i_contact in range(model.nb_rigid_contacts):
            contact_position = vertcat(
                contact_position,
                model.rigid_contact_position(i_contact)(q_sym, parameters)[model.rigid_contact_axes_index(i_contact)],
            )

        constraint = contact_position

        # the jacobian of the constraint
        constraints_jacobian = jacobian(constraint, q_sym)

        # First derivative (velocity)
        velocity_constraint = jacobian(constraint, q_sym) @ q_dot_sym

        # Second derivative (acceleration)
        acceleration_constraint = (
            jacobian(velocity_constraint, q_sym) @ q_dot_sym + jacobian(constraint, q_sym) @ q_ddot_sym
        )

        constraints_func = Function(
            "holonomic_constraints",
            [q_sym, parameters],
            [constraint],
            ["q", "parameters"],
            ["holonomic_constraint"],
        ).expand()

        constraints_jacobian_func = Function(
            "holonomic_constraints_jacobian",
            [q_sym, parameters],
            [constraints_jacobian],
            ["q", "parameters"],
            ["holonomic_constraints_jacobian"],
        ).expand()

        constraints_double_derivative_func = Function(
            "holonomic_constraints_double_derivative",
            [q_sym, q_dot_sym, q_ddot_sym, parameters],
            [acceleration_constraint],
            ["q", "q_dot", "q_ddot", "parameters"],
            ["holonomic_constraints_double_derivative"],
        ).expand()

        return constraints_func, constraints_jacobian_func, constraints_double_derivative_func


class HolonomicConstraintsList(OptionDict):
    """
    A list of holonomic constraints to be sent to HolonomicBiorbdModel.set_holonomic_configuration()

    Methods
    -------
    add(self, key: str, constraints: Function, constraints_jacobian: Function, constraints_double_derivative: Function)
        Add a new holonomic constraint to the dict
    """

    def __init__(self):
        super(HolonomicConstraintsList, self).__init__(sub_type=dict)

    def add(self, key: str, constraints_fcn: Callable, **extra_arguments: Any):
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        key: str
            The name of the optimization variable
        constraints_fcn: HolonomicConstraintsFcn
            The function that generates the holonomic constraints
        """
        super(HolonomicConstraintsList, self)._add(
            key=key,
            constraints_fcn=constraints_fcn,
            extra_arguments=extra_arguments,
        )
