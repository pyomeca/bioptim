"""
This module contains holonomic constraint functions for biomechanical models.
"""

from typing import Any, Callable

from casadi import (
    MX,
    DM,
    Function,
    acos,
    fabs,
    fmax,
    fmin,
    if_else,
    jacobian,
    norm_2,
    sin,
    sqrt,
    trace,
    vertcat,
    pi,
    atan2,
)

from .biomodel import BioModel
from ...misc.options import OptionDict


class HolonomicConstraintsFcn:
    """
    Factory class providing static methods to generate holonomic constraint functions.

    This class contains various holonomic constraint formulations that can be applied to
    biomechanical models. Each method returns a tuple of CasADi Functions representing:
        - The constraint equation Φ(q) = 0
        - The constraint Jacobian ∂Φ/∂q
        - The bias term J̇q̇ for acceleration-level constraints

    Available Constraints
    ---------------------
    - superimpose_markers: Enforce two markers to occupy the same position
    - align_frames: Enforce two reference frames to maintain a specified relative orientation
    - rigid_contacts: Enforce rigid contact constraints to prevent penetration

    See Also
    --------
    HolonomicConstraintsList : Container for managing multiple holonomic constraints
    HolonomicBiorbdModel : Model class that uses these constraints
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
        Generate holonomic constraint functions to superimpose two markers or fix a marker to the origin.

        This constraint enforces that two markers occupy the same position in space (or a marker
        is fixed to the origin), creating a holonomic constraint of the form:
            Φ(q) = marker_1(q) - marker_2(q) = 0

        The constraint can be expressed in either the global frame or a local reference frame,
        and can be applied to specific spatial dimensions via the index parameter.

        Parameters
        ----------
        model : BioModel
            The biomechanical model containing the marker definitions.
        marker_1 : str
            Name of the first marker in the model.
        marker_2 : str, optional
            Name of the second marker. If None, marker_1 is constrained to the origin [0, 0, 0].
        index : slice, default=slice(0, 3)
            Slice object specifying which spatial dimensions to constrain.
            Examples:
                - slice(0, 3): all three dimensions (x, y, z)
                - slice(0, 2): only x and y dimensions
                - slice(2, 3): only z dimension
        local_frame_index : int, optional
            Index of the segment/frame in which to express the constraint.
            If None, markers are expressed in the global (world) frame.
            If specified, both markers are transformed into this local frame before computing the constraint.

        Returns
        -------
        tuple[Function, Function, Function]
            A tuple containing three CasADi Functions:
                - constraints_func: Φ(q) → constraint values (m × 1)
                - constraints_jacobian_func: ∂Φ/∂q → constraint Jacobian (m × n)
                - bias_func: (q, q̇) → J̇q̇ → bias/acceleration term (m × 1)

            where m is the number of constrained dimensions (determined by index),
            and n is the number of generalized coordinates.

        Examples
        --------
        Superimpose two markers in all three dimensions:
        >>> constraint = HolonomicConstraintsFcn.superimpose_markers(
        ...     model=my_model,
        ...     marker_1="hand",
        ...     marker_2="target"
        ... )

        Fix a marker to the origin in x-y plane only:
        >>> constraint = HolonomicConstraintsFcn.superimpose_markers(
        ...     model=my_model,
        ...     marker_1="foot",
        ...     marker_2=None,
        ...     index=slice(0, 2)
        ... )

        See Also
        --------
        align_frames : Constraint to align orientations of two reference frames
        compute_bias_vector : Computes the bias term J̇q̇
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

        bias_vector = compute_bias_vector(constraints_jacobian, q_sym, q_dot_sym)

        bias_func = Function(
            "superimpose_markers_bias",
            [q_sym, q_dot_sym],
            [bias_vector],
            ["q", "q_dot"],
            ["superimpose_markers_bias"],
        ).expand()

        return constraints_func, constraints_jacobian_func, bias_func

    @staticmethod
    def align_frames(
        model: BioModel = None,
        frame_1_idx: int = 0,
        frame_2_idx: int = 1,
        local_frame_idx: int = None,
    ) -> tuple[Function, Function, Function]:
        """
        Generate holonomic constraint functions to align the orientation of two reference frames.

        This constraint enforces that two body-fixed frames maintain a specified relative orientation,
        creating a 3-DOF holonomic constraint on the system's configuration.

        Mathematical Formulation
        ------------------------
        The constraint enforces:
            R₁ᵀ R₂ = R_desired

        where R₁ and R₂ are the rotation matrices of frames 1 and 2, and R_desired is the
        desired relative rotation (identity by default).

        The constraint is formulated using the axis-angle representation of the rotation error.
        For a relative rotation R_rel = R_desired^T (R₁ᵀ R₂), we extract the three independent
        components of the skew-symmetric part:

            S = (R_rel - R_rel^T) / 2

        When the frames are aligned (θ = 0), S = 0. The constraint vector consists of the
        three independent components of S:

            Φ(q) = [S₃₂, -S₃₁, S₂₁]^T = 0

        This formulation:
            - Uses exactly 3 scalar equations (minimal representation)
            - Avoids singularities at θ = 0 through Taylor expansion
            - Is equivalent to the axis-angle formulation: Φ = (θ/sin(θ)) · ω̂

        Parameters
        ----------
        model : BioModel
            The biomechanical model containing the kinematic tree.
        frame_1_idx : int, default=0
            Index of the first segment/frame in the model.
        frame_2_idx : int, default=1
            Index of the second segment/frame whose orientation will be aligned with frame_1.
        local_frame_idx : int, optional
            If specified, both frames are first transformed into this local reference frame
            before computing the relative rotation. When None, the global (world) frame is used.
            This is useful for expressing alignment constraints relative to a moving reference.
        relative_rotation : DM, default=DM.eye(3)
            The desired 3×3 relative rotation matrix between the frames.
            Default is identity, meaning frames should be perfectly aligned.
            For non-identity values, the constraint enforces R₁ᵀ R₂ = relative_rotation.

        Returns
        -------
        tuple[Function, Function, Function]
            A tuple containing three CasADi Functions:
                - constraints_func: Φ(q) → constraint values (3 × 1)
                - constraints_jacobian_func: ∂Φ/∂q → constraint Jacobian (3 × n)
                - bias_func: (q, q̇) → J̇q̇ → bias/acceleration term (3 × 1)

            where n is the number of generalized coordinates.

        Examples
        --------
        Align two segments in the global frame:
        >>> constraint = HolonomicConstraintsFcn.align_frames(
        ...     model=my_model,
        ...     frame_1_idx=2,  # pelvis
        ...     frame_2_idx=5   # torso
        ... )

        Align with a 90-degree rotation about z-axis:
        >>> import numpy as np
        >>> R_z_90 = DM([
        ...     [0, -1, 0],
        ...     [1,  0, 0],
        ...     [0,  0, 1]
        ... ])
        >>> constraint = HolonomicConstraintsFcn.align_frames(
        ...     model=my_model,
        ...     frame_1_idx=2,
        ...     frame_2_idx=5,
        ...     relative_rotation=R_z_90
        ... )

        Notes
        -----
        The Taylor expansion used for θ/sin(θ) provides numerical stability near θ = 0:
            θ/sin(θ) ≈ 1 + θ²/6 + 7θ⁴/360 + 31θ⁶/15120

        This ensures the constraint remains well-conditioned even when the frames are
        nearly aligned.

        See Also
        --------
        superimpose_markers : Constraint to superimpose marker positions
        compute_bias_vector : Computes the bias term J̇q̇

        References
        ----------
        .. [1] Murray, R. M., Li, Z., & Sastry, S. S. (1994). A Mathematical Introduction
               to Robotic Manipulation. CRC Press.
        """
        # Symbolic variables
        q_sym = MX.sym("q", model.nb_q, 1)  # generalized coordinates
        q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)  # velocities
        parameters = model.parameters  # optional model parameters

        # Homogeneous transformation matrices of the two frames
        # Global homogeneous matrices (4×4) of the two frames
        # T1_glob = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)  # shape (4,4)
        # T2_glob = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)  # shape (4,4)

        # If a *local* reference frame is requested we first bring the two frames
        # into that local frame (identical to the logic used for the marker
        # constraint).
        if local_frame_idx is not None:
            # Get the rotation matrix of the local frame in the global frame
            R_loc_glob = model.homogeneous_matrices_in_global(segment_index=local_frame_idx)(q_sym, parameters)[:3, :3]

            # Get the rotation matrices of the two frames in the global frame
            R1_glob = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)[:3, :3]
            R2_glob = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)[:3, :3]

            # Transform the rotation matrices into the local frame
            R1 = R_loc_glob.T @ R1_glob
            R2 = R_loc_glob.T @ R2_glob
        else:
            # If no local frame is specified, use the global frame
            R1 = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)[:3, :3]
            R2 = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)[:3, :3]

        # Relative rotation: R_rel = R1ᵀ·R2    (frame‑1 → frame‑2)
        R_rel = R2.T @ R1  # still a symbolic 3×3 matrix

        # Minimal set of scalar constraints (3 equations)
        # The skew‑symmetric part of a proper rotation is zero when the angle is zero:
        #   S = (R_rel - R_relᵀ) / 2  →  S = 0   ⇔   ω = 0, θ = 0
        # We vectorise the three independent components of S:
        #    S_21, S_31, S_32
        # (any consistent ordering works, we keep the same order used in the
        #  analytical derivation of the constraint in the OP.)
        # cos_theta = (trace(R_rel) - 1) / 2
        # theta = sqrt(2 * (1 - cos_theta) + 1e-15)  # using the first-order expansion of arccos
        theta = sqrt(fmax(3 - trace(R_rel), 0) + 1e-12)  # using the first-order expansion of arccos
        theta_over_sintheta = (
            1 + theta**2 / 6 + 7 * theta**4 / 360 + 31 * theta**6 / 15120
        )  # using the Taylor expansion
        S = theta_over_sintheta * (R_rel - R_rel.T) / 2.0  # still 3×3, skew‑symmetric

        constraint = vertcat(S[2, 1], -S[2, 0], S[1, 0])  # r32 - r23  # r13 - r31  # r21 - r12

        # Jacobian and second derivative (CasADi)
        constraints_jacobian = jacobian(constraint, q_sym)

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

        bias_vector = compute_bias_vector(constraints_jacobian, q_sym, q_dot_sym)

        bias_func = Function(
            "bias_align_frames",
            [q_sym, q_dot_sym],
            [bias_vector],
            ["q", "q_dot"],
            ["bias_align_frames"],
        ).expand()

        return constraints_func, constraints_jacobian_func, bias_func

    @staticmethod
    def align_frames2(
        model,
        frame_1_idx: int = 0,
        frame_2_idx: int = 1,
        local_frame_idx: int | None = None,
        relative_rotation: DM = DM.eye(3),
    ) -> tuple[Function, Function, Function]:
        # Symbolic variables
        q_sym = MX.sym("q", model.nb_q, 1)  # generalized coordinates
        q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)  # velocities
        q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)  # accelerations
        parameters = model.parameters  # optional model parameters

        # Homogeneous transformation matrices of the two frames
        # Global homogeneous matrices (4×4) of the two frames
        # T1_glob = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)  # shape (4,4)
        # T2_glob = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)  # shape (4,4)

        # If a *local* reference frame is requested we first bring the two frames
        # into that local frame (identical to the logic used for the marker
        # constraint).
        if local_frame_idx is not None:
            # Get the rotation matrix of the local frame in the global frame
            R_loc_glob = model.homogeneous_matrices_in_global(segment_index=local_frame_idx)(q_sym, parameters)[:3, :3]

            # Get the rotation matrices of the two frames in the global frame
            R1_glob = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)[:3, :3]
            R2_glob = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)[:3, :3]

            # Transform the rotation matrices into the local frame
            R1 = R_loc_glob.T @ R1_glob
            R2 = R_loc_glob.T @ R2_glob
        else:
            # If no local frame is specified, use the global frame
            R1 = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)[:3, :3]
            R2 = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)[:3, :3]

        # Relative rotation: R_rel = R1ᵀ·R2    (frame‑1 → frame‑2)
        R_rel = R1.T @ R2  # still a symbolic 3×3 matrix

        # Error in relative rotation: R_rel - R_desired
        # R_error = R_rel @ relative_rotation.T
        R_err = relative_rotation.T @ R_rel

        # Minimal set of scalar constraints (3 equations)
        # The skew‑symmetric part of a proper rotation is zero when the angle is zero:
        #   S = (R_rel - R_relᵀ) / 2  →  S = 0   ⇔   ω = 0, θ = 0
        # We vectorise the three independent components of S:
        #    S_21, S_31, S_32
        # (any consistent ordering works, we keep the same order used in the
        #  analytical derivation of the constraint in the OP.)
        cos_theta = (trace(R_err) - 1) / 2

        # Clip to the admissible interval to avoid acos‑NaN caused by rounding
        cos_theta_clip = fmax(fmin(cos_theta, 1.0), -1.0)
        theta = acos(cos_theta_clip)  # 0 ≤ θ ≤ π

        # Scaling factor = θ / (2·sinθ) – robust for θ≈0 and θ≈π
        smooth_transition = 1e-1  # tolerance for singular limits
        transition_epsilon = 2e-2
        hard_switch = (smooth_transition - transition_epsilon) / 2
        is_small = theta < hard_switch

        # regular case
        sin_theta = sin(theta)
        factor_regular = theta / sin_theta  # normal case
        # factor_small = MX(0.5)  # limit θ→0   (θ/(2·sinθ) → ½)

        # small theta case
        theta_small = sqrt(fmax(3 - trace(R_rel), 0) + 1e-12)  # using the first-order expansion of arccos
        theta_over_sintheta_taylor = (
            1 + theta_small**2 / 6 + 7 * theta_small**4 / 360 + 31 * theta_small**6 / 15120
        )  # using the Taylor expansion

        # factor = if_else(is_small, theta_over_sintheta_taylor, factor_regular)
        # t = fmin(fmax(0, (theta - epsilon / 2) / (epsilon / 2)), 1)
        # s = t * t * (3 - 2 * t)  # Smoothstep
        # factor = s * theta_over_sintheta_taylor + (1 - s) * factor_regular
        t = fmin(
            fmax(0, (theta - (smooth_transition - transition_epsilon / 2)) / transition_epsilon), 1
        )  # Shift and scale
        s = t * t * (3 - 2 * t)  # Smoothstep
        factor_smooth = (1 - s) * theta_over_sintheta_taylor + s * factor_regular

        factor = if_else(is_small, theta_over_sintheta_taylor, factor_smooth)

        S = factor * (R_err - R_err.T) / 2.0  # 3×3 MX (or dummy for π)

        constraint = vertcat(S[2, 1], -S[2, 0], S[1, 0])  # r32 - r23  # r13 - r31  # r21 - r12

        # Jacobian and second derivative (CasADi)
        constraints_jacobian = jacobian(constraint, q_sym)

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

        bias_vector = compute_bias_vector(constraints_jacobian, q_sym, q_dot_sym)

        bias_func = Function(
            "bias_align_frames",
            [q_sym, q_dot_sym],
            [bias_vector],
            ["q", "q_dot"],
            ["bias_align_frames"],
        ).expand()

        return constraints_func, constraints_jacobian_func, bias_func

    @staticmethod
    def align_frames3(
        model,
        frame_1_idx: int = 0,
        frame_2_idx: int = 1,
        local_frame_idx: int | None = None,
        relative_rotation: DM = DM.eye(3),
    ) -> tuple[Function, Function, Function]:
        # Symbolic variables
        q_sym = MX.sym("q", model.nb_q, 1)  # generalized coordinates
        q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)  # velocities
        q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)  # accelerations
        parameters = model.parameters  # optional model parameters

        # Homogeneous transformation matrices of the two frames
        # Global homogeneous matrices (4×4) of the two frames
        # T1_glob = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)  # shape (4,4)
        # T2_glob = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)  # shape (4,4)

        # If a *local* reference frame is requested we first bring the two frames
        # into that local frame (identical to the logic used for the marker
        # constraint).
        if local_frame_idx is not None:
            # Get the rotation matrix of the local frame in the global frame
            R_loc_glob = model.homogeneous_matrices_in_global(segment_index=local_frame_idx)(q_sym, parameters)[:3, :3]

            # Get the rotation matrices of the two frames in the global frame
            R1_glob = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)[:3, :3]
            R2_glob = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)[:3, :3]

            # Transform the rotation matrices into the local frame
            R1 = R_loc_glob.T @ R1_glob
            R2 = R_loc_glob.T @ R2_glob
        else:
            # If no local frame is specified, use the global frame
            R1 = model.homogeneous_matrices_in_global(segment_index=frame_1_idx)(q_sym, parameters)[:3, :3]
            R2 = model.homogeneous_matrices_in_global(segment_index=frame_2_idx)(q_sym, parameters)[:3, :3]

        # Relative rotation: R_rel = R1ᵀ·R2    (frame‑1 → frame‑2)
        R_rel = R1.T @ R2  # still a symbolic 3×3 matrix

        # Error in relative rotation: R_rel - R_desired
        # R_error = R_rel @ relative_rotation.T
        R_err = relative_rotation.T @ R_rel

        # Minimal set of scalar constraints (3 equations)
        # The skew‑symmetric part of a proper rotation is zero when the angle is zero:
        #   S = (R_rel - R_relᵀ) / 2  →  S = 0   ⇔   ω = 0, θ = 0
        # We vectorise the three independent components of S:
        #    S_21, S_31, S_32
        # (any consistent ordering works, we keep the same order used in the
        #  analytical derivation of the constraint in the OP.)
        cos_theta = (trace(R_err) - 1) / 2

        # Clip to the admissible interval to avoid acos‑NaN caused by rounding
        cos_theta_clip = fmax(fmin(cos_theta, 1.0), -1.0)
        theta = acos(cos_theta_clip)  # 0 ≤ θ ≤ π

        # Scaling factor = θ / (2·sinθ) – robust for θ≈0 and θ≈π
        smooth_transition = 1e-1  # tolerance for singular limits
        transition_epsilon = 2e-2
        hard_switch = (smooth_transition - transition_epsilon) / 2
        is_small = theta < hard_switch

        smooth_transition_pi = pi - 1e-3
        transition_epsilon_pi = 2e-4
        near_pi_low_transition = 2e-3
        below_pi_lower_trans = fabs(theta - pi) > near_pi_low_transition
        hard_switch_pi = 1e-4
        is_near_pi = fabs(theta - pi) < hard_switch_pi

        # regular case
        sin_theta = sin(fmax(theta, 1e-6))
        factor_regular = theta / sin_theta  # normal case
        # factor_small = MX(0.5)  # limit θ→0   (θ/(2·sinθ) → ½)

        # small theta case
        theta_small = sqrt(3 - trace(R_err) + 1e-15)  # using the first-order expansion of arccos
        theta_over_sintheta_taylor = (
            1 + theta_small**2 / 6 + 7 * theta_small**4 / 360 + 31 * theta_small**6 / 15120
        )  # using the Taylor expansion

        # factor = if_else(is_small, theta_over_sintheta_taylor, factor_regular)
        # t = fmin(fmax(0, (theta - epsilon / 2) / (epsilon / 2)), 1)
        # s = t * t * (3 - 2 * t)  # Smoothstep
        # factor = s * theta_over_sintheta_taylor + (1 - s) * factor_regular
        t = fmin(
            fmax(0, (theta - (smooth_transition - transition_epsilon / 2)) / transition_epsilon), 1
        )  # Shift and scale
        s = t * t * (3 - 2 * t)  # Smoothstep
        factor_smooth = (1 - s) * theta_over_sintheta_taylor + s * factor_regular

        factor = if_else(is_small, theta_over_sintheta_taylor, factor_smooth)

        S = factor * (R_err - R_err.T) / 2.0  # 3×3 MX (or dummy for π)

        # 180° fallback – extract the axis from (R_err + I)
        def normalize(v):
            """Normalise a vector (works for MX, SX, DM)."""
            n = norm_2(v)
            return if_else(fabs(n) < 1e-12, v, v / n)

        # Compute the val_singular values
        val_singular = vertcat(
            2 * R_err[0, 0] - trace(R_err) + 1, 2 * R_err[1, 1] - trace(R_err) + 1, 2 * R_err[2, 2] - trace(R_err) + 1
        )

        # Choose the column with the largest val_singular value (most numerically stable)
        axis_idx = if_else(
            val_singular[0] >= val_singular[1],
            if_else(val_singular[0] >= val_singular[2], 0, 2),
            if_else(val_singular[1] >= val_singular[2], 1, 2),
        )
        # Compute the i-th component of the axis
        omega_i = if_else(
            axis_idx == 0,
            sqrt((R_err[0, 0] + 1) / 2 + 1e-8),
            if_else(
                axis_idx == 1,
                sqrt((R_err[1, 1] + 1) / 2 + 1e-8),
                sqrt((R_err[2, 2] + 1) / 2 + 1e-8),
            ),
        )

        # Compute the sign of the i-th component
        sign = if_else(
            axis_idx == 0,
            if_else(R_err[2, 1] - R_err[1, 2] > 0, 1, -1),
            if_else(
                axis_idx == 1,
                if_else(R_err[0, 2] - R_err[2, 0] > 0, 1, -1),
                if_else(R_err[1, 0] - R_err[0, 1] > 0, 1, -1),
            ),
        )

        # Compute the i-th component with the correct sign
        omega_i = sign * omega_i

        # Compute the other components of the axis
        omega_j = if_else(
            axis_idx == 0,
            vertcat((R_err[0, 1] + R_err[1, 0]) / (4 * omega_i), (R_err[0, 2] + R_err[2, 0]) / (4 * omega_i)),
            if_else(
                axis_idx == 1,
                vertcat((R_err[1, 0] + R_err[0, 1]) / (4 * omega_i), (R_err[1, 2] + R_err[2, 1]) / (4 * omega_i)),
                vertcat((R_err[2, 0] + R_err[0, 2]) / (4 * omega_i), (R_err[2, 1] + R_err[1, 2]) / (4 * omega_i)),
            ),
        )

        # Assemble the axis vector
        axis = if_else(
            axis_idx == 0,
            vertcat(omega_i, omega_j[0], omega_j[1]),
            if_else(
                axis_idx == 1,
                vertcat(omega_j[0], omega_i, omega_j[1]),
                vertcat(omega_j[0], omega_j[1], omega_i),
            ),
        )

        # Normalize the axis vector
        axis = normalize(axis)

        # Compute the angle
        # theta = 2 * atan2(
        #     norm_2(axis),
        #     (R_err[2, 1] - R_err[1, 2]) / (4 * axis[0]),
        # )

        theta = 2 * atan2(
            norm_2(axis),
            if_else(
                axis_idx == 0,
                (R_err[2, 1] - R_err[1, 2]) / (4 * axis[0]),
                if_else(
                    axis_idx == 1,
                    (R_err[0, 2] - R_err[2, 0]) / (4 * axis[1]),
                    (R_err[1, 0] - R_err[0, 1]) / (4 * axis[2]),
                ),
            ),
        )

        t_pi = fmin(fmax(0, (theta - (smooth_transition_pi - transition_epsilon_pi / 2)) / transition_epsilon_pi), 1)
        s_pi = t_pi * t_pi * (3 - 2 * t_pi)  # Smoothstep

        # Constraints
        regular_constraint = vertcat(S[2, 1], -S[2, 0], S[1, 0])
        # near_pi_constraint = pi * normalize(axis)
        near_pi_constraint = theta * axis

        constraint_smooth_pi = vertcat(
            (1 - s_pi) * S[2, 1] + s_pi * near_pi_constraint[0],
            -(1 - s_pi) * S[2, 0] - s_pi * near_pi_constraint[1],
            (1 - s_pi) * S[1, 0] + s_pi * near_pi_constraint[2],
        )

        # constraint = if_else(
        #     is_near_pi, near_pi_constraint, if_else(below_pi_lower_trans, regular_constraint, constraint_smooth_pi)
        # )
        constraint = if_else(
            below_pi_lower_trans, regular_constraint, if_else(is_near_pi, near_pi_constraint, constraint_smooth_pi)
        )
        # constraint = if_else(is_near_pi, near_pi_constraint, regular_constraint)

        # Jacobian and second derivative (CasADi)
        constraints_jacobian = jacobian(constraint, q_sym)

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

        bias_vector = compute_bias_vector(constraints_jacobian, q_sym, q_dot_sym)

        bias_func = Function(
            "bias_align_frames",
            [q_sym, q_dot_sym],
            [bias_vector],
            ["q", "q_dot"],
            ["bias_align_frames"],
        ).expand()

        return constraints_func, constraints_jacobian_func, bias_func

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

        bias_vector = compute_bias_vector(constraints_jacobian, q_sym, q_dot_sym)

        bias_func = Function(
            "rigid_contacts_bias",
            [q_sym, q_dot_sym, parameters],
            [bias_vector],
            ["q", "q_dot", "parameters"],
            ["rigid_contacts_bias"],
        ).expand()

        return constraints_func, constraints_jacobian_func, bias_func


class HolonomicConstraintsList(OptionDict):
    """
    Container for managing multiple holonomic constraints.

    This class stores a collection of holonomic constraints that can be applied to a
    HolonomicBiorbdModel. Each constraint is identified by a unique key and includes
    the constraint function along with any additional parameters.

    Methods
    -------
    add(key, constraints_fcn, **extra_arguments)
        Add a new holonomic constraint to the collection.

    Examples
    --------
    >>> constraints = HolonomicConstraintsList()
    >>> constraints.add(
    ...     "marker_constraint",
    ...     HolonomicConstraintsFcn.superimpose_markers,
    ...     marker_1="hand",
    ...     marker_2="target"
    ... )

    See Also
    --------
    HolonomicConstraintsFcn : Factory class for creating constraint functions
    HolonomicBiorbdModel.set_holonomic_configuration : Method that uses this list
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


def compute_bias_vector(constraints_jacobian: MX, q_sym: MX, q_dot_sym: MX) -> MX:
    """
    Compute the bias vector of the holonomic constraint acceleration equation using the Hessian method.

    For holonomic constraints Φ(q) = 0, the acceleration-level constraint is:
        Φ̈ = J(q)q̈ + J̇(q)q̇ = 0

    This function computes the bias term J̇(q)q̇ through the Hessian tensor of the constraint.

    Mathematical Background
    -----------------------
    The time derivative of the constraint Jacobian can be computed using the chain rule:
        J̇ = ∂J/∂q · q̇ = Σₖ (∂J/∂qₖ) q̇ₖ

    where ∂J/∂q is a 3rd-order tensor (Hessian) with dimensions (m × n × n):
        - m: number of constraints
        - n: number of generalized coordinates

    For each constraint i, the bias term is computed as a quadratic form:
        (J̇q̇)ᵢ = Σⱼ Σₖ (∂Jᵢⱼ/∂qₖ) q̇ₖ q̇ⱼ = q̇ᵀ Hᵢ q̇

    where Hᵢ is the Hessian matrix of the i-th constraint Jacobian row.

    Implementation Note
    -------------------
    This method has O(n²) complexity per constraint. For systems where efficiency is critical,
    spatial algebra methods (O(n)) could be considered, though the Hessian approach is more
    straightforward for arbitrary holonomic constraints.

    Parameters
    ----------
    constraints_jacobian : MX
        The Jacobian matrix of the constraints, shape (m × n), where:
            J = ∂Φ/∂q
    q_sym : MX
        Symbolic variable for generalized coordinates, shape (n × 1).
        Required to compute the Hessian ∂J/∂q through symbolic differentiation.
    q_dot_sym : MX
        Symbolic variable for generalized velocities, shape (n × 1).
        Used in the quadratic form q̇ᵀ H q̇.

    Returns
    -------
    MX
        The bias vector J̇q̇, shape (m × 1), representing the velocity-dependent
        acceleration terms in the constraint equation.

    See Also
    --------
    HolonomicBiorbdModel.holonomic_constraints_bias : Uses this function to compute bias terms
    HolonomicConstraintsFcn.superimpose_markers : Example constraint that generates bias functions
    HolonomicConstraintsFcn.align_frames : Example constraint that generates bias functions

    References
    ----------
    .. [1] Featherstone, R. (2008). Rigid Body Dynamics Algorithms. Springer.
    .. [2] Docquier, N., Poncelet, A., and Fisette, P. (2013). ROBOTRAN: a powerful symbolic
           generator of multibody models. Mech. Sci., 4, 199–219.
    """
    Jdot_qdot = []
    for i in range(constraints_jacobian.shape[0]):
        hessian = jacobian(constraints_jacobian[i, :], q_sym)
        Jdot_qdot.append(q_dot_sym.T @ hessian @ q_dot_sym)
    Jdot_qdot = vertcat(*Jdot_qdot)
    return Jdot_qdot
