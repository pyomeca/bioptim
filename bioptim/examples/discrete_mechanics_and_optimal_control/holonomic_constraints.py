"""
This class contains different holonomic constraint function.
"""
from casadi import MX, Function, jacobian, vertcat

from bioptim import BiorbdModelHolonomic


class HolonomicConstraintFcn:
    """
    This class contains different holonomic constraint.
    """

    @staticmethod
    def superimpose_markers(
            biorbd_model: BiorbdModelHolonomic, marker_1: str, marker_2: str = None, index: slice = slice(0, 3),
            local_frame_index: int = None
    ) -> tuple[Function, Function, Function]:
        """
        Generate the constraint functions to superimpose two markers.

        Parameters
        ----------
        biorbd_model: BiorbdModelCustomHolonomic
            The biorbd model.
        marker_1: str
            The name of the first marker.
        marker_2: str
            The name of the second marker. If None, the constraint will be to superimpose the first marker to the
            origin.
        index: slice
            The index of the markers to superimpose.


        Returns
        -------
        The constraint function, its jacobian and its double derivative.
        """

        # symbolic variables to create the functions
        q_sym = MX.sym("q", biorbd_model.nb_q, 1)
        q_dot_sym = MX.sym("q_dot", biorbd_model.nb_qdot, 1)
        q_ddot_sym = MX.sym("q_ddot", biorbd_model.nb_qdot, 1)

        # symbolic markers in global frame
        marker_1_sym = biorbd_model.marker(q_sym, index=biorbd_model.marker_index(marker_1))
        if marker_2 is not None:
            marker_2_sym = biorbd_model.marker(q_sym, index=biorbd_model.marker_index(marker_2))

        else:
            marker_2_sym =MX([0, 0, 0])

            # if local frame is provided, the markers are expressed in the same local frame
        if local_frame_index is not None:
            jcs_t = biorbd_model.homogeneous_matrices_in_global(q_sym, local_frame_index, inverse=True)
            marker_1_sym = (jcs_t.to_mx() @ vertcat(marker_1_sym, 1))[:3]
            marker_2_sym = (jcs_t.to_mx() @ vertcat(marker_2_sym, 1))[:3]

        # the constraint is the distance between the two markers, set to zero
        constraint = (marker_1_sym - marker_2_sym)[index]
        # the jacobian of the constraint
        constraint_jacobian = jacobian(constraint, q_sym)

        constraint_func = Function(
            "holonomic_constraint",
            [q_sym],
            [constraint],
            ["q"],
            ["holonomic_constraint"],
        ).expand()

        constraint_jacobian_func = Function(
            "holonomic_constraint_jacobian",
            [q_sym],
            [constraint_jacobian],
            ["q"],
            ["holonomic_constraint_jacobian"],
        ).expand()

        # the double derivative of the constraint
        constraint_double_derivative = (
                constraint_jacobian_func(q_sym) @ q_ddot_sym + constraint_jacobian_func(q_dot_sym) @ q_dot_sym
        )

        constraint_double_derivative_func = Function(
            "holonomic_constraint_double_derivative",
            [q_sym, q_dot_sym, q_ddot_sym],
            [constraint_double_derivative],
            ["q", "q_dot", "q_ddot"],
            ["holonomic_constraint_double_derivative"],
        ).expand()

        return constraint_func, constraint_jacobian_func, constraint_double_derivative_func
