"""
This class contains different holonomic constraint function.
"""

from typing import Any, Callable

from casadi import MX, Function, jacobian, vertcat

from bioptim.models.protocols.biomodel import BioModel
from bioptim.misc.options import OptionDict


class HolonomicConstraintsFcn:
    """
    This class contains different holonomic constraint.
    """

    @staticmethod
    def superimpose_markers(
        model: BioModel,
        marker_1: str,
        marker_2: str = None,
        index: slice = slice(0, 3),
        local_frame_index: int = None,
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
        parameters = model.parameters

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

        # the double derivative of the constraint
        constraints_double_derivative = (
            constraints_jacobian_func(q_sym) @ q_ddot_sym + constraints_jacobian_func(q_dot_sym) @ q_dot_sym
        )

        constraints_double_derivative_func = Function(
            "holonomic_constraints_double_derivative",
            [q_sym, q_dot_sym, q_ddot_sym, parameters],
            [constraints_double_derivative],
            ["q", "q_dot", "q_ddot", "parameters"],
            ["holonomic_constraints_double_derivative"],
        ).expand()

        return constraints_func, constraints_jacobian_func, constraints_double_derivative_func

    @staticmethod
    def rigid_contacts(
        model: BioModel,
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
            ["q", "q_dot", "q_ddot"],
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

    def add(self, key: str, constraints_fcn: Callable, **kwargs: Any):
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        key: str
            The name of the optimization variable
        constraints_fcn: HolonomicConstraintsFcn
            The function that generates the holonomic constraints
        """
        constraints, constraints_jacobian, constraints_double_derivative = constraints_fcn(**kwargs)
        super(HolonomicConstraintsList, self)._add(
            key=key,
            constraints=constraints,
            constraints_jacobian=constraints_jacobian,
            constraints_double_derivative=constraints_double_derivative,
        )
