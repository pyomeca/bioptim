from typing import Any
from math import inf
import inspect

import biorbd_casadi as biorbd
from casadi import horzcat, vertcat, SX, Function, atan2, dot, cross, sqrt

from .penalty_option import PenaltyOption
from .penalty_controller import PenaltyController
from ..misc.enums import Node, Axis, ControlType, IntegralApproximation


class PenaltyFunctionAbstract:
    """
    Internal implementation of the penalty functions

    Methods
    -------
    add(ocp: OptimalControlProgram, nlp: NonLinearProgram)
        Add a new penalty to the list (abstract)
    set_idx_columns(penalty: PenaltyOption, controller: PenaltyController, index: str | int | list | tuple, _type: str)
        Simple penalty.cols setter for marker index and names
    set_axes_rows(penalty: PenaltyOption, axes: list | tuple)
        Simple penalty.cols setter for marker index and names
    _check_idx(name: str, elements: list | tuple | int, max_n_elements: int = inf, min_n_elements: int = 0)
        Generic sanity check for requested dimensions.
        If the function returns, everything is okay
    validate_penalty_time_index(penalty: PenaltyOption, controller: PenaltyController)
        Check for any non sense in the requested times for the penalty. Raises an error if so
    get_type()
        Returns the type of the penalty (abstract)
    get_dt(nlp)
        Return the dt of the penalty (abstract
    """

    class Functions:
        """
        Implementation of all the generic penalty functions
        """

        @staticmethod
        def minimize_states(penalty: PenaltyOption, controller: PenaltyController, key: str):
            """
            Minimize the states variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            key: str
                The name of the state to minimize
            """

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            if (
                penalty.integration_rule != IntegralApproximation.TRAPEZOIDAL
                and penalty.integration_rule != IntegralApproximation.TRUE_TRAPEZOIDAL
            ):
                # todo: for trapezoidal integration
                penalty.add_target_to_plot(controller=controller, combine_to=f"{key}_states")
            penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

            return controller.states[key].cx_start

        @staticmethod
        def minimize_controls(penalty: PenaltyOption, controller: PenaltyController, key: str):
            """
            Minimize the joint torque part of the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            key: str
                The name of the controls to minimize
            """

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            if penalty.integration_rule == IntegralApproximation.RECTANGLE:
                # TODO: for trapezoidal integration (This should not be done here but in _set_penalty_function)
                penalty.add_target_to_plot(controller=controller, combine_to=f"{key}_controls")
            penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

            return controller.controls[key].cx_start

        @staticmethod
        def minimize_fatigue(penalty: PenaltyOption, controller: PenaltyController, key: str):
            """
            Minimize the states variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            key: str
                The name of the state to minimize
            """

            return PenaltyFunctionAbstract.Functions.minimize_states(penalty, controller, f"{key}_mf")

        @staticmethod
        def minimize_markers(
            penalty: PenaltyOption,
            controller: PenaltyController,
            marker_index: tuple | list | int | str = None,
            axes: tuple | list = None,
            reference_jcs: str | int = None,
        ):
            """
            Minimize a marker set.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            marker_index: tuple | list | int | str
                The index of markers to minimize, can be int or str.
                penalty.cols should not be defined if marker_index is defined
            axes: list
                The axes to minimize, default XYZ
            reference_jcs: int | str
                The index or name of the segment to use as reference. Default [None] is the global coordinate system
            """

            # Adjust the cols and rows
            PenaltyFunctionAbstract.set_idx_columns(penalty, controller, marker_index, "marker")
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.plot_target = False

            # Compute the position of the marker in the requested reference frame (None for global)
            q = controller.states["q"].mx
            model = controller.model
            jcs_t = (
                biorbd.RotoTrans()
                if reference_jcs is None
                else model.homogeneous_matrices_in_global(q, reference_jcs, inverse=True)
            )

            markers = []
            for m in model.markers(q):
                markers_in_jcs = jcs_t.to_mx() @ vertcat(m, 1)
                markers = horzcat(markers, markers_in_jcs[:3])

            markers_objective = controller.mx_to_cx("markers", markers, controller.states["q"])
            return markers_objective

        @staticmethod
        def minimize_markers_velocity(
            penalty: PenaltyOption,
            controller: PenaltyController,
            marker_index: tuple | list | int | str = None,
            axes: tuple | list = None,
            reference_jcs: str | int = None,
        ):
            """
            Minimize a marker set velocity by computing the actual velocity of the markers
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            marker_index: tuple | list | int | str
                The index of markers to minimize, can be int or str.
                penalty.cols should not be defined if marker_index is defined
            axes: tuple | list
                The axes to project on. Default is all axes
            reference_jcs: int | str
                The index or name of the segment to use as reference. Default [None] is the global coordinate system
            """

            # Adjust the cols and rows
            PenaltyFunctionAbstract.set_idx_columns(penalty, controller, marker_index, "marker")
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            # Add the penalty in the requested reference frame. None for global
            q_mx = controller.states["q"].mx
            qdot_mx = controller.states["qdot"].mx

            # todo: return all MX, shouldn't it be a list of MX, I think there is an inconsistency here
            markers = controller.model.marker_velocities(q_mx, qdot_mx, reference_index=reference_jcs)

            markers_objective = controller.mx_to_cx(
                "markers_velocity", markers, controller.states["q"], controller.states["qdot"]
            )
            return markers_objective

        @staticmethod
        def superimpose_markers(
            penalty: PenaltyOption,
            controller: PenaltyController,
            first_marker: str | int,
            second_marker: str | int,
            axes: tuple | list = None,
        ):
            """
            Minimize the distance between two markers
            By default this function is quadratic, meaning that it minimizes distance between them.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            first_marker: str | int
                The name or index of one of the two markers
            second_marker: str | int
                The name or index of one of the two markers
            axes: tuple | list
                The axes to project on. Default is all axes
            """

            first_marker_idx = (
                controller.model.marker_index(first_marker) if isinstance(first_marker, str) else first_marker
            )
            second_marker_idx = (
                controller.model.marker_index(second_marker) if isinstance(second_marker, str) else second_marker
            )
            PenaltyFunctionAbstract._check_idx(
                "marker", [first_marker_idx, second_marker_idx], controller.model.nb_markers
            )
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            diff_markers = controller.model.marker(
                controller.states["q"].mx, second_marker_idx
            ) - controller.model.marker(controller.states["q"].mx, first_marker_idx)

            return controller.mx_to_cx(
                f"diff_markers",
                diff_markers,
                controller.states["q"],
            )

        @staticmethod
        def proportional_states(
            penalty: PenaltyOption,
            controller: PenaltyController,
            key: str,
            first_dof: int,
            second_dof: int,
            coef: float,
            first_dof_intercept: float = 0,
            second_dof_intercept: float = 0,
        ):
            """
            Introduce a proportionality between two variables (e.g. one variable is twice the other)
            By default this function is quadratic, meaning that it minimizes the difference of this proportion.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            key: str
                The name of the state to minimize
            first_dof: int
                The index of the first variable
            second_dof: int
                The index of the second variable
            coef: float
                The proportion coefficient between the two variables
            first_dof_intercept: float
                The intercept of the first variable
            second_dof_intercept: float
                The intercept of the second variable

            Formula = v[first_dof] - first_dof_intercept = coef * (v[second_dof] - second_dof_intercept)
            """

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            states = controller.states[key].cx_start
            return (states[first_dof, :] - first_dof_intercept) - coef * (states[second_dof, :] - second_dof_intercept)

        @staticmethod
        def proportional_controls(
            penalty: PenaltyOption,
            controller: PenaltyController,
            key: str,
            first_dof: int,
            second_dof: int,
            coef: float,
            first_dof_intercept: float = 0,
            second_dof_intercept: float = 0,
        ):
            """
            Introduce a proportionality between two variables (e.g. one variable is twice the other)
            By default this function is quadratic, meaning that it minimizes the difference of this proportion.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            key: str
                The name of the control to minimize
            first_dof: int
                The index of the first variable
            second_dof: int
                The index of the second variable
            coef: float
                The proportion coefficient between the two variables
            first_dof_intercept: float
                The intercept of the first variable
            second_dof_intercept: float
                The intercept of the second variable

            Formula = v[first_dof] - first_dof_intercept = coef * (v[second_dof] - second_dof_intercept)
            """

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            controls = controller.controls[key].cx_start
            return (controls[first_dof, :] - first_dof_intercept) - coef * (
                controls[second_dof, :] - second_dof_intercept
            )

        @staticmethod
        def minimize_qddot(penalty: PenaltyOption, controller: PenaltyController):
            """
            Minimize the states velocity by comparing the state at a node and at the next node.
            By default this function is quadratic, meaning that it minimizes the difference.
            Indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            """

            penalty.quadratic = True

            if "qddot" not in controller.states and "qddot" not in controller.controls:
                return controller.dynamics(
                    controller.states.cx_start, controller.controls.cx_start, controller.parameters.cx_start
                )[controller.states["qdot"].index, :]
            elif "qddot" in controller.states:
                return controller.states["qddot"].cx_start
            elif "qddot" in controller.controls:
                return controller.controls["qddot"].cx_start

        @staticmethod
        def minimize_predicted_com_height(_: PenaltyOption, controller: PenaltyController):
            """
            Minimize the prediction of the center of mass maximal height from the parabolic equation,
            assuming vertical axis is Z (2): CoM_dot[2]**2 / (2 * -g) + com[2]
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            _: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            """

            g = controller.model.gravity[2]
            com = controller.model.center_of_mass(controller.states["q"].mx)
            com_dot = controller.model.center_of_mass_velocity(controller.states["q"].mx, controller.states["qdot"].mx)
            com_height = (com_dot[2] * com_dot[2]) / (2 * -g) + com[2]
            com_height_cx = controller.mx_to_cx(
                "com_height", com_height, controller.states["q"], controller.states["qdot"]
            )
            return com_height_cx

        @staticmethod
        def minimize_com_position(penalty: PenaltyOption, controller: PenaltyController, axes: tuple | list = None):
            """
            Adds the objective that the position of the center of mass of the model should be minimized.
            If no axes is specified, the squared-norm of the center_of_mass's position is minimized.
            Otherwise, the projection of the center_of_mass's position on the specified axes are minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            axes: tuple | list
                The axes to project on. Default is all axes
            """

            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            com_cx = controller.mx_to_cx("com", controller.model.center_of_mass, controller.states["q"])
            return com_cx

        @staticmethod
        def minimize_com_velocity(penalty: PenaltyOption, controller: PenaltyController, axes: tuple | list = None):
            """
            Adds the objective that the velocity of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the center_of_mass's velocity is minimized.
            Otherwise, the projection of the center_of_mass's velocity on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            axes: tuple | list
                The axes to project on. Default is all axes
            """

            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            com_dot_cx = controller.mx_to_cx(
                "com_dot", controller.model.center_of_mass_velocity, controller.states["q"], controller.states["qdot"]
            )
            return com_dot_cx

        @staticmethod
        def minimize_com_acceleration(penalty: PenaltyOption, controller: PenaltyController, axes: tuple | list = None):
            """
            Adds the objective that the velocity of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the center_of_mass's velocity is minimized.
            Otherwise, the projection of the center_of_mass's velocity on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            axes: tuple | list
                The axes to project on. Default is all axes
            """

            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            if "qddot" not in controller.states and "qddot" not in controller.controls:
                com_ddot = controller.model.center_of_mass_acceleration(
                    controller.states["q"].mx,
                    controller.states["qdot"].mx,
                    controller.dynamics(controller.states.mx, controller.controls.mx, controller.parameters.mx)[
                        controller.states["qdot"].index, :
                    ],
                )
                # TODO scaled?
                var = []
                var.extend([controller.states[key] for key in controller.states])
                var.extend([controller.controls[key] for key in controller.controls])
                var.extend([controller.parameters[key] for key in controller.parameters])
                return controller.mx_to_cx("com_ddot", com_ddot, *var)
            else:
                qddot = controller.states["qddot"] if "qddot" in controller.states else controller.controls["qddot"]
                return controller.mx_to_cx(
                    "com_ddot",
                    controller.model.center_of_mass_acceleration,
                    controller.states["q"],
                    controller.states["qdot"],
                    qddot,
                )

        @staticmethod
        def minimize_angular_momentum(penalty: PenaltyOption, controller: PenaltyController, axes: tuple | list = None):
            """
            Adds the objective that the angular momentum of the model in the global reference frame should be minimized.
            If no axis is specified, the three components of the angular momentum are minimized.
            Otherwise, the projection of the angular momentum on the specified axis is minimized.
            By default this function is quadratic, meaning that it minimizes towards zero.
            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            axes: tuple | list
                The axes to project on. Default is all axes
            """
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            angular_momentum_cx = controller.mx_to_cx(
                "angular_momentum",
                controller.model.angular_momentum,
                controller.states["q"],
                controller.states["qdot"],
            )
            return angular_momentum_cx

        @staticmethod
        def minimize_linear_momentum(penalty: PenaltyOption, controller: PenaltyController, axes: tuple | list = None):
            """
            Adds the objective that the linear momentum of the model in the global reference frame should be minimized.
            If no axis is specified, the three components of the linear momentum are minimized.
            Otherwise, the projection of the linear momentum on the specified axis is minimized.
            By default this function is quadratic, meaning that it minimizes towards zero.
            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            axes: tuple | list
                The axes to project on. Default is all axes
            """

            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            com_velocity = controller.mx_to_cx(
                "com_velocity",
                controller.model.center_of_mass_velocity,
                controller.states["q"],
                controller.states["qdot"],
            )
            if isinstance(com_velocity, SX):
                mass = Function("mass", [], [controller.model.mass]).expand()
                mass = mass()["o0"]
            else:
                mass = controller.model.mass
            linear_momentum_cx = com_velocity * mass
            return linear_momentum_cx

        @staticmethod
        def minimize_contact_forces(
            penalty: PenaltyOption, controller: PenaltyController, contact_index: tuple | list | int | str = None
        ):
            """
            Minimize the contact forces computed from dynamics with contact
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            contact_index: tuple | list
                The index of contact to minimize, must be an int or a list.
                penalty.cols should not be defined if contact_index is defined
            """

            if controller.get_nlp.contact_forces_func is None:
                raise RuntimeError("minimize_contact_forces requires a contact dynamics")

            PenaltyFunctionAbstract.set_axes_rows(penalty, contact_index)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            contact_force = controller.get_nlp.contact_forces_func(
                controller.states.cx_start, controller.controls.cx_start, controller.parameters.cx_start
            )
            return contact_force

        @staticmethod
        def minimize_soft_contact_forces(
            penalty: PenaltyOption, controller: PenaltyController, contact_index: tuple | list | int | str = None
        ):
            """
            Minimize the soft contact forces computed from dynamics with contact
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            contact_index: tuple | list
                The index of contact to minimize, must be an int or a list.
                penalty.cols should not be defined if contact_index is defined
            """

            if controller.get_nlp.soft_contact_forces_func is None:
                raise RuntimeError("minimize_contact_forces requires a soft contact dynamics")

            PenaltyFunctionAbstract.set_axes_rows(penalty, contact_index)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            force_idx = []
            for i_sc in range(controller.model.nb_soft_contacts):
                force_idx.append(3 + (6 * i_sc))
                force_idx.append(4 + (6 * i_sc))
                force_idx.append(5 + (6 * i_sc))
            soft_contact_force = controller.get_nlp.soft_contact_forces_func(
                controller.states.cx_start, controller.controls.cx_start, controller.parameters.cx_start
            )
            return soft_contact_force[force_idx]

        @staticmethod
        def track_segment_with_custom_rt(
            penalty: PenaltyOption, controller: PenaltyController, segment: int | str, rt: int
        ):
            """
            Minimize the difference of the euler angles extracted from the coordinate system of a segment
            and a RT (e.g. IMU). By default this function is quadratic, meaning that it minimizes the difference.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            segment: int | str
                The name or index of the segment
            rt: int
                The index of the RT in the bioMod
            """
            from ..interfaces.biorbd_model import BiorbdModel

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            segment_index = controller.model.segment_index(segment) if isinstance(segment, str) else segment

            if not isinstance(controller.model, BiorbdModel):
                raise NotImplementedError(
                    "The track_segment_with_custom_rt penalty can only be called with a BiorbdModel"
                )
            model: BiorbdModel = controller.model
            r_seg_transposed = model.model.globalJCS(controller.states["q"].mx, segment_index).rot().transpose()
            r_rt = model.model.RT(controller.states["q"].mx, rt).rot()
            angles_diff = biorbd.Rotation.toEulerAngles(r_seg_transposed * r_rt, "zyx").to_mx()

            angle_objective = controller.mx_to_cx(f"track_segment", angles_diff, controller.states["q"])
            return angle_objective

        @staticmethod
        def track_marker_with_segment_axis(
            penalty: PenaltyOption,
            controller: PenaltyController,
            marker: int | str,
            segment: int | str,
            axis: Axis,
        ):
            """
            Track a marker using a segment, that is aligning an axis toward the marker
            By default this function is quadratic, meaning that it minimizes the difference.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            marker: int
                Name or index of the marker to be tracked
            segment: int
                Name or index of the segment to align with the marker
            axis: Axis
                The axis that should be tracking the marker
            """

            if not isinstance(axis, Axis):
                raise RuntimeError("axis must be a bioptim.Axis")

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            marker_idx = controller.model.marker_index(marker) if isinstance(marker, str) else marker
            segment_idx = controller.model.segment_index(segment) if isinstance(segment, str) else segment

            # Get the marker in rt reference frame
            marker = controller.model.marker(controller.states["q"].mx, marker_idx, segment_idx)
            marker_objective = controller.mx_to_cx("marker", marker, controller.states["q"])

            # To align an axis, the other must be equal to 0
            if penalty.rows is not None:
                raise ValueError("rows cannot be defined in track_marker_with_segment_axis")
            penalty.rows = [ax for ax in [Axis.X, Axis.Y, Axis.Z] if ax != axis]

            return marker_objective

        @staticmethod
        def minimize_segment_rotation(
            penalty: PenaltyOption,
            controller: PenaltyController,
            segment: int | str,
            axes: list | tuple = None,
        ):
            """
            Track the orientation of a segment in the global with the sequence XYZ.
            By default, this function is quadratic, meaning that it minimizes towards the target.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            segment: int
                Name or index of the segment to align with the marker
            axes: list | tuple
                The axis that the JCS rotation should be tracked
            """
            from ..interfaces.biorbd_model import BiorbdModel

            if penalty.derivative == True:
                raise RuntimeWarning(
                    "To minimize the velocity of the segment rotation, it would be safer (Euler angles related problems) to use MINIMIZE_SEGMENT_ANGLUAR_VELOCITY instead."
                )

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            segment_idx = controller.model.segment_index(segment) if isinstance(segment, str) else segment

            if not isinstance(controller.model, BiorbdModel):
                raise NotImplementedError("The minimize_segment_rotation penalty can only be called with a BiorbdModel")
            model: BiorbdModel = controller.model
            jcs_segment = model.model.globalJCS(controller.states["q"].mx, segment_idx).rot()
            angles_segment = biorbd.Rotation.toEulerAngles(jcs_segment, "xyz").to_mx()

            if axes is None:
                axes = [Axis.X, Axis.Y, Axis.Z]
            else:
                for ax in axes:
                    if not isinstance(ax, Axis):
                        raise RuntimeError("axes must be a list of bioptim.Axis")

            segment_rotation_objective = controller.mx_to_cx(
                "segment_rotation", angles_segment[axes], controller.states["q"]
            )

            return segment_rotation_objective

        @staticmethod
        def minimize_segment_velocity(
            penalty: PenaltyOption,
            controller: PenaltyController,
            segment: int | str,
            axes: list | tuple = None,
        ):
            """
            Track the orientation of a segment.
            By default, this function is quadratic, meaning that it minimizes towards the target.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            segment: int
                Name or index of the segment to align with the marker
            axes: list | tuple
                The axis that the JCS rotation should be tracked
            """
            from ..interfaces.biorbd_model import BiorbdModel

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            segment_idx = controller.model.segment_index(segment) if isinstance(segment, str) else segment

            if not isinstance(controller.model, BiorbdModel):
                raise NotImplementedError(
                    "The minimize_segments_velocity penalty can only be called with a BiorbdModel"
                )
            model: BiorbdModel = controller.model
            segment_angular_velocity = model.segment_angular_velocity(
                controller.states["q"].mx, controller.states["qdot"].mx, segment_idx
            )

            if axes is None:
                axes = [Axis.X, Axis.Y, Axis.Z]
            else:
                for ax in axes:
                    if not isinstance(ax, Axis):
                        raise RuntimeError("axes must be a list of bioptim.Axis")

            segment_velocity_objective = controller.mx_to_cx(
                "segment_velocity",
                segment_angular_velocity[axes],
                controller.states["q"],
                controller.states["qdot"],
            )

            return segment_velocity_objective

        @staticmethod
        def track_vector_orientations_from_markers(
            penalty: PenaltyOption,
            controller: PenaltyController,
            vector_0_marker_0: int | str,
            vector_0_marker_1: int | str,
            vector_1_marker_0: int | str,
            vector_1_marker_1: int | str,
        ):
            """
            Aligns two vectors together.
            The first vector is defined by vector_0_marker_1 - vector_0_marker_0.
            The second vector is defined by vector_1_marker_1 - vector_1_marker_0.
            Note that is minimizes the angle between the two vectors, thus it is not possible ti specify an axis.
            By default, this function is quadratic, meaning that it minimizes the angle between the two vectors.
            WARNING: please be careful as there is a discontinuity when the two vectors are orthogonal.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            vector_0_marker_0: int | str
                Name or index of the first marker of the first vector
            vector_0_marker_1: int | str
                Name or index of the second marker of the first vector
            vector_1_marker_0: int | str
                Name or index of the first marker of the second vector
            vector_1_marker_1: int | str
                Name or index of the second marker of the second vector
            """

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            vector_0_marker_0_idx = (
                controller.model.marker_index(vector_0_marker_0)
                if isinstance(vector_0_marker_0, str)
                else vector_0_marker_0
            )
            vector_0_marker_1_idx = (
                controller.model.marker_index(vector_0_marker_1)
                if isinstance(vector_0_marker_1, str)
                else vector_0_marker_1
            )
            vector_1_marker_0_idx = (
                controller.model.marker_index(vector_1_marker_0)
                if isinstance(vector_1_marker_0, str)
                else vector_1_marker_0
            )
            vector_1_marker_1_idx = (
                controller.model.marker_index(vector_1_marker_1)
                if isinstance(vector_1_marker_1, str)
                else vector_1_marker_1
            )

            vector_0_marker_0_position = controller.model.marker(controller.states["q"].mx, vector_0_marker_0_idx)
            vector_0_marker_1_position = controller.model.marker(controller.states["q"].mx, vector_0_marker_1_idx)
            vector_1_marker_0_position = controller.model.marker(controller.states["q"].mx, vector_1_marker_0_idx)
            vector_1_marker_1_position = controller.model.marker(controller.states["q"].mx, vector_1_marker_1_idx)

            vector_0 = vector_0_marker_1_position - vector_0_marker_0_position
            vector_1 = vector_1_marker_1_position - vector_1_marker_0_position
            cross_prod = cross(vector_0, vector_1)
            cross_prod_norm = sqrt(cross_prod[0] ** 2 + cross_prod[1] ** 2 + cross_prod[2] ** 2)
            out = atan2(cross_prod_norm, dot(vector_0, vector_1))

            return controller.mx_to_cx("vector_orientations_difference", out, controller.states["q"])

        @staticmethod
        def continuity(penalty: PenaltyOption, controller: PenaltyController | list):
            if controller.control_type in (ControlType.CONSTANT, ControlType.NONE):
                u = controller.controls.cx_start
            elif controller.control_type == ControlType.LINEAR_CONTINUOUS:
                # TODO: For cx_end take the previous node
                u = horzcat(controller.controls.cx_start, controller.controls.cx_end)
            else:
                raise NotImplementedError(f"Dynamics with {controller.control_type} is not implemented yet")

            if isinstance(penalty.node, (list, tuple)) and len(penalty.node) != 1:
                raise RuntimeError("continuity should be called one node at a time")

            penalty.expand = controller.get_nlp.dynamics_type.expand

            if len(penalty.node_idx) > 1 and not controller.ocp.assume_phase_dynamics:
                raise NotImplementedError(
                    f"Length of node index superior to 1 is not implemented yet,"
                    f" actual length {len(penalty.node_idx[0])} "
                )

            continuity = controller.states.cx_end
            if controller.get_nlp.ode_solver.is_direct_collocation:
                cx = horzcat(*([controller.states.cx_start] + controller.states.cx_intermediates_list))
                continuity -= controller.integrate(x0=cx, p=u, params=controller.parameters.cx_start)["xf"]
                continuity = vertcat(
                    continuity,
                    controller.integrate(x0=cx, p=u, params=controller.parameters.cx_start)["defects"],
                )
                penalty.integrate = True

            else:
                continuity -= controller.integrate(
                    x0=controller.states.cx_start, p=u, params=controller.parameters.cx_start
                )["xf"]

            penalty.explicit_derivative = True
            penalty.multi_thread = True

            return continuity

        @staticmethod
        def custom(penalty: PenaltyOption, controller: PenaltyController | list, **parameters: Any):
            """
            A user defined penalty function

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            parameters: dict
                Any parameters that should be pass to the custom function
            """

            invalid_keywords = [
                "phase",
                "list_index",
                "name",
                "type",
                "params",
                "node",
                "quadratic",
                "index",
                "target",
                "min_bound",
                "max_bound",
                "function",
                "weighted_function",
                "custom_function",
                "weight",
                "expand",
            ]
            for keyword in inspect.signature(penalty.custom_function).parameters:
                if keyword in invalid_keywords:
                    raise TypeError(f"{keyword} is a reserved word and cannot be used in a custom function signature")

            val = penalty.custom_function(controller, **parameters)
            if isinstance(val, (list, tuple)):
                if (hasattr(penalty, "min_bound") and penalty.min_bound is not None) or (
                    hasattr(penalty, "max_bound") and penalty.max_bound is not None
                ):
                    raise RuntimeError(
                        "You cannot have non linear bounds for custom constraints and min_bound or max_bound defined"
                    )
                penalty.min_bound = val[0]
                penalty.max_bound = val[2]
                val = val[1]

            return val

    @staticmethod
    def add(ocp, nlp):
        """
        Add a new penalty to the list (abstract)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """
        raise RuntimeError("add cannot be called from an abstract class")

    @staticmethod
    def set_idx_columns(
        penalty: PenaltyOption, controller: PenaltyController, index: str | int | list | tuple, _type: str
    ):
        """
        Simple penalty.cols setter for marker index and names

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        controller: PenaltyController
            The penalty node elements
        index: str | int | list | tuple
            The marker to index
        _type: str
            The type of penalty (for raise error message purpose)
        """

        if penalty.cols is not None and index is not None:
            raise ValueError(f"It is not possible to define cols and {_type}_index since they are the same variable")

        penalty.cols = index if index is not None else penalty.cols
        if penalty.cols is not None:
            penalty.cols = [penalty.cols] if not isinstance(penalty.cols, (tuple, list)) else penalty.cols
            # Convert to int if it is str
            if _type == "marker":
                penalty.cols = [
                    cols if isinstance(cols, int) else controller.model.marker_index(cols) for cols in penalty.cols
                ]

    @staticmethod
    def set_axes_rows(penalty: PenaltyOption, axes: list | tuple):
        """
        Simple penalty.cols setter for marker index and names

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        axes: list | tuple
            The marker to index
        """

        if penalty.rows is not None and axes is not None:
            raise ValueError("It is not possible to define rows and axes since they are the same variable")
        penalty.rows = axes if axes is not None else penalty.rows

    @staticmethod
    def _check_idx(name: str, elements: list | tuple | int, max_n_elements: int = inf, min_n_elements: int = 0):
        """
        Generic sanity check for requested dimensions.
        If the function returns, everything is okay

        Parameters
        name: str
            Name of the element
        elements: list | tuple | int
            Index of the slicing of the array variable
        max_n_elements: int
            The maximal shape of the element
        min_n_elements: int
            The maximal shape of the element
        """

        if not isinstance(elements, (list, tuple)):
            elements = (elements,)
        for element in elements:
            if not isinstance(element, int):
                raise RuntimeError(f"{element} is not a valid index for {name}, it must be an integer")
            if element < min_n_elements or element >= max_n_elements:
                raise RuntimeError(
                    f"{element} is not a valid index for {name}, it must be between "
                    f"{min_n_elements} and {max_n_elements - 1}."
                )

    @staticmethod
    def validate_penalty_time_index(penalty: PenaltyOption, controller: PenaltyController):
        """
        Check for any nonsense in the requested times for the penalty. Raises an error if so

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        controller: PenaltyController
            The penalty node elements
        """

        func = penalty.type
        node = penalty.node
        # Everything that is suspicious in terms of the span of the penalty function ca be checked here
        if (
            func == PenaltyFunctionAbstract.Functions.minimize_controls
            or func == PenaltyFunctionAbstract.Functions.proportional_controls
            or func == PenaltyFunctionAbstract.Functions.minimize_qddot
        ):
            if node == Node.END or (isinstance(node, int) and node >= controller.ns):
                raise RuntimeError("No control u at last node")

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty (abstract)
        """

        raise RuntimeError("get_type cannot be called from an abstract class")

    @staticmethod
    def get_dt(nlp):
        """
        Return the dt of the penalty (abstract
        """

        raise RuntimeError("get_dt cannot be called from an abstract class")

    @staticmethod
    def penalty_nature() -> str:
        """
        Get the nature of the penalty

        Returns
        -------
        The penalty in str format
        """

        raise RuntimeError("penalty_nature cannot be called from an abstract class")
