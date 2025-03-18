import inspect
from typing import Any

from casadi import horzcat, vertcat, Function, MX_eye, SX_eye, SX, jacobian, trace, if_else
from math import inf

from .penalty_controller import PenaltyController
from .penalty_option import PenaltyOption
from ..misc.enums import Node, Axis, ControlType, QuadratureRule
from ..models.protocols.stochastic_biomodel import StochasticBioModel


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
                penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                and penalty.integration_rule != QuadratureRule.TRAPEZOIDAL
            ):
                # todo: for trapezoidal integration
                penalty.add_target_to_plot(controller=controller, combine_to=f"{key}_states")
            penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

            # TODO: We should scale the target here!
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

            if penalty.integration_rule == QuadratureRule.RECTANGLE_LEFT:
                # TODO: for trapezoidal integration (This should not be done here but in _set_penalty_function)
                penalty.add_target_to_plot(controller=controller, combine_to=f"{key}_controls")
            penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

            # TODO: We should scale the target here!
            return controller.controls[key].cx_start

        @staticmethod
        def minimize_power(
            penalty: PenaltyOption,
            controller: PenaltyController,
            key_control: str,
        ):
            """
            Minimize a product of states variables x control variable; e.g. joint power; muscle power.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            key_controls: str
                The name of the control to select
            """

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

            controls = controller.controls[key_control].cx_start  # select only actuacted states
            if key_control == "tau":
                return controls * controller.qdot
            elif key_control == "muscles":
                muscles_dot = controller.model.muscle_velocity()(
                    controller.q, controller.qdot, controller.parameters.cx
                )

                return controls * muscles_dot

        @staticmethod
        def minimize_algebraic_states(penalty: PenaltyOption, controller: PenaltyController, key: str):
            """
            Minimize a algebraic_states variable.
            By default, this function is quadratic, meaning that it minimizes towards the target.
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
            penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

            return controller.algebraic_states[key].cx_start

        @staticmethod
        def stochastic_minimize_expected_feedback_efforts(penalty: PenaltyOption, controller: PenaltyController):
            """
            This function computes the expected effort due to the motor command and feedback gains for a given sensory noise
            magnitude.
            It is computed as Jacobian(effort, states) @ cov @ Jacobian(effort, states).T +
                                Jacobian(efforts, motor_noise) @ sigma_w @ Jacobian(efforts, motor_noise).T

            Parameters
            ----------
            controller : PenaltyController
                Controller to be used to compute the expected effort.
            """

            if penalty.target is not None:
                raise RuntimeError("It is not possible to use a target for the expected feedback effort.")

            CX_eye = SX_eye if controller.ocp.cx == SX else MX_eye
            sensory_noise_matrix = controller.model.sensory_noise_magnitude * CX_eye(
                controller.model.sensory_noise_magnitude.shape[0]
            )

            # create the casadi function to be evaluated
            # Get the symbolic variables
            if "cholesky_cov" in controller.controls.keys():
                l_cov_matrix = StochasticBioModel.reshape_to_cholesky_matrix(
                    controller.controls["cholesky_cov"].cx_start, controller.model.matrix_shape_cov_cholesky
                )
                cov_matrix = l_cov_matrix @ l_cov_matrix.T
            elif "cov" in controller.controls.keys():
                cov_matrix = StochasticBioModel.reshape_to_matrix(
                    controller.controls["cov"].cx_start, controller.model.matrix_shape_cov
                )
            else:
                raise RuntimeError(
                    "The covariance matrix must be provided in the controls to compute the expected efforts."
                )

            k_matrix = StochasticBioModel.reshape_to_matrix(
                controller.controls["k"].cx_start, controller.model.matrix_shape_k
            )

            # Compute the expected effort
            trace_k_sensor_k = trace(k_matrix @ sensory_noise_matrix @ k_matrix.T)

            e_fb_mx = controller.model.compute_torques_from_noise_and_feedback(
                nlp=controller.get_nlp,
                time=controller.time.cx,
                states=controller.states_scaled.cx,
                controls=controller.controls_scaled.cx,
                parameters=controller.parameters_scaled.cx,
                algebraic_states=controller.algebraic_states_scaled.cx,
                numerical_timeseries=controller.numerical_timeseries.cx,
                sensory_noise=controller.model.sensory_noise_magnitude,
                motor_noise=controller.model.motor_noise_magnitude,
            )

            jac_e_fb_x = jacobian(e_fb_mx, controller.states_scaled.cx)

            jac_e_fb_x_cx = Function(
                "jac_e_fb_x",
                [
                    controller.t_span.cx,
                    controller.states_scaled.cx,
                    controller.controls_scaled.cx,
                    controller.parameters_scaled.cx,
                    controller.algebraic_states_scaled.cx,
                    controller.numerical_timeseries.cx,
                ],
                [jac_e_fb_x],
            )(
                controller.t_span.cx,
                controller.states.cx_start,
                controller.controls.cx_start,
                controller.parameters.cx_start,
                controller.algebraic_states.cx_start,
                controller.numerical_timeseries.cx_start,
            )

            trace_jac_p_jack = trace(jac_e_fb_x_cx @ cov_matrix @ jac_e_fb_x_cx.T)

            expectedEffort_fb_mx = trace_jac_p_jack + trace_k_sensor_k

            return expectedEffort_fb_mx

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
            CX_eye = SX_eye if controller.ocp.cx == SX else MX_eye
            jcs_t = (
                CX_eye(4)
                if reference_jcs is None
                else controller.model.homogeneous_matrices_in_global(reference_jcs, inverse=True)(
                    controller.q, controller.parameters.cx
                )
            )

            markers = controller.model.markers()(controller.q, controller.parameters.cx)
            markers_in_jcs = []
            for i in range(markers.shape[1]):
                marker_in_jcs = jcs_t @ vertcat(markers[:, i], 1)
                markers_in_jcs = horzcat(markers_in_jcs, marker_in_jcs[:3])

            return markers_in_jcs

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
            markers = controller.model.markers_velocities(reference_index=reference_jcs)(
                controller.q, controller.qdot, controller.parameters.cx
            )

            return markers

        @staticmethod
        def minimize_markers_acceleration(
            penalty: PenaltyOption,
            controller: PenaltyController,
            marker_index: tuple | list | int | str = None,
            axes: tuple | list = None,
            reference_jcs: str | int = None,
        ):
            """
            Minimize a marker set acecleration by computing the actual acceleration of the markers
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

            PenaltyFunctionAbstract.set_idx_columns(penalty, controller, marker_index, "marker")
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            qddot = PenaltyFunctionAbstract._get_qddot(controller, "cx")

            markers = controller.model.markers_accelerations(reference_index=reference_jcs)(
                controller.q, controller.qdot, qddot, controller.parameters.cx
            )

            return markers

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

            diff_markers = controller.model.marker(second_marker_idx)(
                controller.q, controller.parameters.cx
            ) - controller.model.marker(first_marker_idx)(controller.q, controller.parameters.cx)

            return diff_markers

        @staticmethod
        def superimpose_markers_velocity(
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

            marker_velocities = controller.model.markers_velocities()(
                controller.q, controller.qdot, controller.parameters.cx
            )
            marker_1 = marker_velocities[:, first_marker_idx]
            marker_2 = marker_velocities[:, second_marker_idx]

            diff_markers = marker_2 - marker_1

            return diff_markers

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

            return PenaltyFunctionAbstract._get_qddot(controller, "cx_start")

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

            g = controller.model.gravity()(controller.parameters.cx)[2]
            com = controller.model.center_of_mass()(controller.q, controller.parameters.cx)
            com_dot = controller.model.center_of_mass_velocity()(
                controller.q, controller.qdot, controller.parameters.cx
            )
            com_height = (com_dot[2] * com_dot[2]) / (2 * -g) + com[2]
            return com_height

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

            return controller.model.center_of_mass()(controller.q, controller.parameters.cx)

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

            return controller.model.center_of_mass_velocity()(controller.q, controller.qdot, controller.parameters.cx)

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

            qddot = PenaltyFunctionAbstract._get_qddot(controller, "cx")

            marker = controller.model.center_of_mass_acceleration()(
                controller.q, controller.qdot, qddot, controller.parameters.cx
            )

            return marker

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

            return controller.model.angular_momentum()(controller.q, controller.qdot, controller.parameters.cx)

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

            com_velocity = controller.model.center_of_mass_velocity()(
                controller.q, controller.qdot, controller.parameters.cx
            )
            mass = controller.model.mass()(controller.parameters.cx)
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
                controller.time.cx,
                controller.states.cx_start,
                controller.controls.cx_start,
                controller.parameters.cx,
                controller.algebraic_states.cx_start,
                controller.numerical_timeseries.cx,
            )
            return contact_force

        @staticmethod
        def minimize_sum_reaction_forces(
            penalty: PenaltyOption,
            controller: PenaltyController,
            contact_index: tuple[str, ...] | tuple[int, ...] | list[str | int],
        ):
            """
            Simulate force plate data from the contact forces computed through the dynamics with contact.
            For example, in the case of gait, it allows to match the experimental ground reaction forces.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
            Please note that the contact index (number of the contact point in the model) must be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            contact_index: tuple[str | int] | list[str | int]
                The index of contact to minimize, must be an int or a list.
                penalty.cols should not be defined if contact_index is defined
            """

            if penalty.target is not None and penalty.target.shape[0] != 3:
                raise RuntimeError("The target for the ground reaction forces must be of size 3 x n_shooting")

            if not isinstance(contact_index, (tuple, list)):
                raise RuntimeError("contact_index must be a tuple or a list")
            if not all(isinstance(contact, (str, int)) for contact in contact_index):
                raise RuntimeError("contact_index must be a tuple or a list of str or int")

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            forces_on_each_point = controller.model.forces_on_each_rigid_contact_point()(
                controller.q, controller.qdot, controller.tau, controller.external_forces, controller.parameters.cx
            )

            total_force = controller.cx.zeros(3, 1)
            for contact in contact_index:
                idx = controller.model.contact_index(contact) if isinstance(contact, str) else contact
                total_force += forces_on_each_point[:, idx]

            return total_force

        @staticmethod
        def minimize_center_of_pressure(
            penalty: PenaltyOption,
            controller: PenaltyController,
            contact_index: tuple[str, ...] | tuple[int, ...] | list[str | int],
        ):
            """
            Simulate the center of pressure from force plate data from the contact forces computed through the dynamics with contact
            It is computed as a mean position of the contact points weighted by the magnitude of the contact force (for each axis independently).
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
            Please note that the contact index of the appropriate foot must be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            contact_index: tuple[str | int] | list[str | int]
                The index of contact to minimize, must be an int or a list.
            """

            if penalty.target is not None and penalty.target.shape[0] != 3:
                raise RuntimeError("The target for the center of pressure must be of size 3 x n_shooting")

            if not isinstance(contact_index, (tuple, list)):
                raise RuntimeError("contact_index must be a tuple or a list")
            if not all(isinstance(contact, (str, int)) for contact in contact_index):
                raise RuntimeError("contact_index must be a tuple or a list of str or int")

            # PenaltyFunctionAbstract.set_axes_rows(penalty, contact_index)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            forces_on_each_point = controller.model.forces_on_each_rigid_contact_point()(
                controller.q, controller.qdot, controller.tau, controller.external_forces, controller.parameters.cx
            )

            total_force = controller.cx.zeros(3, 1)
            position_of_each_point = None
            weighted_sum = controller.cx.zeros(3, 1)
            for contact in contact_index:
                idx = controller.model.contact_index(contact) if isinstance(contact, str) else contact

                # Compute the sum of the forces on the points of interest
                total_force += forces_on_each_point[:, idx]

                # Get the position of all the contact points of interest
                this_contact_position = controller.model.rigid_contact_position(idx)(
                    controller.q, controller.parameters.cx
                )
                position_of_each_point = (
                    horzcat(position_of_each_point, this_contact_position)
                    if position_of_each_point is not None
                    else this_contact_position
                )

                # Weighted sum
                weighted_sum += forces_on_each_point[:, idx] * this_contact_position

            # Compute the mean position weighted by the force magnitude
            center_of_pressure = controller.cx.zeros(3, 1)
            for i_component in range(3):
                # Avoid division by zero if the force is too small
                center_of_pressure[i_component] = if_else(
                    total_force[i_component] ** 2 < 1e-8, 0, weighted_sum[i_component] / total_force[i_component]
                )

            return center_of_pressure

        @staticmethod
        def minimize_contact_forces_end_of_interval(
            penalty: PenaltyOption, controller: PenaltyController, contact_index: tuple | list | int | str = None
        ):
            """
            Minimize the contact forces at the end of the interval computed by integrating the dynamics with contact.
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

            if controller.control_type != ControlType.CONSTANT:
                raise NotImplementedError(
                    f"This constraint is only useful for ControlType.CONSTANT controls. For any other dynamics, you should constraint the contact at the begining of the interval."
                )

            if controller.algebraic_states.cx_start.shape[0] != 0:
                raise NotImplementedError(
                    "This constraint is not implemented for problems with algebraic states as you should provide their dynamics to integrate it as well."
                )

            PenaltyFunctionAbstract.set_axes_rows(penalty, contact_index)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            t_span = controller.t_span.cx
            cx = (
                horzcat(*([controller.states.cx_start] + controller.states.cx_intermediates_list))
                if controller.get_nlp.ode_solver.is_direct_collocation
                else controller.states.cx_start
            )
            end_of_interval_states = controller.integrate(
                t_span=t_span,
                x0=cx,
                u=controller.controls.cx_start,
                p=controller.parameters.cx,
                a=controller.algebraic_states.cx_start,
                d=controller.numerical_timeseries.cx,
            )["xf"]

            contact_force = controller.get_nlp.contact_forces_func(
                controller.time.cx,
                end_of_interval_states,
                controller.controls.cx_start,
                controller.parameters.cx,
                controller.algebraic_states.cx_start,
                controller.numerical_timeseries.cx,
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

            if controller.get_nlp.model.nb_soft_contacts == 0:
                raise RuntimeError("minimize_contact_forces requires a soft contact")

            PenaltyFunctionAbstract.set_axes_rows(penalty, contact_index)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            force_idx = []
            for i_sc in range(controller.model.nb_soft_contacts):
                force_idx.append(3 + (6 * i_sc))
                force_idx.append(4 + (6 * i_sc))
                force_idx.append(5 + (6 * i_sc))

            # Note to the developers: the .expand() should be an option in the future
            # But for now, removing it breaks the tests as it introduces NaNs in the NLP
            soft_contact_force = controller.get_nlp.model.soft_contact_forces().expand()(
                controller.states["q"].cx,
                controller.states["qdot"].cx,
                controller.parameters.cx,
            )

            return soft_contact_force[force_idx]

        @staticmethod
        def track_segment_with_custom_rt(
            penalty: PenaltyOption,
            controller: PenaltyController,
            segment: int | str,
            rt_index: int,
            sequence: str = "zyx",
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
            rt_index: int
                The index of the RT in the bioMod
            sequence: str
                The sequence of the euler angles (default="zyx")
            """

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            segment_index = controller.model.segment_index(segment) if isinstance(segment, str) else segment

            r_seg = controller.model.homogeneous_matrices_in_global(segment_index=segment_index)(
                controller.q, controller.parameters.cx
            )[:3, :3]
            r_seg_transposed = r_seg.T
            r_rt = controller.model.rt(rt_index=rt_index)(controller.q, controller.parameters.cx)[:3, :3]

            angles_diff = controller.model.rotation_matrix_to_euler_angles(sequence=sequence)(r_seg_transposed @ r_rt)

            return angles_diff

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
            segment_index = controller.model.segment_index(segment) if isinstance(segment, str) else segment

            # Get the marker in segment_index reference frame
            marker = controller.model.marker(marker_idx, segment_index)(controller.q, controller.parameters.cx)

            # To align an axis, the other must be equal to 0
            if not penalty.rows_is_set:
                if penalty.rows is not None:
                    raise ValueError("rows cannot be defined in track_marker_with_segment_axis")
                penalty.rows = [ax for ax in [Axis.X, Axis.Y, Axis.Z] if ax != axis]
                penalty.rows_is_set = True

            return marker

        @staticmethod
        def minimize_segment_rotation(
            penalty: PenaltyOption,
            controller: PenaltyController,
            segment: int | str,
            axes: list | tuple = None,
            sequence: str = "xyz",
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
            sequence: str
                The sequence of the euler angles (default="xyz")
            """
            from ..models.biorbd.biorbd_model import BiorbdModel

            if penalty.derivative == True:
                raise RuntimeWarning(
                    "To minimize the velocity of the segment rotation, it would be safer (Euler angles related problems) to use MINIMIZE_SEGMENT_ANGLUAR_VELOCITY instead."
                )

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            segment_index = controller.model.segment_index(segment) if isinstance(segment, str) else segment

            jcs_segment = controller.model.homogeneous_matrices_in_global(segment_index)(
                controller.q, controller.parameters.cx
            )[:3, :3]
            angles_segment = controller.model.rotation_matrix_to_euler_angles(sequence)(jcs_segment)

            if axes is None:
                axes = [Axis.X, Axis.Y, Axis.Z]
            else:
                for ax in axes:
                    if not isinstance(ax, Axis):
                        raise RuntimeError("axes must be a list of bioptim.Axis")

            return angles_segment[axes]

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
            from ..models.biorbd.biorbd_model import BiorbdModel

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic

            segment_index = controller.model.segment_index(segment) if isinstance(segment, str) else segment

            if not isinstance(controller.model, BiorbdModel):
                raise NotImplementedError(
                    "The minimize_segments_velocity penalty can only be called with a BiorbdModel"
                )
            model: BiorbdModel = controller.model
            segment_angular_velocity = model.segment_angular_velocity(segment_index)(
                controller.q, controller.qdot, controller.parameters.cx
            )

            if axes is None:
                axes = [Axis.X, Axis.Y, Axis.Z]
            else:
                for ax in axes:
                    if not isinstance(ax, Axis):
                        raise RuntimeError("axes must be a list of bioptim.Axis")

            return segment_angular_velocity[axes]

        @staticmethod
        def state_continuity(penalty: PenaltyOption, controller: PenaltyController | list):
            if controller.control_type in (
                ControlType.CONSTANT,
                ControlType.CONSTANT_WITH_LAST_NODE,
            ):
                u = controller.controls.cx_start
            elif controller.control_type == ControlType.LINEAR_CONTINUOUS:
                # TODO: For cx_end take the previous node
                u = horzcat(controller.controls.cx_start, controller.controls.cx_end)
            else:
                raise NotImplementedError(f"Dynamics with {controller.control_type} is not implemented yet")

            if isinstance(penalty.node, (list, tuple)) and len(penalty.node) != 1:
                raise RuntimeError("continuity should be called one node at a time")

            penalty.expand = controller.get_nlp.dynamics_type.expand_continuity

            t_span = controller.t_span.cx
            continuity = controller.states.cx_end
            if controller.get_nlp.ode_solver.is_direct_collocation:
                cx = horzcat(*([controller.states.cx_start] + controller.states.cx_intermediates_list))

                integrated = controller.integrate(
                    t_span=t_span,
                    x0=cx,
                    u=u,
                    p=controller.parameters.cx,
                    a=controller.algebraic_states.cx_start,
                    d=controller.numerical_timeseries.cx,
                )
                continuity -= integrated["xf"]
                continuity = vertcat(continuity, integrated["defects"])

                penalty.integrate = True

            else:
                continuity -= controller.integrate(
                    t_span=t_span,
                    x0=controller.states.cx_start,
                    u=u,
                    p=controller.parameters.cx_start,
                    a=controller.algebraic_states.cx_start,
                    d=controller.numerical_timeseries.cx,
                )["xf"]

            penalty.phase = controller.phase_idx
            penalty.explicit_derivative = True
            penalty.multi_thread = True

            return continuity

        @staticmethod
        def first_collocation_point_equals_state(penalty: PenaltyOption, controller: PenaltyController | list):
            """
            Ensures that the first collocation helper is equal to the states at the shooting node.
            This is a necessary constraint for COLLOCATION with duplicate_starting_point.
            """
            collocation_helper = controller.states.cx_intermediates_list[0]
            states = controller.states.cx_start
            return collocation_helper - states

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
                "extra_parameters",
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
                "is_stochastic",
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
                        "You cannot have non linear bounds for custom constraints and min_bound or max_bound defined.\n"
                        "Please note that you may run into this error message if phase_dynamics was set "
                        "to PhaseDynamics.ONE_PER_NODE. One workaround is to define your penalty one node at a time instead of "
                        "using the built-in ALL_SHOOTING (or something similar)."
                    )
                penalty.min_bound = val[0]
                penalty.max_bound = val[2]
                val = val[1]

            return val

        @staticmethod
        def minimize_parameter(penalty: PenaltyOption, controller: PenaltyController, key: str = "all"):
            """
            Minimize the specified parameter.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            controller: PenaltyController
                The penalty node elements
            key: str
                The name of the parameter to minimize
            """

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

            return controller.parameters.cx if key is None or key == "all" else controller.parameters[key].cx

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
        if penalty.cols_is_set:
            return

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
        penalty.cols_is_set = True

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
        if penalty.rows_is_set:
            return

        if penalty.rows is not None and axes is not None:
            if penalty.rows != axes:
                raise ValueError("It is not possible to define rows and axes since they are the same variable")
        penalty.rows = axes if axes is not None else penalty.rows

        penalty.rows_is_set = True

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
        Return the dt of the penalty (abstract)
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

    @staticmethod
    def _get_qddot(controller: PenaltyController, attribute: str):
        """
        Returns the generalized acceleration by either fetching it directly
        from the controller's states or controls or from the controller's dynamics.

        Parameters
        ----------
        controller : object
            The controller that has 'states', 'controls', 'parameters', and 'algebraic_states' attributes.
        attribute : str
            Specifies which attribute ('cx_start' or 'mx') to use for the extraction.
        """

        if "qddot" not in controller.states and "qddot" not in controller.controls:
            source_qdot = controller.states if "qdot" in controller.states else controller.controls
            return controller.dynamics(
                getattr(controller.time, attribute),
                getattr(controller.states, attribute),
                getattr(controller.controls, attribute),
                getattr(controller.parameters, attribute),
                getattr(controller.algebraic_states, attribute),
                getattr(controller.numerical_timeseries, attribute),
            )[source_qdot["qdot"].index, :]

        source = controller.states if "qddot" in controller.states else controller.controls
        return getattr(source["qddot"], attribute)
