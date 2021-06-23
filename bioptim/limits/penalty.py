from typing import Union, Any
from math import inf
import inspect

import biorbd
from casadi import horzcat, vertcat

from .penalty_option import PenaltyOption
from .penalty_node import PenaltyNodeList
from ..interfaces.biorbd_interface import BiorbdInterface
from ..misc.enums import Node, Axis
from ..optimization.optimization_variable import OptimizationVariable


class PenaltyFunctionAbstract:
    """
    Internal implementation of the penalty functions

    Methods
    -------
    add(ocp: OptimalControlProgram, nlp: NonLinearProgram)
        Add a new penalty to the list (abstract)
    add_or_replace(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: PenaltyOption)
        Doing some configuration on the penalty and add it to the list of penalty
    _parameter_modifier(penalty: PenaltyOption)
        Apply some default parameters
    _span_checker(penalty: PenaltyOption, pn: PenaltyNodeList)
        Check for any non sense in the requested times for the constraint. Raises an error if so
    _check_and_fill_index(var_idx: Union[list, int], target_size: int, var_name: str = "var")
        Checks if the variable index is consistent with the requested variable.
    _check_and_fill_tracking_data_size(data_to_track: np.ndarray, target_size: int)
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay
    _check_idx(name: str, elements: Union[list, tuple, int], max_n_elements: int = inf, min_n_elements: int = 0)
        Generic sanity check for requested dimensions.
        If the function returns, everything is okay
    add_to_penalty(ocp: OptimalControlProgram, pn: PenaltyNodeList, val: Union[MX, SX, float, int], penalty: PenaltyOption)
        Add the constraint to the penalty pool (abstract)
    clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: PenaltyOption)
        Resets a penalty. A negative penalty index creates a new empty penalty (abstract)
    get_type()
        Returns the type of the penalty (abstract)
    _get_node(nlp: NonLinearProgram, penalty: PenaltyOption)
        Get the actual node (time, X and U) specified in the penalty
    _add_track_data_to_plot(pn: PenaltyNodeList, data: np.ndarray, combine_to: str,
            axes_idx: Union[Mapping, tuple, list] = None)
        Interface to the plot so it can be properly added to the proper plot
    _get_states_or_controls_with_specified_var(pn: PenaltyNodeList, s: str)
        Extracting controls or states depending on if the selected optimisation variables s are defined as states or controls
    """

    class Functions:
        """
        Implementation of all the generic penalty functions

        Methods
        -------
        minimize_states(penalty: PenaltyOption, pn: PenaltyNodeList)
            Minimize the states variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        minimize_markers(penalty: PenaltyOption, , pn: PenaltyNodeList, axis: Axis = (Axis.X, Axis.Y, Axis.Z))
            Minimize a marker set.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        minimize_markers_displacement(penalty: PenaltyOption, pn: PenaltyNodeList, coordinates_system_idx: int = -1)
            Minimize a marker set velocity by comparing the position at a node and at the next node.
            By default this function is quadratic, meaning that it minimizes the difference.
            Indices (default=all_idx) can be specified.
        minimize_markers_velocity(penalty: PenaltyOption, pn: PenaltyNodeList)
            Minimize a marker set velocity by computing the actual velocity of the markers
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        superimpose_markers(penalty: PenaltyOption, pn: PenaltyNodeList, first_marker: Union[int, str], second_marker: Union[int, str]):
            Minimize the distance between two markers
            By default this function is quadratic, meaning that it minimizes distance between them.
        proportional_variable(penalty: PenaltyOption, pn: PenaltyNodeList,
                which_var: str, first_dof: int, second_dof: int, coef: float)
            Introduce a proportionality between two variables (e.g. one variable is twice the other)
            By default this function is quadratic, meaning that it minimizes the difference of this proportion.
        minimize_torque(penalty: PenaltyOption, pn: PenaltyNodeList)
            Minimize the joint torque part of the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        minimize_predicted_com_height(penalty: PenaltyOption, pn: PenaltyNodeList)
            Minimize the prediction of the center of mass maximal height from the parabolic equation,
            assuming vertical axis is Z (2): CoM_dot[2]**2 / (2 * -g) + CoM[2]
            By default this function is not quadratic, meaning that it minimizes towards infinity.
        minimize_com_position(penalty: PenaltyOption, pn: PenaltyNodeList, axis: Axis = None)
            Adds the objective that the position of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the CoM's position is minimized.
            Otherwise, the projection of the CoM's position on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.
        minimize_com_velocity(penalty: PenaltyOption, pn: PenaltyNodeList, axis: Axis = None)
            Adds the objective that the velocity of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the CoM's velocity is minimized.
            Otherwise, the projection of the CoM's velocity on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.
        minimize_contact_forces(penalty: PenaltyOption, pn: PenaltyNodeList)
            Minimize the contact forces computed from dynamics with contact
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        track_segment_with_custom_rt(penalty: PenaltyOption, pn: PenaltyNodeList, segment: int, rt_idx: int)
            Minimize the difference of the euler angles extracted from the coordinate system of a segment
            and a RT (e.g. IMU). By default this function is quadratic, meaning that it minimizes the difference.
        track_marker_with_segment_axis(penalty: PenaltyOption, pn: PenaltyNodeList, marker: str, segment: str, axis: Axis)
            Track a marker using a segment, that is aligning an axis toward the marker
            By default this function is quadratic, meaning that it minimizes the difference.
        custom(penalty: PenaltyOption, pn: PenaltyNodeList, **parameters: dict)
            A user defined penalty function
        """

        @staticmethod
        def minimize_states(penalty: PenaltyOption, all_pn: PenaltyNodeList, names: str = "all"):
            """
            Minimize the states variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            names: str
                The name of the state to minimize. Default "all"
            """

            PenaltyFunctionAbstract.Functions._minimize_optim_var(penalty, all_pn, names, "states")

        @staticmethod
        def minimize_controls(penalty: PenaltyOption, all_pn: PenaltyNodeList, names: Union[str, list] = "all"):
            """
            Minimize the joint torque part of the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            names: Union[str, list]
                The name of the controls to minimize
            """

            PenaltyFunctionAbstract.Functions._minimize_optim_var(penalty, all_pn, names, "controls")

        @staticmethod
        def _minimize_optim_var(penalty: PenaltyOption, all_pn: PenaltyNodeList, names: Union[str, list], suffix: str):
            """
            Minimize the joint torque part of the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            names: Union[str, list]
                The name of the controls to minimize
            suffix: str
                If the optim_var is 'states' or 'controls'
            """

            if suffix == "states":
                optim_var = all_pn.nlp.states
                var = all_pn.x
            elif suffix == "controls":
                optim_var = all_pn.nlp.controls
                var = all_pn.u
            else:
                raise ValueError("suffix can only be 'states' or 'controls'")
            names = optim_var.keys() if names == "all" else names

            fcn = vertcat(*[optim_var[name].cx for name in names])
            combined_to = None if isinstance(names, (list, tuple)) else f"{names}_{suffix}"
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(fcn, all_pn=all_pn, combine_to=combined_to, target_ns=len(var))

            if combined_to is None:
                penalty.add_multiple_target_to_plot(names, suffix, all_pn)

        @staticmethod
        def minimize_markers(
            penalty: PenaltyOption, all_pn: PenaltyNodeList, marker_index: Union[tuple, list, int, str] = None, axes: Union[tuple, list] = None, reference_jcs: Union[str, int] = None
        ):
            """
            Minimize a marker set.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            marker_index: Union[tuple, list, int, str]
                The index of markers to minimize, can be int or str.
                penalty.cols should not be defined if marker_index is defined
            axes: list
                The axes to minimize, default XYZ
            reference_jcs: Union[int, str]
                The index or name of the segment to use as reference. Default [None] is the global coordinate system
            """

            # Adjust the cols and rows
            PenaltyFunctionAbstract.set_idx_columns(penalty, all_pn, marker_index, "marker")
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)

            # Compute the position of the marker in the requested reference frame (None for global)
            nlp = all_pn.nlp
            q_mx = nlp.states["q"].mx
            model = nlp.model
            jcs_t = biorbd.RotoTrans() if reference_jcs is None else model.globalJCS(q_mx, reference_jcs).transpose()
            markers = horzcat(*[m.to_mx() for m in model.markers(q_mx) if m.applyRT(jcs_t) is None])

            markers_objective = BiorbdInterface.mx_to_cx("markers", markers, nlp.states["q"])
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(markers_objective, all_pn)

        @staticmethod
        def minimize_markers_velocity(penalty: PenaltyOption, all_pn: PenaltyNodeList, marker_index: Union[tuple, list, int, str] = None, axes: Union[tuple, list] = None, reference_jcs: Union[str, int] = None):
            """
            Minimize a marker set velocity by computing the actual velocity of the markers
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            marker_index: Union[tuple, list, int, str]
                The index of markers to minimize, can be int or str.
                penalty.cols should not be defined if marker_index is defined
            axes: Union[tuple, list]
                The axes to project on. Default is all axes
            reference_jcs: Union[int, str]
                The index or name of the segment to use as reference. Default [None] is the global coordinate system
            """

            # Adjust the cols and rows
            PenaltyFunctionAbstract.set_idx_columns(penalty, all_pn, marker_index, "marker")
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)

            # Add the penalty in the requested reference frame. None for global
            nlp = all_pn.nlp
            q_mx = nlp.states["q"].mx
            qdot_mx = nlp.states["qdot"].mx
            model = nlp.model
            jcs_t = biorbd.RotoTrans() if reference_jcs is None else model.globalJCS(q_mx, reference_jcs).transpose()
            markers = horzcat(*[m.to_mx() for m in model.markersVelocity(q_mx, qdot_mx) if m.applyRT(jcs_t) is None])

            markers_objective = BiorbdInterface.mx_to_cx("markersVel", markers, nlp.states["q"], nlp.states["qdot"])
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(markers_objective, all_pn)

        @staticmethod
        def superimpose_markers(
            penalty: PenaltyOption,
            all_pn: PenaltyNodeList,
            first_marker: Union[str, int],
            second_marker: Union[str, int],
            axes: Union[tuple, list] = None,
        ):
            """
            Minimize the distance between two markers
            By default this function is quadratic, meaning that it minimizes distance between them.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            first_marker: Union[str, int]
                The name or index of one of the two markers
            second_marker: Union[str, int]
                The name or index of one of the two markers
            axes: Union[tuple, list]
                The axes to project on. Default is all axes
            """

            nlp = all_pn.nlp
            first_marker_idx = (
                biorbd.marker_index(nlp.model, first_marker) if isinstance(first_marker, str) else first_marker
            )
            second_marker_idx = (
                biorbd.marker_index(nlp.model, second_marker) if isinstance(second_marker, str) else second_marker
            )
            PenaltyFunctionAbstract._check_idx("marker", [first_marker_idx, second_marker_idx], nlp.model.nbMarkers())
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)

            marker_0 = BiorbdInterface.mx_to_cx(f"markers_{first_marker}", nlp.model.marker, nlp.states["q"], first_marker)
            marker_1 = BiorbdInterface.mx_to_cx(f"markers_{second_marker}", nlp.model.marker, nlp.states["q"], second_marker)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(marker_1 - marker_0, all_pn)

        @staticmethod
        def proportional_states(
            penalty: PenaltyOption,
            all_pn: PenaltyNodeList,
            first_dof: int,
            second_dof: int,
            coef: float,
        ):
            """
            Introduce a proportionality between two variables (e.g. one variable is twice the other)
            By default this function is quadratic, meaning that it minimizes the difference of this proportion.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            first_dof: int
                The index of the first variable
            second_dof: int
                The index of the second variable
            coef: float
                The proportion coefficient such that v[first_dof] = coef * v[second_dof]
            """

            PenaltyFunctionAbstract.Functions._proportional_variable(penalty, all_pn, first_dof, second_dof, coef, all_pn.nlp.states.cx, "proportional_states")

        @staticmethod
        def proportional_controls(
                penalty: PenaltyOption,
                all_pn: PenaltyNodeList,
                first_dof: int,
                second_dof: int,
                coef: float,
        ):
            """
            Introduce a proportionality between two variables (e.g. one variable is twice the other)
            By default this function is quadratic, meaning that it minimizes the difference of this proportion.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            first_dof: int
                The index of the first variable
            second_dof: int
                The index of the second variable
            coef: float
                The proportion coefficient such that v[first_dof] = coef * v[second_dof]
            """

            PenaltyFunctionAbstract.Functions._proportional_variable(penalty, all_pn, first_dof, second_dof, coef, all_pn.nlp.controls.cx, "proportional_controls")

        @staticmethod
        def _proportional_variable(
                penalty: PenaltyOption,
                all_pn: PenaltyNodeList,
                first_dof: int,
                second_dof: int,
                coef: float,
                var_cx: OptimizationVariable,
                var_type: str,
        ):
            """
            Introduce a proportionality between two variables (e.g. one variable is twice the other)
            By default this function is quadratic, meaning that it minimizes the difference of this proportion.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            first_dof: int
                The index of the first variable
            second_dof: int
                The index of the second variable
            coef: float
                The proportion coefficient such that v[first_dof] = coef * v[second_dof]
            """

            if penalty.rows is not None:
                raise ValueError(f"rows should not be defined for {var_type}")

            if penalty.cols is not None:
                raise ValueError(f"cols should not be defined for {var_type}")

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(var_cx[first_dof, :] - coef * var_cx[second_dof, :], all_pn)

        @staticmethod
        def minimize_qddot(penalty: PenaltyOption, all_pn: PenaltyNodeList):
            """
            Minimize the states velocity by comparing the state at a node and at the next node.
            By default this function is quadratic, meaning that it minimizes the difference.
            Indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            """

            nlp = all_pn.nlp
            penalty.set_penalty(all_pn.nlp.dynamics_func(nlp.states.cx, nlp.controls.cx, nlp.parameters.cx), all_pn)

        @staticmethod
        def minimize_predicted_com_height(penalty: PenaltyOption, all_pn: PenaltyNodeList):
            """
            Minimize the prediction of the center of mass maximal height from the parabolic equation,
            assuming vertical axis is Z (2): CoM_dot[2]**2 / (2 * -g) + CoM[2]
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            """

            nlp = all_pn.nlp
            g = nlp.model.getGravity().to_mx()[2]
            com = nlp.model.CoM(nlp.states["q"].mx).to_mx()
            com_dot = nlp.model.CoMdot(nlp.states["q"].mx, nlp.states["qdot"].mx).to_mx()
            com_height = (com_dot[2] * com_dot[2]) / (2 * -g) + com[2]
            com_height_cx = BiorbdInterface.mx_to_cx("com_height", com_height, nlp.states["q"], nlp.states["qdot"])
            penalty.set_penalty(com_height_cx, all_pn)

        @staticmethod
        def minimize_com_position(penalty: PenaltyOption, all_pn: PenaltyNodeList, axes: Union[tuple, list] = None):
            """
            Adds the objective that the position of the center of mass of the model should be minimized.
            If no axes is specified, the squared-norm of the CoM's position is minimized.
            Otherwise, the projection of the CoM's position on the specified axes are minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            axes: Union[tuple, list]
                The axes to project on. Default is all axes
            """

            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)

            com_cx = BiorbdInterface.mx_to_cx("com", all_pn.nlp.model.CoM, all_pn.nlp.states["q"])
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(com_cx, all_pn)

        @staticmethod
        def minimize_com_velocity(penalty: PenaltyOption, all_pn: PenaltyNodeList, axes: Union[tuple, list] = None):
            """
            Adds the objective that the velocity of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the CoM's velocity is minimized.
            Otherwise, the projection of the CoM's velocity on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            axes: Union[tuple, list]
                The axes to project on. Default is all axes
            """

            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)

            nlp = all_pn.nlp
            com_dot_cx = BiorbdInterface.mx_to_cx("com_dot", nlp.model.CoMdot, nlp.states["q"], nlp.states["qdot"])
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(com_dot_cx, all_pn)

        @staticmethod
        def minimize_contact_forces(penalty: PenaltyOption, all_pn: PenaltyNodeList, contact_index: Union[tuple, list, int, str] = None, axes: Union[tuple, list] = None):
            """
            Minimize the contact forces computed from dynamics with contact
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            contact_index: Union[tuple, list]
                The index of contact to minimize, must be an int.
                penalty.cols should not be defined if contact_index is defined
            axes: Union[tuple, list]
                The axes to project on. Default is all axes
            """

            nlp = all_pn.nlp
            if nlp.contact_forces_func is None:
                raise RuntimeError("minimize_contact_forces requires a contact dynamics")

            PenaltyFunctionAbstract.set_idx_columns(penalty, all_pn, contact_index, "contact")
            PenaltyFunctionAbstract.set_axes_rows(penalty, axes)

            contact_force = nlp.contact_forces_func(nlp.states.cx, nlp.controls.cx, nlp.parameters.cx)
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(contact_force, all_pn)

        @staticmethod
        def track_segment_with_custom_rt(
            penalty: PenaltyOption, all_pn: PenaltyNodeList, segment: Union[int, str], rt: int
        ):
            """
            Minimize the difference of the euler angles extracted from the coordinate system of a segment
            and a RT (e.g. IMU). By default this function is quadratic, meaning that it minimizes the difference.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            segment: Union[int, str]
                The name or index of the segment
            rt: int
                The index of the RT in the bioMod
            """

            nlp = all_pn.nlp
            segment_index = biorbd.segment_index(nlp.model, segment) if isinstance(segment, str) else segment

            r_seg = nlp.model.globalJCS(nlp.states["q"].mx, segment_index).rot()
            r_rt = nlp.model.RT(nlp.states["q"].mx, rt).rot()
            angles_diff = biorbd.Rotation_toEulerAngles(r_seg.transpose() * r_rt, "zyx").to_mx()

            angle_objective = BiorbdInterface.mx_to_cx(f"track_segment", angles_diff, nlp.states["q"])
            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(angle_objective, all_pn)

        @staticmethod
        def track_marker_with_segment_axis(
            penalty: PenaltyOption,
            all_pn: PenaltyNodeList,
            marker: Union[int, str],
            segment: Union[int, str],
            axis: Axis,
        ):
            """
            Track a marker using a segment, that is aligning an axis toward the marker
            By default this function is quadratic, meaning that it minimizes the difference.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
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

            nlp = all_pn.nlp
            marker_idx = biorbd.marker_index(nlp.model, marker) if isinstance(marker, str) else marker
            segment_idx = biorbd.segment_index(nlp.model, segment) if isinstance(segment, str) else segment

            # Get the marker in rt reference frame
            jcs = nlp.model.globalJCS(nlp.states["q"].mx, segment_idx)
            marker = nlp.model.marker(nlp.states["q"].mx, marker_idx)
            marker.applyRT(jcs.transpose())
            marker_objective = BiorbdInterface.mx_to_cx("marker", marker.to_mx(), nlp.states["q"])

            # To align an axis, the other must be equal to 0
            if penalty.rows is not None:
                raise ValueError("rows cannot be defined in track_marker_with_segment_axis")
            penalty.rows = [ax for ax in [Axis.X, Axis.Y, Axis.Z] if ax != axis]

            penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
            penalty.set_penalty(marker_objective, all_pn)

        @staticmethod
        def custom(penalty: PenaltyOption, all_pn: Union[PenaltyNodeList, list], **parameters: Any):
            """
            A user defined penalty function

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
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
                "sliced_target",
                "min_bound",
                "max_bound",
                "custom_function",
                "weight",
            ]
            for keyword in inspect.signature(penalty.custom_function).parameters:
                if keyword in invalid_keywords:
                    raise TypeError(f"{keyword} is a reserved word and cannot be used in a custom function signature")

            if penalty.node == Node.TRANSITION:
                raise NotImplementedError("Node transition is not implemented yet")

            val = penalty.custom_function(all_pn, **parameters)
            if isinstance(val, (list, tuple)):
                if (hasattr(penalty, "min_bound") and penalty.min_bound is not None) or (hasattr(penalty, "max_bound") and penalty.max_bound is not None):
                    raise RuntimeError(
                        "You cannot have non linear bounds for custom constraints and min_bound or max_bound defined"
                    )
                penalty.min_bound = val[0]
                penalty.max_bound = val[2]
                val = val[1]

            penalty.set_penalty(val, all_pn)

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
    def set_idx_columns(penalty: PenaltyOption, all_pn: PenaltyNodeList, index: Union[str, int, list, tuple], _type: str):
        """
        Simple penalty.cols setter for marker index and names

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        all_pn: PenaltyNodeList
            The penalty node elements
        index: Union[str, int, list, tuple]
            The marker to index
        """

        if penalty.cols is not None and index is not None:
            raise ValueError(f"It is not possible to define cols and {_type}_index since they are the same variable")
        penalty.cols = index if index is not None else penalty.cols
        if penalty.cols is not None:
            penalty.cols = [penalty.cols] if not isinstance(penalty, (tuple, list)) else penalty.cols
            # Convert to int if it is str
            if _type == "marker":
                penalty.cols = [cols if isinstance(cols, int) else biorbd.marker_index(all_pn.nlp.model, cols) for cols in penalty.cols]

    @staticmethod
    def set_axes_rows(penalty: PenaltyOption, axes: Union[list, tuple]):
        """
        Simple penalty.cols setter for marker index and names

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        axes: Union[list, tuple]
            The marker to index
        """

        if penalty.rows is not None and axes is not None:
            raise ValueError("It is not possible to define rows and axes since they are the same variable")
        penalty.rows = axes if axes is not None else penalty.rows

    @staticmethod
    def _check_idx(name: str, elements: Union[list, tuple, int], max_n_elements: int = inf, min_n_elements: int = 0):
        """
        Generic sanity check for requested dimensions.
        If the function returns, everything is okay

        Parameters
        name: str
            Name of the element
        elements: Union[list, tuple, int]
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
    def adjust_penalty_parameters(penalty: PenaltyOption):
        """
        Apply some default parameters

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        """
        pass

    @staticmethod
    def validate_penalty_time_index(penalty: PenaltyOption, pn: PenaltyNodeList):
        """
        Check for any non sense in the requested times for the penalty. Raises an error if so

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        pn: PenaltyNodeList
            The penalty node elements
        """

        func = penalty.type.value[0]
        node = penalty.node
        # Everything that is suspicious in terms of the span of the penalty function ca be checked here
        if (
            func == PenaltyFunctionAbstract.Functions.minimize_controls
            or func == PenaltyFunctionAbstract.Functions.proportional_controls
        ):
            if node == Node.END or (isinstance(node, int) and node >= pn.nlp.ns):
                raise RuntimeError("No control u at last node")

    @staticmethod
    def clear_penalty(ocp, nlp, penalty: PenaltyOption):
        """
        Resets a penalty. A negative penalty index creates a new empty penalty (abstract)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        penalty: PenaltyOption
            The actual penalty to declare
        """

        raise RuntimeError("_reset_penalty cannot be called from an abstract class")

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty (abstract)
        """

        raise RuntimeError("get_type cannot be called from an abstract class")
