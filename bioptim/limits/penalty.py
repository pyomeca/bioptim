from typing import Union, Any
from enum import Enum
from math import inf
import inspect

import biorbd
from casadi import vertcat, MX, SX

from .penalty_option import PenaltyOption
from .penalty_node import PenaltyNodeList
from ..misc.enums import Node, Axis
from ..misc.mapping import Mapping
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
        minimize_markers(penalty: PenaltyOption, , pn: PenaltyNodeList, axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z))
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
            penalty.set_penalty(fcn, all_pn=all_pn, combine_to=combined_to, target_ns=len(var))

            if combined_to is None:
                penalty.add_multiple_target_to_plot(names, suffix, all_pn)

        @staticmethod
        def minimize_markers(
            penalty: PenaltyOption, all_pn: PenaltyNodeList, marker_index: Union[tuple, list] = None,  axis_to_track: Union[tuple, list] = (Axis.X, Axis.Y, Axis.Z)
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
            marker_index: Union[tuple, list]
                The index of markers to minimize, can be int or str.
                penalty.cols should not be defined if marker_index is defined
            axis_to_track: list
                The axis to minimize, default XYZ
            """

            # Adjust the cols and rows
            PenaltyFunctionAbstract.set_marker_idx_columns(penalty, all_pn, marker_index)
            PenaltyFunctionAbstract.set_marker_idx_rows(penalty, axis_to_track)

            # Add the penalty
            markers_obj = all_pn.nlp.add_casadi_func("marker", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
            penalty.set_penalty(markers_obj, all_pn)

        @staticmethod
        def minimize_markers_displacement(
            penalty: PenaltyOption, pn: PenaltyNodeList, coordinates_system_idx: int = -1
        ):
            """
            Minimize a marker set velocity by comparing the position at a node and at the next node.
            By default this function is quadratic, meaning that it minimizes the difference.
            Indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            pn: PenaltyNodeList
                The penalty node elements
            coordinates_system_idx: int
                The index of the segment to use as reference. Default [-1] is the global coordinate system
            """

            nlp = pn.nlp
            n_rts = nlp.model.nbSegment()

            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                penalty.index, nlp.model.nbMarkers(), "markers_idx"
            )

            nlp.add_casadi_func("biorbd_markers", nlp.model.markers, nlp.states["q"].mx)
            if coordinates_system_idx >= 0:
                if coordinates_system_idx >= n_rts:
                    raise RuntimeError(
                        f"coordinates_system_idx ({coordinates_system_idx}) cannot be higher than {n_rts - 1}"
                    )
                nlp.add_casadi_func(
                    f"globalJCS_{coordinates_system_idx}",
                    nlp.model.globalJCS,
                    nlp.states["q"].mx,
                    coordinates_system_idx,
                )

            for i in range(len(pn.x) - 1):
                q_0 = nlp.variable_mappings["q"].to_second.map(pn.x[i][pn.nlp.states["q"].index, :])
                q_1 = nlp.variable_mappings["q"].to_second.map(pn.x[i + 1][pn.nlp.states["q"].index, :])

                if coordinates_system_idx < 0:
                    jcs_0_T = nlp.cx.eye(4)
                    jcs_1_T = nlp.cx.eye(4)

                elif coordinates_system_idx < n_rts:
                    jcs_0 = nlp.casadi_func[f"globalJCS_{coordinates_system_idx}"](q_0)
                    jcs_0_T = vertcat(horzcat(jcs_0[:3, :3], -jcs_0[:3, :3] @ jcs_0[:3, 3]), horzcat(0, 0, 0, 1))

                    jcs_1 = nlp.casadi_func[f"globalJCS_{coordinates_system_idx}"](q_1)
                    jcs_1_T = vertcat(horzcat(jcs_1[:3, :3], -jcs_1[:3, :3] @ jcs_1[:3, 3]), horzcat(0, 0, 0, 1))

                else:
                    raise RuntimeError(
                        f"Wrong choice of coordinates_system_idx. (Negative values refer to global coordinates system, "
                        f"positive values must be between 0 and {n_rts})"
                    )

                val = jcs_1_T @ vertcat(
                    nlp.casadi_func["biorbd_markers"](q_1)[:, markers_idx], nlp.cx.ones(1, markers_idx.shape[0])
                ) - jcs_0_T @ vertcat(
                    nlp.casadi_func["biorbd_markers"](q_0)[:, markers_idx], nlp.cx.ones(1, markers_idx.shape[0])
                )
                penalty.type.get_type().add_to_penalty(pn.ocp, pn, val[:3, :], penalty)

        @staticmethod
        def minimize_markers_velocity(penalty: PenaltyOption, all_pn: PenaltyNodeList, marker_index: Union[tuple, list] = None,  axis_to_track: Union[tuple, list] = (Axis.X, Axis.Y, Axis.Z)):
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
            marker_index: Union[tuple, list]
                The index of markers to minimize, can be int or str.
                penalty.cols should not be defined if marker_index is defined
            axis_to_track: list
                The axis to minimize, default XYZ
            """

            # Adjust the cols and rows
            PenaltyFunctionAbstract.set_marker_idx_columns(penalty, all_pn, marker_index)
            PenaltyFunctionAbstract.set_marker_idx_rows(penalty, axis_to_track)

            # Add the penalty
            markers_obj = all_pn.nlp.add_casadi_func("marker", all_pn.nlp.model.markersVelocity, all_pn.nlp.states["q"], all_pn.nlp.states["qdot"])
            penalty.set_penalty(markers_obj, all_pn)

        @staticmethod
        def superimpose_markers(
            penalty: PenaltyOption,
            all_pn: PenaltyNodeList,
            first_marker: Union[str, int],
            second_marker: Union[str, int],
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
            """

            nlp = all_pn.nlp
            first_marker_idx = (
                biorbd.marker_index(nlp.model, first_marker) if isinstance(first_marker, str) else first_marker
            )
            second_marker_idx = (
                biorbd.marker_index(nlp.model, second_marker) if isinstance(second_marker, str) else second_marker
            )
            PenaltyFunctionAbstract._check_idx("marker", [first_marker_idx, second_marker_idx], nlp.model.nbMarkers())

            marker_0 = nlp.add_casadi_func(f"markers_{first_marker}", nlp.model.marker, nlp.states["q"], first_marker)
            marker_1 = nlp.add_casadi_func(f"markers_{second_marker}", nlp.model.marker, nlp.states["q"], second_marker)
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
            com_height_cx = nlp.add_casadi_func("com_height", com_height, nlp.states["q"], nlp.states["qdot"])
            penalty.set_penalty(com_height_cx, all_pn)

        @staticmethod
        def minimize_com_position(penalty: PenaltyOption, all_pn: PenaltyNodeList, axis: Axis = None):
            """
            Adds the objective that the position of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the CoM's position is minimized.
            Otherwise, the projection of the CoM's position on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            all_pn: PenaltyNodeList
                The penalty node elements
            axis: Axis
                The axis to project on. Default is all axes
            """

            nlp = all_pn.nlp
            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(penalty.target, (1, len(all_pn.x)))

            nlp.add_casadi_func("biorbd_CoM", nlp.model.CoM, nlp.states["q"].mx)
            for i, pn in enumerate(all_pn):
                q = nlp.variable_mappings["q"].to_second.map(pn["q"])
                CoM = nlp.casadi_func["biorbd_CoM"](q)

                if axis is None:
                    CoM_proj = CoM
                elif not isinstance(axis, Axis):
                    raise RuntimeError("axis must be a bioptim.Axis")
                else:
                    CoM_proj = CoM[axis]

                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(all_pn.ocp, all_pn, CoM_proj, penalty)

        @staticmethod
        def minimize_com_velocity(penalty: PenaltyOption, all_pn: PenaltyNodeList, axis: Axis = None):
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
            axis: Axis
                The axis to project on. Default is all axes
            """

            nlp = all_pn.nlp
            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(penalty.target, (1, len(all_pn.x)))

            nlp.add_casadi_func("biorbd_CoM_dot", nlp.model.CoMdot, nlp.states["q"].mx, nlp.states["qdot"].mx)
            for i, pn in enumerate(all_pn):
                q = nlp.variable_mappings["q"].to_second.map(pn["q"])
                qdot = nlp.variable_mappings["qdot"].to_second.map(pn["qdot"])
                CoM_dot = nlp.casadi_func["biorbd_CoM_dot"](q, qdot)

                if axis is None:
                    CoM_dot_proj = CoM_dot[0] ** 2 + CoM_dot[1] ** 2 + CoM_dot[2] ** 2
                elif not isinstance(axis, Axis):
                    raise RuntimeError("axis must be a bioptim.Axis")
                else:
                    CoM_dot_proj = CoM_dot[axis]

                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(all_pn.ocp, all_pn, CoM_dot_proj, penalty)

        @staticmethod
        def minimize_contact_forces(penalty: PenaltyOption, all_pn: PenaltyNodeList):
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
            """

            n_contact = all_pn.nlp.model.nbContacts()
            contacts_idx = PenaltyFunctionAbstract._check_and_fill_index(penalty.index, n_contact, "contacts_idx")

            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    penalty.target, (len(contacts_idx), len(all_pn.u))
                )

                PenaltyFunctionAbstract._add_track_data_to_plot(
                    all_pn, target, combine_to="contact_forces", axes_idx=Mapping(contacts_idx)
                )

            for i, v in enumerate(all_pn.u):
                force = all_pn.nlp.contact_forces_func(all_pn.x[i], all_pn.u[i], all_pn.p)
                val = force[contacts_idx]
                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(all_pn.ocp, all_pn, val, penalty)

        @staticmethod
        def track_segment_with_custom_rt(
            penalty: PenaltyOption, all_pn: PenaltyNodeList, segment: Union[int, str], rt_idx: int
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
            rt_idx: int
                The index of the RT
            """

            nlp = all_pn.nlp
            segment_idx = biorbd.segment_index(nlp.model, segment) if isinstance(segment, str) else segment

            PenaltyFunctionAbstract._check_idx("segment", segment_idx, nlp.model.nbSegment())
            PenaltyFunctionAbstract._check_idx("rt", rt_idx, nlp.model.nbRTs())

            def biorbd_meta_func(q: Union[MX, SX], segment_idx: int, rt_idx: int):
                """
                Compute the Euler angles between a segment and a RT

                Parameters
                ----------
                q: Union[MX, SX]
                    The generalized coordinates of the system
                segment_idx: int
                    The index of the segment
                rt_idx: int
                    The index of the RT

                Returns
                -------
                The Euler angles between a segment and a RT
                """
                r_seg = nlp.model.globalJCS(q, segment_idx).rot()
                r_rt = nlp.model.RT(q, rt_idx).rot()
                return biorbd.Rotation_toEulerAngles(r_seg.transpose() * r_rt, "zyx").to_mx()

            nlp.add_casadi_func(
                f"track_segment_with_custom_rt_{segment_idx}", biorbd_meta_func, nlp.states["q"].mx, segment_idx, rt_idx
            )

            for pn in all_pn:
                q = nlp.variable_mappings["q"].to_second.map(pn["q"])
                val = nlp.casadi_func[f"track_segment_with_custom_rt_{segment_idx}"](q)
                penalty.type.get_type().add_to_penalty(all_pn.ocp, all_pn, val, penalty)

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

            def biorbd_meta_func(q, segment_idx, marker_idx):
                r_rt = nlp.model.globalJCS(q, segment_idx)
                marker = nlp.model.marker(q, marker_idx)
                marker.applyRT(r_rt.transpose())
                return marker.to_mx()

            nlp.add_casadi_func(
                f"track_marker_with_segment_axis_{segment_idx}_{marker_idx}",
                biorbd_meta_func,
                nlp.states["q"].mx,
                segment_idx,
                marker_idx,
            )
            nq = len(nlp.variable_mappings["q"].to_first)
            for pn in all_pn:
                q = nlp.variable_mappings["q"].to_second.map(pn["q"])
                marker = nlp.casadi_func[f"track_marker_with_segment_axis_{segment_idx}_{marker_idx}"](q)
                for axe in Axis:
                    if axe != axis:
                        # To align an axis, the other must be equal to 0
                        val = marker[axe, 0]
                        penalty.type.get_type().add_to_penalty(all_pn.ocp, all_pn, val, penalty)

        @staticmethod
        def custom(penalty: PenaltyOption, nodes: Union[PenaltyNodeList, list], **parameters: Any):
            """
            A user defined penalty function

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            nodes: PenaltyNodeList
                The penalty node elements
            parameters: dict
                Any parameters that should be pass to the custom function
            """

            keywords = [
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
                "get_all_nodes_at_once",
            ]
            for keyword in keywords:
                if keyword in inspect.signature(penalty.custom_function).parameters:
                    raise TypeError(f"{keyword} is a reserved word and cannot be used in a custom function signature")

            has_bound = (
                True
                if (hasattr(penalty, "min_bound") and penalty.min_bound is not None)
                or (hasattr(penalty, "max_bound") and penalty.max_bound is not None)
                else False
            )

            if penalty.get_all_nodes_at_once:
                nodes = [nodes]  # Trick the next for loop into sending everything at once

            for node in nodes:
                val = penalty.custom_function(node, **parameters)
                if val is None:
                    continue

                if isinstance(val, (list, tuple)):
                    if has_bound:
                        raise RuntimeError(
                            "You cannot have non linear bounds for custom constraints "
                            "and min_bound or max_bound defined"
                        )
                    penalty.min_bound = val[0]
                    penalty.max_bound = val[2]
                    val = val[1]
                if penalty.get_all_nodes_at_once:
                    nlp = nodes[0].nlp if penalty.node != Node.TRANSITION else None
                    penalty.type.get_type().add_to_penalty(node[0].ocp, nlp, val, penalty)
                else:
                    penalty.type.get_type().add_to_penalty(node.ocp, nodes, val, penalty)

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
    def set_marker_idx_columns(penalty: PenaltyOption, all_pn: PenaltyNodeList, marker_index: Union[str, int, list, tuple]):
        """
        Simple penalty.cols setter for marker index and names

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        all_pn: PenaltyNodeList
            The penalty node elements
        marker_index: Union[str, int, list, tuple]
            The marker to index
        """

        if penalty.cols is not None and marker_index is not None:
            raise ValueError("It is not possible to define cols and marker_index since they are the same variable")
        penalty.cols = marker_index if marker_index is not None else penalty.cols
        if penalty.cols is not None:
            penalty.cols = [penalty.cols] if not isinstance(penalty, (tuple, list)) else penalty.cols
            # Convert to int if it is str
            penalty.cols = [cols if isinstance(cols, int) else biorbd.marker_index(all_pn.nlp.model, cols) for cols in penalty.cols]

    @staticmethod
    def set_marker_idx_rows(penalty: PenaltyOption, axis_to_track: Union[list, tuple]):
        """
        Simple penalty.cols setter for marker index and names

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        axis_to_track: Union[list, tuple]
            The marker to index
        """

        if penalty.rows is not None and axis_to_track is not None:
            raise ValueError("It is not possible to define rows and axis_to_track since they are the same variable")
        penalty.rows = axis_to_track if axis_to_track is not None else penalty.rows

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

        func = penalty.type.value[0]
        # Everything that should change the entry parameters depending on the penalty can be added here
        if penalty.quadratic is None:
            if (
                func == PenaltyType.MINIMIZE_STATE
                or func == PenaltyType.MINIMIZE_MARKERS
                or func == PenaltyType.MINIMIZE_MARKERS_DISPLACEMENT
                or func == PenaltyType.MINIMIZE_MARKERS_VELOCITY
                or func == PenaltyType.SUPERIMPOSE_MARKERS
                or func == PenaltyType.PROPORTIONAL_STATE
                or func == PenaltyType.PROPORTIONAL_CONTROL
                or func == PenaltyType.MINIMIZE_CONTROL
                or func == PenaltyType.MINIMIZE_CONTACT_FORCES
                or func == PenaltyType.TRACK_SEGMENT_WITH_CUSTOM_RT
                or func == PenaltyType.TRACK_MARKER_WITH_SEGMENT_AXIS
                or func == PenaltyType.MINIMIZE_COM_POSITION
                or func == PenaltyType.MINIMIZE_COM_VELOCITY
            ):
                penalty.quadratic = True
            else:
                penalty.quadratic = False

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
            func == PenaltyType.PROPORTIONAL_CONTROL
            or func == PenaltyType.MINIMIZE_CONTROL
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


class PenaltyType(Enum):
    """
    Selection of valid penalty functions
    """

    MINIMIZE_STATE = PenaltyFunctionAbstract.Functions.minimize_states
    TRACK_STATE = MINIMIZE_STATE
    MINIMIZE_CONTROL = PenaltyFunctionAbstract.Functions.minimize_controls
    TRACK_CONTROL = MINIMIZE_CONTROL
    MINIMIZE_MARKERS = PenaltyFunctionAbstract.Functions.minimize_markers
    TRACK_MARKERS = MINIMIZE_MARKERS
    SUPERIMPOSE_MARKERS = PenaltyFunctionAbstract.Functions.superimpose_markers

    MINIMIZE_MARKERS_DISPLACEMENT = PenaltyFunctionAbstract.Functions.minimize_markers_displacement
    MINIMIZE_MARKERS_VELOCITY = PenaltyFunctionAbstract.Functions.minimize_markers_velocity
    TRACK_MARKERS_VELOCITY = MINIMIZE_MARKERS_VELOCITY
    PROPORTIONAL_STATE = PenaltyFunctionAbstract.Functions.proportional_states
    PROPORTIONAL_CONTROL = PenaltyFunctionAbstract.Functions.proportional_controls
    MINIMIZE_QDDOT = PenaltyFunctionAbstract.Functions.minimize_qddot
    MINIMIZE_PREDICTED_COM_HEIGHT = PenaltyFunctionAbstract.Functions.minimize_predicted_com_height

    MINIMIZE_CONTACT_FORCES = PenaltyFunctionAbstract.Functions.minimize_contact_forces
    TRACK_CONTACT_FORCES = MINIMIZE_CONTACT_FORCES
    MINIMIZE_COM_POSITION = PenaltyFunctionAbstract.Functions.minimize_com_position
    MINIMIZE_COM_VELOCITY = PenaltyFunctionAbstract.Functions.minimize_com_velocity
    TRACK_SEGMENT_WITH_CUSTOM_RT = PenaltyFunctionAbstract.Functions.track_segment_with_custom_rt
    TRACK_MARKER_WITH_SEGMENT_AXIS = PenaltyFunctionAbstract.Functions.track_marker_with_segment_axis
    CUSTOM = PenaltyFunctionAbstract.Functions.custom
