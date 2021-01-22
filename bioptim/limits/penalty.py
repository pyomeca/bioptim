from typing import Callable, Union
from enum import Enum
from math import inf
import inspect

import numpy as np
import biorbd
from casadi import vertcat, horzcat, MX, SX

from ..misc.enums import Node, Axis, PlotType, ControlType
from ..misc.mapping import Mapping
from ..misc.options import OptionGeneric


class PenaltyOption(OptionGeneric):
    """
    A placeholder for a penalty

    Attributes
    ----------
    node: Node
        The node within a phase on which the penalty is acting on
    quadratic: bool
        If the penalty is quadratic
    index: index
        The component index the penalty is acting on
    target: np.array(target)
        A target to track for the penalty
    sliced_target: np.array(target)
        The sliced version of the target to track, this is the one actually tracked
    custom_function: function
        A user defined function to call to get the penalty
    """

    def __init__(
        self,
        penalty,
        phase: int = 0,
        node: Node = Node.DEFAULT,
        target: np.ndarray = None,
        quadratic: bool = None,
        index: int = None,
        custom_function: Callable = None,
        **params,
    ):
        """
        Parameters
        ----------
        penalty: PenaltyType
            The actual penalty
        phase: int
            The phase the penalty is acting on
        node: Node
            The node within a phase on which the penalty is acting on
        target: np.ndarray
            A target to track for the penalty
        quadratic: bool
            If the penalty is quadratic
        index: int
            The component index the penalty is acting on
        custom_function: Callable
            A user defined function to call to get the penalty
        **params: dict
            Generic parameters for the penalty
        """

        super(PenaltyOption, self).__init__(phase=phase, type=penalty, **params)
        self.node = node
        self.quadratic = quadratic

        self.index = index
        self.target = np.array(target) if np.any(target) else None
        self.sliced_target = None  # This one is the sliced node from the target. This is what is actually tracked

        self.custom_function = custom_function


class PenaltyFunctionAbstract:
    """
    Internal implementation of the penalty functions

    Methods
    -------
    add(ocp: OptimalControlProgram, nlp: NonLinearProgram)
        Add a new penalty to the list (abstract)
    add_or_replace(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: PenaltyOption)
        Doing some configuration on the penalty and add it to the list of penalty
    _add_to_casadi_func(nlp: NonLinearProgram, name: str, function: Callable, *all_param)
        Add to the pool of declared casadi function. If the function already exists, it is skipped
    _parameter_modifier(penalty: PenaltyOption)
        Apply some default parameters
    _span_checker(penalty: PenaltyOption, nlp: NonLinearProgram)
        Check for any non sense in the requested times for the constraint. Raises an error if so
    _check_and_fill_index(var_idx: Union[list, int], target_size: int, var_name: str = "var")
        Checks if the variable index is consistent with the requested variable.
    _check_and_fill_tracking_data_size(data_to_track: np.ndarray, target_size: int)
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay
    _check_idx(name: str, elements: Union[list, tuple, int], max_nb_elements: int = inf, min_nb_elements: int = 0)
        Generic sanity check for requested dimensions.
        If the function returns, everything is okay
    add_to_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, val: Union[MX, SX], penalty: PenaltyOption)
        Add the constraint to the penalty pool (abstract)
    clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: PenaltyOption)
        Resets a penalty. A negative penalty index creates a new empty penalty (abstract)
    get_type()
        Returns the type of the penalty (abstract)
    _get_node(nlp: NonLinearProgram, penalty: PenaltyOption)
        Get the actual node (time, X and U) specified in the penalty
    _add_track_data_to_plot(ocp: OptimalControlProgram, nlp: NonLinearProgram, data: np.ndarray, combine_to: str, axes_idx: Union[Mapping, tuple, list] = None)
        Interface to the plot so it can be properly added to the proper plot
    """

    class Functions:
        """
        Implementation of all the generic penalty functions

        Methods
        -------
        minimize_states(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
            Minimize the states variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        minimize_markers(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z))
            Minimize a marker set.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        minimize_markers_displacement(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], coordinates_system_idx: int = -1)
            Minimize a marker set velocity by comparing the position at a node and at the next node.
            By default this function is quadratic, meaning that it minimizes the difference.
            Indices (default=all_idx) can be specified.
        minimize_markers_velocity(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
            Minimize a marker set velocity by computing the actual velocity of the markers
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        track_markers(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], first_marker_idx: int, second_marker_idx: int):
            Minimize the distance between two markers
            By default this function is quadratic, meaning that it minimizes distance between them.
        proportional_variable(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], which_var: str, first_dof: int, second_dof: int, coef: float)
            Introduce a proportionality between two variables (e.g. one variable is twice the other)
            By default this function is quadratic, meaning that it minimizes the difference of this proportion.
        minimize_torque(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
            Minimize the joint torque part of the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        minimize_torque_derivative(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
            Minimize the joint torque velocity by comparing the torque at a node and at the next node.
            By default this function is quadratic, meaning that it minimizes the difference.
            Indices (default=all_idx) can be specified.
        minimize_muscles_control(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
            Minimize the muscles part of the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        minimize_all_controls(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
            Minimize the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        minimize_predicted_com_height(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
            Minimize the prediction of the center of mass maximal height from the parabolic equation,
            assuming vertical axis is Z (2): CoM_dot[2]**2 / (2 * -g) + CoM[2]
            By default this function is not quadratic, meaning that it minimizes towards infinity.
        minimize_com_position(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], axis: Axis = None)
            Adds the objective that the position of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the CoM's position is minimized.
            Otherwise, the projection of the CoM's position on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.
        minimize_com_velocity(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], axis: Axis = None)
            Adds the objective that the velocity of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the CoM's velocity is minimized.
            Otherwise, the projection of the CoM's velocity on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.
        minimize_contact_forces(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
            Minimize the contact forces computed from dynamics with contact
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.
        track_segment_with_custom_rt(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], segment_idx: int, rt_idx: int)
            Minimize the difference of the euler angles extracted from the coordinate system of a segment and a RT (e.g. IMU)
            By default this function is quadratic, meaning that it minimizes the difference.
        track_marker_with_segment_axis(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], marker_idx: int, segment_idx: int, axis: Axis)
            Track a marker using a segment, that is aligning an axis toward the marker
            By default this function is quadratic, meaning that it minimizes the difference.
        custom(penalty: PenaltyOption, ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX], **parameters)
            A user defined penalty function
        """

        @staticmethod
        def minimize_states(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
        ):
            """
            Minimize the states variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            """

            states_idx = PenaltyFunctionAbstract._check_and_fill_index(penalty.index, nlp.nx, "state_idx")
            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    penalty.target, (len(states_idx), len(x))
                )

                # Prepare the plot
                prev_idx = 0  # offset due to previous states
                for s in nlp.var_states:
                    state_idx = []
                    for i, idx in enumerate(states_idx):
                        if prev_idx <= idx < nlp.var_states[s] + prev_idx:
                            state_idx.append([idx - prev_idx, i])
                    state_idx = np.array(state_idx)
                    if state_idx.shape[0] > 0:
                        mapping = Mapping(state_idx[:, 0])
                        PenaltyFunctionAbstract._add_track_data_to_plot(
                            ocp, nlp, target[state_idx[:, 1], :], combine_to=s, axes_idx=mapping
                        )
                    prev_idx += nlp.var_states[s]

            for i, v in enumerate(x):
                val = v[states_idx]
                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def minimize_markers(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            axis_to_track: Axis = (Axis.X, Axis.Y, Axis.Z),
        ):
            """
            Minimize a marker set.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            axis_to_track: Axis
                The axis the penalty is acting on
            """

            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                penalty.index, nlp.model.nbMarkers(), "markers_idx"
            )
            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    penalty.target, (3, len(markers_idx), len(x))
                )
            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_markers", nlp.model.markers, nlp.q)
            nq = nlp.mapping["q"].to_first.len
            for i, v in enumerate(x):
                q = nlp.mapping["q"].to_second.map(v[:nq])
                val = nlp.casadi_func["biorbd_markers"](q)[axis_to_track, markers_idx]
                penalty.sliced_target = target[axis_to_track, :, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def minimize_markers_displacement(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            coordinates_system_idx: int = -1,
        ):
            """
            Minimize a marker set velocity by comparing the position at a node and at the next node.
            By default this function is quadratic, meaning that it minimizes the difference.
            Indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            coordinates_system_idx: int
                The index of the segment to use as reference. Default [-1] is the global coordinate system
            """

            nq = nlp.mapping["q"].to_first.len
            nb_rts = nlp.model.nbSegment()

            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                penalty.index, nlp.model.nbMarkers(), "markers_idx"
            )

            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_markers", nlp.model.markers, nlp.q)
            if coordinates_system_idx >= 0:
                if coordinates_system_idx >= nb_rts:
                    raise RuntimeError(
                        f"coordinates_system_idx ({coordinates_system_idx}) cannot be higher than {nb_rts - 1}"
                    )
                PenaltyFunctionAbstract._add_to_casadi_func(
                    nlp, f"globalJCS_{coordinates_system_idx}", nlp.model.globalJCS, nlp.q, coordinates_system_idx
                )

            for i in range(len(x) - 1):
                q_0 = nlp.mapping["q"].to_second.map(x[i][:nq])
                q_1 = nlp.mapping["q"].to_second.map(x[i + 1][:nq])

                if coordinates_system_idx < 0:
                    jcs_0_T = nlp.CX.eye(4)
                    jcs_1_T = nlp.CX.eye(4)

                elif coordinates_system_idx < nb_rts:
                    jcs_0 = nlp.casadi_func[f"globalJCS_{coordinates_system_idx}"](q_0)
                    jcs_0_T = vertcat(horzcat(jcs_0[:3, :3], -jcs_0[:3, :3] @ jcs_0[:3, 3]), horzcat(0, 0, 0, 1))

                    jcs_1 = nlp.casadi_func[f"globalJCS_{coordinates_system_idx}"](q_1)
                    jcs_1_T = vertcat(horzcat(jcs_1[:3, :3], -jcs_1[:3, :3] @ jcs_1[:3, 3]), horzcat(0, 0, 0, 1))

                else:
                    raise RuntimeError(
                        f"Wrong choice of coordinates_system_idx. (Negative values refer to global coordinates system, "
                        f"positive values must be between 0 and {nb_rts})"
                    )

                val = jcs_1_T @ vertcat(
                    nlp.casadi_func["biorbd_markers"](q_1)[:, markers_idx], nlp.CX.ones(1, markers_idx.shape[0])
                ) - jcs_0_T @ vertcat(
                    nlp.casadi_func["biorbd_markers"](q_0)[:, markers_idx], nlp.CX.ones(1, markers_idx.shape[0])
                )
                penalty.type.get_type().add_to_penalty(ocp, nlp, val[:3, :], penalty)

        @staticmethod
        def minimize_markers_velocity(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
        ):
            """
            Minimize a marker set velocity by computing the actual velocity of the markers
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            """

            n_q = nlp.shape["q"]
            n_qdot = nlp.shape["q_dot"]
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                penalty.index, nlp.model.nbMarkers(), "markers_idx"
            )

            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    penalty.target, (3, len(markers_idx), len(x))
                )

            for m in markers_idx:
                PenaltyFunctionAbstract._add_to_casadi_func(
                    nlp, f"biorbd_markerVelocity_{m}", nlp.model.markerVelocity, nlp.q, nlp.q_dot, int(m)
                )

            for i, v in enumerate(x):
                for m in markers_idx:
                    val = nlp.casadi_func[f"biorbd_markerVelocity_{m}"](v[:n_q], v[n_q : n_q + n_qdot])
                    penalty.sliced_target = target[:, m, i] if target is not None else None
                    penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def track_markers(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            first_marker_idx: int,
            second_marker_idx: int,
        ):
            """
            Minimize the distance between two markers
            By default this function is quadratic, meaning that it minimizes distance between them.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            first_marker_idx: int
                The index of one of the two markers
            second_marker_idx: int
                The index of one of the two markers
            """

            PenaltyFunctionAbstract._check_idx("marker", [first_marker_idx, second_marker_idx], nlp.model.nbMarkers())
            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "markers", nlp.model.markers, nlp.q)
            nq = nlp.mapping["q"].to_first.len
            for v in x:
                q = nlp.mapping["q"].to_second.map(v[:nq])
                first_marker = nlp.casadi_func["markers"](q)[:, first_marker_idx]
                second_marker = nlp.casadi_func["markers"](q)[:, second_marker_idx]

                val = first_marker - second_marker
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def proportional_variable(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            which_var: str,
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
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            which_var: str
                If the proportion is on 'states' or on 'controls'
            first_dof: int
                The index of the first variable
            second_dof: int
                The index of the second variable
            coef: float
                The proportion coefficient such that v[first_dof] = coef * v[second_dof]
            """
            """
            Adds proportionality constraint between the elements (states or controls) chosen.
            :param nlp: An instance of the OptimalControlProgram class.
            :param V: List of states or controls at nodes on which this constraint must be applied.
            :param which_var: Type of the variable constrained to be proportional. (string) ("states" or "controls")
            :param first_dof: Index of the first state or control on which this constraint must be applied. (integer)
            :param second_dof: Index of the second state or control on which this constraint must be applied. (integer)
            :param coef: Coefficient of proportionality between the two states or controls. (float)
            """

            if which_var == "states":
                ux = x
                nb_val = nlp.nx
            elif which_var == "controls":
                ux = u
                nb_val = nlp.nu
            else:
                raise RuntimeError("Wrong choice of which_var")

            PenaltyFunctionAbstract._check_idx("dof", (first_dof, second_dof), nb_val)
            if not isinstance(coef, (int, float)):
                raise RuntimeError("coef must be an int or a float")

            for v in ux:
                v = nlp.mapping["q"].to_second.map(v)
                val = v[first_dof] - coef * v[second_dof]
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def minimize_torque(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
        ):
            """
            Minimize the joint torque part of the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            """

            n_tau = nlp.shape["tau"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(penalty.index, n_tau, "controls_idx")

            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    penalty.target, (len(controls_idx), len(u))
                )
                PenaltyFunctionAbstract._add_track_data_to_plot(
                    ocp, nlp, target, combine_to="tau", axes_idx=Mapping(controls_idx)
                )

            for i, v in enumerate(u):
                val = v[controls_idx]
                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def minimize_torque_derivative(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
        ):
            """
            Minimize the joint torque velocity by comparing the torque at a node and at the next node.
            By default this function is quadratic, meaning that it minimizes the difference.
            Indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            """

            n_tau = nlp.shape["tau"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(penalty.index, n_tau, "controls_idx")

            for i in range(len(u) - 1):
                val = u[i + 1][controls_idx] - u[i][controls_idx]
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def minimize_muscles_control(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
        ):
            """
            Minimize the muscles part of the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            """

            muscles_idx = PenaltyFunctionAbstract._check_and_fill_index(
                penalty.index, nlp.shape["muscle"], "muscles_idx"
            )

            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    penalty.target, (len(muscles_idx), len(u))
                )

                PenaltyFunctionAbstract._add_track_data_to_plot(
                    ocp, nlp, target, combine_to="muscles_control", axes_idx=Mapping(muscles_idx)
                )

            # Add the nbTau offset to the muscle index
            muscles_idx_plus_tau = [idx + nlp.shape["tau"] for idx in muscles_idx]
            for i, v in enumerate(u):
                val = v[muscles_idx_plus_tau]
                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def minimize_all_controls(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
        ):
            """
            Minimize the control variables.
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            """

            n_u = nlp.nu
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(penalty.index, n_u, "muscles_idx")

            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    penalty.target, (len(controls_idx), len(u))
                )

            for i, v in enumerate(u):
                val = v[controls_idx]
                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def minimize_predicted_com_height(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
        ):
            """
            Minimize the prediction of the center of mass maximal height from the parabolic equation,
            assuming vertical axis is Z (2): CoM_dot[2]**2 / (2 * -g) + CoM[2]
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            """

            g = -9.81  # Todo: Get the gravity from biorbd
            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_CoM", nlp.model.CoM, nlp.q)
            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_CoMdot", nlp.model.CoMdot, nlp.q, nlp.q_dot)
            for i, v in enumerate(x):
                q = nlp.mapping["q"].to_second.map(v[: nlp.shape["q"]])
                q_dot = nlp.mapping["q_dot"].to_second.map(v[nlp.shape["q"] :])
                CoM = nlp.casadi_func["biorbd_CoM"](q)
                CoM_dot = nlp.casadi_func["biorbd_CoMdot"](q, q_dot)
                CoM_height = (CoM_dot[2] * CoM_dot[2]) / (2 * -g) + CoM[2]
                penalty.type.get_type().add_to_penalty(ocp, nlp, CoM_height, penalty)

        @staticmethod
        def minimize_com_position(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            axis: Axis = None,
        ):
            """
            Adds the objective that the position of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the CoM's position is minimized.
            Otherwise, the projection of the CoM's position on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            axis: Axis
                The axis to project on. Default is all axes
            """

            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(penalty.target, (1, len(x)))

            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_CoM", nlp.model.CoM, nlp.q)
            for i, v in enumerate(x):
                q = nlp.mapping["q"].to_second.map(v[: nlp.shape["q"]])
                CoM = nlp.casadi_func["biorbd_CoM"](q)

                if axis == None:
                    CoM_proj = CoM
                elif not isinstance(axis, Axis):
                    raise RuntimeError("axis must be a bioptim.Axis")
                else:
                    CoM_proj = CoM[axis]

                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, CoM_proj, penalty)

        @staticmethod
        def minimize_com_velocity(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            axis: Axis = None,
        ):
            """
            Adds the objective that the velocity of the center of mass of the model should be minimized.
            If no axis is specified, the squared-norm of the CoM's velocity is minimized.
            Otherwise, the projection of the CoM's velocity on the specified axis is minimized.
            By default this function is not quadratic, meaning that it minimizes towards infinity.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            axis: Axis
                The axis to project on. Default is all axes
            """

            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(penalty.target, (1, len(x)))

            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_CoMdot", nlp.model.CoMdot, nlp.q, nlp.q_dot)
            for i, v in enumerate(x):
                q = nlp.mapping["q"].to_second.map(v[: nlp.shape["q"]])
                q_dot = nlp.mapping["q_dot"].to_second.map(v[nlp.shape["q"] :])
                CoM_dot = nlp.casadi_func["biorbd_CoMdot"](q, q_dot)

                if axis == None:
                    CoM_dot_proj = CoM_dot[0] ** 2 + CoM_dot[1] ** 2 + CoM_dot[2] ** 2
                elif not isinstance(axis, Axis):
                    raise RuntimeError("axis must be a bioptim.Axis")
                else:
                    CoM_dot_proj = CoM_dot[axis]

                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, CoM_dot_proj, penalty)

        @staticmethod
        def minimize_contact_forces(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
        ):
            """
            Minimize the contact forces computed from dynamics with contact
            By default this function is quadratic, meaning that it minimizes towards the target.
            Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            """

            n_contact = nlp.model.nbContacts()
            contacts_idx = PenaltyFunctionAbstract._check_and_fill_index(penalty.index, n_contact, "contacts_idx")

            target = None
            if penalty.target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    penalty.target, (len(contacts_idx), len(u))
                )

                PenaltyFunctionAbstract._add_track_data_to_plot(
                    ocp, nlp, target, combine_to="contact_forces", axes_idx=Mapping(contacts_idx)
                )

            for i, v in enumerate(u):
                force = nlp.contact_forces_func(x[i], u[i], p)
                val = force[contacts_idx]
                penalty.sliced_target = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def track_segment_with_custom_rt(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            segment_idx: int,
            rt_idx: int,
        ):
            """
            Minimize the difference of the euler angles extracted from the coordinate system of a segment and a RT (e.g. IMU)
            By default this function is quadratic, meaning that it minimizes the difference.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            segment_idx: int
                The index of the segment
            rt_idx: int
                The index of the RT
            """

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

            PenaltyFunctionAbstract._add_to_casadi_func(
                nlp, f"align_segment_with_custom_rt_{segment_idx}", biorbd_meta_func, nlp.q, segment_idx, rt_idx
            )

            nq = nlp.mapping["q"].to_first.len
            for v in x:
                q = nlp.mapping["q"].to_second.map(v[:nq])
                val = nlp.casadi_func[f"align_segment_with_custom_rt_{segment_idx}"](q)
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def track_marker_with_segment_axis(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            marker_idx: int,
            segment_idx: int,
            axis: Axis,
        ):
            """
            Track a marker using a segment, that is aligning an axis toward the marker
            By default this function is quadratic, meaning that it minimizes the difference.

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
            marker_idx: int
                Index of the marker to be tracked
            segment_idx: int
                Index of the segment to align with the marker
            axis: Axis
                The axis that should be tracking the marker
            """

            if not isinstance(axis, Axis):
                raise RuntimeError("axis must be a bioptim.Axis")

            def biorbd_meta_func(q, segment_idx, marker_idx):
                r_rt = nlp.model.globalJCS(q, segment_idx)
                marker = nlp.model.marker(q, marker_idx)
                marker.applyRT(r_rt.transpose())
                return marker.to_mx()

            PenaltyFunctionAbstract._add_to_casadi_func(
                nlp,
                f"align_marker_with_segment_axis_{segment_idx}_{marker_idx}",
                biorbd_meta_func,
                nlp.q,
                segment_idx,
                marker_idx,
            )
            nq = nlp.mapping["q"].to_first.len
            for v in x:
                q = nlp.mapping["q"].to_second.map(v[:nq])
                marker = nlp.casadi_func[f"align_marker_with_segment_axis_{segment_idx}_{marker_idx}"](q)
                for axe in Axis:
                    if axe != axis:
                        # To align an axis, the other must be equal to 0
                        val = marker[axe, 0]
                        penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def custom(
            penalty: PenaltyOption,
            ocp,
            nlp,
            t: list,
            x: list,
            u: list,
            p: Union[MX, SX],
            **parameters,
        ):
            """
            A user defined penalty function

            Parameters
            ----------
            penalty: PenaltyOption
                The actual penalty to declare
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            t: list
                Time indices, maximum value being the number of shooting point + 1
            x: list
                References to the state variables
            u: list
                References to the control variables
            p: Union[MX, SX]
                References to the parameter variables
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
            ]
            for keyword in keywords:
                if keyword in inspect.signature(penalty.custom_function).parameters:
                    raise TypeError(f"{keyword} is a reserved word and cannot be used in a custom function signature")

            val = penalty.custom_function(ocp, nlp, t, x, u, p, **parameters)
            if isinstance(val, tuple):
                if penalty.min_bound is not None or penalty.max_bound is not None:
                    raise RuntimeError(
                        "You cannot have non linear bounds for custom constraints and min_bound or max_bound defined"
                    )
                penalty.min_bound = val[0]
                penalty.max_bound = val[2]
                val = val[1]

            penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty)

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
    def add_or_replace(ocp, nlp, penalty: PenaltyOption):
        """
        Doing some configuration on the penalty and add it to the list of penalty

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        penalty: PenaltyOption
            The actual penalty to declare
        """

        t, x, u = PenaltyFunctionAbstract._get_node(nlp, penalty)
        penalty_type = penalty.type.get_type()

        penalty_type._span_checker(penalty, nlp)
        penalty_type._parameter_modifier(penalty)

        penalty_type.clear_penalty(ocp, nlp, penalty)
        penalty.type.value[0](penalty, ocp, nlp, t, x, u, nlp.p, **penalty.params)

    @staticmethod
    def _add_to_casadi_func(nlp, name: str, function: Callable, *all_param):
        """
        Add to the pool of declared casadi function. If the function already exists, it is skipped

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        name: str
            The unique name of the function to add to the casadi functions pool
        function: function
            The biorbd function to add
        all_param: dict
            Any parameters to pass to the biorbd function
        """
        if name in nlp.casadi_func:
            return
        else:
            nlp.casadi_func[name] = biorbd.to_casadi_func(name, function, *all_param)

    @staticmethod
    def _parameter_modifier(penalty: PenaltyOption):
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
                or func == PenaltyType.ALIGN_MARKERS
                or func == PenaltyType.PROPORTIONAL_STATE
                or func == PenaltyType.PROPORTIONAL_CONTROL
                or func == PenaltyType.MINIMIZE_TORQUE
                or func == PenaltyType.MINIMIZE_MUSCLES_CONTROL
                or func == PenaltyType.MINIMIZE_ALL_CONTROLS
                or func == PenaltyType.MINIMIZE_CONTACT_FORCES
                or func == PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT
                or func == PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS
                or func == PenaltyType.MINIMIZE_TORQUE_DERIVATIVE
                or func == PenaltyType.MINIMIZE_COM_POSITION
                or func == PenaltyType.MINIMIZE_COM_VELOCITY
            ):
                penalty.quadratic = True
            else:
                penalty.quadratic = False

        if func == PenaltyType.PROPORTIONAL_STATE:
            penalty.params["which_var"] = "states"
        if func == PenaltyType.PROPORTIONAL_CONTROL:
            penalty.params["which_var"] = "controls"

    @staticmethod
    def _span_checker(penalty: PenaltyOption, nlp):
        """
        Check for any non sense in the requested times for the constraint. Raises an error if so

        Parameters
        ----------
        penalty: PenaltyOption
            The actual penalty to declare
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        """

        func = penalty.type.value[0]
        node = penalty.node
        # Everything that is suspicious in terms of the span of the penalty function ca be checked here
        if (
            func == PenaltyType.PROPORTIONAL_CONTROL
            or func == PenaltyType.MINIMIZE_TORQUE
            or func == PenaltyType.MINIMIZE_MUSCLES_CONTROL
            or func == PenaltyType.MINIMIZE_ALL_CONTROLS
        ):
            if node == Node.END or node == nlp.ns:
                raise RuntimeError("No control u at last node")

    @staticmethod
    def _check_and_fill_index(var_idx: Union[list, int], target_size: int, var_name: str = "var"):
        """
        Checks if the variable index is consistent with the requested variable.

        Parameters
        ----------
        var_idx: Union[list, int]
            Indices of the variable
        target_size: int
            The size of the variable array
        var_name: str
            The type of variable, it is use for raise message purpose

        Returns
        -------
        The formatted indices
        """

        if var_idx is None:
            var_idx = range(target_size)
        else:
            if isinstance(var_idx, int):
                var_idx = [var_idx]
            if max(var_idx) > target_size:
                raise RuntimeError(f"{var_name} in cannot be higher than nx ({target_size})")
        out = np.array(var_idx)
        if not np.issubdtype(out.dtype, np.integer):
            raise RuntimeError(f"{var_name} must be a list of integer")
        return out

    @staticmethod
    def _check_and_fill_tracking_data_size(data_to_track: np.ndarray, target_size: Union[list, tuple]):
        """
        Checks if the variable index is consistent with the requested variable.
        If the function returns, all is okay

        Parameters
        ----------
        data_to_track: np.ndarray
            The data to track matrix
        target_size: Union[list, tuple]
            The expected shape (n, m) of the data to track
        """

        if data_to_track is not None:
            if len(data_to_track.shape) == 1:
                raise RuntimeError(
                    f"data_to_track cannot be a vector (it can be a matrix with time dimension equals to 1 though)"
                )
            if data_to_track.shape[1] == 1:
                data_to_track = np.repeat(data_to_track, target_size[1], axis=1)

            if data_to_track.shape != target_size:
                raise RuntimeError(
                    f"data_to_track {data_to_track.shape} does not correspond to expected size {target_size}"
                )
        else:
            raise RuntimeError("data_to_track is None and that should not happen, please contact a developer")
        return data_to_track

    @staticmethod
    def _check_idx(name: str, elements: Union[list, tuple, int], max_nb_elements: int = inf, min_nb_elements: int = 0):
        """
        Generic sanity check for requested dimensions.
        If the function returns, everything is okay

        Parameters
        name: str
            Name of the element
        elements: Union[list, tuple, int]
            Index of the slicing of the array variable
        max_nb_elements: int
            The maximal shape of the element
        min_nb_elements: int
            The maximal shape of the element
        """

        if not isinstance(elements, (list, tuple)):
            elements = (elements,)
        for element in elements:
            if not isinstance(element, int):
                raise RuntimeError(f"{element} is not a valid index for {name}, it must be an integer")
            if element < min_nb_elements or element >= max_nb_elements:
                raise RuntimeError(
                    f"{element} is not a valid index for {name}, it must be between {min_nb_elements} and {max_nb_elements - 1}."
                )

    @staticmethod
    def add_to_penalty(ocp, nlp, val: Union[MX, SX], penalty: PenaltyOption):
        """
        Add the constraint to the penalty pool (abstract)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        val: Union[MX, SX]
            The actual constraint to add
        penalty: PenaltyOption
            The actual penalty to declare
        """

        raise RuntimeError("_add_to_penalty cannot be called from an abstract class")

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

        raise RuntimeError("_get_type cannot be called from an abstract class")

    @staticmethod
    def _get_node(nlp, penalty: PenaltyOption):
        """
        Get the actual node (time, X and U) specified in the penalty

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        penalty: PenaltyOption
            The actual penalty to declare

        Returns
        -------
        The actual node (time, X and U) specified in the penalty
        """

        if not isinstance(penalty.node, (list, tuple)):
            penalty.node = (penalty.node,)
        t = []
        x = []
        u = []
        for node in penalty.node:
            if isinstance(node, int):
                if node < 0 or node > nlp.ns:
                    raise RuntimeError(f"Invalid node, {node} must be between 0 and {nlp.ns}")
                t.append(node)
                x.append(nlp.X[node])
                if (
                    nlp.control_type == ControlType.CONSTANT and node != nlp.ns
                ) or nlp.control_type != ControlType.CONSTANT:
                    u.append(nlp.U[node])

            elif node == Node.START:
                t.append(0)
                x.append(nlp.X[0])
                u.append(nlp.U[0])

            elif node == Node.MID:
                if nlp.ns % 2 == 1:
                    raise (ValueError("Number of shooting points must be even to use MID"))
                t.append(nlp.ns // 2)
                x.append(nlp.X[nlp.ns // 2])
                u.append(nlp.U[nlp.ns // 2])

            elif node == Node.INTERMEDIATES:
                for i in range(1, nlp.ns - 1):
                    t.append(i)
                    x.append(nlp.X[i])
                    u.append(nlp.U[i])

            elif node == Node.END:
                t.append(nlp.ns)
                x.append(nlp.X[nlp.ns])

            elif node == Node.ALL:
                t.extend([i for i in range(nlp.ns + 1)])
                for i in range(nlp.ns):
                    x.append(nlp.X[i])
                    u.append(nlp.U[i])
                x.append(nlp.X[nlp.ns])

            else:
                raise RuntimeError(" is not a valid node")
        return t, x, u

    @staticmethod
    def _add_track_data_to_plot(
        ocp,
        nlp,
        data: np.ndarray,
        combine_to: str,
        axes_idx: Union[Mapping, tuple, list] = None,
    ):
        """
        Interface to the plot so it can be properly added to the proper plot

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        data: np.ndarray
            The actual tracking data to plot
        combine_to: str
            The name of the underlying plot to combine the tracking data to
        axes_idx: Union[Mapping, tuple, list]
            The index of the matplotlib axes
        """

        if (isinstance(data, np.ndarray) and not data.any()) or (not isinstance(data, np.ndarray) and not data):
            return

        if data.shape[1] == nlp.ns:
            data = np.c_[data, data[:, -1]]
        ocp.add_plot(
            combine_to,
            lambda x, u, p: data,
            color="tab:red",
            linestyle=".-",
            plot_type=PlotType.STEP,
            phase=nlp.phase_idx,
            axes_idx=axes_idx,
        )


class PenaltyType(Enum):
    """
    Selection of valid penalty functions
    """

    MINIMIZE_STATE = PenaltyFunctionAbstract.Functions.minimize_states
    TRACK_STATE = MINIMIZE_STATE
    MINIMIZE_MARKERS = PenaltyFunctionAbstract.Functions.minimize_markers
    TRACK_MARKERS = MINIMIZE_MARKERS
    MINIMIZE_MARKERS_DISPLACEMENT = PenaltyFunctionAbstract.Functions.minimize_markers_displacement
    MINIMIZE_MARKERS_VELOCITY = PenaltyFunctionAbstract.Functions.minimize_markers_velocity
    TRACK_MARKERS_VELOCITY = MINIMIZE_MARKERS_VELOCITY
    ALIGN_MARKERS = PenaltyFunctionAbstract.Functions.track_markers
    PROPORTIONAL_STATE = PenaltyFunctionAbstract.Functions.proportional_variable
    PROPORTIONAL_CONTROL = PenaltyFunctionAbstract.Functions.proportional_variable
    MINIMIZE_TORQUE = PenaltyFunctionAbstract.Functions.minimize_torque
    TRACK_TORQUE = MINIMIZE_TORQUE
    MINIMIZE_TORQUE_DERIVATIVE = PenaltyFunctionAbstract.Functions.minimize_torque_derivative
    MINIMIZE_MUSCLES_CONTROL = PenaltyFunctionAbstract.Functions.minimize_muscles_control
    TRACK_MUSCLES_CONTROL = MINIMIZE_MUSCLES_CONTROL
    MINIMIZE_ALL_CONTROLS = PenaltyFunctionAbstract.Functions.minimize_all_controls
    TRACK_ALL_CONTROLS = MINIMIZE_ALL_CONTROLS
    MINIMIZE_CONTACT_FORCES = PenaltyFunctionAbstract.Functions.minimize_contact_forces
    TRACK_CONTACT_FORCES = MINIMIZE_CONTACT_FORCES
    MINIMIZE_PREDICTED_COM_HEIGHT = PenaltyFunctionAbstract.Functions.minimize_predicted_com_height
    MINIMIZE_COM_POSITION = PenaltyFunctionAbstract.Functions.minimize_com_position
    MINIMIZE_COM_VELOCITY = PenaltyFunctionAbstract.Functions.minimize_com_velocity
    ALIGN_SEGMENT_WITH_CUSTOM_RT = PenaltyFunctionAbstract.Functions.track_segment_with_custom_rt
    ALIGN_MARKER_WITH_SEGMENT_AXIS = PenaltyFunctionAbstract.Functions.track_marker_with_segment_axis
    CUSTOM = PenaltyFunctionAbstract.Functions.custom
