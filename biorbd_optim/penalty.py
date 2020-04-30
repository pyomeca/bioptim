from enum import Enum
from math import inf

import numpy as np
import biorbd

from .enums import Instant
from .enums import Axe


class PenaltyFunctionAbstract:
    class Functions:
        @staticmethod
        def minimize_states(penalty_type, ocp, nlp, t, x, u, data_to_track=(), states_idx=(), **extra_param):
            states_idx = PenaltyFunctionAbstract._check_and_fill_index(states_idx, nlp["nx"], "state_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"] + 1, len(states_idx)]
            )

            for i, v in enumerate(x):
                val = v[states_idx] - data_to_track[t[i], states_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_markers(penalty_type, ocp, nlp, t, x, u, markers_idx=(), data_to_track=(), **extra_param):
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp["model"].nbMarkers(), "markers_idx"
            )
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [3, max(markers_idx) + 1, nlp["ns"] + 1]
            )

            nq = nlp["q_mapping"].reduce.len
            for i, v in enumerate(x):
                q = nlp["q_mapping"].expand.map(v[:nq])
                val = nlp["model"].markers(q)[:, markers_idx] - data_to_track[:, markers_idx, t[i]]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_markers_displacement(penalty_type, ocp, nlp, t, x, u, markers_idx=(), **extra_param):
            n_q = nlp["nbQ"]
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp["model"].nbMarkers(), "markers_idx"
            )

            for i in range(len(x) - 1):
                val = (
                    nlp["model"].markers(x[i + 1][:n_q])[:, markers_idx]
                    - nlp["model"].markers(x[i][:n_q])[:, markers_idx]
                )
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_markers_velocity(penalty_type, ocp, nlp, t, x, u, markers_idx=(), data_to_track=(), **extra_param):
            n_q = nlp["nbQ"]
            n_qdot = nlp["nbQdot"]
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp["model"].nbMarkers(), "markers_idx"
            )
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [3, max(markers_idx) + 1, nlp["ns"] + 1]
            )

            for i, v in enumerate(x):
                val = (
                    nlp["model"].markerVelocity(v[:n_q], v[n_q : n_q + n_qdot], markers_idx).to_mx()
                    - data_to_track[:, markers_idx, t[i]]
                )
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def align_markers(penalty_type, ocp, nlp, t, x, u, first_marker, second_marker, **extra_param):
            """
            Adds the constraint that the two markers must be coincided at the desired instant(s).
            :param nlp: An OptimalControlProgram class.
            :param x: List of instant(s).
            :param policy: Tuple of indices of two markers.
            """
            PenaltyFunctionAbstract._check_idx("marker", [first_marker, second_marker], nlp["model"].nbMarkers())

            nq = nlp["q_mapping"].reduce.len
            for v in x:
                q = nlp["q_mapping"].expand.map(v[:nq])
                marker1 = nlp["model"].marker(q, first_marker).to_mx()
                marker2 = nlp["model"].marker(q, second_marker).to_mx()

                val = marker1 - marker2
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def proportional_variable(
            penalty_type, ocp, nlp, t, x, u, which_var, first_dof, second_dof, coef, **extra_param
        ):
            """
            Adds proportionality constraint between the elements (states or controls) chosen.
            :param nlp: An instance of the OptimalControlProgram class.
            :param V: List of states or controls at instants on which this constraint must be applied.
            :param policy: A tuple or a tuple of tuples (also works with lists) whose first two elements
            are the indexes of elements to be linked proportionally.
            The third element of each tuple (policy[i][2]) is the proportionality coefficient.
            """
            if which_var == "states":
                ux = x
                nb_val = nlp["nx"]
            elif which_var == "controls":
                ux = u
                nb_val = nlp["nu"]
            else:
                raise RuntimeError("Wrong choice of which_var")

            PenaltyFunctionAbstract._check_idx("dof", (first_dof, second_dof), nb_val)
            if not isinstance(coef, (int, float)):
                raise RuntimeError("coef must be an int or a float")

            for v in ux:
                v = nlp["q_mapping"].expand.map(v)
                val = v[first_dof] - coef * v[second_dof]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_torque(penalty_type, ocp, nlp, t, x, u, controls_idx=(), data_to_track=(), **extra_param):
            n_tau = nlp["nbTau"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_tau, "controls_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"], max(controls_idx) + 1]
            )

            for i, v in enumerate(u):
                val = v[controls_idx] - data_to_track[t[i], controls_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_muscles_control(penalty_type, ocp, nlp, t, x, u, muscles_idx=(), data_to_track=(), **extra_param):
            n_tau = nlp["nbTau"]
            nb_muscle = nlp["nbMuscle"]
            muscles_idx = PenaltyFunctionAbstract._check_and_fill_index(muscles_idx, nb_muscle, "muscles_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"], max(muscles_idx) + 1]
            )

            # Add the nbTau offset to the muscle index
            muscles_idx_plus_tau = [idx + n_tau for idx in muscles_idx]
            for i, v in enumerate(u):
                val = v[muscles_idx_plus_tau] - data_to_track[t[i], muscles_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_all_controls(penalty_type, ocp, nlp, t, x, u, controls_idx=(), data_to_track=(), **extra_param):
            n_u = nlp["nu"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_u, "muscles_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"], max(controls_idx) + 1]
            )

            for i, v in enumerate(u):
                val = v[controls_idx] - data_to_track[t[i], controls_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_predicted_com_height(penalty_type, ocp, nlp, t, x, u, **extra_param):
            g = -9.81  # get gravity from biorbd

            for i, v in enumerate(x):
                q = nlp["q_mapping"].expand.map(v[: nlp["nbQ"]])
                q_dot = nlp["q_dot_mapping"].expand.map(v[nlp["nbQ"] :])
                CoM = nlp["model"].CoM(q).to_mx()
                CoM_dot = nlp["model"].CoMdot(q, q_dot).to_mx()
                CoM_height = (CoM_dot[2] * CoM_dot[2]) / (2 * -g) + CoM[2]

                penalty_type._add_to_penalty(ocp, nlp, CoM_height, **extra_param)

        @staticmethod
        def align_segment_with_custom_rt(penalty_type, ocp, nlp, t, x, u, segment_idx, rt_idx, **extra_param):
            """
            Adds the constraint that the RT and the segment must be aligned at the desired instant(s).
            :param nlp: An OptimalControlProgram class.
            :param X: List of instant(s).
            :param policy: Tuple of indices of segment and rt.
            """
            PenaltyFunctionAbstract._check_idx("segment", segment_idx, nlp["model"].nbSegment())
            PenaltyFunctionAbstract._check_idx("rt", rt_idx, nlp["model"].nbRTs())

            nq = nlp["q_mapping"].reduce.len
            for v in x:
                q = nlp["q_mapping"].expand.map(v[:nq])
                r_seg = nlp["model"].globalJCS(q, segment_idx).rot()
                r_rt = nlp["model"].RT(q, rt_idx).rot()
                val = biorbd.Rotation_toEulerAngles(r_seg.transpose() * r_rt, "zyx").to_mx()
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def align_marker_with_segment_axis(
            penalty_type, ocp, nlp, t, x, u, marker_idx, segment_idx, axis, **extra_param
        ):
            if not isinstance(axis, Axe):
                raise RuntimeError("axis must be a biorbd_optim.Axe")

            nq = nlp["q_mapping"].reduce.len
            for v in x:
                q = nlp["q_mapping"].expand.map(v[:nq])

                r_rt = nlp["model"].globalJCS(q, segment_idx)
                n_seg = nlp["model"].marker(q, marker_idx)
                n_seg.applyRT(r_rt.transpose())
                n_seg = n_seg.to_mx()

                for axe in Axe:
                    if axe != axis:
                        # To align an axis, the other must be equal to 0
                        val = n_seg[axe, 0]
                        penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def custom(penalty_type, ocp, nlp, t, x, u, **parameters):
            func = parameters["function"]
            weight = None
            if "weight" in parameters.keys():
                weight = parameters["weight"]
            del parameters["function"]
            del parameters["weight"]
            val = func(ocp, nlp, t, x, u, **parameters)
            if weight is not None:
                parameters["weight"] = weight
            penalty_type._add_to_penalty(ocp, nlp, val, **parameters)

    @staticmethod
    def add(ocp, nlp):
        raise RuntimeError("add cannot be called from an abstract class")

    @staticmethod
    def _add(ocp, nlp, key=None):
        for parameters in nlp[key]:
            t, x, u = PenaltyFunctionAbstract.__get_instant(nlp, parameters)
            penalty_function = parameters["type"].value[0]
            penalty_type = parameters["type"]._get_type()
            instant = parameters["instant"]
            del parameters["instant"], parameters["type"]

            penalty_type._span_checker(penalty_function, instant, nlp)
            penalty_type._parameter_modifier(penalty_function, parameters)

            penalty_function(penalty_type, ocp, nlp, t, x, u, **parameters)

    @staticmethod
    def _parameter_modifier(penalty_function, parameters):
        # Everything that should change the entry parameters depending on the penalty can be added here
        if (
            penalty_function == PenaltyType.MINIMIZE_STATE
            or penalty_function == PenaltyType.MINIMIZE_MARKERS
            or penalty_function == PenaltyType.MINIMIZE_MARKERS_DISPLACEMENT
            or penalty_function == PenaltyType.MINIMIZE_MARKERS_VELOCITY
            or penalty_function == PenaltyType.ALIGN_MARKERS
            or penalty_function == PenaltyType.PROPORTIONAL_STATE
            or penalty_function == PenaltyType.PROPORTIONAL_CONTROL
            or penalty_function == PenaltyType.MINIMIZE_TORQUE
            or penalty_function == PenaltyType.MINIMIZE_MUSCLES_CONTROL
            or penalty_function == PenaltyType.MINIMIZE_ALL_CONTROLS
            or penalty_function == PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT
            or penalty_function == PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS
        ):
            if "quadratic" not in parameters.keys():
                parameters["quadratic"] = True

        if penalty_function == PenaltyType.PROPORTIONAL_STATE:
            parameters["which_var"] = "states"
        if penalty_function == PenaltyType.PROPORTIONAL_CONTROL:
            parameters["which_var"] = "controls"

    @staticmethod
    def _span_checker(penalty_function, instant, nlp):
        # Everything that is suspicious in terms of the span of the penalty function ca be checked here
        if (
            penalty_function == PenaltyType.PROPORTIONAL_CONTROL
            or penalty_function == PenaltyType.MINIMIZE_TORQUE
            or penalty_function == PenaltyType.MINIMIZE_MUSCLES_CONTROL
            or penalty_function == PenaltyType.MINIMIZE_ALL_CONTROLS
        ):
            if instant == Instant.END or instant == nlp["ns"]:
                raise RuntimeError("No control u at last node")

    @staticmethod
    def _check_and_fill_index(var_idx, target_size, var_name="var"):
        if var_idx == ():
            var_idx = range(target_size)
        else:
            if isinstance(var_idx, int):
                var_idx = [var_idx]
            if max(var_idx) > target_size:
                raise RuntimeError(f"{var_name} in minimize_states cannot be higher than nx ({target_size})")
        return var_idx

    @staticmethod
    def _check_and_fill_tracking_data_size(data_to_track, target_size):
        if data_to_track == ():
            data_to_track = np.zeros(target_size)
        else:
            if len(data_to_track.shape) != len(target_size):
                if target_size[1] == 1 and len(data_to_track.shape) == 1:
                    # If we have a vector it is still okay
                    data_to_track = data_to_track.reshape(data_to_track.shape[0], 1)
                else:
                    raise RuntimeError(
                        f"data_to_track {data_to_track.shape}don't correspond to expected minimum size {target_size}"
                    )
            for i in range(len(target_size)):
                if data_to_track.shape[i] < target_size[i]:
                    raise RuntimeError(
                        f"data_to_track {data_to_track.shape} don't correspond to expected minimum size {target_size}"
                    )
        return data_to_track

    @staticmethod
    def _check_idx(name, elements, max_bound=inf, min_bound=0):
        if not isinstance(elements, (list, tuple)):
            elements = (elements,)
        for element in elements:
            if not isinstance(element, int):
                raise RuntimeError(f"{element} is not a valid index for {name}, it must be an integer")
            if element < min_bound or element >= max_bound:
                raise RuntimeError(
                    f"{element} is not a valid index for {name}, it must be between 0 and {max_bound - 1}."
                )

    @staticmethod
    def _add_to_penalty(ocp, nlp, val, **extra_param):
        raise RuntimeError("_add_to_penalty cannot be called from an abstract class")

    @staticmethod
    def _get_type():
        raise RuntimeError("_get_type cannot be called from an abstract class")

    @staticmethod
    def __get_instant(nlp, constraint):
        if not isinstance(constraint["instant"], (list, tuple)):
            constraint["instant"] = (constraint["instant"],)
        t = []
        x = []
        u = []
        for node in constraint["instant"]:
            if isinstance(node, int):
                if node < 0 or node > nlp["ns"]:
                    raise RuntimeError(f"Invalid instant, {node} must be between 0 and {nlp['ns']}")
                t.append(node)
                x.append(nlp["X"][node])
                u.append(nlp["U"][node])

            elif node == Instant.START:
                t.append(0)
                x.append(nlp["X"][0])
                u.append(nlp["U"][0])

            elif node == Instant.MID:
                if nlp["ns"] % 2 == 1:
                    raise (ValueError("Number of shooting points must be even to use MID"))
                t.append(nlp["X"][nlp["ns"] // 2])
                x.append(nlp["X"][nlp["ns"] // 2])
                u.append(nlp["U"][nlp["ns"] // 2])

            elif node == Instant.INTERMEDIATES:
                for i in range(1, nlp["ns"] - 1):
                    t.append(i)
                    x.append(nlp["X"][i])
                    u.append(nlp["U"][i])

            elif node == Instant.END:
                t.append(nlp["X"][nlp["ns"]])
                x.append(nlp["X"][nlp["ns"]])

            elif node == Instant.ALL:
                t.extend([i for i in range(nlp["ns"] + 1)])
                for i in range(nlp["ns"]):
                    x.append(nlp["X"][i])
                    u.append(nlp["U"][i])
                x.append(nlp["X"][nlp["ns"]])

            else:
                raise RuntimeError(" is not a valid instant")
        return t, x, u


class PenaltyType(Enum):
    """
    Different conditions between biorbd geometric structures.
    """

    MINIMIZE_STATE = PenaltyFunctionAbstract.Functions.minimize_states
    TRACK_STATE = MINIMIZE_STATE
    MINIMIZE_MARKERS = PenaltyFunctionAbstract.Functions.minimize_markers
    TRACK_MARKERS = MINIMIZE_MARKERS
    MINIMIZE_MARKERS_DISPLACEMENT = PenaltyFunctionAbstract.Functions.minimize_markers_displacement
    MINIMIZE_MARKERS_VELOCITY = PenaltyFunctionAbstract.Functions.minimize_markers_velocity
    TRACK_MARKERS_VELOCITY = MINIMIZE_MARKERS_VELOCITY
    ALIGN_MARKERS = PenaltyFunctionAbstract.Functions.align_markers
    PROPORTIONAL_STATE = PenaltyFunctionAbstract.Functions.proportional_variable
    PROPORTIONAL_CONTROL = PenaltyFunctionAbstract.Functions.proportional_variable
    MINIMIZE_TORQUE = PenaltyFunctionAbstract.Functions.minimize_torque
    TRACK_TORQUE = MINIMIZE_TORQUE
    MINIMIZE_MUSCLES_CONTROL = PenaltyFunctionAbstract.Functions.minimize_muscles_control
    TRACK_MUSCLES_CONTROL = MINIMIZE_MUSCLES_CONTROL
    MINIMIZE_ALL_CONTROLS = PenaltyFunctionAbstract.Functions.minimize_all_controls
    TRACK_ALL_CONTROLS = MINIMIZE_ALL_CONTROLS
    MINIMIZE_PREDICTED_COM_HEIGHT = PenaltyFunctionAbstract.Functions.minimize_predicted_com_height
    ALIGN_SEGMENT_WITH_CUSTOM_RT = PenaltyFunctionAbstract.Functions.align_segment_with_custom_rt
    ALIGN_MARKER_WITH_SEGMENT_AXIS = PenaltyFunctionAbstract.Functions.align_marker_with_segment_axis
    CUSTOM = PenaltyFunctionAbstract.Functions.custom
