from enum import Enum
from math import inf

import numpy as np
from casadi import horzcat, horzsplit, MX
import biorbd

from .enums import Instant
from .enums import Axe


class PenaltyFunctionAbstract:
    class Type(Enum):
        """
        Different conditions between biorbd geometric structures.
        """

        MINIMIZE_STATE = "minimize_state"
        MINIMIZE_MARKERS = "minimize_markers"
        MINIMIZE_MARKERS_DISPLACEMENT = "minimize_markers_displacement"
        MINIMIZE_MARKERS_VELOCITY = "minimize_markers_velocity"
        ALIGN_MARKERS = "align_match"
        PROPORTIONAL_STATE = "proportional_state"
        PROPORTIONAL_CONTROL = "proportional_control"
        MINIMIZE_TORQUE = "minimize_torque"
        MINIMIZE_MUSCLES = "minimize_muscles"
        MINIMIZE_ALL_CONTROLS = "minimize_all_controls"
        MINIMIZE_PREDICTED_COM_HEIGHT = "minimize_predicted_com_height"
        ALIGN_SEGMENT_WITH_CUSTOM_RT = "align_segment_with_custom_rt"
        ALIGN_MARKER_WITH_SEGMENT_AXIS = "align_marker_with_segment_axis"
        CUSTOM = "custom"

    class Functions:
        @staticmethod
        def minimize_states(_type, ocp, nlp, t, x, data_to_track=(), states_idx=(), weight=1):
            states_idx = PenaltyFunctionAbstract._check_and_fill_index(states_idx, nlp["nx"], "state_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(data_to_track, [nlp["ns"] + 1, len(states_idx)])

            for i, v in enumerate(horzsplit(x, 1)):
                val = v[states_idx] - data_to_track[t[i], states_idx]
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def minimize_markers(_type, ocp, nlp, t, x, markers_idx=(), data_to_track=(), weight=1):
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp["model"].nbMarkers(), "markers_idx"
            )
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [3, max(markers_idx) + 1, nlp["ns"] + 1]
            )

            nq = nlp["q_mapping"].reduce.len
            for i, v in enumerate(horzsplit(x, 1)):
                q = nlp["q_mapping"].expand.map(v[:nq])
                val = nlp["model"].markers(q)[:, markers_idx] - data_to_track[:, markers_idx, t[i]]
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def minimize_markers_displacement(_type, ocp, nlp, x, markers_idx=(), weight=1):
            n_q = nlp["nbQ"]
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(markers_idx, nlp["model"].nbMarkers(), "markers_idx")

            X = horzsplit(x, 1)
            for i in range(len(X)-1):
                val = nlp["model"].markers(X[i+1][:n_q])[:, markers_idx] - nlp["model"].markers(X[i][:n_q])[:, markers_idx]
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def minimize_markers_velocity(_type, ocp, nlp, t, x, markers_idx=(), data_to_track=(), weight=1):
            n_q = nlp["nbQ"]
            n_qdot = nlp["nbQdot"]
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(markers_idx, nlp["model"].nbMarkers(), "markers_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [3, max(markers_idx) + 1, nlp["ns"] + 1]
            )

            for i, v in enumerate(horzsplit(x, 1)):
                val = nlp["model"].markerVelocity(v[:n_q], v[n_q : n_q + n_qdot], markers_idx).to_mx() - data_to_track[:, markers_idx, t[i]]
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def align_markers(_type, ocp, nlp, x, first_marker, second_marker, weight=1):
            """
            Adds the constraint that the two markers must be coincided at the desired instant(s).
            :param nlp: An OptimalControlProgram class.
            :param x: List of instant(s).
            :param policy: Tuple of indices of two markers.
            """
            PenaltyFunctionAbstract._check_idx("marker", [first_marker, second_marker], nlp["model"].nbMarkers())

            nq = nlp["q_mapping"].reduce.len
            for v in horzsplit(x, 1):
                q = nlp["q_mapping"].expand.map(v[:nq])
                marker1 = nlp["model"].marker(q, first_marker).to_mx()
                marker2 = nlp["model"].marker(q, second_marker).to_mx()

                val = marker1 - marker2
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def proportional_variable(_type, ocp, nlp, UX, first_dof, second_dof, coef, weight=1):
            """
            Adds proportionality constraint between the elements (states or controls) chosen.
            :param nlp: An instance of the OptimalControlProgram class.
            :param V: List of states or controls at instants on which this constraint must be applied.
            :param policy: A tuple or a tuple of tuples (also works with lists) whose first two elements
            are the indexes of elements to be linked proportionally.
            The third element of each tuple (policy[i][2]) is the proportionality coefficient.
            """
            PenaltyFunctionAbstract._check_idx("dof", (first_dof, second_dof), UX.rows())
            if not isinstance(coef, (int, float)):
                raise RuntimeError("coef must be an int or a float")

            for v in horzsplit(UX, 1):
                v = nlp["q_mapping"].expand.map(v)
                val = v[first_dof] - coef * v[second_dof]
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def minimize_torque(_type, ocp, nlp, t, u, controls_idx=(), data_to_track=(), weight=1):
            n_tau = nlp["nbTau"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_tau, "controls_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(data_to_track, [nlp["ns"], max(controls_idx) + 1])

            for i, v in enumerate(horzsplit(u, 1)):
                val = v[controls_idx] - data_to_track[t[i], controls_idx]
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def minimize_muscles(_type, ocp, nlp, t, u, muscles_idx=(), data_to_track=(), weight=1):
            n_tau = nlp["nbTau"]
            nb_muscle = nlp["nbMuscle"]
            muscles_idx = PenaltyFunctionAbstract._check_and_fill_index(muscles_idx, nb_muscle, "muscles_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(data_to_track, [nlp["ns"], max(muscles_idx) + 1])

            # Add the nbTau offset to the muscle index
            muscles_idx_plus_tau = [idx + n_tau for idx in muscles_idx]
            for i, v in enumerate(horzsplit(u, 1)):
                val = v[muscles_idx_plus_tau] - data_to_track[t[i], muscles_idx]
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def minimize_all_controls(_type, ocp, nlp, t, u, controls_idx=(), data_to_track=(), weight=1):
            n_u = nlp["nu"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_u, "muscles_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"], max(controls_idx) + 1])

            for i, v in enumerate(horzsplit(u, 1)):
                val = v[controls_idx] - data_to_track[t[i], controls_idx]
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def minimize_predicted_com_height(_type, ocp, nlp, x, weight=1):
            g = -9.81  # get gravity from biorbd

            for i, v in enumerate(horzsplit(x, 1)):
                q = nlp["q_mapping"].expand.map(v[: nlp["nbQ"]])
                q_dot = nlp["q_dot_mapping"].expand.map(v[nlp["nbQ"]:])
                CoM = nlp["model"].CoM(q).to_mx()
                CoM_dot = nlp["model"].CoMdot(q, q_dot).to_mx()
                CoM_height = (CoM_dot[2] * CoM_dot[2]) / (2 * -g) + CoM[2]

                _type._add_to_goal(ocp, nlp, CoM_height, weight)

        @staticmethod
        def align_segment_with_custom_rt(_type, ocp, nlp, x, segment_idx, rt_idx, weight=1):
            """
            Adds the constraint that the RT and the segment must be aligned at the desired instant(s).
            :param nlp: An OptimalControlProgram class.
            :param X: List of instant(s).
            :param policy: Tuple of indices of segment and rt.
            """
            PenaltyFunctionAbstract._check_idx("segment", segment_idx, nlp["model"].nbSegment())
            PenaltyFunctionAbstract._check_idx("rt", rt_idx, nlp["model"].nbRTs())

            nq = nlp["q_mapping"].reduce.len
            for v in horzsplit(x, 1):
                q = nlp["q_mapping"].expand.map(v[:nq])
                r_seg = nlp["model"].globalJCS(q, segment_idx).rot()
                r_rt = nlp["model"].RT(q, rt_idx).rot()
                val = biorbd.Rotation_toEulerAngles(r_seg.transpose() * r_rt, "zyx").to_mx()
                _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def align_marker_with_segment_axis(_type, ocp, nlp, X, marker_idx, segment_idx, axis, weight=1):
            if not isinstance(axis, Axe):
                raise RuntimeError("axis must be a biorbd_optim.Axe")

            nq = nlp["q_mapping"].reduce.len
            for x in horzsplit(X, 1):
                q = nlp["q_mapping"].expand.map(x[:nq])

                r_rt = nlp["model"].globalJCS(q, segment_idx)
                n_seg = nlp["model"].marker(q, marker_idx)
                n_seg.applyRT(r_rt.transpose())
                n_seg = n_seg.to_mx()

                for axe in Axe:
                    if axe != axis:
                        # To align an axis, the other must be equal to 0
                        val = n_seg[axe, 0]
                        _type._add_to_goal(ocp, nlp, val, weight)

        @staticmethod
        def custom(_type, ocp, nlp, t, x, u, weight=1, **parameters):
            func = parameters["function"]
            del parameters["function"]
            X = horzsplit(x, 1)
            U = horzsplit(u, 1)
            val = func(ocp, nlp, t, X, U, **parameters)
            _type._add_to_goal(ocp, nlp, val, weight)

    @staticmethod
    def _add(ocp, nlp, key=None):
        if nlp[key] is None:
            return

        for parameters in nlp[key]:
            t, x, u = PenaltyFunctionAbstract.__get_instant(nlp, parameters)
            _type = parameters["type"]
            func_type = _type._get_type()
            instant = parameters["instant"]
            del parameters["instant"], parameters["type"]

            if _type == func_type.MINIMIZE_STATE:
                PenaltyFunctionAbstract.Functions.minimize_states(
                    func_type, ocp, nlp, t, x, **parameters)

            elif _type == func_type.MINIMIZE_MARKERS:
                PenaltyFunctionAbstract.Functions.minimize_markers(
                    func_type, ocp, nlp, t, x, **parameters)

            elif _type == func_type.MINIMIZE_MARKERS_DISPLACEMENT:
                PenaltyFunctionAbstract.Functions.minimize_markers_displacement(
                    func_type, ocp, nlp, x, **parameters)

            elif _type == func_type.MINIMIZE_MARKERS_VELOCITY:
                PenaltyFunctionAbstract.Functions.minimize_markers_velocity(
                    func_type, ocp, nlp, t, x, **parameters)

            elif _type == func_type.ALIGN_MARKERS:
                PenaltyFunctionAbstract.Functions.align_markers(
                    func_type, ocp, nlp, x, **parameters)

            elif _type == func_type.PROPORTIONAL_STATE:
                PenaltyFunctionAbstract.Functions.proportional_variable(
                    func_type, ocp, nlp, x, **parameters)

            elif _type == func_type.PROPORTIONAL_CONTROL:
                if instant == Instant.END or instant == nlp["ns"]:
                    raise RuntimeError("No control u at last node")
                PenaltyFunctionAbstract.Functions.proportional_variable(
                    func_type, ocp, nlp, u, **parameters)

            elif _type == func_type.MINIMIZE_TORQUE:
                if instant == Instant.END or instant == nlp["ns"]:
                    raise RuntimeError("No control u at last node")
                PenaltyFunctionAbstract.Functions.minimize_torque(
                    func_type, ocp, nlp, t, u, **parameters)

            elif _type == func_type.MINIMIZE_MUSCLES:
                if instant == Instant.END or instant == nlp["ns"]:
                    raise RuntimeError("No control u at last node")
                PenaltyFunctionAbstract.Functions.minimize_muscles(
                    func_type, ocp, nlp, t, u, **parameters)

            elif _type == func_type.MINIMIZE_ALL_CONTROLS:
                if instant == Instant.END or instant == nlp["ns"]:
                    raise RuntimeError("No control u at last node")
                PenaltyFunctionAbstract.Functions.minimize_all_controls(
                    func_type, ocp, nlp, t, u, **parameters)

            elif _type == func_type.MINIMIZE_PREDICTED_COM_HEIGHT:
                PenaltyFunctionAbstract.Functions.minimize_predicted_com_height(
                    func_type, ocp, nlp, x, **parameters)

            elif _type == func_type.ALIGN_SEGMENT_WITH_CUSTOM_RT:
                PenaltyFunctionAbstract.Functions.align_segment_with_custom_rt(
                    func_type, ocp, nlp, x, **parameters)

            elif _type == func_type.ALIGN_MARKER_WITH_SEGMENT_AXIS:
                PenaltyFunctionAbstract.Functions.align_marker_with_segment_axis(
                    func_type, ocp, nlp, x, **parameters)

            elif _type == func_type.CUSTOM:
                PenaltyFunctionAbstract.Functions.custom(
                    func_type, ocp, nlp, t, x, u, **parameters)

            else:
                yield _type, instant, t, x, u, parameters

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
                    f"{element} is not a valid index for {name}, it must be between 0 and {max_bound - 1}.")

    @staticmethod
    def _add_to_goal(ocp, nlp, val, weight):
        raise RuntimeError("__add_to_goal cannot be called from an abstract class")

    @staticmethod
    def _get_type():
        raise RuntimeError("_get_type cannot be called from an abstract class")

    @staticmethod
    def __get_instant(nlp, constraint):
        if not isinstance(constraint["instant"], (list, tuple)):
            constraint["instant"] = (constraint["instant"],)
        t = []
        x = MX()
        u = MX()
        for node in constraint["instant"]:
            if isinstance(node, int):
                if node < 0 or node > nlp["ns"]:
                    raise RuntimeError(f"Invalid instant, {node} must be between 0 and {nlp['ns']}")
                t.append(node)
                x = horzcat(x, nlp["X"][node])
                u = horzcat(u, nlp["U"][node])

            elif node == Instant.START:
                t.append(0)
                x = horzcat(x, nlp["X"][0])
                u = horzcat(u, nlp["U"][0])

            elif node == Instant.MID:
                if nlp["ns"] % 2 == 1:
                    raise (ValueError("Number of shooting points must be even to use MID"))
                t.append(nlp["X"][nlp["ns"] // 2])
                x = horzcat(x, nlp["X"][nlp["ns"] // 2])
                u = horzcat(u, nlp["U"][nlp["ns"] // 2])

            elif node == Instant.INTERMEDIATES:
                for i in range(1, nlp["ns"] - 1):
                    t.append(i)
                    x = horzcat(x, nlp["X"][i])
                    u = horzcat(u, nlp["U"][i])

            elif node == Instant.END:
                t.append(nlp["X"][nlp["ns"]])
                x = horzcat(x, nlp["X"][nlp["ns"]])

            elif node == Instant.ALL:
                t.extend([i for i in range(nlp["ns"] + 1)])
                for i in range(nlp["ns"]):
                    x = horzcat(x, nlp["X"][i])
                    u = horzcat(u, nlp["U"][i])
                x = horzcat(x, nlp["X"][nlp["ns"]])

            else:
                raise RuntimeError(" is not a valid instant")
        return t, x, u
