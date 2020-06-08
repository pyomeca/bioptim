from enum import Enum
from math import inf

import numpy as np
import biorbd
import casadi

from .enums import Instant, Axe, PlotType
from .mapping import Mapping


class PenaltyFunctionAbstract:
    class Functions:
        @staticmethod
        def minimize_states(penalty_type, ocp, nlp, t, x, u, p, data_to_track=(), states_idx=(), **extra_param):
            """
            Adds the objective that the specific states should be minimized.
            It is possible to track states, in this case the objective is to minimize
            the mismatch between the optimized states and the reference states (data_to_track).
            :param data_to_track: Reference states for tracking. (list of lists of float)
            :param states_idx: Index of the states to minimize. (list of integers)
            """
            states_idx = PenaltyFunctionAbstract._check_and_fill_index(states_idx, nlp["nx"], "state_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"] + 1, max(states_idx) + 1]
            )

            for i, v in enumerate(x):
                # print(i)
                # print(v)
                val = v[states_idx] - data_to_track[t[i], states_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

            PenaltyFunctionAbstract._add_track_data_to_plot(
                ocp, nlp, data_to_track.T, combine_to="q", axes_idx=Mapping(states_idx)
            )

        @staticmethod
        def minimize_markers(
            penalty_type,
            ocp,
            nlp,
            t,
            x,
            u,
            p,
            axis_to_track=(Axe.X, Axe.Y, Axe.Z),
            markers_idx=(),
            data_to_track=(),
            **extra_param,
        ):
            """
            Adds the objective that the specific markers should be minimized.
            It is possible to track markers, in this case the objective is to minimize
            the mismatch between the optimized markers positions and the reference markers positions (data_to_track).
            :param markers_idx: Index of the markers to minimize. (list of integers)
            :param data_to_track: Reference markers positions for tracking. (list of lists of float)
            :axis_to_track: Index of axis to keep while tracking (default track 3d trajectories)
            """
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp["model"].nbMarkers(), "markers_idx"
            )
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [3, max(markers_idx) + 1, nlp["ns"] + 1]
            )

            nq = nlp["q_mapping"].reduce.len
            for i, v in enumerate(x):
                q = nlp["q_mapping"].expand.map(v[:nq])
                data_marker = data_to_track[:, markers_idx, t[i]]
                val = nlp["model"].markers(q)[axis_to_track, markers_idx] - data_marker[axis_to_track, :]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_markers_displacement(
            penalty_type, ocp, nlp, t, x, u, p, coordinates_system_idx=-1, markers_idx=(), **extra_param
        ):
            """
            Adds the objective that the specific markers displacement (difference between the position of the
            markers at each neighbour frame)should be minimized.
            :coordinates_system_idx: Index of the segment in which to project to displacement
            :param markers_idx: Index of the markers to minimize. (list of integers)
            """
            n_q = nlp["nbQ"]
            nb_rts = nlp["model"].nbSegment()
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp["model"].nbMarkers(), "markers_idx"
            )
            for i in range(len(x) - 1):
                if coordinates_system_idx < 0:
                    inv_jcs_1 = casadi.MX.eye(4)
                    inv_jcs_0 = casadi.MX.eye(4)
                elif coordinates_system_idx < nb_rts:
                    jcs_1 = nlp["model"].globalJCS(x[i + 1][:n_q], coordinates_system_idx).to_mx()
                    inv_jcs_1 = casadi.vertcat(
                        casadi.horzcat(jcs_1[:3, :3], -jcs_1[:3, :3] @ jcs_1[:3, 3]), casadi.horzcat(0, 0, 0, 1)
                    )
                    jcs_0 = nlp["model"].globalJCS(x[i][:n_q], coordinates_system_idx).to_mx()
                    inv_jcs_0 = casadi.vertcat(
                        casadi.horzcat(jcs_0[:3, :3], -jcs_0[:3, :3] @ jcs_0[:3, 3]), casadi.horzcat(0, 0, 0, 1)
                    )
                else:
                    raise RuntimeError(
                        f"Wrong choice of coordinates_system_idx. (Negative values refer to global coordinates system, "
                        f"positive values must be between 0 and {nb_rts})"
                    )
                val = inv_jcs_1 @ casadi.vertcat(
                    nlp["model"].markers(x[i + 1][:n_q])[:, markers_idx], 1
                ) - inv_jcs_0 @ casadi.vertcat(nlp["model"].markers(x[i][:n_q])[:, markers_idx], 1)
                penalty_type._add_to_penalty(ocp, nlp, val[:3], **extra_param)

        @staticmethod
        def minimize_markers_velocity(
            penalty_type, ocp, nlp, t, x, u, p, markers_idx=(), data_to_track=(), **extra_param
        ):
            """
            Adds the objective that the specific markers velocity should be minimized.
            It is possible to track markers velocity, in this case the objective is to minimize
            the mismatch between the optimized markers velocities and the reference markers velocities (data_to_track).
            :param markers_idx: Index of the markers to minimize. (list of integers)
            :param data_to_track: Reference markers velocities for tracking. (list of lists of float)
            """
            n_q = nlp["nbQ"]
            n_qdot = nlp["nbQdot"]
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp["model"].nbMarkers(), "markers_idx"
            )
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [3, max(markers_idx) + 1, nlp["ns"] + 1]
            )

            for m in markers_idx:
                for i, v in enumerate(x):
                    val = (
                        nlp["model"].markerVelocity(v[:n_q], v[n_q : n_q + n_qdot], m).to_mx()
                        - data_to_track[:, markers_idx, t[i]]
                    )
                    penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def align_markers(penalty_type, ocp, nlp, t, x, u, p, first_marker_idx, second_marker_idx, **extra_param):
            """
            Adds the constraint that the two markers must be coincided at the desired instant(s).
            :param nlp: An OptimalControlProgram class.
            :param x: List of instant(s).
            :param first_marker_idx: Index of the first marker (integer).
            :param second_marker_idx: Index of the second marker (integer).
            """
            PenaltyFunctionAbstract._check_idx(
                "marker", [first_marker_idx, second_marker_idx], nlp["model"].nbMarkers()
            )

            nq = nlp["q_mapping"].reduce.len
            for v in x:
                q = nlp["q_mapping"].expand.map(v[:nq])
                first_marker = nlp["model"].marker(q, first_marker_idx).to_mx()
                second_marker = nlp["model"].marker(q, second_marker_idx).to_mx()

                val = first_marker - second_marker
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def proportional_variable(
            penalty_type, ocp, nlp, t, x, u, p, which_var, first_dof, second_dof, coef, **extra_param
        ):
            """
            Adds proportionality constraint between the elements (states or controls) chosen.
            :param nlp: An instance of the OptimalControlProgram class.
            :param V: List of states or controls at instants on which this constraint must be applied.
            :param which_var: Type of the variable constrained to be proportional. (string) ("states" or "controls")
            :param first_dof: Index of the first state or control on which this constraint must be applied. (integer)
            :param second_dof: Index of the second state or control on which this constraint must be applied. (integer)
            :param coef: Coefficient of proportionality between the two states or controls. (float)
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
        def minimize_torque(penalty_type, ocp, nlp, t, x, u, p, controls_idx=(), data_to_track=(), **extra_param):
            """
            Adds the objective that the specific torques should be minimized.
            It is possible to track torques, in this case the objective is to minimize
            the mismatch between the optimized torques and the reference torques (data_to_track).
            :param controls_idx: Index of the controls to minimize. (list of integers)
            :param data_to_track: Reference torques for tracking. (list of lists of float)
            """
            n_tau = nlp["nbTau"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_tau, "controls_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"], max(controls_idx) + 1]
            )

            for i, v in enumerate(u):
                val = v[controls_idx] - data_to_track[t[i], controls_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

            PenaltyFunctionAbstract._add_track_data_to_plot(
                ocp, nlp, data_to_track.T, combine_to="tau", axes_idx=Mapping(controls_idx)
            )

        @staticmethod
        def minimize_muscles_control(
            penalty_type, ocp, nlp, t, x, u, p, muscles_idx=(), data_to_track=(), **extra_param
        ):
            """
            Adds the objective that the specific muscle controls should be minimized.
            It is possible to track muscle activation, in this case the objective is to minimize
            the mismatch between the optimized muscle controls and the reference muscle activation (data_to_track).
            :param muscles_idx: Index of the muscles which the activation in minimized. (list of integers)
            :param data_to_track: Reference muscle activation for tracking. (list of lists of float)
            """
            muscles_idx = PenaltyFunctionAbstract._check_and_fill_index(muscles_idx, nlp["nbMuscle"], "muscles_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"], max(muscles_idx) + 1]
            )

            # Add the nbTau offset to the muscle index
            muscles_idx_plus_tau = [idx + nlp["nbTau"] for idx in muscles_idx]
            for i, v in enumerate(u):
                val = v[muscles_idx_plus_tau] - data_to_track[t[i], muscles_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

            PenaltyFunctionAbstract._add_track_data_to_plot(
                ocp, nlp, data_to_track.T, combine_to="muscles_control", axes_idx=Mapping(muscles_idx)
            )

        @staticmethod
        def minimize_all_controls(penalty_type, ocp, nlp, t, x, u, p, controls_idx=(), data_to_track=(), **extra_param):
            """
            Adds the objective that all the controls should be minimized.
            It is possible to track controls, in this case the objective is to minimize
            the mismatch between the optimized controls and the reference controls (data_to_track).
            :param controls_idx: Index of the controls to minimize. (list of integers)
            :param data_to_track: Reference controls for tracking. (list of lists of float)
            """
            n_u = nlp["nu"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_u, "muscles_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"], max(controls_idx) + 1]
            )

            for i, v in enumerate(u):
                val = v[controls_idx] - data_to_track[t[i], controls_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def minimize_predicted_com_height(penalty_type, ocp, nlp, t, x, u, p, **extra_param):
            """
            Adds the objective that the minimal height of the center of mass of the model should be minimized.
            The height is assumed to be the third axis.
            """
            g = -9.81  # get gravity from biorbd

            for i, v in enumerate(x):
                q = nlp["q_mapping"].expand.map(v[: nlp["nbQ"]])
                q_dot = nlp["q_dot_mapping"].expand.map(v[nlp["nbQ"] :])
                CoM = nlp["model"].CoM(q).to_mx()
                CoM_dot = nlp["model"].CoMdot(q, q_dot).to_mx()
                CoM_height = (CoM_dot[2] * CoM_dot[2]) / (2 * -g) + CoM[2]
                penalty_type._add_to_penalty(ocp, nlp, CoM_height, **extra_param)

        @staticmethod
        def minimize_contact_forces(
            penalty_type, ocp, nlp, t, x, u, p, contacts_idx=(), data_to_track=(), **extra_param
        ):
            """
            Adds the objective that the contact force should be minimized.
            It is possible to track contact forces, in this case the objective is to minimize
            the mismatch between the optimized contact forces and the reference contact forces (data_to_track).
            :param contacts_idx: Index of the component of the force to be minimized. (integer)
            :param data_to_track: Reference contact forces for tracking. (list of lists of float)
            """
            n_contact = nlp["model"].nbContacts()
            contacts_idx = PenaltyFunctionAbstract._check_and_fill_index(contacts_idx, n_contact, "contacts_idx")
            data_to_track = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                data_to_track, [nlp["ns"], max(contacts_idx) + 1]
            )

            for i, v in enumerate(u):
                force = nlp["contact_forces_func"](x[i], u[i], p)
                val = force[contacts_idx] - data_to_track[t[i], contacts_idx]
                penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

            PenaltyFunctionAbstract._add_track_data_to_plot(
                ocp, nlp, data_to_track.T, combine_to="contact_forces", axes_idx=Mapping(contacts_idx)
            )

        @staticmethod
        def align_segment_with_custom_rt(penalty_type, ocp, nlp, t, x, u, p, segment_idx, rt_idx, **extra_param):
            """
            Adds the constraint that the local reference frame and the segment must be aligned at the desired
            instant(s).
            :param nlp: An OptimalControlProgram class.
            :param X: List of instant(s).
            :param segment_idx: Index of the segment to be aligned. (integer)
            :param rt_idx: Index of the local reference frame to be aligned. (integer)
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
            penalty_type, ocp, nlp, t, x, u, p, marker_idx, segment_idx, axis, **extra_param
        ):
            """
            Adds the constraint that the marker and the segment must be aligned at the desired
            instant(s).
            :param marker_idx: Index of the marker to be aligned. (integer)
            :param segment_idx: Index of the segment to be aligned. (integer)
            :param axis: Axis of the segment to be aligned. (biorbd_optim.Axe)
            """
            if not isinstance(axis, Axe):
                raise RuntimeError("axis must be a biorbd_optim.Axe")

            nq = nlp["q_mapping"].reduce.len
            for v in x:
                q = nlp["q_mapping"].expand.map(v[:nq])

                r_rt = nlp["model"].globalJCS(q, segment_idx)
                marker = nlp["model"].marker(q, marker_idx)
                marker.applyRT(r_rt.transpose())
                marker = marker.to_mx()

                for axe in Axe:
                    if axe != axis:
                        # To align an axis, the other must be equal to 0
                        val = marker[axe, 0]
                        penalty_type._add_to_penalty(ocp, nlp, val, **extra_param)

        @staticmethod
        def custom(penalty_type, ocp, nlp, t, x, u, p, **parameters):
            """
            Adds a custom penalty function (objective or constraint).
            :param parameters: parameters["function"] -> Penalty function (CasADi function),
            parameters["penalty_idx"] -> Index of the penalty (integer), parameters["weight"] -> Weight of the penalty
            (float)
            """
            func = parameters["function"]
            weight = None
            penalty_idx = parameters["penalty_idx"]
            if "weight" in parameters.keys():
                weight = parameters["weight"]
                del parameters["weight"]
            del parameters["function"]
            del parameters["penalty_idx"]
            val = func(ocp, nlp, t, x, u, p, **parameters)
            if weight is not None:
                parameters["weight"] = weight
            parameters["penalty_idx"] = penalty_idx
            penalty_type._add_to_penalty(ocp, nlp, val, **parameters)

    @staticmethod
    def add(ocp, nlp):
        raise RuntimeError("add cannot be called from an abstract class")

    @staticmethod
    def add_or_replace(ocp, nlp, penalty, penalty_idx):
        """
        Adds a penalty at the index penalty_index. If a penalty already exists at this index, it replaces it by the
        new penalty.
        :param penalty: Penalty to be added. (instance of PenaltyFunctionAbstract class)
        :param penalty_idx: Index to add the penalty. (integer)
        """
        t, x, u = PenaltyFunctionAbstract._get_instant(nlp, penalty)
        penalty_function = penalty["type"].value[0]
        penalty_type = penalty["type"]._get_type()
        instant = penalty["instant"]
        del penalty["instant"], penalty["type"]

        penalty_type._span_checker(penalty_function, instant, nlp)
        penalty_type._parameter_modifier(penalty_function, penalty)

        penalty_idx = penalty_type._reset_penalty(ocp, nlp, penalty_idx)
        penalty_function(penalty_type, ocp, nlp, t, x, u, nlp["p"], penalty_idx=penalty_idx, **penalty)

    @staticmethod
    def _parameter_modifier(penalty_function, parameters):
        """
        Modifies parameters entries if needed.
        :param penalty_function: Penalty function to be checked (instance of PenaltyType class)
        :param parameters: Parameters to be checked. If parameters["quadratic"] is not defined, it sets it to True.
        (bool)
        """
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
            or penalty_function == PenaltyType.MINIMIZE_CONTACT_FORCES
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
        """
        Raises errors if the time span is not consistent with the problem definition.
        (There can not be any control at the last time node)
        :param penalty_function: Penalty function. (instance of PenaltyType class)
        :param instant: Instant at which the penalty is applied. (instance of Instant class)
        """
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
        """
        Checks if the variable index is consistent and sets it to var_index.
        :param var_idx: Index of the variable. (integer)
        :param target_size: Current size of the variable array. (integer)
        :param var_name: Name of the variable. (string)
        :return: var_idx: New index of the variable. (integer)
        """
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
        """
        Raises errors if the size of the data_to_track array is inconsistent.
        Reshape data_to_track array if the shape is inconsistent.
        :param data_to_track: Data used for tracking. (list of lists)
        :param target_size: Size of the variable array. (integer)
        :return: data_to_track -> Data used for tracking. (numpy array of size target_size)
        """
        if (isinstance(data_to_track, np.ndarray) and data_to_track.any()) or data_to_track:
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
        else:
            data_to_track = np.zeros(target_size)
        return data_to_track

    @staticmethod
    def _check_idx(name, elements, max_bound=inf, min_bound=0):
        """

        :param name: Name of the array variable. (string)
        :param elements: Index of the targeted spot in the array variable. (integer)
        :param max_bound: Maximal index of the targeted spot in the array variable. (integer)
        :param min_bound: Minimal index of the targeted spot in the array variable. (integer)
        """
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
    def continuity(ocp):
        # Dynamics must be continuous between phases
        for pt in ocp.state_transitions:
            penalty_idx = pt["base"]._reset_penalty(ocp, None, -1)
            state_transition_function = pt["type"]
            val = state_transition_function(ocp, **pt)
            pt["base"]._add_to_penalty(ocp, None, val, penalty_idx, **pt)

    @staticmethod
    def _add_to_penalty(ocp, nlp, val, penalty_idx, **extra_param):
        raise RuntimeError("_add_to_penalty cannot be called from an abstract class")

    @staticmethod
    def _reset_penalty(ocp, nlp, penalty_idx):
        raise RuntimeError("_reset_penalty cannot be called from an abstract class")

    @staticmethod
    def _get_type():
        raise RuntimeError("_get_type cannot be called from an abstract class")

    @staticmethod
    def _get_instant(nlp, constraint):
        """
        Initializes x (states), u (controls) and t (time) with user provided initial guesses.
        :param constraint: constraint["instant"] -> time nodes precision. (integer or instance of Instant class)
        (integer, Instant.START, Instant.MID, Instant.INTERMEDIATES, Instant.END or Instant.ALL)
        :return t: Time nodes. (list)
        :return x: States. (list of lists)
        :return u: Controls. (list of lists)
        """
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
                t.append(nlp["ns"] // 2)
                x.append(nlp["X"][nlp["ns"] // 2])
                u.append(nlp["U"][nlp["ns"] // 2])

            elif node == Instant.INTERMEDIATES:
                for i in range(1, nlp["ns"] - 1):
                    t.append(i)
                    x.append(nlp["X"][i])
                    u.append(nlp["U"][i])

            elif node == Instant.END:
                t.append(nlp["ns"])
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

    @staticmethod
    def _add_track_data_to_plot(ocp, nlp, data, combine_to, axes_idx=None):
        """
        Adds the tracked data to the graphs.
        :param data: Tracked data. (numpy array)
        :param combine_to: Plot in which to add the tracked data.
        :param axes_idx: Index of the axis in which to add the tracked data. (integer)
        """
        if (isinstance(data, np.ndarray) and not data.any()) or (not isinstance(data, np.ndarray) and not data):
            return

        if data.shape[1] == nlp["ns"]:
            data = np.c_[data, data[:, -1]]
        ocp.add_plot(
            combine_to,
            lambda x, u: data,
            color="tab:red",
            plot_type=PlotType.STEP,
            phase_number=nlp["phase_idx"],
            axes_idx=axes_idx,
        )


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
    MINIMIZE_CONTACT_FORCES = PenaltyFunctionAbstract.Functions.minimize_contact_forces
    TRACK_CONTACT_FORCES = MINIMIZE_CONTACT_FORCES
    MINIMIZE_PREDICTED_COM_HEIGHT = PenaltyFunctionAbstract.Functions.minimize_predicted_com_height
    ALIGN_SEGMENT_WITH_CUSTOM_RT = PenaltyFunctionAbstract.Functions.align_segment_with_custom_rt
    ALIGN_MARKER_WITH_SEGMENT_AXIS = PenaltyFunctionAbstract.Functions.align_marker_with_segment_axis
    CUSTOM = PenaltyFunctionAbstract.Functions.custom
