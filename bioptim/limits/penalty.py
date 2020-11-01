from enum import Enum
from math import inf

import numpy as np
import biorbd
from casadi import vertcat, horzcat

from ..misc.enums import Instant, Axe, PlotType, ControlType
from ..misc.mapping import Mapping


class PenaltyFunctionAbstract:
    class Functions:
        @staticmethod
        def minimize_states(penalty, ocp, nlp, t, x, u, p, target=None, states_idx=(), **extra_param):
            """
            Adds the objective that the specific states should be minimized.
            It is possible to track states, in this case the objective is to minimize
            the mismatch between the optimized states and the reference states (target).
            :param target: Reference states for tracking. (list of lists of float)
            :param states_idx: Index of the states to minimize. (list of integers)
            """
            states_idx = PenaltyFunctionAbstract._check_and_fill_index(states_idx, nlp.nx, "state_idx")
            if target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(target, (len(states_idx), len(x)))

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
                target_tp = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, target=target_tp, **extra_param)

        @staticmethod
        def minimize_markers(
            penalty,
            ocp,
            nlp,
            t,
            x,
            u,
            p,
            axis_to_track=(Axe.X, Axe.Y, Axe.Z),
            markers_idx=(),
            target=None,
            **extra_param,
        ):
            """
            Adds the objective that the specific markers should be minimized.
            It is possible to track markers, in this case the objective is to minimize
            the mismatch between the optimized markers positions and the reference markers positions (target).
            :param markers_idx: Index of the markers to minimize. (list of integers)
            :param target: Reference markers positions for tracking. (list of lists of float)
            :axis_to_track: Index of axis to keep while tracking (default track 3d trajectories)
            """
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp.model.nbMarkers(), "markers_idx"
            )
            if target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    target, (3, len(markers_idx), len(x))
                )
            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_markers", nlp.model.markers, nlp.q)
            nq = nlp.mapping["q"].reduce.len
            for i, v in enumerate(x):
                q = nlp.mapping["q"].expand.map(v[:nq])
                val = nlp.casadi_func["biorbd_markers"](q)[axis_to_track, markers_idx]
                target_tp = target[axis_to_track, :, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, target=target_tp, **extra_param)

        @staticmethod
        def minimize_markers_displacement(
            penalty, ocp, nlp, t, x, u, p, coordinates_system_idx=-1, markers_idx=(), **extra_param
        ):
            """
            Adds the objective that the specific markers displacement (difference between the position of the
            markers at each neighbour frame)should be minimized.
            :coordinates_system_idx: Index of the segment in which to project to displacement
            :param markers_idx: Index of the markers to minimize. (list of integers)
            """

            nq = nlp.mapping["q"].reduce.len
            nb_rts = nlp.model.nbSegment()

            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp.model.nbMarkers(), "markers_idx"
            )

            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_markers", nlp.model.markers, nlp.q)
            if coordinates_system_idx >= 0 and coordinates_system_idx < nb_rts:
                PenaltyFunctionAbstract._add_to_casadi_func(
                    nlp, f"globalJCS_{coordinates_system_idx}", nlp.model.globalJCS, nlp.q, coordinates_system_idx
                )

            for i in range(len(x) - 1):
                q_0 = nlp.mapping["q"].expand.map(x[i][:nq])
                q_1 = nlp.mapping["q"].expand.map(x[i + 1][:nq])

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

                val = jcs_1_T @ vertcat(nlp.casadi_func["biorbd_markers"](q_1)[:, markers_idx], 1) - jcs_0_T @ vertcat(
                    nlp.casadi_func["biorbd_markers"](q_0)[:, markers_idx], 1
                )
                penalty.type.get_type().add_to_penalty(ocp, nlp, val[:3], penalty, **extra_param)

        @staticmethod
        def minimize_markers_velocity(penalty, ocp, nlp, t, x, u, p, markers_idx=(), target=None, **extra_param):
            """
            Adds the objective that the specific markers velocity should be minimized.
            It is possible to track markers velocity, in this case the objective is to minimize
            the mismatch between the optimized markers velocities and the reference markers velocities (data_to_track).
            :param markers_idx: Index of the markers to minimize. (list of integers)
            :param data_to_track: Reference markers velocities for tracking. (list of lists of float)
            """
            n_q = nlp.shape["q"]
            n_qdot = nlp.shape["q_dot"]
            markers_idx = PenaltyFunctionAbstract._check_and_fill_index(
                markers_idx, nlp.model.nbMarkers(), "markers_idx"
            )

            if target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(
                    target, (3, len(markers_idx), len(x))
                )

            for m in markers_idx:
                # TODO This should be available natively in biorbd instead of
                # a for loop (major time lost if lot of markers)
                PenaltyFunctionAbstract._add_to_casadi_func(
                    nlp, f"biorbd_markerVelocity_{m}", nlp.model.markerVelocity, nlp.q, nlp.q_dot, int(m)
                )

            for i, v in enumerate(x):
                for m in markers_idx:
                    val = nlp.casadi_func[f"biorbd_markerVelocity_{m}"](v[:n_q], v[n_q : n_q + n_qdot])
                    target_tp = target[:, m, i] if target is not None else None
                    penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, target=target_tp, **extra_param)

        @staticmethod
        def align_markers(penalty, ocp, nlp, t, x, u, p, first_marker_idx, second_marker_idx, **extra_param):
            """
            Adds the constraint that the two markers must be coincided at the desired instant(s).
            :param nlp: An OptimalControlProgram class.
            :param x: List of instant(s).
            :param first_marker_idx: Index of the first marker (integer).
            :param second_marker_idx: Index of the second marker (integer).
            """
            PenaltyFunctionAbstract._check_idx("marker", [first_marker_idx, second_marker_idx], nlp.model.nbMarkers())
            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "markers", nlp.model.markers, nlp.q)
            nq = nlp.mapping["q"].reduce.len
            for v in x:
                q = nlp.mapping["q"].expand.map(v[:nq])
                first_marker = nlp.casadi_func["markers"](q)[:, first_marker_idx]
                second_marker = nlp.casadi_func["markers"](q)[:, second_marker_idx]

                val = first_marker - second_marker
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, **extra_param)

        @staticmethod
        def proportional_variable(penalty, ocp, nlp, t, x, u, p, which_var, first_dof, second_dof, coef, **extra_param):
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
                v = nlp.mapping["q"].expand.map(v)
                val = v[first_dof] - coef * v[second_dof]
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, **extra_param)

        @staticmethod
        def minimize_torque(penalty, ocp, nlp, t, x, u, p, controls_idx=(), target=None, **extra_param):
            """
            Adds the objective that the specific torques should be minimized.
            It is possible to track torques, in this case the objective is to minimize
            the mismatch between the optimized torques and the reference torques (target).
            :param controls_idx: Index of the controls to minimize. (list of integers)
            :param target: Reference torques for tracking. (list of lists of float)
            """
            n_tau = nlp.shape["tau"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_tau, "controls_idx")

            if target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(target, (len(controls_idx), len(u)))
                PenaltyFunctionAbstract._add_track_data_to_plot(
                    ocp, nlp, target, combine_to="tau", axes_idx=Mapping(controls_idx)
                )

            for i, v in enumerate(u):
                val = v[controls_idx]
                target_tp = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, target=target_tp, **extra_param)

        @staticmethod
        def minimize_torque_derivative(penalty, ocp, nlp, t, x, u, p, controls_idx=(), **extra_param):
            """
            Adds the objective that the specific torques should be minimized.
            It is possible to track torques, in this case the objective is to minimize
            the mismatch between the optimized torques and the reference torques (data_to_track).
            :param controls_idx: Index of the controls to minimize. (list of integers)
            :param data_to_track: Reference torques for tracking. (list of lists of float)
            """
            n_tau = nlp.shape["tau"]
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_tau, "controls_idx")

            for i in range(len(u) - 1):
                val = u[i + 1][controls_idx] - u[i][controls_idx]
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, **extra_param)

        @staticmethod
        def minimize_muscles_control(penalty, ocp, nlp, t, x, u, p, muscles_idx=(), target=None, **extra_param):
            """
            Adds the objective that the specific muscle controls should be minimized.
            It is possible to track muscle activation, in this case the objective is to minimize
            the mismatch between the optimized muscle controls and the reference muscle activation (data_to_track).
            :param muscles_idx: Index of the muscles which the activation in minimized. (list of integers)
            :param data_to_track: Reference muscle activation for tracking. (list of lists of float)
            """
            muscles_idx = PenaltyFunctionAbstract._check_and_fill_index(muscles_idx, nlp.shape["muscle"], "muscles_idx")

            if target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(target, (len(muscles_idx), len(u)))

                PenaltyFunctionAbstract._add_track_data_to_plot(
                    ocp, nlp, target, combine_to="muscles_control", axes_idx=Mapping(muscles_idx)
                )

            # Add the nbTau offset to the muscle index
            muscles_idx_plus_tau = [idx + nlp.shape["tau"] for idx in muscles_idx]
            for i, v in enumerate(u):
                val = v[muscles_idx_plus_tau]
                target_tp = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, target=target_tp, **extra_param)

        @staticmethod
        def minimize_all_controls(penalty, ocp, nlp, t, x, u, p, controls_idx=(), target=None, **extra_param):
            """
            Adds the objective that all the controls should be minimized.
            It is possible to track controls, in this case the objective is to minimize
            the mismatch between the optimized controls and the reference controls (data_to_track).
            :param controls_idx: Index of the controls to minimize. (list of integers)
            :param data_to_track: Reference controls for tracking. (list of lists of float)
            """
            n_u = nlp.nu
            controls_idx = PenaltyFunctionAbstract._check_and_fill_index(controls_idx, n_u, "muscles_idx")

            if target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(target, (len(controls_idx), len(u)))

            for i, v in enumerate(u):
                val = v[controls_idx]
                target_tp = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, target=target_tp, **extra_param)

        @staticmethod
        def minimize_predicted_com_height(penalty, ocp, nlp, t, x, u, p, **extra_param):
            """
            Adds the objective that the minimal height of the center of mass of the model should be minimized.
            The height is assumed to be the third axis.
            """
            g = -9.81  # get gravity from biorbd
            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_CoM", nlp.model.CoM, nlp.q)
            PenaltyFunctionAbstract._add_to_casadi_func(nlp, "biorbd_CoMdot", nlp.model.CoMdot, nlp.q, nlp.q_dot)
            for i, v in enumerate(x):
                q = nlp.mapping["q"].expand.map(v[: nlp.shape["q"]])
                q_dot = nlp.mapping["q_dot"].expand.map(v[nlp.shape["q"] :])
                CoM = nlp.casadi_func["biorbd_CoM"](q)
                CoM_dot = nlp.casadi_func["biorbd_CoMdot"](q, q_dot)
                CoM_height = (CoM_dot[2] * CoM_dot[2]) / (2 * -g) + CoM[2]
                penalty.type.get_type().add_to_penalty(ocp, nlp, CoM_height, penalty, **extra_param)

        @staticmethod
        def minimize_contact_forces(penalty, ocp, nlp, t, x, u, p, contacts_idx=(), target=None, **extra_param):
            """
            Adds the objective that the contact force should be minimized.
            It is possible to track contact forces, in this case the objective is to minimize
            the mismatch between the optimized contact forces and the reference contact forces (data_to_track).
            :param contacts_idx: Index of the component of the force to be minimized. (integer)
            :param data_to_track: Reference contact forces for tracking. (list of lists of float)
            """
            n_contact = nlp.model.nbContacts()
            contacts_idx = PenaltyFunctionAbstract._check_and_fill_index(contacts_idx, n_contact, "contacts_idx")

            if target is not None:
                target = PenaltyFunctionAbstract._check_and_fill_tracking_data_size(target, (len(contacts_idx), len(u)))

                PenaltyFunctionAbstract._add_track_data_to_plot(
                    ocp, nlp, target, combine_to="contact_forces", axes_idx=Mapping(contacts_idx)
                )

            for i, v in enumerate(u):
                force = nlp.contact_forces_func(x[i], u[i], p)
                val = force[contacts_idx]
                target_tp = target[:, i] if target is not None else None
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, target=target_tp, **extra_param)

        @staticmethod
        def align_segment_with_custom_rt(penalty, ocp, nlp, t, x, u, p, segment_idx, rt_idx, **extra_param):
            """
            Adds the constraint that the local reference frame and the segment must be aligned at the desired
            instant(s).
            :param nlp: An OptimalControlProgram class.
            :param X: List of instant(s).
            :param segment_idx: Index of the segment to be aligned. (integer)
            :param rt_idx: Index of the local reference frame to be aligned. (integer)
            """
            PenaltyFunctionAbstract._check_idx("segment", segment_idx, nlp.model.nbSegment())
            PenaltyFunctionAbstract._check_idx("rt", rt_idx, nlp.model.nbRTs())

            def biorbd_meta_func(q, segment_idx, rt_idx):
                r_seg = nlp.model.globalJCS(q, segment_idx).rot()
                r_rt = nlp.model.RT(q, rt_idx).rot()
                return biorbd.Rotation_toEulerAngles(r_seg.transpose() * r_rt, "zyx").to_mx()

            PenaltyFunctionAbstract._add_to_casadi_func(
                nlp, f"align_segment_with_custom_rt_{segment_idx}", biorbd_meta_func, nlp.q, segment_idx, rt_idx
            )

            nq = nlp.mapping["q"].reduce.len
            for v in x:
                q = nlp.mapping["q"].expand.map(v[:nq])
                val = nlp.casadi_func[f"align_segment_with_custom_rt_{segment_idx}"](q)
                penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, **extra_param)

        @staticmethod
        def align_marker_with_segment_axis(penalty, ocp, nlp, t, x, u, p, marker_idx, segment_idx, axis, **extra_param):
            """
            Adds the constraint that the marker and the segment must be aligned at the desired
            instant(s).
            :param marker_idx: Index of the marker to be aligned. (integer)
            :param segment_idx: Index of the segment to be aligned. (integer)
            :param axis: Axis of the segment to be aligned. (bioptim.Axe)
            """
            if not isinstance(axis, Axe):
                raise RuntimeError("axis must be a bioptim.Axe")

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
            nq = nlp.mapping["q"].reduce.len
            for v in x:
                q = nlp.mapping["q"].expand.map(v[:nq])
                marker = nlp.casadi_func[f"align_marker_with_segment_axis_{segment_idx}_{marker_idx}"](q)
                for axe in Axe:
                    if axe != axis:
                        # To align an axis, the other must be equal to 0
                        val = marker[axe, 0]
                        penalty.type.get_type().add_to_penalty(ocp, nlp, val, penalty, **extra_param)

        @staticmethod
        def custom(penalty, ocp, nlp, t, x, u, p, **parameters):
            """
            Adds a custom penalty function (objective or constraint).
            :param parameters: parameters["function"] -> Penalty function (CasADi function),
            parameters["penalty"] -> Index of the penalty (integer), parameters.weight -> Weight of the penalty
            (float)
            """
            if "min_bound" in parameters:
                min_bound = parameters["min_bound"]
                del parameters["min_bound"]
            else:
                min_bound = 0
            if "max_bound" in parameters:
                max_bound = parameters["max_bound"]
                del parameters["max_bound"]
            else:
                max_bound = 0

            val = penalty.custom_function(ocp, nlp, t, x, u, p, **parameters)
            penalty.type.get_type().add_to_penalty(
                ocp, nlp, val, penalty, min_bound=min_bound, max_bound=max_bound, **parameters
            )

    @staticmethod
    def add(ocp, nlp):
        raise RuntimeError("add cannot be called from an abstract class")

    @staticmethod
    def add_or_replace(ocp, nlp, penalty):
        """
        Adds a penalty at the index penalty_index. If a penalty already exists at this index, it replaces it by the
        new penalty.
        :param penalty: Penalty to be added. (instance of PenaltyFunctionAbstract class)
        """
        t, x, u = PenaltyFunctionAbstract._get_instant(nlp, penalty)
        penalty_function = penalty.type.value[0]
        penalty_type = penalty.type.get_type()
        instant = penalty.instant

        penalty_type._span_checker(penalty_function, instant, nlp)
        penalty_type._parameter_modifier(penalty_function, penalty)

        penalty_type.clear_penalty(ocp, nlp, penalty)
        penalty_function(penalty, ocp, nlp, t, x, u, nlp.p, **penalty.params)

    @staticmethod
    def _add_to_casadi_func(nlp, name, function, *all_param):
        if name in nlp.casadi_func:
            return
        else:
            nlp.casadi_func[name] = biorbd.to_casadi_func(name, function, *all_param)

    @staticmethod
    def _parameter_modifier(penalty_function, parameters):
        """
        Modifies parameters entries if needed.
        :param penalty_function: Penalty function to be checked (instance of PenaltyType class)
        :param parameters: Parameters to be checked. If parameters.quadratic is not defined, it sets it to True.
        (bool)
        """
        # Everything that should change the entry parameters depending on the penalty can be added here
        if parameters.quadratic is None:
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
                or penalty_function == PenaltyType.MINIMIZE_TORQUE_DERIVATIVE
            ):
                parameters.quadratic = True
            else:
                parameters.quadratic = False

        if penalty_function == PenaltyType.PROPORTIONAL_STATE:
            parameters.params["which_var"] = "states"
        if penalty_function == PenaltyType.PROPORTIONAL_CONTROL:
            parameters.params["which_var"] = "controls"

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
            if instant == Instant.END or instant == nlp.ns:
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
                raise RuntimeError(f"{var_name} in cannot be higher than nx ({target_size})")
        out = np.array(var_idx)
        if not np.issubdtype(out.dtype, np.integer):
            raise RuntimeError(f"{var_name} must be a list of integer")
        return out

    @staticmethod
    def _check_and_fill_tracking_data_size(data_to_track, target_size):
        """
        Raises errors if the size of the data_to_track array is inconsistent.
        Reshape data_to_track array if the shape is inconsistent.
        :param data_to_track: Data used for tracking. (list of lists)
        :param target_size: Size of the variable array. (integer)
        :return: data_to_track -> Data used for tracking. (numpy array of size target_size)
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
                    f"data_to_track {data_to_track.shape} doesn't correspond to expected minimum size {target_size}"
                )
        else:
            raise ValueError("COUCOU!!!!")
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
        raise RuntimeError("continuity cannot be called from an abstract class")

    @staticmethod
    def add_to_penalty(ocp, nlp, val, penalty, **extra_arguments):
        raise RuntimeError("_add_to_penalty cannot be called from an abstract class")

    @staticmethod
    def clear_penalty(ocp, nlp, penalty):
        raise RuntimeError("_reset_penalty cannot be called from an abstract class")

    @staticmethod
    def get_type():
        raise RuntimeError("_get_type cannot be called from an abstract class")

    @staticmethod
    def _get_instant(nlp, constraint):
        """
        Initializes x (states), u (controls) and t (time) with user provided initial guesses.
        :param constraint: constraint.instant -> time nodes precision. (integer or instance of Instant class)
        (integer, Instant.START, Instant.MID, Instant.INTERMEDIATES, Instant.END or Instant.ALL)
        :return t: Time nodes. (list)
        :return x: States. (list of lists)
        :return u: Controls. (list of lists)
        """
        if not isinstance(constraint.instant, (list, tuple)):
            constraint.instant = (constraint.instant,)
        t = []
        x = []
        u = []
        for node in constraint.instant:
            if isinstance(node, int):
                if node < 0 or node > nlp.ns:
                    raise RuntimeError(f"Invalid instant, {node} must be between 0 and {nlp.ns}")
                t.append(node)
                x.append(nlp.X[node])
                if (
                    nlp.control_type == ControlType.CONSTANT and node != nlp.ns
                ) or nlp.control_type != ControlType.CONSTANT:
                    u.append(nlp.U[node])

            elif node == Instant.START:
                t.append(0)
                x.append(nlp.X[0])
                u.append(nlp.U[0])

            elif node == Instant.MID:
                if nlp.ns % 2 == 1:
                    raise (ValueError("Number of shooting points must be even to use MID"))
                t.append(nlp.ns // 2)
                x.append(nlp.X[nlp.ns // 2])
                u.append(nlp.U[nlp.ns // 2])

            elif node == Instant.INTERMEDIATES:
                for i in range(1, nlp.ns - 1):
                    t.append(i)
                    x.append(nlp.X[i])
                    u.append(nlp.U[i])

            elif node == Instant.END:
                t.append(nlp.ns)
                x.append(nlp.X[nlp.ns])

            elif node == Instant.ALL:
                t.extend([i for i in range(nlp.ns + 1)])
                for i in range(nlp.ns):
                    x.append(nlp.X[i])
                    u.append(nlp.U[i])
                x.append(nlp.X[nlp.ns])

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

        if data.shape[1] == nlp.ns:
            data = np.c_[data, data[:, -1]]
        ocp.add_plot(
            combine_to,
            lambda x, u, p: data,
            color="tab:red",
            linestyle=".-",
            plot_type=PlotType.STEP,
            phase_number=nlp.phase_idx,
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
    MINIMIZE_TORQUE_DERIVATIVE = PenaltyFunctionAbstract.Functions.minimize_torque_derivative
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
