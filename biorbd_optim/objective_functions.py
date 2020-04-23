import casadi
import numpy as np


class ObjectiveFunction:
    @staticmethod
    def add_objective_functions(ocp, nlp):
        for objective_function in nlp["objective_functions"]:
            if not isinstance(objective_function, dict):
                raise RuntimeError(f"{objective_function} is not a dictionary")
            type = objective_function["type"]
            del objective_function["type"]
            type(ocp, nlp, **objective_function)

        if ocp.is_cyclic_objective:
            ObjectiveFunction.cyclic(ocp)

    @staticmethod
    def minimize_states(ocp, nlp, weight=1, states_idx=(), data_to_track=()):
        states_idx = ObjectiveFunction.__check_var_size(states_idx, nlp["nx"], "state_idx")
        data_to_track = ObjectiveFunction.__check_tracking_data_size(data_to_track, [nlp["ns"] + 1, len(states_idx)])

        for i in range(nlp["ns"] + 1):
            ocp.J += (
                casadi.dot(
                    nlp["X"][i][states_idx] - data_to_track[i, states_idx],
                    nlp["X"][i][states_idx] - data_to_track[i, states_idx],
                )
                * nlp["dt"]
                * nlp["dt"]
                * weight
            )

    @staticmethod
    def minimize_markers(ocp, nlp, weight=1, markers_idx=(), data_to_track=()):
        n_q = nlp["nbQ"]
        markers_idx = ObjectiveFunction.__check_var_size(markers_idx, nlp["model"].nbMarkers(), "markers_idx")
        data_to_track = ObjectiveFunction.__check_tracking_data_size(
            data_to_track, [3, len(markers_idx), nlp["ns"] + 1]
        )

        for i in range(nlp["ns"] + 1):
            for j in markers_idx:
                ocp.J += (
                    casadi.dot(
                        nlp["model"].marker(nlp["X"][i][:n_q], j).to_mx() - data_to_track[:, j, i],
                        nlp["model"].marker(nlp["X"][i][:n_q], j).to_mx() - data_to_track[:, j, i],
                    )
                    * nlp["dt"]
                    * nlp["dt"]
                    * weight
                )

    @staticmethod
    def minimize_markers_displacement(ocp, nlp, weight=1, markers_idx=()):
        n_q = nlp["nbQ"]
        markers_idx = ObjectiveFunction.__check_var_size(markers_idx, nlp["model"].nbMarkers(), "markers_idx")

        for i in range(nlp["ns"]):
            for j in markers_idx:
                ocp.J += (
                        casadi.dot(
                            nlp["model"].marker(nlp["X"][i][:n_q], j).to_mx() - nlp["model"].marker(nlp["X"][i + 1][:n_q], j).to_mx(),
                            nlp["model"].marker(nlp["X"][i][:n_q], j).to_mx() - nlp["model"].marker(nlp["X"][i + 1][:n_q], j).to_mx(),
                        )
                        * nlp["dt"]
                        * nlp["dt"]
                        * weight
                )

    @staticmethod
    def minimize_markers_velocity(ocp, nlp, weight=1, markers_idx=()):
        n_q = nlp["nbQ"]
        n_qdot = nlp["nbQdot"]
        markers_idx = ObjectiveFunction.__check_var_size(markers_idx, nlp["model"].nbMarkers(), "markers_idx")

        for i in range(nlp["ns"] + 1):
            for j in markers_idx:
                ocp.J += (
                    casadi.dot(
                        nlp["model"].markerVelocity(nlp["X"][i][:n_q], nlp["X"][i][n_q:n_q+n_qdot], j).to_mx(),
                        nlp["model"].markerVelocity(nlp["X"][i][:n_q], nlp["X"][i][n_q:n_q+n_qdot], j).to_mx(),
                    )
                    * nlp["dt"]
                    * nlp["dt"]
                    * weight
                )

    @staticmethod
    def minimize_torque(ocp, nlp, weight=1, controls_idx=(), data_to_track=()):
        n_tau = nlp["nbTau"]
        controls_idx = ObjectiveFunction.__check_var_size(controls_idx, n_tau, "controls_idx")
        data_to_track = ObjectiveFunction.__check_tracking_data_size(data_to_track, [nlp["ns"], len(controls_idx)])

        for i in range(nlp["ns"]):
            ocp.J += (
                casadi.dot(
                    nlp["U"][i][controls_idx] - data_to_track[i, controls_idx],
                    nlp["U"][i][controls_idx] - data_to_track[i, controls_idx],
                )
                * nlp["dt"]
                * nlp["dt"]
                * weight
            )

    @staticmethod
    def minimize_muscle(ocp, nlp, weight=1, muscles_idx=(), data_to_track=()):
        n_tau = nlp["nbTau"]
        nb_muscle = nlp["nbMuscle"]
        muscles_idx = ObjectiveFunction.__check_var_size(muscles_idx, nb_muscle, "muscles_idx")
        data_to_track = ObjectiveFunction.__check_tracking_data_size(data_to_track, [nlp["ns"], len(muscles_idx)])

        # Add the nbTau offset to the muscle index
        muscles_idx_plus_tau = [idx + n_tau for idx in muscles_idx]
        for i in range(nlp["ns"]):
            ocp.J += (
                casadi.dot(
                    nlp["U"][i][muscles_idx_plus_tau] - data_to_track[i, muscles_idx],
                    nlp["U"][i][muscles_idx_plus_tau] - data_to_track[i, muscles_idx],
                )
                * nlp["dt"]
                * nlp["dt"]
                * weight
            )

    @staticmethod
    def minimize_all_controls(ocp, nlp, weight=1):
        for i in range(nlp["ns"]):
            ocp.J += casadi.dot(nlp["U"][i], nlp["U"][i]) * nlp["dt"] * nlp["dt"] * weight

    @staticmethod
    def cyclic(ocp, weight=1):

        if ocp.nlp[0]["nx"] != ocp.nlp[-1]["nx"]:
            raise RuntimeError("Cyclic constraint without same nx is not supported yet")

        ocp.J += (
            casadi.dot(ocp.nlp[-1]["X"][-1] - ocp.nlp[0]["X"][0], ocp.nlp[-1]["X"][-1] - ocp.nlp[0]["X"][0]) * weight
        )

    @staticmethod
    def minimize_distance_between_two_markers(ocp, nlp, first_marker, second_marker, weight=1, node=-1):

        q = nlp["q_mapping"].expand.map(nlp["X"][node][: nlp["nbQ"]])
        marker0 = nlp["model"].marker(q, first_marker).to_mx()
        marker1 = nlp["model"].marker(q, second_marker).to_mx()

        ocp.J += casadi.dot(marker0 - marker1, marker0 - marker1) * weight

    @staticmethod
    def maximize_predicted_height_jump(ocp, nlp, weight=1, node=-1):
        g = -9.81  # get gravity from biorbd
        q = nlp["q_mapping"].expand.map(nlp["X"][node][: nlp["nbQ"]])
        q_dot = nlp["q_dot_mapping"].expand.map(nlp["X"][node][nlp["nbQ"] :])
        CoM = nlp["model"].CoM(q).to_mx()
        CoM_dot = nlp["model"].CoMdot(q, q_dot).to_mx()
        jump_height = (CoM_dot[2] * CoM_dot[2]) / (2 * -g) + CoM[2]

        ocp.J -= jump_height * weight

    @staticmethod
    def __check_var_size(var_idx, target_size, var_name="var"):
        if var_idx == ():
            var_idx = range(target_size)
        else:
            if isinstance(var_idx, int):
                var_idx = [var_idx]
            if max(var_idx) > target_size:
                raise RuntimeError(f"{var_name} in minimize_states cannot be higher than nx ({target_size})")
        return var_idx

    @staticmethod
    def __check_tracking_data_size(data_to_track, target_size):
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
