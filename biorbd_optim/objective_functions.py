import casadi
import numpy as np


class ObjectiveFunction:
    @staticmethod
    def minimize_states(ocp, nlp, weight=1, states_idx=(), data_to_track=()):
        states_idx = ObjectiveFunction.__check_var_size(states_idx, nlp["nx"], "state_idx")
        data_to_track = ObjectiveFunction.__check_tracking_data_size(data_to_track, [nlp["ns"] + 1, len(states_idx)])

        for i in range(nlp["ns"] + 1):
            ocp.J += (
                casadi.dot(nlp["X"][i][states_idx] - data_to_track[i, states_idx],
                           nlp["X"][i][states_idx] - data_to_track[i, states_idx])
                * nlp["dt"] * nlp["dt"] * weight
            )

    @staticmethod
    def minimize_markers(ocp, nlp, weight=1, markers_idx=(), data_to_track=()):
        n_q = nlp["nbQ"]
        n_mark = nlp["model"].nbMarkers()
        markers_idx = ObjectiveFunction.__check_var_size(markers_idx, n_mark, "markers_idx")
        data_to_track = ObjectiveFunction.__check_tracking_data_size(data_to_track, [3, len(markers_idx), nlp["ns"] + 1])

        for i in range(nlp["ns"] + 1):
            for j in range(n_mark):
                ocp.J += (
                    casadi.dot(nlp["model"].marker(nlp["X"][i][:n_q], j).to_mx() - data_to_track[:, j, i],
                               nlp["model"].marker(nlp["X"][i][:n_q], j).to_mx() - data_to_track[:, j, i]) * nlp["dt"] * nlp["dt"] * weight
                )

    @staticmethod
    def minimize_torque(ocp, nlp, weight=1, controls_idx=(), data_to_track=()):
        n_tau = nlp["nbTau"]
        controls_idx = ObjectiveFunction.__check_var_size(controls_idx, n_tau, "controls_idx")
        data_to_track = ObjectiveFunction.__check_tracking_data_size(data_to_track, [nlp["ns"], len(controls_idx)])

        for i in range(nlp["ns"]):
            ocp.J += (
                casadi.dot(nlp["U"][i][controls_idx] - data_to_track[i, controls_idx],
                           nlp["U"][i][controls_idx] - data_to_track[i, controls_idx],)
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
        raise RuntimeError("cyclic objective function not implemented yet")

    @staticmethod
    def cyclic(ocp, nlp, weight=1):
        raise RuntimeError("cyclic objective function not implemented yet")

    @staticmethod
    def minimize_final_distance_between_two_markers(ocp, nlp, weight=1, markers=()):
        if not isinstance(markers, (list, tuple)) or len(markers) != 2:
            raise RuntimeError(
                "minimize_distance_between_two_markers expect markers to be a list of 2 marker indices"
            )
        q = nlp["dof_mapping"].expand(nlp["X"][nlp["ns"]][: nlp["nbQ"]])
        marker0 = nlp["model"].marker(q, markers[0]).to_mx()
        marker1 = nlp["model"].marker(q, markers[1]).to_mx()

        ocp.J += (
            casadi.dot(marker0 - marker1, marker0 - marker1)
            * nlp["dt"]
            * nlp["dt"]
            * weight
        )

    @staticmethod
    def __check_var_size(var_idx, target_size, var_name="var"):
        if var_idx == ():
            var_idx = range(target_size)
        else:
            if isinstance(var_idx, int):
                var_idx = [var_idx]
            if max(var_idx) > target_size:
                raise RuntimeError(var_name + " in minimize_states cannot be higher than nx (" + target_size + ")")
        return var_idx

    @staticmethod
    def __check_tracking_data_size(data_to_track, target_size):
        if data_to_track == ():
            data_to_track = np.zeros(target_size)
        else:
            if len(data_to_track.shape) != len(target_size):
                raise RuntimeError("data_to_track " + str(data_to_track.shape)
                                   + " don't correspond to expected minimum size " + str(target_size))
            for i in range(len(target_size)):
                if data_to_track.shape[i] < target_size[i]:
                    raise RuntimeError("data_to_track " + str(data_to_track.shape)
                                       + " don't correspond to expected minimum size " + str(target_size))
        return data_to_track
