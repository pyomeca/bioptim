import numpy as np
from scipy import interpolate


class Data:
    class Phase:
        def __init__(self, time, phase):
            self.node = [node.reshape(node.shape[0], 1) for node in phase.T]
            self.nb_elements = phase.shape[0]
            self.t = np.linspace(time[0], time[1], len(self.node))
            self.nb_t = self.t.shape[0]

    def __init__(self):
        self.phase = []
        self.nb_elements = -1
        self.has_same_nb_elements = True

    def to_matrix(self, idx=(), phase_idx=(), node_idx=(), concatenate_phases=True):
        if not self.phase:
            return np.ndarray((0, 1))

        phase_idx = phase_idx if isinstance(phase_idx, (list, tuple)) else [phase_idx]
        if self.has_same_nb_elements and concatenate_phases:
            node_idx = node_idx if isinstance(node_idx, (list, tuple)) else [node_idx]
            idx = idx if isinstance(idx, (list, tuple)) else [idx]

            range_phases = range(len(self.phase)) if phase_idx == () else phase_idx
            range_idx = range(self.nb_elements) if idx == () else idx

            data = np.ndarray((len(range_idx), 0))
            for idx_phase in range_phases:
                if idx_phase < range_phases[-1]:
                    range_nodes = range(self.phase[idx_phase].nb_t - 1) if node_idx == () else node_idx
                else:
                    range_nodes = range(self.phase[idx_phase].nb_t) if node_idx == () else node_idx
                for idx_node in range_nodes:
                    node = self.phase[idx_phase].node[idx_node][range_idx, :]
                    data = np.concatenate((data, node), axis=1)
        else:
            data = [Data.to_matrix(idx=idx, phase_idx=phase, node_idx=node_idx, concatenate_phases=False) for phase in phase_idx]

        return data

    def get_time_per_phase(self, phases=(), concatenate=False):
        if self.phase == []:
            return np.ndarray((0,))

        phases = phases if isinstance(phases, (list, tuple)) else [phases]
        range_phases = range(len(self.phase)) if phases == () else phases
        if not concatenate:
            return [self.phase[idx_phase].t for idx_phase in range_phases]
        else:
            t = [self.phase[idx_phase].t for idx_phase in range_phases]
            t_concat = []
            for t_tp in t:
                t_concat.extend(t_tp[:-1])
            t_concat.extend([t[-1][-1]])
            return np.array(t_concat)

    @staticmethod
    def get_data_from_V(ocp, V, phase_idx=None, integrate=False, interpolate_nb_frames=-1, concatenate=True):
        V_array = np.array(V).squeeze()

        if phase_idx is None:
            phase_idx = range(len(ocp.nlp))
        elif isinstance(phase_idx, int):
            phase_idx = [phase_idx]
        offsets = [0]
        for i, nlp in enumerate(ocp.nlp):
            offsets.append(offsets[i] + nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * (nlp["ns"]))

        data_states, data_controls = {}, {}
        for i in phase_idx:
            nlp = ocp.nlp[i]
            for key in nlp["has_states"].keys():
                if key not in data_states.keys():
                    data_states[key] = Data()

            for key in nlp["has_controls"].keys():
                if key not in data_controls.keys():
                    data_controls[key] = Data()

            V_phase = np.array(V_array[offsets[i] : offsets[i + 1]])
            nb_var = nlp["nx"] + nlp["nu"]
            offset = 0

            for key in nlp["has_states"]:
                data_states[key]._append_phase(
                    (nlp["t0"], nlp["tf"]),
                    Data._get_phase(V_phase, nlp["has_states"][key], nlp["ns"] + 1, offset, nb_var, False),
                )
                offset += nlp["has_states"][key]

            for key in nlp["has_controls"]:
                data_controls[key]._append_phase(
                    (nlp["t0"], nlp["tf"]),
                    Data._get_phase(V_phase, nlp["has_controls"][key], nlp["ns"], offset, nb_var, True),
                )
                offset += nlp["has_controls"][key]

        if integrate:
            data_states = Data._get_data_integrated_from_V(ocp, data_states, data_controls)

        if interpolate_nb_frames > 0:
            if integrate:
                raise RuntimeError("interpolate values are not compatible yet with integrated values")
            data_states = Data._get_data_interpolated_from_V(data_states, interpolate_nb_frames, concatenate)

        return data_states, data_controls

    @staticmethod
    def _get_data_integrated_from_V(ocp, data_states, data_controls):
        for idx_phase in range(ocp.nb_phases):
            dt = ocp.nlp[idx_phase]["dt"]
            nlp = ocp.nlp[idx_phase]
            for idx_node in reversed(range(ocp.nlp[idx_phase]["ns"])):
                x0 = Data._vertcat(data_states, list(nlp["has_states"].keys()), idx_phase, idx_node)
                p = Data._vertcat(data_controls, list(nlp["has_controls"].keys()), idx_phase, idx_node)
                xf_dof = np.array(ocp.nlp[idx_phase]["dynamics"](x0=x0, p=p)["xf"])  # Integrate

                offset = 0
                for key in nlp["has_states"]:
                    data_states[key]._horzcat_node(
                        dt, xf_dof[offset : offset + nlp["has_states"][key]], idx_phase, idx_node
                    )
                    offset += nlp["has_states"][key]
        return data_states

    @staticmethod
    def _get_data_interpolated_from_V(data_states, nb_frames, concatenate):
        for key in data_states:
            t = data_states[key].get_time_per_phase(concatenate=concatenate)
            d = data_states[key].to_matrix(concatenate_phases=concatenate)
            if not isinstance(d, list):
                t = [t]
                d = [d]

            for idx_phase in range(len(d)):
                t_phase = t[idx_phase]
                t_int = np.linspace(t_phase[0], t_phase[-1], nb_frames)
                x_phase = d[idx_phase]

                x_interpolate = np.ndarray((data_states[key].nb_elements, nb_frames))
                for j in range(data_states[key].nb_elements):
                    s = interpolate.splrep(t_phase, x_phase[j, :])
                    x_interpolate[j, :] = interpolate.splev(t_int, s)
                data_states[key].phase[idx_phase] = x_interpolate
        return data_states

    def _horzcat_node(self, dt, x_to_add, idx_phase, idx_node):
        self.phase[idx_phase].t = np.concatenate(
            (
                self.phase[idx_phase].t[: idx_node + 1],
                [self.phase[idx_phase].t[idx_node] + dt],
                self.phase[idx_phase].t[idx_node + 1 :],
            )
        )
        self.phase[idx_phase].node[idx_node] = np.concatenate((self.phase[idx_phase].node[idx_node], x_to_add), axis=1)

    @staticmethod
    def _get_phase(V_phase, var_size, nb_nodes, offset, nb_variables, duplicate_last_column):
        """
        Extracts variables from V.
        :param V_phase: numpy array : Extract of V for a phase.
        """
        array = np.ndarray((var_size, nb_nodes))
        for dof in range(var_size):
            array[dof, :] = V_phase[offset + dof :: nb_variables]

        if duplicate_last_column:
            return np.c_[array, array[:, -1]]
        else:
            return array

    @staticmethod
    def _vertcat(data, keys, phases=(), nodes=()):
        data_concat = data[keys[0]].to_matrix(phase_idx=phases, node_idx=nodes)
        for k in range(1, len(keys)):
            data_concat = np.concatenate((data_concat, data[keys[k]].to_matrix(phase_idx=phases, node_idx=nodes)))
        return data_concat

    def _append_phase(self, time, phase):
        self.phase.append(Data.Phase(time, phase))
        if self.nb_elements < 0:
            self.nb_elements = self.phase[-1].nb_elements

        if self.nb_elements != self.phase[-1].nb_elements:
            self.has_same_nb_elements = False
