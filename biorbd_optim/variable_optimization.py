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

    @staticmethod
    def vertcat(first_data, second_data, phases=(), nodes=()):
        first_array = first_data.to_matrix(phases=phases, nodes=nodes)
        second_array = second_data.to_matrix(phases=phases, nodes=nodes)
        return np.concatenate((first_array, second_array))

    def to_matrix(self, idx=(), phases=(), nodes=(), concatenate_phases=True):
        if self.phase == []:
            return np.ndarray((0, 1))

        phases = phases if isinstance(phases, (list, tuple)) else [phases]
        if self.has_same_nb_elements and concatenate_phases:
            nodes = nodes if isinstance(nodes, (list, tuple)) else [nodes]
            idx = idx if isinstance(idx, (list, tuple)) else [idx]

            range_phases = range(len(self.phase)) if phases == () else phases
            range_idx = range(self.nb_elements) if idx == () else idx

            data = np.ndarray((len(range_idx), 0))
            for idx_phase in range_phases:
                range_nodes = range(self.phase[idx_phase].nb_t) if nodes == () else nodes
                for idx_node in range_nodes:
                    node = self.phase[idx_phase].node[idx_node][range_idx, :]
                    data = np.concatenate((data, node), axis=1)
        else:
            data = [Data.to_matrix(idx=idx, phases=phase, nodes=nodes, concatenate_phases=False) for phase in phases]

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
    def get_data_from_V(ocp, V, num_phase=None, integrate=False, interpolate_nb_frames=-1, concatenate=True):
        V_array = np.array(V).squeeze()

        if num_phase is None:
            num_phase = range(len(ocp.nlp))
        elif isinstance(num_phase, int):
            num_phase = [num_phase]
        offsets = [0]
        for i, nlp in enumerate(ocp.nlp):
            offsets.append(offsets[i] + nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * (nlp["ns"]))

        data = {"q": Data(), "q_dot": Data(), "tau": Data()}
        if sum([nlp["has_muscles"] for nlp in ocp.nlp]):
            data["muscle"] = Data()

        for i in num_phase:
            nlp = ocp.nlp[i]
            V_phase = np.array(V_array[offsets[i] : offsets[i + 1]])
            nb_var = nlp["nx"] + nlp["nu"]

            data["q"]._append_phase((nlp["t0"], nlp["tf"]), Data._get_phase(V_phase, nlp["nbQ"], nlp["ns"] + 1, 0, nb_var, False))
            data["q_dot"]._append_phase((nlp["t0"], nlp["tf"]), Data._get_phase(V_phase, nlp["nbQdot"], nlp["ns"] + 1, nlp["nbQ"], nb_var, False))
            data["tau"]._append_phase((nlp["t0"], nlp["tf"]), Data._get_phase(V_phase, nlp["nbTau"], nlp["ns"], nlp["nx"], nb_var, True))
            if nlp["has_muscles"]:
                data["muscle"] = Data()
                data["muscle"]._append_phase((nlp["t0"], nlp["tf"]), Data._get_phase(V_phase, nlp["nbMuscle"], nlp["ns"], nlp["nx"] + nlp["nbTau"], nb_var, True,))

        if integrate:
            data = Data._get_data_integrated_from_V(ocp, data)

        if interpolate_nb_frames > 0:
            if integrate:
                raise RuntimeError("interpolate values are not compatible yet with integrated values")

            for key in data.keys():
                t = data[key].get_time_per_phase(concatenate=concatenate)
                d = data[key].to_matrix(concatenate_phases=concatenate)
                if not isinstance(d, list):
                    t = [t]
                    d = [d]

                for idx_phase in range(len(d)):
                    t_phase = t[idx_phase]
                    t_int = np.linspace(t_phase[0], t_phase[-1], interpolate_nb_frames)
                    x_phase = d[idx_phase]

                    x_interpolate = np.ndarray((data[key].nb_elements, interpolate_nb_frames))
                    for j in range(data[key].nb_elements):
                        s = interpolate.splrep(t_phase, x_phase[j, :])
                        x_interpolate[j, :] = interpolate.splev(t_int, s)
                    data[key].phase[idx_phase] = x_interpolate
        return data

    @staticmethod
    def _get_data_integrated_from_V(ocp, data):
        for idx_phase in range(ocp.nb_phases):
            dt = ocp.nlp[idx_phase]["dt"]
            for idx_node in reversed(range(ocp.nlp[idx_phase]["ns"])):
                x0 = Data.vertcat(data["q"], data["q_dot"], idx_phase, idx_node)
                if ocp.nlp[idx_phase]["has_muscles"]:
                    p = Data.vertcat(data["tau"], data["muscle"], idx_phase, idx_node)
                else:
                    p = data["q"].to_matrix(phases=idx_phase, nodes=idx_node)

                xf_dof = np.array(ocp.nlp[idx_phase]["dynamics"](x0=x0, p=p)["xf"])  # Integrate

                data["q"]._horzcat_node(dt, xf_dof[:ocp.nlp[idx_phase]["nbQ"]], idx_phase, idx_node)
                data["q_dot"]._horzcat_node(dt, xf_dof[ocp.nlp[idx_phase]["nbQ"]:], idx_phase, idx_node)
        return data

    def _horzcat_node(self, dt, x_to_add, idx_phase, idx_node):
        self.phase[idx_phase].t = np.concatenate((self.phase[idx_phase].t[:idx_node+1], [self.phase[idx_phase].t[idx_node]+dt], self.phase[idx_phase].t[idx_node+1:]))
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

    def _append_phase(self, time, phase):
        self.phase.append(Data.Phase(time, phase))
        if self.nb_elements < 0:
            self.nb_elements = self.phase[-1].nb_elements

        if self.nb_elements != self.phase[-1].nb_elements:
            self.has_same_nb_elements = False

