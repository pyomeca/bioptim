import numpy as np
from scipy import interpolate


class Data:
    class Phase:
        def __init__(self, time, phase):
            """
            Initializes phases with user provided information.
            :param time: Duration of the movement optimized. (float)
            :param phase: Phases information. (list of tuples)
            """
            self.node = [node.reshape(node.shape[0], 1) for node in phase.T]
            self.nb_elements = phase.shape[0]
            self.t = time
            self.nb_t = self.t.shape[0]

    def __init__(self):
        self.phase = []
        self.nb_elements = -1
        self.has_same_nb_elements = True

    def to_matrix(self, idx=(), phase_idx=(), node_idx=(), concatenate_phases=True):
        """
        Conversion of lists int matrix.
        :param idx: Index of the target in the destination variable matrix. (integer)
        :param phase_idx: Index of the phase targeted in the origin variable list. (integer)
        :param node_idx: Index of the node targetedin the origin variable list. (integer)
        :param concatenate_phases: If True, concatenates the phases into one big phase. (bool)
        """
        if not self.phase:
            return np.ndarray((0, 1))

        phase_idx = phase_idx if isinstance(phase_idx, (list, tuple)) else [phase_idx]
        range_phases = range(len(self.phase)) if phase_idx == () else phase_idx
        if (self.has_same_nb_elements and concatenate_phases) or len(range_phases) == 1:
            node_idx = node_idx if isinstance(node_idx, (list, tuple)) else [node_idx]
            idx = idx if isinstance(idx, (list, tuple)) else [idx]

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
            data = [
                self.to_matrix(idx=idx, phase_idx=phase, node_idx=node_idx, concatenate_phases=False)
                for phase in range_phases
            ]

        return data

    def set_time_per_phase(self, new_t):
        """
        Sets the new time vector in function of the duration of the movement.
        :param new_t: Duration of the movement. (float)
        """
        for i, phase in enumerate(self.phase):
            phase.t = np.linspace(new_t[i][0], new_t[i][1], len(phase.node))

    def get_time_per_phase(self, phases=(), concatenate=False):
        """
        Sets the new time vector for each phases.
        :param phases: Phases. (list of tuple)
        :param concatenate: If True, concatenates all the phases into one big phase. (bool)
        :return t: Time vector.
        """
        if not self.phase:
            return np.ndarray((0,))

        phases = phases if isinstance(phases, (list, tuple)) else [phases]
        range_phases = range(len(self.phase)) if phases == () else phases
        if not concatenate:
            t = [self.phase[idx_phase].t for idx_phase in range_phases]
            if len(t) == 1:
                t = t[0]
            return t
        else:
            t = [self.phase[idx_phase].t for idx_phase in range_phases]
            t_concat = np.array(t[0])
            for t_idx in range(1, len(t)):
                t_concat = np.concatenate((t_concat[:-1], t_concat[-1] + t[t_idx]))
            return np.array(t_concat)

    @staticmethod
    def get_data(
        ocp,
        sol_x,
        get_states=True,
        get_controls=True,
        get_parameters=False,
        phase_idx=None,
        integrate=False,
        interpolate_nb_frames=-1,
        concatenate=True,
    ):
        """
        Rearranges the solution states, controls and parameters of hte optimization into a list (out).
        :param sol_x: Solution of the optimization. (dictionary)
        :param get_states: If True, states are included in the out list. (bool)
        :param get_controls: If True, controls are included in the out list. (bool)
        :param get_parameters: If True, parameters are included in the out list. (bool)
        :param phase_idx: Index of the phase. (integer)
        :param integrate: If True, solution is integrated between nodes. (bool)
        :param interpolate_nb_frames: Number of frames to interpolate the solution to. (integer)
        :param concatenate: If True, concatenates all phases into one big phase. (bool)
        :return out: Rearranged ist of the solution. (list)
        """
        if isinstance(sol_x, dict) and "x" in sol_x:
            sol_x = sol_x["x"]

        data_states, data_controls, data_parameters = Data.get_data_object(
            ocp, sol_x, phase_idx, integrate, interpolate_nb_frames, concatenate
        )

        out = []
        if get_states:
            data_states_out = {}
            for key in data_states:
                data_states_out[key] = data_states[key].to_matrix(concatenate_phases=False)
            out.append(data_states_out)

        if get_controls:
            data_controls_out = {}
            for key in data_controls:
                data_controls_out[key] = data_controls[key].to_matrix(concatenate_phases=False)
            out.append(data_controls_out)

        if get_parameters:
            out.append(data_parameters)

        if len(out) == 1:
            return out[0]
        else:
            return out

    @staticmethod
    def get_data_object(ocp, V, phase_idx=None, integrate=False, interpolate_nb_frames=-1, concatenate=True):
        """

        :param V:
        :param phase_idx: Index of the phase. (integer)
        :param integrate: If True V is integrated between nodes. (bool)
        :param interpolate_nb_frames: Number of frames to interpolate the solution to. (integer)
        :param concatenate: If True, concatenates all phases into one big phase. (bool)
        :return: data_states -> Optimal states. (dictionary), data_controls -> Optimal controls. (dictionary)
        and data_parameters -> Optimal parameters. (dictionary)
        """
        V_array = np.array(V).squeeze()
        data_states, data_controls, data_parameters = {}, {}, {}
        phase_time = [nlp["tf"] for nlp in ocp.nlp]

        if phase_idx is None:
            phase_idx = range(len(ocp.nlp))
        elif isinstance(phase_idx, int):
            phase_idx = [phase_idx]

        offset = 0
        for key in ocp.param_to_optimize:
            if ocp.param_to_optimize[key]:
                nb_param = ocp.param_to_optimize[key]["size"]
                data_parameters[key] = np.array(V[offset : offset + nb_param])
                offset += nb_param

                if key == "time":
                    cmp = 0
                    for i in range(len(phase_time)):
                        if isinstance(phase_time[i], ocp.CX):
                            phase_time[i] = data_parameters["time"][cmp, 0]
                            cmp += 1

        offsets = [offset]
        for i, nlp in enumerate(ocp.nlp):
            offsets.append(offsets[i] + nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * (nlp["ns"]))

        for i in phase_idx:
            nlp = ocp.nlp[i]
            for key in nlp["var_states"].keys():
                if key not in data_states.keys():
                    data_states[key] = Data()

            for key in nlp["var_controls"].keys():
                if key not in data_controls.keys():
                    data_controls[key] = Data()

            V_phase = np.array(V_array[offsets[i] : offsets[i + 1]])
            nb_var = nlp["nx"] + nlp["nu"]
            offset = 0

            for key in nlp["var_states"]:
                data_states[key]._append_phase(
                    (0, phase_time[i]),
                    Data._get_phase(V_phase, nlp["var_states"][key], nlp["ns"] + 1, offset, nb_var, False),
                )
                offset += nlp["var_states"][key]

            for key in nlp["var_controls"]:
                data_controls[key]._append_phase(
                    (0, phase_time[i]),
                    Data._get_phase(V_phase, nlp["var_controls"][key], nlp["ns"], offset, nb_var, True),
                )
                offset += nlp["var_controls"][key]

        if integrate:
            data_states = Data._get_data_integrated_from_V(ocp, data_states, data_controls, data_parameters)

        if concatenate:
            data_states = Data._data_concatenated(data_states)
            data_controls = Data._data_concatenated(data_controls)

        if interpolate_nb_frames > 0:
            if integrate:
                raise RuntimeError("interpolate values are not compatible yet with integrated values")
            data_states = Data._get_data_interpolated_from_V(data_states, interpolate_nb_frames)

        return data_states, data_controls, data_parameters

    @staticmethod
    def _get_data_integrated_from_V(ocp, data_states, data_controls, data_parameters):
        """
        Integrates data between nodes.
        :param data_states: Optimal states. (dictionary)
        :param data_controls: Optimal controls. (dictionary)
        :return: data_states -> Integrated between node optimal states. (dictionary)
        """
        # Check if time is optimized
        for idx_phase in range(ocp.nb_phases):
            dt = ocp.nlp[idx_phase]["dt"]
            nlp = ocp.nlp[idx_phase]
            for idx_node in reversed(range(ocp.nlp[idx_phase]["ns"])):
                x0 = Data._vertcat(data_states, list(nlp["var_states"].keys()), idx_phase, idx_node)
                p = Data._vertcat(data_controls, list(nlp["var_controls"].keys()), idx_phase, idx_node)
                params = Data._vertcat(data_parameters, [key for key in ocp.param_to_optimize if key != "time"])
                xf_dof = np.array(ocp.nlp[idx_phase]["dynamics"][idx_node](x0=x0, p=p, params=params)["xall"])

                offset = 0
                for key in nlp["var_states"]:
                    data_states[key]._horzcat_node(
                        dt, xf_dof[offset : offset + nlp["var_states"][key], 1:], idx_phase, idx_node
                    )
                    offset += nlp["var_states"][key]
        return data_states

    @staticmethod
    def _data_concatenated(data):
        """
        Concatenates phases into one big phase.
        :param data: Variable to concatenate. (dictionary)
        :return: data -> Variable concatenated. (dictionary)
        """
        for key in data:
            if data[key].has_same_nb_elements:
                data[key].phase = [
                    Data.Phase(
                        data[key].get_time_per_phase(concatenate=True), data[key].to_matrix(concatenate_phases=True)
                    )
                ]
        return data

    @staticmethod
    def _get_data_interpolated_from_V(data_states, nb_frames):
        """
        Interpolates data between nodes.
        :param data_states: Optimal states. (dictionary)
        :param nb_frames: Number of frames to interpolate to. (integer)
        :return: data_states -> Integrated between node optimal states. (dictionary)
        """
        for key in data_states:
            t = data_states[key].get_time_per_phase(concatenate=False)
            d = data_states[key].to_matrix(concatenate_phases=False)
            if not isinstance(d, (tuple, list)):
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
                data_states[key].phase[idx_phase] = Data.Phase(t_int, x_interpolate)

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
        def get_matrix(elem):
            if isinstance(elem, Data):
                return elem.to_matrix(phase_idx=phases, node_idx=nodes)
            else:
                return elem

        if keys:
            data_concat = get_matrix(data[keys[0]])
            for k in range(1, len(keys)):
                data_concat = np.concatenate((data_concat, get_matrix(data[keys[k]])))
            return data_concat
        else:
            return np.empty((0, 0))

    def _append_phase(self, time, phase):
        time = np.linspace(time[0], time[1], len(phase[0]))
        self.phase.append(Data.Phase(time, phase))
        if self.nb_elements < 0:
            self.nb_elements = self.phase[-1].nb_elements

        if self.nb_elements != self.phase[-1].nb_elements:
            self.has_same_nb_elements = False
