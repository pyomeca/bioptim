from typing import Union

import numpy as np
from scipy import interpolate
from .enums import ControlType


class Data:
    """
    Data manipulation and storage. Mostly for internal purpose

    Attributes
    ----------
    phase: list[Phase]
        A collection of phases.
    nb_elements: int
        The number of expected phases
    has_same_nb_elements: bool
        Variable that make sure the len(phase) and nb_elements are corresponding

    Methods
    -------
    to_matrix(self, idx: Union[int, list, tuple] = (), phase_idx: Union[int, list, tuple] = (), node_idx: Union[int, list, tuple] = (), concatenate_phases: bool = True) -> np.ndarray
        Parse the data into a np.ndarray
    set_time_per_phase(self, new_t: list)
        Set the time vector of the phase
    get_time_per_phase(self, phases: Union[int, list, tuple] = (), concatenate: bool = False) -> np.ndarray
        Get the time for each phase
    get_data(ocp: OptimalControlProgram, sol_x: dict, get_states: bool = True, get_controls: bool = True, get_parameters: bool = False, phase_idx: Union[int, list, tuple] = None, integrate: bool = False, interpolate_nb_frames: int = -1, concatenate: bool = True,) -> tuple
        Comprehensively parse the data from a solution
    get_data_object(ocp: OptimalControlProgram, V: np.ndarray, phase_idx: Union[int, list, tuple] = None, integrate: bool = False, interpolate_nb_frames: int = -1, concatenate: bool = True) -> tuple
        Parse an unstructured vector of data of data into their list of Phase format
    _get_data_integrated_from_V(ocp: OptimalControlProgram, data_states: dict, data_controls: dict, data_parameters: dict) -> dict
        Integrates the states
    _data_concatenated(data: dict) -> dict
        Concatenate all the phases
    _get_data_interpolated_from_V(data_states: dict, nb_frames: int) -> dict
        Interpolate the states
    _horzcat_node(self, dt: float, x_to_add: np.ndarray, idx_phase: int, idx_node: int)
        Concatenate the nodes of a Phase into a np.ndarray
    _get_phase(V_phase: np.ndarray, var_size: int, nb_nodes: int, offset: int, nb_variables: int, duplicate_last_column: bool) -> np.ndarray
        Extract the data of a specific phase from an unstructured vector of data
    _vertcat(data: np.ndarray, keys: str, phases: Union[int, list, tuple] = (), nodes: Union[int, list, tuple] = ())
        Add new elements (rows) to the data
    _append_phase(self, time: np.ndarray, phase: "Data.Phase")
        Add a new phase to the phase list
    """

    class Phase:
        """

        Attributes
        ----------
        node: list[np.ndarray]
            The actual values stored by nodes
        nb_elements:
            The number of expected nodes
        t: np.array
            The time vector
        nb_t: int
            The len of the time vector
        """

        def __init__(self, time: np.ndarray, phase: np.ndarray):
            """
            Parameters
            ----------
            time: np.ndarray
                The time vector
            phase: np.ndarray
                The values of the nodes
            """
            self.node = [node.reshape(node.shape[0], 1) for node in phase.T]
            self.nb_elements = phase.shape[0]
            self.t = time
            self.nb_t = self.t.shape[0]

    def __init__(self):
        """
        Parameters
        ----------
        """
        self.phase = []
        self.nb_elements = -1
        self.has_same_nb_elements = True

    def to_matrix(
        self,
        idx: Union[int, list, tuple] = (),
        phase_idx: Union[int, list, tuple] = (),
        node_idx: Union[int, list, tuple] = (),
        concatenate_phases: bool = True,
    ) -> np.ndarray:
        """
        Parse the data into a np.ndarray

        Parameters
        ----------
        idx: Union[int, list, tuple]
            The indices of the rows to keep
        phase_idx: Union[int, list, tuple]
            The phases to keep
        node_idx: Union[int, list, tuple]
            The nodes in the phases to keep
        concatenate_phases: bool
            If the phases should be concatenated [True] or in a list [False]
        Returns
        -------
        The data parsed into a np.ndarray
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

    def set_time_per_phase(self, new_t: list):
        """
        Set the time vector of the phase

        Parameters
        ----------
        new_t: list[list[int, int]]
            The list of initial and final times for all the phases
        """

        for i, phase in enumerate(self.phase):
            phase.t = np.linspace(new_t[i][0], new_t[i][1], len(phase.node))

    def get_time_per_phase(self, phases: Union[int, list, tuple] = (), concatenate: bool = False) -> np.ndarray:
        """
        Get the time for each phase

        Parameters
        ----------
        phases: Union[int, list, tuple]
            The phases to get the time from
        concatenate: bool
            If all the time should be concatenated [True] or in a list [False]
        Returns
        -------
        The time for each phase
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
        sol_x: dict,
        get_states: bool = True,
        get_controls: bool = True,
        get_parameters: bool = False,
        phase_idx: Union[int, list, tuple] = None,
        integrate: bool = False,
        interpolate_nb_frames: int = -1,
        concatenate: bool = True,
    ) -> tuple:
        """
        Comprehensively parse the data from a solution

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        sol_x: dict
            The dictionary of solution
        get_states: bool
            If the function should return the states
        get_controls: bool
            If the function should return the controls
        get_parameters: bool
            If the function should return the parameters
        phase_idx: Union[int, list, tuple]
            The index of the phase to get the data from
        integrate: bool
            If the data should be integrate (returns the points at each time step of the RK)
        interpolate_nb_frames: int
            If the data should be interpolated to change the frame rate
        concatenate: bool
            If the phases should be concatenated into one matrix [True] or returned in a list [False]

        Returns
        -------
        The data comprehensively parsed
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
    def get_data_object(
        ocp,
        V: np.ndarray,
        phase_idx: Union[int, list, tuple] = None,
        integrate: bool = False,
        interpolate_nb_frames: int = -1,
        concatenate: bool = True,
    ) -> tuple:
        """
        Parse an unstructured vector of data of data into their list of Phase format

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        V: np.ndarray
            The unstructured vector of data of data
        phase_idx: Union[int, list, tuple]
            Index of the phases to return
        integrate: bool
            If V should be integrated between nodes
        interpolate_nb_frames: int
            Number of frames to interpolate the solution
        concatenate: bool
            If the data should be return in one matrix [True] or in a list [False]

        Returns
        -------
        The states, controls and parameters in list of Phase format
        """
        V_array = np.array(V).squeeze()
        data_states, data_controls, data_parameters = {}, {}, {}
        phase_time = [nlp.tf for nlp in ocp.nlp]

        if phase_idx is None:
            phase_idx = range(len(ocp.nlp))
        elif isinstance(phase_idx, int):
            phase_idx = [phase_idx]

        offset = 0
        for key in ocp.param_to_optimize:
            if ocp.param_to_optimize[key]:
                nb_param = ocp.param_to_optimize[key].size
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
            if nlp.control_type == ControlType.CONSTANT:
                offsets.append(offsets[i] + nlp.nx * (nlp.ns + 1) + nlp.nu * (nlp.ns))
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                offsets.append(offsets[i] + (nlp.nx + nlp.nu) * (nlp.ns + 1))
            else:
                raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

        for i in phase_idx:
            nlp = ocp.nlp[i]
            for key in nlp.var_states.keys():
                if key not in data_states.keys():
                    data_states[key] = Data()

            for key in nlp.var_controls.keys():
                if key not in data_controls.keys():
                    data_controls[key] = Data()

            V_phase = np.array(V_array[offsets[i] : offsets[i + 1]])
            nb_var = nlp.nx + nlp.nu
            offset = 0

            for key in nlp.var_states:
                data_states[key]._append_phase(
                    (0, phase_time[i]),
                    Data._get_phase(V_phase, nlp.var_states[key], nlp.ns + 1, offset, nb_var, False),
                )
                offset += nlp.var_states[key]

            for key in nlp.var_controls:
                if nlp.control_type == ControlType.CONSTANT:
                    data_controls[key]._append_phase(
                        (0, phase_time[i]),
                        Data._get_phase(V_phase, nlp.var_controls[key], nlp.ns, offset, nb_var, True),
                    )
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    data_controls[key]._append_phase(
                        (0, phase_time[i]),
                        Data._get_phase(V_phase, nlp.var_controls[key], nlp.ns + 1, offset, nb_var, False),
                    )
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")
                offset += nlp.var_controls[key]

        if integrate:
            data_states = Data._get_data_integrated_from_V(ocp, data_states, data_controls, data_parameters)

        if concatenate:
            data_states = Data._data_concatenated(data_states)
            data_controls = Data._data_concatenated(data_controls)

        if interpolate_nb_frames > 0:
            if integrate:
                raise RuntimeError("interpolate values are not compatible yet with integrated values")
            data_states = Data._get_data_interpolated_from_V(data_states, interpolate_nb_frames)
            data_controls = Data._get_data_interpolated_from_V(data_controls, interpolate_nb_frames)

        return data_states, data_controls, data_parameters

    @staticmethod
    def _get_data_integrated_from_V(ocp, data_states: dict, data_controls: dict, data_parameters: dict) -> dict:
        """
        Integrates the states

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        data_states: dict
            A dictionary of all the states
        data_controls: dict
            A dictionary of all the controls
        data_parameters: dict
            A dictionary of all the parameters

        Returns
        -------
        The dictionary of states integrated
        """

        # Check if time is optimized
        for idx_phase in range(ocp.nb_phases):
            dt = ocp.nlp[idx_phase].dt
            nlp = ocp.nlp[idx_phase]
            for idx_node in reversed(range(ocp.nlp[idx_phase].ns)):
                x0 = Data._vertcat(data_states, list(nlp.var_states.keys()), idx_phase, idx_node)
                if nlp.control_type == ControlType.CONSTANT:
                    p = Data._vertcat(data_controls, list(nlp.var_controls.keys()), idx_phase, idx_node)
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    p = Data._vertcat(data_controls, list(nlp.var_controls.keys()), idx_phase, [idx_node, idx_node + 1])
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")
                params = Data._vertcat(data_parameters, [key for key in ocp.param_to_optimize])
                xf_dof = np.array(ocp.nlp[idx_phase].dynamics[idx_node](x0=x0, p=p, params=params)["xall"])

                offset = 0
                for key in nlp.var_states:
                    data_states[key]._horzcat_node(
                        dt, xf_dof[offset : offset + nlp.var_states[key], 1:], idx_phase, idx_node
                    )
                    offset += nlp.var_states[key]
        return data_states

    @staticmethod
    def _data_concatenated(data: dict) -> dict:
        """
        Concatenate all the phases

        Parameters
        ----------
        data: dict
            The dictionary of data

        Returns
        -------
        The new dictionary of data concatenated
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
    def _get_data_interpolated_from_V(data_states: dict, nb_frames: int) -> dict:
        """
        Interpolate the states

        Parameters
        ----------
        data_states: dict
            A dictionary of all the states
        nb_frames: int
            The number of frames to interpolate the data

        Returns
        -------
        The dictionary of states interpolated
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

    def _horzcat_node(self, dt: float, x_to_add: np.ndarray, idx_phase: int, idx_node: int):
        """
        Concatenate the nodes of a Phase into a np.ndarray

        Parameters
        ----------
        dt: float
            The delta time of the concatenated values
        x_to_add: np.ndarray
            The data to concatenate
        idx_phase: int
            The index of phase in which the node to add
        idx_node
            The index of the node before the concatation point
        """

        self.phase[idx_phase].t = np.concatenate(
            (
                self.phase[idx_phase].t[: idx_node + 1],
                [self.phase[idx_phase].t[idx_node] + dt],
                self.phase[idx_phase].t[idx_node + 1 :],
            )
        )
        self.phase[idx_phase].node[idx_node] = np.concatenate((self.phase[idx_phase].node[idx_node], x_to_add), axis=1)

    @staticmethod
    def _get_phase(
        V_phase: np.ndarray, var_size: int, nb_nodes: int, offset: int, nb_variables: int, duplicate_last_column: bool
    ) -> np.ndarray:
        """
        Extract the data of a specific phase from an unstructured vector of data

        Parameters
        ----------
        V_phase: np.ndarray
            The unstructured vector of data
        var_size: int
            The size of the variable to extract
        nb_nodes:
            The number of node to extract
        offset:
            The index of the first element to extract
        nb_variables:
            The number of variable to skip
        duplicate_last_column:
            If the last column should be duplicated

        Returns
        -------
        The data in the form of a np.ndarray
        """

        array = np.ndarray((var_size, nb_nodes))
        for dof in range(var_size):
            array[dof, :] = V_phase[offset + dof :: nb_variables]

        if duplicate_last_column:
            return np.c_[array, array[:, -1]]
        else:
            return array

    @staticmethod
    def _vertcat(
        data: np.ndarray, keys: str, phases: Union[int, list, tuple] = (), nodes: Union[int, list, tuple] = ()
    ):
        """
        Add new elements (rows) to the data

        Parameters
        ----------
        data: np.ndarray
            The rows to add to the data
        keys: str
            The name of the data to add
        phases: Union[int, list, tuple]
            The phases to add the data to
        nodes: Union[int, list, tuple]
            The nodes to add the data to
        """

        def get_matrix(elem: Union[Data.Phase, np.ndarray]):
            """
            Converts the data into matrix if needed

            Parameters
            ----------
            elem: Union[Data.Phase, np.ndarray]
                The data to convert

            Returns
            -------
            The data in the matrix format
            """

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

    def _append_phase(self, time: np.ndarray, phase: "Data.Phase"):
        """
        Add a new phase to the phase list

        Parameters
        ----------
        time: np.ndarray
            The time vector
        phase: "Data.Phase"
            The phase to concatenate
        """

        time = np.linspace(time[0], time[1], len(phase[0]))
        self.phase.append(Data.Phase(time, phase))
        if self.nb_elements < 0:
            self.nb_elements = self.phase[-1].nb_elements

        if self.nb_elements != self.phase[-1].nb_elements:
            self.has_same_nb_elements = False
