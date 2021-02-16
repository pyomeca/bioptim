from typing import Union

import numpy as np
from scipy import interpolate as sci_interp

from .enums import ControlType
from ..optimization.variable import OptimizationVariable


class Data:
    """
    Data manipulation and storage

    Methods
    -------
    to_matrix(self, idx: Union[int, list, tuple] = (), phase_idx: Union[int, list, tuple] = (),
            node_idx: Union[int, list, tuple] = (), concatenate_phases: bool = True) -> np.ndarray
        Parse the data into a np.ndarray
    set_time_per_phase(self, new_t: list)
        Set the time vector of the phase
    get_time_per_phase(self, phases: Union[int, list, tuple] = (), concatenate: bool = False) -> np.ndarray
        Get the time for each phase
    get_data(ocp: OptimalControlProgram, sol_x: dict, get_states: bool = True, get_controls: bool = True,
            get_parameters: bool = False, phase_idx: Union[int, list, tuple] = None, integrate: bool = False,
            interpolate_n_frames: int = -1, concatenate: bool = True,) -> tuple
        Comprehensively parse the data from a solution
    get_data_object(ocp: OptimalControlProgram, V: np.ndarray, phase_idx: Union[int, list, tuple] = None,
            integrate: bool = False, interpolate_n_frames: int = -1, concatenate: bool = True) -> tuple
        Parse an unstructured vector of data of data into their list of Phase format
    _get_data_integrated_from_V(ocp: OptimalControlProgram, data_states: dict,
            data_controls: dict, data_parameters: dict) -> dict
        Integrates the states
    _data_concatenated(data: dict) -> dict
        Concatenate all the phases
    _get_data_interpolated_from_V(data_states: dict, n_frames: int) -> dict
        Interpolate the states
    _horzcat_node(self, dt: float, x_to_add: np.ndarray, idx_phase: int, idx_node: int)
        Concatenate the nodes of a Phase into a np.ndarray
    _get_phase(V_phase: np.ndarray, var_size: int, n_nodes: int, offset: int, n_variables: int,
            duplicate_last_column: bool) -> np.ndarray
        Extract the data of a specific phase from an unstructured vector of data
    _vertcat(data: np.ndarray, keys: str, phases: Union[int, list, tuple] = (), nodes: Union[int, list, tuple] = ())
        Add new elements (rows) to the data
    _append_phase(self, time: np.ndarray, phase: "Data.Phase")
        Add a new phase to the phase list
    """

    @staticmethod
    def get_data(
        ocp,
        sol: Union[dict, np.ndarray],
        get_states: bool = True,
        get_controls: bool = True,
        get_parameters: bool = False,
        phase_idx: Union[int, list, tuple] = None,
        integrate: bool = False,
        interpolate_n_frames: int = 0,
        concatenate: bool = True,
    ) -> Union[dict, list]:
        """
        Comprehensively parse the data from a solution

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        sol: Union[dict, np.ndarray]
            The dictionary of solution or the v formatted vector
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
        interpolate_n_frames: int
            If the data should be interpolated to change the frame rate
        concatenate: bool
            If the phases should be concatenated into one matrix [True] or returned in a list [False]

        Returns
        -------
        The data comprehensively parsed
        """

        if isinstance(sol, dict) and "x" in sol:
            sol = sol["x"]

        phase_idx = OptimizationVariable.phase_index_to_slice(ocp, phase_idx)
        phase_time = ocp.v.extract_phase_time(sol)

        data_states, data_controls, data_parameters = OptimizationVariable.to_dictionaries(ocp.v, sol, phase_idx)
        Data._complete_control(ocp, data_controls)

        if integrate:
            data_states = Data._integrate(ocp, data_states, data_controls, data_parameters)

        if concatenate:
            data_states = Data._concatenate(ocp, data_states, phase_idx)
            data_controls = Data._concatenate(ocp, data_controls, phase_idx)
            phase_time = [0] + [sum([phase_time[i+1] for i in phase_idx])]

        if interpolate_n_frames > 0:
            if integrate:
                raise RuntimeError("interpolate values are not compatible yet with integrated values")
            data_states = Data._interpolate(data_states, interpolate_n_frames, phase_time)
            data_controls = Data._interpolate(data_controls, interpolate_n_frames, phase_time)

        out = []
        if get_states:
            if len(data_states) == 1:
                out.append(data_states[0])
            else:
                out.append(data_states)

        if get_controls:
            if len(data_controls) == 1:
                out.append(data_controls[0])
            else:
                out.append(data_controls)

        if get_parameters:
            out.append(data_parameters)

        if len(out) == 1:
            return out[0]
        else:
            return out

    @staticmethod
    def _complete_control(ocp, data_controls):
        for p, nlp in enumerate(ocp.nlp):
            if nlp.control_type == ControlType.CONSTANT:
                for key in data_controls[p]:
                    data_controls[p][key] = np.concatenate((data_controls[p][key], data_controls[p][key][:, -1][:, np.newaxis]), axis=1)

    @staticmethod
    def _integrate(ocp, data_states: dict, data_controls: dict, data_parameters: dict) -> list:
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

        data_states_out = []
        for _ in range(len(data_states)):
            data_states_out.append({})

        params = data_parameters["all"]
        for p in range(len(data_states)):
            n_steps = ocp.nlp[p].n_integration_steps + 1

            for key in data_states[p]:
                shape = data_states[p][key].shape
                data_states_out[p][key] = np.ndarray((shape[0], (shape[1] - 1) * n_steps + 1))

            # Integrate
            for n in range(ocp.nlp[p].ns):
                x0 = data_states[p]["all"][:, n]
                u = data_controls[p]["all"][:, n]
                data_states_out[p]["all"] = np.array(ocp.nlp[p].dynamics[n](x0=x0, p=u, params=params)["xall"])
                off = 0
                for key in ocp.nlp[p].var_states:
                    data_states_out[p][key][:, n*n_steps: (n+1)*n_steps] = data_states_out[p]["all"][off:off+ocp.nlp[p].var_states[key], :]
                    off += ocp.nlp[p].var_states[key]

            # Copy last states
            for key in ocp.nlp[p].var_states:
                data_states_out[p][key][:, -1] = data_states[p][key][:, -1]

        return data_states_out

    @staticmethod
    def _concatenate(ocp, data: list, phase_idx) -> list:
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

        # Sanity check (all phases must contain the same keys with the same dimensions)
        keys = data[0].keys()
        sizes = [data[0][d].shape[0] for d in data[0]]
        for d in data:
            if d.keys() != keys or [d[key].shape[0] for key in d] != sizes:
                raise RuntimeError("Program dimension must be coherent across phases to concatenate them")

        data_out = [{}]
        for i, key in enumerate(keys):
            data_out[0][key] = np.ndarray((sizes[i], 0))

        for p in phase_idx:
            d = data[p]
            for key in d:
                data_out[0][key] = np.concatenate((data_out[0][key], d[key][:, :ocp.nlp[p].ns]), axis=1)
        for key in data[-1]:
            data_out[0][key] = np.concatenate((data_out[0][key], data[-1][key][:, -1][:, np.newaxis]), axis=1)

        return data_out

    @staticmethod
    def _interpolate(data: dict, n_frames: int, phase_time) -> list:
        """
        Interpolate the states

        Parameters
        ----------
        data: dict
            A dictionary of all the data to interpolate
        n_frames: int
            The number of frames to interpolate the data

        Returns
        -------
        The dictionary of states interpolated
        """

        data_out = []
        for _ in range(len(data)):
            data_out.append({})
        for idx_phase in range(len(data)):
            x_phase = data[idx_phase]["all"]
            n_elements = x_phase.shape[0]

            t_phase = np.linspace(phase_time[idx_phase], phase_time[idx_phase + 1], x_phase.shape[1])
            t_int = np.linspace(phase_time[0], phase_time[-1], n_frames)

            x_interpolate = np.ndarray((n_elements, n_frames))
            for j in range(n_elements):
                s = sci_interp.splrep(t_phase, x_phase[j, :])
                x_interpolate[j, :] = sci_interp.splev(t_int, s)
            data_out[idx_phase]["all"] = x_interpolate

            offset = 0
            for key in data[idx_phase]:
                if key == "all":
                    continue
                n_elements = data[idx_phase][key].shape[0]
                data_out[idx_phase][key] = data_out[idx_phase]["all"][offset: offset + n_elements]
                offset += n_elements

        return data_out

