from typing import Union

import numpy as np
from scipy import interpolate as sci_interp

from bioptim.misc.enums import ControlType, DataType
from bioptim.optimization.variable import OptimizationVariable


class Solution:
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

    def __init__(self, ocp, sol):
        self.ocp = ocp

        self.vector = sol["x"] if isinstance(sol, dict) and "x" in sol else sol
        self.cost = sol["f"] if isinstance(sol, dict) and "f" in sol else None
        self.constraints = sol["g"] if isinstance(sol, dict) and "g" in sol else None

        self.lam_g = sol["lam_g"] if isinstance(sol, dict) and "lam_g" in sol else None
        self.lam_p = sol["lam_p"] if isinstance(sol, dict) and "lam_p" in sol else None
        self.lam_x = sol["lam_x"] if isinstance(sol, dict) and "lam_x" in sol else None

        # Extract the data now for further use
        self._data_states, self._data_controls, self._data_parameters = self.ocp.v.to_dictionaries(self.vector)
        self._complete_control()
        self.phase_time = self.ocp.v.extract_phase_time(self.vector)

    @property
    def states(self):
        return self._data_states[0] if len(self._data_states) == 1 else self._data_states

    @property
    def controls(self):
        return self._data_controls[0] if len(self._data_controls) == 1 else self._data_controls

    @property
    def parameters(self):
        return self._data_parameters

    def integrate(self, concatenate: bool = False) -> list:
        """
        Integrates the states

        Returns
        -------
        The dictionary of states integrated
        """
        # This will become relevant when concatenate is used
        phase_time = [0] + [sum([self.phase_time[i+1] for i in range(self.ocp.n_phases)])]

        ocp = self.ocp
        data_states_out = []
        for _ in range(len(self._data_states)):
            data_states_out.append({})

        params = self._data_parameters["all"]
        for p in range(len(self._data_states)):
            n_steps = ocp.nlp[p].n_integration_steps + 1

            for key in self._data_states[p]:
                shape = self._data_states[p][key].shape
                data_states_out[p][key] = np.ndarray((shape[0], (shape[1] - 1) * n_steps + 1))

            # Integrate
            for n in range(ocp.nlp[p].ns):
                x0 = self._data_states[p]["all"][:, n]
                u = self._data_controls[p]["all"][:, n]
                data_states_out[p]["all"] = np.array(ocp.nlp[p].dynamics[n](x0=x0, p=u, params=params)["xall"])
                off = 0
                for key in ocp.nlp[p].var_states:
                    data_states_out[p][key][:, n*n_steps: (n+1)*n_steps] = data_states_out[p]["all"][off:off+ocp.nlp[p].var_states[key], :]
                    off += ocp.nlp[p].var_states[key]

            # Copy last states
            for key in ocp.nlp[p].var_states:
                data_states_out[p][key][:, -1] = self._data_states[p][key][:, -1]

        return data_states_out[0] if len(data_states_out) == 1 else data_states_out

    def interpolate(self, n_frames: int, data_type=DataType.STATES) -> list:
        """
        Interpolate the states

        Parameters
        ----------
        n_frames: int
            The number of frames to interpolate the data

        Returns
        -------
        The dictionary of states interpolated
        """

        if data_type == DataType.STATES:
            data = self._data_states
        elif data_type == DataType.CONTROLS:
            data = self._data_controls
        else:
            raise ValueError("Interpolate can only be called with DataType.STATES or DataType.CONTROLS")

        data_out = []
        for _ in range(len(data)):
            data_out.append({})
        for idx_phase in range(len(data)):
            x_phase = data[idx_phase]["all"]
            n_elements = x_phase.shape[0]

            t_phase = np.linspace(self.phase_time[idx_phase], self.phase_time[idx_phase + 1], x_phase.shape[1])
            t_int = np.linspace(self.phase_time[0], self.phase_time[-1], n_frames)

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

        return data_out[0] if len(data_out) == 1 else data_out

    def concatenate_phases(self, data) -> dict:
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

        if isinstance(data, dict):
            return data

        # Sanity check (all phases must contain the same keys with the same dimensions)
        keys = data[0].keys()
        sizes = [data[0][d].shape[0] for d in data[0]]
        for d in data:
            if d.keys() != keys or [d[key].shape[0] for key in d] != sizes:
                raise RuntimeError("Program dimension must be coherent across phases to concatenate them")

        data_out = [{}]
        for i, key in enumerate(keys):
            data_out[0][key] = np.ndarray((sizes[i], 0))

        for p in range(self.ocp.n_phases):
            d = data[p]
            for key in d:
                data_out[0][key] = np.concatenate((data_out[0][key], d[key][:, :self.ocp.nlp[p].ns]), axis=1)
        for key in data[-1]:
            data_out[0][key] = np.concatenate((data_out[0][key], data[-1][key][:, -1][:, np.newaxis]), axis=1)

        return data_out[0]

    def _complete_control(self):
        for p, nlp in enumerate(self.ocp.nlp):
            if nlp.control_type == ControlType.CONSTANT:
                for key in self._data_controls[p]:
                    self._data_controls[p][key] = np.concatenate((self._data_controls[p][key], self._data_controls[p][key][:, -1][:, np.newaxis]), axis=1)
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                pass
            else:
                raise NotImplementedError(f"ControlType {nlp.control_type} is not implemented  in _complete_control")
