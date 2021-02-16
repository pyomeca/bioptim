from typing import Any

import numpy as np
from scipy import interpolate as sci_interp
from casadi import Function
from matplotlib import pyplot as plt

from ..misc.enums import ControlType, DataType
from ..misc.utils import check_version


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
        self.time_to_optimize = sol["time_tot"] if isinstance(sol, dict) and "time_tot" in sol else None
        self.iterations = sol["iter"] if isinstance(sol, dict) and "iter" in sol else None

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

    def graphs(
        self, automatically_organize: bool = True, adapt_graph_size_to_bounds: bool = False, show_now: bool = True
    ):
        """
        Prepare the graphs of the simulation

        Parameters
        ----------
        automatically_organize: bool
            If the figures should be spread on the screen automatically
        adapt_graph_size_to_bounds: bool
            If the plot should adapt to bounds (True) or to data (False)
        show_now: bool
            If the show method should be called. This is blocking

        Returns
        -------

        """
        plot_ocp = self.ocp.prepare_plots(automatically_organize, adapt_graph_size_to_bounds)
        plot_ocp.update_data(self.vector)
        if show_now:
            plt.show()

    def animate(self, n_frames: int = 80, show_now: bool = True, **kwargs: Any) -> list:
        """
        An interface to animate solution with bioviz

        Parameters
        ----------
        n_frames: int
            The number of frames to interpolate to
        show_now: bool
            If the bioviz exec() function should be called. This is blocking
        kwargs: dict
            Any parameters to pass to bioviz

        Returns
        -------
            A list of bioviz structures (one for each phase)
        """

        try:
            import bioviz
        except ModuleNotFoundError:
            raise RuntimeError("bioviz must be install to animate the model")
        check_version(bioviz, "2.0.1", "2.1.0")
        data_interpolate = self.interpolate(n_frames) if n_frames > 0 else self.states
        if not isinstance(data_interpolate["q"], (list, tuple)):
            data_interpolate["q"] = [data_interpolate["q"]]

        all_bioviz = []
        for idx_phase, data in enumerate(data_interpolate["q"]):
            all_bioviz.append(bioviz.Viz(loaded_model=self.ocp.nlp[idx_phase].model, **kwargs))
            all_bioviz[-1].load_movement(self.ocp.nlp[idx_phase].mapping["q"].to_second.map(data))

        if show_now:
            b_is_visible = [True] * len(all_bioviz)
            while sum(b_is_visible):
                for i, b in enumerate(all_bioviz):
                    if b.vtk_window.is_active:
                        b.update()
                    else:
                        b_is_visible[i] = False
        else:
            return all_bioviz

    def print(self, data_type: DataType = DataType.ALL):
        def print_objective_functions(ocp, sol):
            """
            Print the values of each objective function to the console
            """

            def __extract_objective(pen: dict):
                """
                Extract objective function from a penalty

                Parameters
                ----------
                pen: dict
                    The penalty to extract the value from

                Returns
                -------
                The value extract
                """

                # TODO: This should be done in bounds and objective functions, so it is available for all the code
                val_tp = Function("val_tp", [ocp.v.vector], [pen["val"]]).expand()(sol.vector)
                if pen["target"] is not None:
                    # TODO Target should be available to constraint?
                    nan_idx = np.isnan(pen["target"])
                    pen["target"][nan_idx] = 0
                    val_tp -= pen["target"]
                    if np.any(nan_idx):
                        val_tp[np.where(nan_idx)] = 0

                if pen["objective"].quadratic:
                    val_tp *= val_tp

                val = np.sum(val_tp)

                dt = Function("dt", [ocp.v.vector], [pen["dt"]]).expand()(sol.vector)
                val_weighted = pen["objective"].weight * val * dt
                return val, val_weighted

            print(f"\n---- COST FUNCTION VALUES ----")
            has_global = False
            running_total = 0
            for J in ocp.J:
                has_global = True
                val = []
                val_weighted = []
                for j in J:
                    out = __extract_objective(j)
                    val.append(out[0])
                    val_weighted.append(out[1])
                sum_val_weighted = sum(val_weighted)
                print(f"{J[0]['objective'].name}: {sum(val)} (weighted {sum_val_weighted})")
                running_total += sum_val_weighted
            if has_global:
                print("")

            for idx_phase, nlp in enumerate(ocp.nlp):
                print(f"PHASE {idx_phase}")
                for J in nlp.J:
                    val = []
                    val_weighted = []
                    for j in J:
                        out = __extract_objective(j)
                        val.append(out[0])
                        val_weighted.append(out[1])
                    sum_val_weighted = sum(val_weighted)
                    print(f"{J[0]['objective'].name}: {sum(val)} (weighted {sum_val_weighted})")
                    running_total += sum_val_weighted
                print("")
            print(f"Sum cost functions: {running_total}")
            print(f"------------------------------")

        def print_constraints(ocp, sol):
            """
            Print the values of each constraints with its lagrange multiplier to the console
            """

            if sol.constraints is None:
                return

            # Todo, min/mean/max
            print(f"\n--------- CONSTRAINTS ---------")
            idx = 0
            has_global = False
            for G in ocp.g:
                has_global = True
                for g in G:
                    next_idx = idx + g["val"].shape[0]
                print(
                    f"{g['constraint'].name}: {np.sum(sol.constraints[idx:next_idx])}"
                )
                idx = next_idx
            if has_global:
                print("")

            for idx_phase, nlp in enumerate(ocp.nlp):
                print(f"PHASE {idx_phase}")
                for G in nlp.g:
                    next_idx = idx
                    for g in G:
                        next_idx += g["val"].shape[0]
                    print(
                        f"{g['constraint'].name}: {np.sum(sol.constraints[idx:next_idx])}"
                    )
                    idx = next_idx
                print("")
            print(f"------------------------------")

        if data_type == DataType.OBJECTIVES:
            print_objective_functions(self.ocp, self)
        elif data_type == DataType.CONSTRAINTS:
            print_constraints(self.ocp, self)
        elif data_type == DataType.ALL:
            self.print(DataType.OBJECTIVES)
            self.print(DataType.CONSTRAINTS)
        else:
            raise ValueError("print can only be called with DataType.OBJECTIVES or DataType.CONSTRAINTS")
