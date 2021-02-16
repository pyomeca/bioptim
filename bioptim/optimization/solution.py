from typing import Any, Union

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
        self._states, self._controls, self._parameters = self.ocp.v.to_dictionaries(self.vector)
        self._complete_control()

        self.phase_time = self.ocp.v.extract_phase_time(self.vector)
        self.ns = [nlp.ns for nlp in self.ocp.nlp]

        self.is_interpolated = False
        self.is_integrated = False
        self.is_concatenated = False
        self.phase_time_original = self.phase_time
        self.ns_original = self.ns
        self._states_original = self._states
        self._controls_original = self._controls

    @property
    def states(self):
        return self._states[0] if len(self._states) == 1 else self._states

    @property
    def controls(self):
        return self._controls[0] if len(self._controls) == 1 else self._controls

    @property
    def parameters(self):
        return self._parameters

    def reset_data(self):
        self.is_interpolated = False
        self.is_integrated = False
        self.is_concatenated = False
        self.phase_time = self.phase_time_original
        self.ns = self.ns_original
        self._states = self._states_original
        self._controls = self._controls_original

    def integrate(self, concatenate: bool = False, apply_to_self: bool = False, continuous: bool = True):
        """
        Integrates the states

        Returns
        -------
        The dictionary of states integrated
        """

        if self.is_interpolated:
            raise RuntimeError("Cannot integrate after interpolating, please use reset_data before integrating")
        if self.is_concatenated:
            raise RuntimeError("Cannot integrate after concatenating, please use reset_data before integrating")

        ns = self.ns
        ocp = self.ocp
        out = []
        for _ in range(len(self._states)):
            out.append({})

        params = self._parameters["all"]
        for p in range(len(self._states)):
            if continuous:
                n_steps = ocp.nlp[p].n_integration_steps if continuous else ocp.nlp[p].n_integration_steps + 1
                ns[p] *= ocp.nlp[p].n_integration_steps
            else:
                n_steps = ocp.nlp[p].n_integration_steps + 1
                ns[p] *= ocp.nlp[p].n_integration_steps + 1

            for key in self._states[p]:
                shape = self._states[p][key].shape
                out[p][key] = np.ndarray((shape[0], (shape[1] - 1) * n_steps + 1))

            # Integrate
            for n in range(ocp.nlp[p].ns):
                x0 = self._states[p]["all"][:, n]
                u = self._controls[p]["all"][:, n]
                cols = range(n*n_steps, (n+1)*n_steps+1) if continuous else range(n*n_steps, (n+1)*n_steps)
                out[p]["all"][:, cols] = np.array(ocp.nlp[p].dynamics[n](x0=x0, p=u, params=params)["xall"])
                off = 0
                for key in ocp.nlp[p].var_states:
                    out[p][key][:, cols] = out[p]["all"][off:off+ocp.nlp[p].var_states[key], cols]
                    off += ocp.nlp[p].var_states[key]

        phase_time = self.phase_time
        if concatenate:
            out, _, phase_time, ns = self._concatenate_phases(out, None, self.phase_time, ns)

        if apply_to_self:
            self.is_integrated = True
            self._states = out
            self.phase_time = phase_time
            self.ns = ns
        return out[0] if len(out) == 1 else out

    def interpolate(self, n_frames: Union[int, list, tuple], apply_to_self: bool = False) -> list:
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

        if isinstance(n_frames, int):
            # Todo interpolate relative to time of the phase and not relative to number of frames in the phase
            is_concatenated = True
            data_states, _, phase_time, ns = self._concatenate_phases(self._states, self._controls, self.phase_time, self.ns)
            n_frames = [n_frames]
        elif isinstance(n_frames, (list, tuple)) and len(n_frames) == len(self._states):
            is_concatenated = False
            data_states = self._states
            phase_time = self.phase_time
            ns = n_frames
        else:
            raise ValueError("n_frames should either be a int to concatenate phases "
                             "or a list of int of the number of phases dimension")

        out = []
        for _ in range(len(data_states)):
            out.append({})
        for p in range(len(data_states)):
            x_phase = data_states[p]["all"]
            n_elements = x_phase.shape[0]

            t_phase = np.linspace(phase_time[p], phase_time[p] + phase_time[p + 1], x_phase.shape[1])
            t_int = np.linspace(t_phase[0], t_phase[-1], n_frames[p])

            x_interpolate = np.ndarray((n_elements, n_frames[p]))
            for j in range(n_elements):
                s = sci_interp.splrep(t_phase, x_phase[j, :])
                x_interpolate[j, :] = sci_interp.splev(t_int, s)
            out[p]["all"] = x_interpolate

            offset = 0
            for key in data_states[p]:
                if key == "all":
                    continue
                n_elements = data_states[p][key].shape[0]
                out[p][key] = out[p]["all"][offset: offset + n_elements]
                offset += n_elements

        if apply_to_self:
            self.is_interpolated = True
            self.is_concatenated = self.is_interpolated or is_concatenated
            self.phase_time = phase_time
            self.ns = ns
            self._states = out
        return out[0] if len(out) == 1 else out

    def concatenate_phases(self, apply_to_self: bool = False):
        out_states, out_controls, phase_time, ns = self._concatenate_phases(self._states, self._controls, self.phase_time, self.ns)
        if apply_to_self:
            self.is_concatenated = True
            self._states = out_states
            self._controls = out_controls
            self.phase_time = phase_time
            self.ns = ns
        return out_states[0], out_controls[0]

    def _concatenate_phases(self, states, controls, phase_time, ns) -> tuple:
        """
        Concatenate all the phases

        Parameters
        ----------

        Returns
        -------
        The new dictionary of data concatenated
        """

        if self.is_concatenated or len(self._states) == 1:
            return self._states, self._controls, phase_time, ns

        def _concat(data, ns):
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
                    data_out[0][key] = np.concatenate((data_out[0][key], d[key][:, :ns[p]]), axis=1)
            for key in data[-1]:
                data_out[0][key] = np.concatenate((data_out[0][key], data[-1][key][:, -1][:, np.newaxis]), axis=1)

            return data_out

        out_states = _concat(states, ns) if states else None
        out_controls = _concat(controls, ns) if controls else None
        phase_time = [0] + [sum([phase_time[i+1] for i in range(self.ocp.n_phases)])]
        ns = [sum(self.ns)]

        return out_states, out_controls, phase_time, ns

    def _complete_control(self):
        for p, nlp in enumerate(self.ocp.nlp):
            if nlp.control_type == ControlType.CONSTANT:
                for key in self._controls[p]:
                    self._controls[p][key] = np.concatenate((self._controls[p][key], self._controls[p][key][:, -1][:, np.newaxis]), axis=1)
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
        data_interpolate = self.interpolate(n_frames) if not isinstance(n_frames, int) or n_frames > 0 else self.states

        if not isinstance(data_interpolate, (list, tuple)):
            data_interpolate = [data_interpolate]

        all_bioviz = []
        for idx_phase, data in enumerate(data_interpolate):
            all_bioviz.append(bioviz.Viz(loaded_model=self.ocp.nlp[idx_phase].model, **kwargs))
            all_bioviz[-1].load_movement(self.ocp.nlp[idx_phase].mapping["q"].to_second.map(data["q"]))

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
