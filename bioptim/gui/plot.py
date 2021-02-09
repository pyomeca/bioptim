from typing import Callable, Union, Any
import multiprocessing as mp
from copy import copy
import tkinter
import pickle
import os
from itertools import accumulate

import numpy as np
from matplotlib import pyplot as plt, lines
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity, DM, Function

from ..limits.path_conditions import Bounds
from ..misc.data import Data
from ..misc.enums import PlotType, ControlType, InterpolationType
from ..misc.mapping import Mapping
from ..misc.utils import check_version


class CustomPlot:
    """
    Interface to create/add plots of the simulation

    Attributes
    ----------
    function: Callable[states, controls, parameters]
        The function to call to update the graph
    type: PlotType
        Type of plot to use
    phase_mappings: Mapping
        The index of the plot across the phases
    legend: Union[tuple[str], list[str]]
        The titles of the graphs
    combine_to: str
        The name of the variable to combine this one with
    color: str
        The color of the line as specified in matplotlib
    linestyle: str
        The style of the line as specified in matplotlib
    ylim: Union[tuple[float, float], list[float, float]]
        The ylim of the axes as specified in matplotlib
    bounds:
        The bounds to show on the graph
    """

    def __init__(
        self,
        update_function: Callable,
        plot_type: PlotType = PlotType.PLOT,
        axes_idx: Union[Mapping, tuple, list] = None,
        legend: Union[tuple, list] = (),
        combine_to: str = None,
        color: str = None,
        linestyle: str = None,
        ylim: Union[tuple, list] = None,
        bounds: Bounds = None,
    ):
        """
        Parameters
        ----------
        update_function: Callable[states, controls, parameters]
            The function to call to update the graph
        plot_type: PlotType
            Type of plot to use
        axes_idx: Union[Mapping, tuple, list]
            The index of the plot across the phases
        legend: Union[tuple[str], list[str]]
            The titles of the graphs
        combine_to: str
            The name of the variable to combine this one with
        color: str
            The color of the line as specified in matplotlib
        linestyle: str
            The style of the line as specified in matplotlib
        ylim: Union[tuple[float, float], list[float, float]]
            The ylim of the axes as specified in matplotlib
        bounds:
            The bounds to show on the graph
        """

        self.function = update_function
        self.type = plot_type
        if axes_idx is None:
            self.phase_mappings = None  # Will be set later
        elif isinstance(axes_idx, (tuple, list)):
            self.phase_mappings = Mapping(axes_idx)
        elif isinstance(axes_idx, Mapping):
            self.phase_mappings = axes_idx
        else:
            raise RuntimeError("phase_mapping must be a list or a Mapping")
        self.legend = legend
        self.combine_to = combine_to
        self.color = color
        self.linestyle = linestyle
        self.ylim = ylim
        self.bounds = bounds


class PlotOcp:
    """
    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the full ocp
    plot_options: dict
        matplotlib options template for specified PlotType
    ydata: list
        The actual current data to be plotted. It is update by update_data
    ns: int
        The total number of shooting points
    t: list[float]
        The time vector
    t_integrated: list[float]
        The time vector integrated
    tf: list[float]
        The times at the end of each phase
    t_idx_to_optimize: list[int]
        The index of the phases where time is a variable to optimize (non constant)
    axes: dict
        The dictionary of handlers to the matplotlib axes
    plots: list
        The list of handlers to the matplotlib plots for the ydata
    plots_vertical_lines: list
        The list of handlers to the matplotlib plots for the phase separators
    plots_bounds: list
        The list of handlers to the matplotlib plots for the bounds
    all_figures: list
        The list of handlers to the matplotlib figures
    automatically_organize: bool
        If the figure should be automatically organized on screen
    self.plot_func: dict
        The dictionary of all the CustomPlot
    self.variable_sizes: list[int]
        The size of all variables. This helps declaring all the plots in advance
    self.adapt_graph_size_to_bounds: bool
        If the plot should adapt to bounds or to ydata
    n_vertical_windows: int
        The number of figure rows
    n_horizontal_windows: int
        The number of figure columns
    top_margin: float
        The space between the top of the screen and the figure when automatically rearrange
    height_step: float
        The height of the figure
    width_step: float
        The width of the figure

    Methods
    -------
    __update_time_vector(self)
        Setup the time and time integrated vector, which is the x-axes of the graphs
    __create_plots(self)
        Setup the plots
    __add_new_axis(self, variable: str, nb: int, n_rows: int, n_cols: int)
        Add a new axis to the axes pool
    _organize_windows(self, n_windows: int)
        Automatically organize the figure across the screen.
    find_phases_intersections(self)
        Finds the intersection between the phases
    show()
        Force the show of the graphs. This is a blocking function
    update_data(self, V: dict)
        Update ydata from the variable a solution structure
    __update_xdata(self)
        Update of the time axes in plots
    __append_to_ydata(self, data: list)
        Parse the data list to create a single list of all ydata that will fit the plots vector
    __update_axes(self)
        Update the plotted data from ydata
    __compute_ylim(min_val: Union[np.ndarray, DM], max_val: Union[np.ndarray, DM], factor: float) -> tuple:
        Dynamically find the ylim
    _generate_windows_size(nb: int) -> tuple[int, int]
        Defines the number of column and rows of subplots from the number of variables to plot.
    """

    def __init__(
        self,
        ocp,
        automatically_organize: bool = True,
        adapt_graph_size_to_bounds: bool = False,
    ):
        """
        Prepares the figures during the simulation

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to show
        automatically_organize: bool
            If the figures should be spread on the screen automatically
        adapt_graph_size_to_bounds: bool
            If the axes should fit the bounds (True) or the data (False)
        """
        for i in range(1, ocp.n_phases):
            if ocp.nlp[0].shape["q"] != ocp.nlp[i].shape["q"]:
                raise RuntimeError("Graphs with nbQ different at each phase is not implemented yet")

        self.ocp = ocp
        self.plot_options = {
            "general_options": {"use_tight_layout": False},
            "non_integrated_plots": {"linestyle": "-.", "markersize": 3},
            "integrated_plots": {"linestyle": "-", "markersize": 3, "linewidth": 1.1},
            "bounds": {"color": "k", "linewidth": 0.4, "linestyle": "-"},
            "grid": {"color": "k", "linestyle": "-", "linewidth": 0.15},
            "vertical_lines": {"color": "k", "linestyle": "--", "linewidth": 1.2},
        }

        self.ydata = []
        self.ns = 0

        self.t = []
        self.t_integrated = []
        if isinstance(self.ocp.original_phase_time, (int, float)):
            self.tf = [self.ocp.original_phase_time]
        else:
            self.tf = list(self.ocp.original_phase_time)
        self.t_idx_to_optimize = []
        for i, nlp in enumerate(self.ocp.nlp):
            if isinstance(nlp.tf, self.ocp.CX):
                self.t_idx_to_optimize.append(i)
        self.__update_time_vector()

        self.axes = {}
        self.plots = []
        self.plots_vertical_lines = []
        self.plots_bounds = []
        self.all_figures = []

        self.automatically_organize = automatically_organize
        self.n_vertical_windows = None
        self.n_horizontal_windows = None
        self.top_margin = None
        self.height_step = None
        self.width_step = None
        self._organize_windows(len(self.ocp.nlp[0].var_states) + len(self.ocp.nlp[0].var_controls))

        self.plot_func = {}
        self.variable_sizes = []
        self.adapt_graph_size_to_bounds = adapt_graph_size_to_bounds
        self.__create_plots()

        horz = 0
        vert = 1 if len(self.all_figures) < self.n_vertical_windows * self.n_horizontal_windows else 0
        for i, fig in enumerate(self.all_figures):
            if self.automatically_organize:
                try:
                    fig.canvas.manager.window.move(
                        int(vert * self.width_step), int(self.top_margin + horz * self.height_step)
                    )
                    vert += 1
                    if vert >= self.n_vertical_windows:
                        horz += 1
                        vert = 0
                except AttributeError:
                    pass
            fig.canvas.draw()
            if self.plot_options["general_options"]["use_tight_layout"]:
                fig.tight_layout()

    def __update_time_vector(self):
        """
        Setup the time and time integrated vector, which is the x-axes of the graphs

        """

        self.t = []
        self.t_integrated = []
        last_t = 0
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            n_int_steps = nlp.n_integration_steps
            dt_ns = self.tf[phase_idx] / nlp.ns
            time_phase_integrated = []
            last_t_int = copy(last_t)
            for _ in range(nlp.ns):
                time_phase_integrated.append(np.linspace(last_t_int, last_t_int + dt_ns, n_int_steps + 1))
                last_t_int += dt_ns
            self.t_integrated.append(time_phase_integrated)

            self.ns += nlp.ns + 1
            time_phase = np.linspace(last_t, last_t + self.tf[phase_idx], nlp.ns + 1)
            last_t += self.tf[phase_idx]
            self.t.append(time_phase)

    def __create_plots(self):
        """
        Setup the plots
        """

        variable_sizes = []
        for i, nlp in enumerate(self.ocp.nlp):
            variable_sizes.append({})
            if nlp.plot:
                for key in nlp.plot:
                    if isinstance(nlp.plot[key], tuple):
                        nlp.plot[key] = nlp.plot[key][0]
                    if nlp.plot[key].phase_mappings is None:
                        size = (
                            nlp.plot[key]
                            .function(np.zeros((nlp.nx, 1)), np.zeros((nlp.nu, 1)), np.zeros((nlp.np, 1)))
                            .shape[0]
                        )
                        nlp.plot[key].phase_mappings = Mapping(range(size))
                    else:
                        size = len(nlp.plot[key].phase_mappings.map_idx)
                    if key not in variable_sizes[i]:
                        variable_sizes[i][key] = size
                    else:
                        variable_sizes[i][key] = max(variable_sizes[i][key], size)
        self.variable_sizes = variable_sizes
        if not variable_sizes:
            # No graph was setup in problem_type
            return

        self.plot_func = {}
        for i, nlp in enumerate(self.ocp.nlp):
            for variable in self.variable_sizes[i]:
                if nlp.plot[variable].combine_to:
                    self.axes[variable] = self.axes[nlp.plot[variable].combine_to]
                    axes = self.axes[variable][1]
                elif i > 0 and variable in self.axes:
                    axes = self.axes[variable][1]
                else:
                    nb = max(
                        [
                            max(nlp.plot[variable].phase_mappings.map_idx) + 1 if variable in nlp.plot else 0
                            for nlp in self.ocp.nlp
                        ]
                    )
                    n_cols, n_rows = PlotOcp._generate_windows_size(nb)
                    axes = self.__add_new_axis(variable, nb, n_rows, n_cols)
                    self.axes[variable] = [nlp.plot[variable], axes]

                t = self.t[i]
                if variable not in self.plot_func:
                    self.plot_func[variable] = [None] * self.ocp.n_phases
                self.plot_func[variable][i] = nlp.plot[variable]

                mapping = self.plot_func[variable][i].phase_mappings.map_idx
                for ctr, k in enumerate(mapping):
                    ax = axes[k]
                    if k < len(self.plot_func[variable][i].legend):
                        axes[k].set_title(self.plot_func[variable][i].legend[k])
                    ax.grid(**self.plot_options["grid"])
                    ax.set_xlim(0, self.t[-1][-1])
                    if nlp.plot[variable].ylim:
                        ax.set_ylim(nlp.plot[variable].ylim)
                    elif self.adapt_graph_size_to_bounds and nlp.plot[variable].bounds:
                        if nlp.plot[variable].bounds.type != InterpolationType.CUSTOM:
                            y_min = nlp.plot[variable].bounds.min[ctr].min()
                            y_max = nlp.plot[variable].bounds.max[ctr].max()
                        else:
                            nlp.plot[variable].bounds.check_and_adjust_dimensions(len(mapping), nlp.ns)
                            y_min = min([nlp.plot[variable].bounds.min.evaluate_at(j)[k] for j in range(nlp.ns)])
                            y_max = max([nlp.plot[variable].bounds.max.evaluate_at(j)[k] for j in range(nlp.ns)])
                        y_range, _ = self.__compute_ylim(y_min, y_max, 1.25)
                        ax.set_ylim(y_range)
                    zero = np.zeros((t.shape[0], 1))
                    plot_type = self.plot_func[variable][i].type
                    if plot_type == PlotType.PLOT:
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:green"
                        self.plots.append(
                            [
                                plot_type,
                                i,
                                ax.plot(t, zero, color=color, zorder=0, **self.plot_options["non_integrated_plots"])[0],
                            ]
                        )
                    elif plot_type == PlotType.INTEGRATED:
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:brown"
                        plots_integrated = []
                        n_int_steps = nlp.n_integration_steps
                        for cmp in range(nlp.ns):
                            plots_integrated.append(
                                ax.plot(
                                    self.t_integrated[i][cmp],
                                    np.zeros(n_int_steps + 1),
                                    color=color,
                                    **self.plot_options["integrated_plots"],
                                )[0]
                            )
                        self.plots.append([plot_type, i, plots_integrated])

                    elif plot_type == PlotType.STEP:
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:orange"
                        linestyle = (
                            self.plot_func[variable][i].linestyle if self.plot_func[variable][i].linestyle else "-"
                        )
                        self.plots.append(
                            [plot_type, i, ax.step(t, zero, linestyle, where="post", color=color, zorder=0)[0]]
                        )
                    else:
                        raise RuntimeError(f"{plot_type} is not implemented yet")

                for j, ax in enumerate(axes):
                    intersections_time = self.find_phases_intersections()
                    for time in intersections_time:
                        self.plots_vertical_lines.append(ax.axvline(time, **self.plot_options["vertical_lines"]))

                    if nlp.plot[variable].bounds and self.adapt_graph_size_to_bounds:
                        if nlp.plot[variable].bounds.type == InterpolationType.EACH_FRAME:
                            ns = nlp.plot[variable].bounds.min.shape[1] - 1
                        else:
                            ns = nlp.ns
                        nlp.plot[variable].bounds.check_and_adjust_dimensions(n_elements=len(mapping), n_shooting=ns)
                        bounds_min = np.array([nlp.plot[variable].bounds.min.evaluate_at(k)[j] for k in range(ns + 1)])
                        bounds_max = np.array([nlp.plot[variable].bounds.max.evaluate_at(k)[j] for k in range(ns + 1)])
                        if bounds_min.shape[0] == nlp.ns:
                            bounds_min = np.concatenate((bounds_min, [bounds_min[-1]]))
                            bounds_max = np.concatenate((bounds_max, [bounds_max[-1]]))

                        self.plots_bounds.append(
                            [ax.step(self.t[i], bounds_min, where="post", **self.plot_options["bounds"]), i]
                        )
                        self.plots_bounds.append(
                            [ax.step(self.t[i], bounds_max, where="post", **self.plot_options["bounds"]), i]
                        )

    def __add_new_axis(self, variable: str, nb: int, n_rows: int, n_cols: int):
        """
        Add a new axis to the axes pool

        Parameters
        ----------
        variable: str
            The name of the graph
        nb: int
            The total number of axes to create
        n_rows: int
            The number of rows for the subplots
        n_cols: int
            The number of columns for the subplots
        """

        if self.automatically_organize:
            self.all_figures.append(plt.figure(variable, figsize=(self.width_step / 100, self.height_step / 131)))
        else:
            self.all_figures.append(plt.figure(variable))
        axes = self.all_figures[-1].subplots(n_rows, n_cols)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for i in range(nb, len(axes)):
            axes[i].remove()
        axes = axes[:nb]

        idx_center = n_rows * n_cols - int(n_cols / 2) - 1
        if idx_center >= len(axes):
            idx_center = len(axes) - 1
        axes[idx_center].set_xlabel("time (s)")

        self.all_figures[-1].tight_layout()
        return axes

    def _organize_windows(self, n_windows: int):
        """
        Automatically organize the figure across the screen.

        Parameters
        ----------
        n_windows: int
            The number of figures to show
        """

        self.n_vertical_windows, self.n_horizontal_windows = PlotOcp._generate_windows_size(n_windows)
        if self.automatically_organize:
            height = tkinter.Tk().winfo_screenheight()
            width = tkinter.Tk().winfo_screenwidth()
            self.top_margin = height / 15
            self.height_step = (height - self.top_margin) / self.n_horizontal_windows
            self.width_step = width / self.n_vertical_windows

    def find_phases_intersections(self):
        """
        Finds the intersection between the phases
        """

        return list(accumulate(self.tf))[:-1]

    @staticmethod
    def show():
        """
        Force the show of the graphs. This is a blocking function
        """

        plt.show()

    def update_data(self, V: dict):
        """
        Update ydata from the variable a solution structure

        Parameters
        ----------
        V: dict
            The data to parse
        """

        self.ydata = []

        data_states, data_controls, data_param = Data.get_data(
            self.ocp, V, get_parameters=True, integrate=True, concatenate=False
        )
        data_param_in_dyn = np.array([data_param[key] for key in data_param if key != "time"]).squeeze()

        for _ in self.ocp.nlp:
            if self.t_idx_to_optimize:
                for i_in_time, i_in_tf in enumerate(self.t_idx_to_optimize):
                    self.tf[i_in_tf] = data_param["time"][i_in_time]
            self.__update_xdata()

        data_states_per_phase, data_controls_per_phase = Data.get_data(self.ocp, V, integrate=True, concatenate=False)
        for i, nlp in enumerate(self.ocp.nlp):
            step_size = nlp.n_integration_steps + 1
            n_elements = nlp.ns * step_size + 1

            state = np.ndarray((0, n_elements))
            for s in nlp.var_states:
                if isinstance(data_states_per_phase[s], (list, tuple)):
                    state = np.concatenate((state, data_states_per_phase[s][i]))
                else:
                    state = np.concatenate((state, data_states_per_phase[s]))
            control = np.ndarray((0, nlp.ns + 1))
            for s in nlp.var_controls:
                if isinstance(data_controls_per_phase[s], (list, tuple)):
                    control = np.concatenate((control, data_controls_per_phase[s][i]))
                else:
                    control = np.concatenate((control, data_controls_per_phase[s]))

            if nlp.control_type == ControlType.CONSTANT:
                u_mod = 1
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                u_mod = 2
            else:
                raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

            for key in self.variable_sizes[i]:
                if self.plot_func[key][i].type == PlotType.INTEGRATED:
                    all_y = []
                    for idx, t in enumerate(self.t_integrated[i]):
                        y_tp = np.empty((self.variable_sizes[i][key], len(t)))
                        y_tp.fill(np.nan)
                        y_tp[:, :] = self.plot_func[key][i].function(
                            state[:, step_size * idx : step_size * (idx + 1)],
                            control[:, idx : idx + u_mod],
                            data_param_in_dyn,
                        )
                        all_y.append(y_tp)

                    for idx in range(len(self.plot_func[key][i].phase_mappings.map_idx)):
                        y_tp = []
                        for y in all_y:
                            y_tp.append(y[idx, :])
                        self.__append_to_ydata([y_tp])
                else:
                    y = np.empty((self.variable_sizes[i][key], len(self.t[i])))
                    y.fill(np.nan)
                    try:
                        y[:, :] = self.plot_func[key][i].function(state[:, ::step_size], control, data_param_in_dyn)
                    except ValueError:
                        raise ValueError(
                            f"Wrong dimensions for plot {key}. Got "
                            f"{self.plot_func[key][i].function(state[:, ::step_size], control, data_param_in_dyn).shape}"
                            f", but expected {y.shape}"
                        )
                    self.__append_to_ydata(y)
        self.__update_axes()

    def __update_xdata(self):
        """
        Update of the time axes in plots
        """

        self.__update_time_vector()
        for plot in self.plots:
            phase_idx = plot[1]
            if plot[0] == PlotType.INTEGRATED:
                for cmp, p in enumerate(plot[2]):
                    p.set_xdata(self.t_integrated[phase_idx][cmp])
                ax = plot[2][-1].axes
            else:
                plot[2].set_xdata(self.t[phase_idx])
                ax = plot[2].axes
            ax.set_xlim(0, self.t[-1][-1])

        if self.plots_bounds:
            for plot_bounds in self.plots_bounds:
                plot_bounds[0][0].set_xdata(self.t[plot_bounds[1]])
                ax = plot_bounds[0][0].axes
                ax.set_xlim(0, self.t[-1][-1])

        intersections_time = self.find_phases_intersections()
        n = len(intersections_time)
        if n > 0:
            for p in range(int(len(self.plots_vertical_lines) / n)):
                for i, time in enumerate(intersections_time):
                    self.plots_vertical_lines[p * n + i].set_xdata([time, time])

    def __append_to_ydata(self, data: Union[list, np.ndarray]):
        """
        Parse the data list to create a single list of all ydata that will fit the plots vector

        Parameters
        ----------
        data: list
            The data list to copy
        """

        for y in data:
            self.ydata.append(y)

    def __update_axes(self):
        """
        Update the plotted data from ydata
        """

        assert len(self.plots) == len(self.ydata)
        for i, plot in enumerate(self.plots):
            y = self.ydata[i]

            if plot[0] == PlotType.INTEGRATED:
                for cmp, p in enumerate(plot[2]):
                    p.set_ydata(y[cmp])
            else:
                plot[2].set_ydata(y)

        for p in self.plots_vertical_lines:
            p.set_ydata((np.nan, np.nan))

        for key in self.axes:
            if (not self.adapt_graph_size_to_bounds) or (self.axes[key][0].bounds is None):
                for i, ax in enumerate(self.axes[key][1]):
                    if not self.axes[key][0].ylim:
                        y_max = -np.inf
                        y_min = np.inf
                        for p in ax.get_children():
                            if isinstance(p, lines.Line2D):
                                y_min = min(y_min, np.min(p.get_ydata()))
                                y_max = max(y_max, np.max(p.get_ydata()))
                        y_range, data_range = self.__compute_ylim(y_min, y_max, 1.25)
                        ax.set_ylim(y_range)
                        ax.set_yticks(
                            np.arange(
                                y_range[0],
                                y_range[1],
                                step=data_range / 4,
                            )
                        )

        for p in self.plots_vertical_lines:
            p.set_ydata((0, 1))

    @staticmethod
    def __compute_ylim(min_val: Union[np.ndarray, DM], max_val: Union[np.ndarray, DM], factor: float) -> tuple:
        """
        Dynamically find the ylim
        Parameters
        ----------
        min_val: Union[np.ndarray, DM]
            The minimal value of the y axis
        max_val: Union[np.ndarray, DM]
            The maximal value of the y axis
        factor: float
            The widening factor of the y range

        Returns
        -------
        The ylim and actual range of ylim (before applying the threshold)
        """

        if np.isnan(min_val) or np.isinf(min_val):
            min_val = 0
        if np.isnan(max_val) or np.isinf(max_val):
            max_val = 1
        data_mean = np.mean((min_val, max_val))
        data_range = max_val - min_val
        if np.abs(data_range) < 0.8:
            data_range = 0.8
        y_range = (factor * data_range) / 2
        y_range = data_mean - y_range, data_mean + y_range
        return y_range, data_range

    @staticmethod
    def _generate_windows_size(nb: int) -> tuple:
        """
        Defines the number of column and rows of subplots from the number of variables to plot.

        Parameters
        ----------
        nb: int
            Number of variables to plot

        Returns
        -------
        The optimized number of rows and columns
        """

        n_rows = int(round(np.sqrt(nb)))
        return n_rows + 1 if n_rows * n_rows < nb else n_rows, n_rows


class ShowResult:
    """
    The main interface to bioptim GUI

    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp to show
    sol: dict
        The solution dictionary

    Methods
    -------
    graphs(self, automatically_organize: bool=True, adapt_graph_size_to_bounds:bool=False, show_now:bool=True)
        Prepare the graphs of the simulation
    animate(self, n_frames:int=80, show_now:bool=True, **kwargs) -> list
        An interface to animate solution with bioviz
    objective_functions(self)
        Print the values of each objective function to the console
    constraints(self)
        Print the values of each constraints with its lagrange multiplier to the console
    keep_matplotlib()
        Allows for non-blocking show of the figure. This works only in Debug
    """

    def __init__(self, ocp, sol: dict):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
        A reference to the ocp to show
        sol: dict
            The solution dictionary
        """
        self.ocp = ocp
        self.sol = sol

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
        plot_ocp = PlotOcp(
            self.ocp,
            automatically_organize=automatically_organize,
            adapt_graph_size_to_bounds=adapt_graph_size_to_bounds,
        )
        plot_ocp.update_data(self.sol["x"])
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
        data_interpolate, data_control = Data.get_data(
            self.ocp, self.sol["x"], integrate=False, interpolate_n_frames=n_frames
        )
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

    def objective_functions(self):
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
            val_tp = Function("val_tp", [self.ocp.V], [pen["val"]]).expand()(self.sol["x"])
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

            dt = Function("dt", [self.ocp.V], [pen["dt"]]).expand()(self.sol["x"])
            val_weighted = pen["objective"].weight * val * dt
            return val, val_weighted

        print(f"\n---- COST FUNCTION VALUES ----")
        has_global = False
        running_total = 0
        for J in self.ocp.J:
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

        for idx_phase, nlp in enumerate(self.ocp.nlp):
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

    def constraints(self):
        """
        Print the values of each constraints with its lagrange multiplier to the console
        """

        # Todo, min/mean/max
        print(f"\n--------- CONSTRAINTS ---------")
        idx = 0
        has_global = False
        for G in self.ocp.g:
            has_global = True
            for g in G:
                next_idx = idx + g["val"].shape[0]
            print(
                f"{g['constraint'].name}: {np.sum(self.sol['g'][idx:next_idx])}"
            )
            idx = next_idx
        if has_global:
            print("")

        for idx_phase, nlp in enumerate(self.ocp.nlp):
            print(f"PHASE {idx_phase}")
            for G in nlp.g:
                next_idx = idx
                for g in G:
                    next_idx += g["val"].shape[0]
                print(
                    f"{g['constraint'].name}: {np.sum(self.sol['g'][idx:next_idx])}"
                )
                idx = next_idx
            print("")
        print(f"------------------------------")

    @staticmethod
    def keep_matplotlib():
        """
        Allows for non-blocking show of the figure. This works only in Debug
        """
        plt.figure(figsize=(0.01, 0.01)).canvas.manager.window.move(1000, 100)
        plt.show()


class OnlineCallback(Callback):
    """
    CasADi interface of Ipopt callbacks

    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp to show
    nx: int
        The number of optimization variables
    ng: int
        The number of constraints
    queue: mp.Queue
        The multiprocessing queue
    plotter: ProcessPlotter
        The callback for plotting for the multiprocessing
    plot_process: mp.Process
        The multiprocessing placeholder

    Methods
    -------
    get_n_in() -> int
        Get the number of variables in
    get_n_out() -> int
        Get the number of variables out
    get_name_in(i: int) -> int
        Get the name of a variable
    get_name_out(_) -> str
        Get the name of the output variable
    get_sparsity_in(self, i: int) -> tuple[int]
        Get the sparsity of a specific variable
    eval(self, arg: Union[list, tuple]) -> list[int]
        Send the current data to the plotter
    """

    def __init__(self, ocp, opts: dict = {}):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to show
        opts: dict
            Option to AnimateCallback method of CasADi
        """
        Callback.__init__(self)
        self.ocp = ocp
        self.nx = self.ocp.V.rows()
        self.ng = 0
        self.construct("AnimateCallback", opts)

        self.queue = mp.Queue()
        self.plotter = self.ProcessPlotter(self.ocp)
        self.plot_process = mp.Process(target=self.plotter, args=(self.queue,), daemon=True)
        self.plot_process.start()

    @staticmethod
    def get_n_in() -> int:
        """
        Get the number of variables in

        Returns
        -------
        The number of variables in
        """

        return nlpsol_n_out()

    @staticmethod
    def get_n_out() -> int:
        """
        Get the number of variables out

        Returns
        -------
        The number of variables out
        """

        return 1

    @staticmethod
    def get_name_in(i: int) -> int:
        """
        Get the name of a variable

        Parameters
        ----------
        i: int
            The index of the variable

        Returns
        -------
        The name of the variable
        """

        return nlpsol_out(i)

    @staticmethod
    def get_name_out(_) -> str:
        """
        Get the name of the output variable

        Returns
        -------
        The name of the output variable
        """

        return "ret"

    def get_sparsity_in(self, i: int) -> tuple:
        """
        Get the sparsity of a specific variable

        Parameters
        ----------
        i: int
            The index of the variable

        Returns
        -------
        The sparsity of the variable
        """

        n = nlpsol_out(i)
        if n == "f":
            return Sparsity.scalar()
        elif n in ("x", "lam_x"):
            return Sparsity.dense(self.nx)
        elif n in ("g", "lam_g"):
            return Sparsity.dense(self.ng)
        else:
            return Sparsity(0, 0)

    def eval(self, arg: Union[list, tuple]) -> list:
        """
        Send the current data to the plotter

        Parameters
        ----------
        arg: Union[list, tuple]
            The data to send

        Returns
        -------
        A list of error index
        """
        send = self.queue.put
        send(arg[0])
        return [0]

    class ProcessPlotter(object):
        """
        The plotter that interface PlotOcp and the multiprocessing

        Attributes
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to show
        pipe: mp.Queue
            The multiprocessing queue to evaluate
        plot: PlotOcp
            The handler on all the figures

        Methods
        -------
        callback(self) -> bool
            The callback to update the graphs
        """

        def __init__(self, ocp):
            """
            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp to show
            """

            self.ocp = ocp

        def __call__(self, pipe: mp.Queue):
            """
            Parameters
            ----------
            pipe: mp.Queue
                The multiprocessing queue to evaluate
            """

            self.pipe = pipe
            self.plot = PlotOcp(self.ocp)
            timer = self.plot.all_figures[0].canvas.new_timer(interval=10)
            timer.add_callback(self.callback)
            timer.start()
            plt.show()

        def callback(self) -> bool:
            """
            The callback to update the graphs

            Returns
            -------
            True if everything went well
            """
            while not self.pipe.empty():
                V = self.pipe.get()
                self.plot.update_data(V)
                Iterations.save(V)
            for i, fig in enumerate(self.plot.all_figures):
                fig.canvas.draw()
            return True


class Iterations:
    """
    Act of iterations

    Methods
    -------
    save(V: np.ndarray)
        Save the current iteration on the hard drive
    """

    @staticmethod
    def save(V: np.ndarray):
        """
        Save the current iteration on the hard drive

        Parameters
        ----------
        V: np.ndarray
            The vector of data to save
        """

        file_path = ".__tmp_bioptim/temp_save_iter.bobo"
        if os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                previews_iterations = pickle.load(file)
            previews_iterations.append(np.array(V))
            with open(file_path, "wb") as file:
                pickle.dump(previews_iterations, file)
