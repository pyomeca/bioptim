from typing import Callable, Any
import multiprocessing as mp
from copy import copy
import tkinter
from itertools import accumulate

import numpy as np
from matplotlib import pyplot as plt, lines
from matplotlib.ticker import StrMethodFormatter
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity, DM

from ..limits.path_conditions import Bounds
from ..limits.multinode_constraint import MultinodeConstraint
from ..misc.enums import PlotType, ControlType, InterpolationType, Shooting, SolutionIntegrator, IntegralApproximation
from ..misc.mapping import Mapping
from ..optimization.solution import Solution


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
    legend: tuple[str] | list[str]
        The titles of the graphs
    combine_to: str
        The name of the variable to combine this one with
    color: str
        The color of the line as specified in matplotlib
    linestyle: str
        The style of the line as specified in matplotlib
    ylim: tuple[float, float] | list[float, float]
        The ylim of the axes as specified in matplotlib
    bounds: Bounds
        The bounds to show on the graph
    node_idx : list
        The node time to be plotted on the graphs
    parameters: Any
        The parameters of the function
    """

    def __init__(
        self,
        update_function: Callable,
        plot_type: PlotType = PlotType.PLOT,
        axes_idx: Mapping | tuple | list = None,
        legend: tuple | list = None,
        combine_to: str = None,
        color: str = None,
        linestyle: str = None,
        ylim: tuple | list = None,
        bounds: Bounds = None,
        node_idx: list = None,
        label: list = None,
        compute_derivative: bool = False,
        integration_rule: IntegralApproximation = IntegralApproximation.RECTANGLE,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        update_function: Callable[states, controls, parameters]
            The function to call to update the graph
        plot_type: PlotType
            Type of plot to use
        axes_idx: Mapping | tuple | list
            The index of the plot across the phases
        legend: tuple[str] | list[str]
            The titles of the graphs
        combine_to: str
            The name of the variable to combine this one with
        color: str
            The color of the line as specified in matplotlib
        linestyle: str
            The style of the line as specified in matplotlib
        ylim: tuple[float, float] | list[float, float]
            The ylim of the axes as specified in matplotlib
        bounds: Bounds
            The bounds to show on the graph
        node_idx: list
            The node time to be plotted on the graphs
        label: list
            Label of the curve to plot (to be added to the legend)
        compute_derivative: bool
            If the function should send the next node with x and u. Prevents from computing all at once (therefore a bit slower)
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
        self.legend = legend if legend is not None else ()
        self.combine_to = combine_to
        self.color = color
        self.linestyle = linestyle
        self.ylim = ylim
        self.bounds = bounds
        self.node_idx = node_idx
        self.label = label
        self.compute_derivative = compute_derivative
        self.integration_rule = integration_rule
        self.parameters = parameters


class PlotOcp:
    """
    Attributes
    ----------
    show_bounds: bool
        If the plot should adapt to bounds or to ydata
    all_figures: list
        The list of handlers to the matplotlib figures
    automatically_organize: bool
        If the figure should be automatically organized on screen
    axes: dict
        The dictionary of handlers to the matplotlib axes
    height_step: float
        The height of the figure
    n_horizontal_windows: int
        The number of figure columns
    ns: int
        The total number of shooting points
    n_vertical_windows: int
        The number of figure rows
    ocp: OptimalControlProgram
        A reference to the full ocp
    plots: list
        The list of handlers to the matplotlib plots for the ydata
    plots_bounds: list
        The list of handlers to the matplotlib plots for the bounds
    plot_func: dict
        The dictionary of all the CustomPlot
    plot_options: dict
        matplotlib options template for specified PlotType
    plots_vertical_lines: list
        The list of handlers to the matplotlib plots for the phase separators
    shooting_type: Shooting
        The type of integration method
    t: list[float]
        The time vector
    tf: list[float]
        The times at the end of each phase
    t_integrated: list[float]
        The time vector integrated
    t_idx_to_optimize: list[int]
        The index of the phases where time is a variable to optimize (non constant)
    top_margin: float
        The space between the top of the screen and the figure when automatically rearrange
    variable_sizes: list[int]
        The size of all variables. This helps declaring all the plots in advance
    width_step: float
        The width of the figure
    ydata: list
        The actual current data to be plotted. It is update by update_data

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
    update_data(self, v: dict)
        Update ydata from the variable a solution structure
    __update_xdata(self)
        Update of the time axes in plots
    __append_to_ydata(self, data: list)
        Parse the data list to create a single list of all ydata that will fit the plots vector
    __update_axes(self)
        Update the plotted data from ydata
    __compute_ylim(min_val: np.ndarray | DM, max_val: np.ndarray | DM, factor: float) -> tuple:
        Dynamically find the ylim
    _generate_windows_size(nb: int) -> tuple[int, int]
        Defines the number of column and rows of subplots from the number of variables to plot.
    """

    def __init__(
        self,
        ocp,
        automatically_organize: bool = True,
        show_bounds: bool = False,
        shooting_type: Shooting = Shooting.MULTIPLE,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
    ):
        """
        Prepares the figures during the simulation

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to show
        automatically_organize: bool
            If the figures should be spread on the screen automatically
        show_bounds: bool
            If the axes should fit the bounds (True) or the data (False)
        shooting_type: Shooting
            The type of integration method
        integrator: SolutionIntegrator
             Use the ode defined by OCP or use a separate integrator provided by scipy

        """
        self.ocp = ocp
        self.plot_options = {
            "general_options": {"use_tight_layout": False},
            "non_integrated_plots": {"linestyle": "-", "markersize": 3, "linewidth": 1.1},
            "integrated_plots": {"linestyle": "-", "markersize": 3, "linewidth": 1.1},
            "point_plots": {"linestyle": None, "marker": ".", "markersize": 15},
            "bounds": {"color": "k", "linewidth": 0.4, "linestyle": "-"},
            "grid": {"color": "k", "linestyle": "-", "linewidth": 0.15},
            "vertical_lines": {"color": "k", "linestyle": "--", "linewidth": 1.2},
        }

        self.ydata = []
        self.ns = 0

        self.t = []
        self.t_integrated = []
        self.integrator = integrator

        if isinstance(self.ocp.original_phase_time, (int, float)):
            self.tf = [self.ocp.original_phase_time]
        else:
            self.tf = list(self.ocp.original_phase_time)
        self.t_idx_to_optimize = []
        for i, nlp in enumerate(self.ocp.nlp):
            if isinstance(nlp.tf, self.ocp.cx):
                self.t_idx_to_optimize.append(i)
        self.__update_time_vector()

        self.axes = {}
        self.plots = []
        self.plots_vertical_lines = []
        self.plots_bounds = []
        self.all_figures = []

        self.automatically_organize = automatically_organize
        self.n_vertical_windows: int | None = None
        self.n_horizontal_windows: int | None = None
        self.top_margin: int | None = None
        self.height_step: int | None = None
        self.width_step: int | None = None
        self._organize_windows(len(self.ocp.nlp[0].states) + len(self.ocp.nlp[0].controls))

        self.plot_func = {}
        self.variable_sizes = []
        self.show_bounds = show_bounds
        self.__create_plots()
        self.shooting_type = shooting_type

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
            n_int_steps = (
                nlp.ode_solver.steps_scipy if self.integrator != SolutionIntegrator.OCP else nlp.ode_solver.steps
            )
            dt_ns = self.tf[phase_idx] / nlp.ns
            time_phase_integrated = []
            last_t_int = copy(last_t)
            for _ in range(nlp.ns):
                if nlp.ode_solver.is_direct_collocation and self.integrator == SolutionIntegrator.OCP:
                    time_phase_integrated.append(np.array(nlp.dynamics[0].step_time) * dt_ns + last_t_int)
                else:
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

        def legend_without_duplicate_labels(ax):
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            if unique:
                ax.legend(*zip(*unique))

        variable_sizes = []
        for i, nlp in enumerate(self.ocp.nlp):
            variable_sizes.append({})
            if nlp.plot:
                for key in nlp.plot:
                    if isinstance(nlp.plot[key], tuple):
                        nlp.plot[key] = nlp.plot[key][0]

                    if nlp.plot[key].phase_mappings is None:
                        node_index = 0  # TODO deal with assume_phase_dynamics=False
                        if nlp.plot[key].node_idx is not None:
                            node_index = nlp.plot[key].node_idx[0]
                        nlp.states.node_index = node_index
                        nlp.states_dot.node_index = node_index
                        nlp.controls.node_index = node_index

                        size = (
                            nlp.plot[key]
                            .function(
                                node_index,
                                np.zeros((nlp.states.shape, 2)),
                                np.zeros((nlp.controls.shape, 2)),
                                np.zeros((nlp.parameters.shape, 2)),
                                **nlp.plot[key].parameters,
                            )
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

        y_min_all = [None for _ in self.variable_sizes[0]]
        y_max_all = [None for _ in self.variable_sizes[0]]
        self.plot_func = {}
        for i, nlp in enumerate(self.ocp.nlp):
            for var_idx, variable in enumerate(self.variable_sizes[i]):
                if nlp.plot[variable].combine_to:
                    self.axes[variable] = self.axes[nlp.plot[variable].combine_to]
                    axes = self.axes[variable][1]
                elif i > 0 and variable in self.axes:
                    axes = self.axes[variable][1]
                else:
                    nb = max(
                        [
                            len(nlp.plot[variable].phase_mappings.map_idx) if variable in nlp.plot else 0
                            for nlp in self.ocp.nlp
                        ]
                    )
                    n_cols, n_rows = PlotOcp._generate_windows_size(nb)
                    axes = self.__add_new_axis(variable, nb, n_rows, n_cols)
                    self.axes[variable] = [nlp.plot[variable], axes]
                    if not y_min_all[var_idx]:
                        y_min_all[var_idx] = [np.inf] * nb
                        y_max_all[var_idx] = [-np.inf] * nb

                if variable not in self.plot_func:
                    self.plot_func[variable] = [
                        nlp_tp.plot[variable] if variable in nlp_tp.plot else None for nlp_tp in self.ocp.nlp
                    ]
                if not self.plot_func[variable][i]:
                    continue

                mapping = nlp.plot[variable].phase_mappings.map_idx
                for ctr, axe_index in enumerate(mapping):
                    ax = axes[axe_index]
                    if ctr < len(nlp.plot[variable].legend):
                        ax.set_title(nlp.plot[variable].legend[ctr])
                    ax.grid(**self.plot_options["grid"])
                    ax.set_xlim(0, self.t[-1][-1])
                    if nlp.plot[variable].ylim:
                        ax.set_ylim(nlp.plot[variable].ylim)
                    elif self.show_bounds and nlp.plot[variable].bounds:
                        if nlp.plot[variable].bounds.type != InterpolationType.CUSTOM:
                            y_min = nlp.plot[variable].bounds.min[ctr, :].min()
                            y_max = nlp.plot[variable].bounds.max[ctr, :].max()
                        else:
                            nlp.plot[variable].bounds.check_and_adjust_dimensions(len(mapping), nlp.ns)
                            y_min = min([nlp.plot[variable].bounds.min.evaluate_at(j)[ctr] for j in range(nlp.ns)])
                            y_max = max([nlp.plot[variable].bounds.max.evaluate_at(j)[ctr] for j in range(nlp.ns)])
                        if y_min.__array__()[0] < y_min_all[var_idx][ctr]:
                            y_min_all[var_idx][ctr] = y_min
                        if y_max.__array__()[0] > y_max_all[var_idx][ctr]:
                            y_max_all[var_idx][ctr] = y_max

                        y_range, _ = self.__compute_ylim(y_min_all[var_idx][ctr], y_max_all[var_idx][ctr], 1.25)
                        ax.set_ylim(y_range)

                    plot_type = self.plot_func[variable][i].type
                    t = self.t[i][nlp.plot[variable].node_idx] if plot_type == PlotType.POINT else self.t[i]
                    if self.plot_func[variable][i].label:
                        label = self.plot_func[variable][i].label
                    else:
                        label = None

                    if plot_type == PlotType.PLOT:
                        zero = np.zeros((t.shape[0], 1))
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:green"
                        self.plots.append(
                            [
                                plot_type,
                                i,
                                ax.plot(
                                    t,
                                    zero,
                                    color=color,
                                    zorder=0,
                                    label=label,
                                    **self.plot_options["non_integrated_plots"],
                                )[0],
                            ]
                        )
                    elif plot_type == PlotType.INTEGRATED:
                        plots_integrated = []
                        n_int_steps = (
                            nlp.ode_solver.steps_scipy
                            if self.integrator != SolutionIntegrator.OCP
                            else nlp.ode_solver.steps
                        )
                        zero = np.zeros(n_int_steps + 1)
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:brown"
                        for cmp in range(nlp.ns):
                            plots_integrated.append(
                                ax.plot(
                                    self.t_integrated[i][cmp],
                                    zero,
                                    color=color,
                                    label=label,
                                    **self.plot_options["integrated_plots"],
                                )[0]
                            )
                        self.plots.append([plot_type, i, plots_integrated])

                    elif plot_type == PlotType.STEP:
                        zero = np.zeros((t.shape[0], 1))
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:orange"
                        linestyle = (
                            self.plot_func[variable][i].linestyle if self.plot_func[variable][i].linestyle else "-"
                        )
                        self.plots.append(
                            [
                                plot_type,
                                i,
                                ax.step(t, zero, linestyle, where="post", color=color, zorder=0, label=label)[0],
                            ]
                        )
                    elif plot_type == PlotType.POINT:
                        zero = np.zeros((t.shape[0], 1))
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:purple"
                        self.plots.append(
                            [
                                plot_type,
                                i,
                                ax.plot(
                                    t, zero, color=color, zorder=0, label=label, **self.plot_options["point_plots"]
                                )[0],
                                variable,
                            ]
                        )
                    else:
                        raise RuntimeError(f"{plot_type} is not implemented yet")

                    legend_without_duplicate_labels(ax)

                for j, ax in enumerate(axes):
                    intersections_time = self.find_phases_intersections()
                    for time in intersections_time:
                        self.plots_vertical_lines.append(ax.axvline(time, **self.plot_options["vertical_lines"]))

                    if nlp.plot[variable].bounds and self.show_bounds:
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
        for ax in axes:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1f}"))  # 1 decimal places
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

    def update_data(self, v: dict):
        """
        Update ydata from the variable a solution structure

        Parameters
        ----------
        v: dict
            The data to parse
        """

        self.ydata = []

        sol = Solution(self.ocp, v)

        if all([nlp.ode_solver.is_direct_collocation for nlp in self.ocp.nlp]):
            # no need to integrate when using direct collocation
            data_states = sol.states
            data_time = sol._generate_time()
        elif all([nlp.ode_solver.is_direct_shooting for nlp in self.ocp.nlp]):
            integrated = sol.integrate(
                shooting_type=self.shooting_type,
                keep_intermediate_points=True,
                integrator=self.integrator,
            )
            data_states = integrated.states
            data_time = integrated._time_vector
        else:
            raise NotImplementedError("Graphs are not implemented when mixing direct collocation and direct shooting")

        data_controls = sol.controls
        data_params = sol.parameters
        data_params_in_dyn = np.array([data_params[key] for key in data_params if key != "all"]).reshape(-1, 1)

        for _ in self.ocp.nlp:
            if self.t_idx_to_optimize:
                for i_in_time, i_in_tf in enumerate(self.t_idx_to_optimize):
                    self.tf[i_in_tf] = float(data_params["time"][i_in_time, 0])
            self.__update_xdata()

        for i, nlp in enumerate(self.ocp.nlp):
            step_size = (
                nlp.ode_solver.steps_scipy + 1
                if self.integrator != SolutionIntegrator.OCP
                else nlp.ode_solver.steps + 1
            )

            n_elements = data_time[i].shape[0]
            state = np.ndarray((0, n_elements))
            for s in nlp.states:
                if nlp.use_states_from_phase_idx == nlp.phase_idx:
                    if isinstance(data_states, (list, tuple)):
                        state = np.concatenate((state, data_states[i][s]))
                    else:
                        state = np.concatenate((state, data_states[s]))
            control = np.ndarray((0, nlp.ns + 1))
            for s in nlp.controls:
                if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                    if isinstance(data_controls, (list, tuple)):
                        control = np.concatenate((control, data_controls[i][s]))
                    else:
                        control = np.concatenate((control, data_controls[s]))

            for key in self.variable_sizes[i]:
                if not self.plot_func[key][i]:
                    continue
                if self.plot_func[key][i].label:
                    if self.plot_func[key][i].label[:16] == "PHASE_TRANSITION":
                        self.ydata.append(np.zeros(np.shape(state)[0]))
                        continue
                x_mod = (
                    1
                    if self.plot_func[key][i].compute_derivative
                    or self.plot_func[key][i].integration_rule == IntegralApproximation.TRAPEZOIDAL
                    or self.plot_func[key][i].integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
                    else 0
                )
                u_mod = (
                    1
                    if (nlp.control_type == ControlType.LINEAR_CONTINUOUS or self.plot_func[key][i].compute_derivative)
                    and not ("OBJECTIVES" in key or "CONSTRAINTS" in key or "PHASE_TRANSITION" in key)
                    or (
                        (
                            self.plot_func[key][i].integration_rule == IntegralApproximation.TRAPEZOIDAL
                            or self.plot_func[key][i].integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
                        )
                        and nlp.control_type == ControlType.LINEAR_CONTINUOUS
                    )
                    else 0
                )

                if self.plot_func[key][i].type == PlotType.INTEGRATED:
                    all_y = []
                    for idx, t in enumerate(self.t_integrated[i]):
                        y_tp = np.empty((self.variable_sizes[i][key], len(t)))
                        y_tp.fill(np.nan)

                        val = self.plot_func[key][i].function(
                            idx,
                            state[:, step_size * idx : step_size * (idx + 1) + x_mod],
                            control[:, idx : idx + u_mod + 1],
                            data_params_in_dyn,
                            **self.plot_func[key][i].parameters,
                        )

                        if self.plot_func[key][i].compute_derivative:
                            # This is a special case since derivative is not properly integrated
                            val = np.repeat(val, y_tp.shape[1])[np.newaxis, :]

                        if val.shape != y_tp.shape:
                            raise RuntimeError(
                                f"Wrong dimensions for plot {key}. Got {val.shape}, but expected {y_tp.shape}"
                            )
                        y_tp[:, :] = val
                        all_y.append(y_tp)

                    for idx in range(len(self.plot_func[key][i].phase_mappings.map_idx)):
                        y_tp = []
                        for y in all_y:
                            y_tp.append(y[idx, :])
                        self.__append_to_ydata([y_tp])

                elif self.plot_func[key][i].type == PlotType.POINT:
                    for i_var in range(self.variable_sizes[i][key]):
                        if self.plot_func[key][i].parameters["penalty"].multinode_constraint:
                            y = np.array([np.nan])
                            penalty: MultinodeConstraint = self.plot_func[key][i].parameters["penalty"]
                            phase_1 = penalty.nodes_phase[1]
                            phase_2 = penalty.nodes_phase[0]
                            node_idx_1 = penalty.all_nodes_index[1]
                            node_idx_2 = penalty.all_nodes_index[0]
                            x_phase_1 = data_states[phase_1]["all"][:, node_idx_1 * step_size]
                            x_phase_2 = data_states[phase_2]["all"][:, node_idx_2 * step_size]
                            u_phase_1 = data_controls[phase_1]["all"][:, node_idx_1]
                            u_phase_2 = data_controls[phase_2]["all"][:, node_idx_2]
                            val = self.plot_func[key][i].function(
                                self.plot_func[key][i].node_idx[0],
                                np.hstack((x_phase_1, x_phase_2)),
                                np.hstack(
                                    (
                                        u_phase_1,
                                        u_phase_2,
                                    )
                                ),
                                data_params_in_dyn,
                                **self.plot_func[key][i].parameters,
                            )
                            y[0] = val[i_var]
                        else:
                            y = np.empty((len(self.plot_func[key][i].node_idx),))
                            y.fill(np.nan)
                            for i_node, node_idx in enumerate(self.plot_func[key][i].node_idx):
                                if self.plot_func[key][i].parameters["penalty"].transition:
                                    val = self.plot_func[key][i].function(
                                        node_idx,
                                        np.hstack(
                                            (
                                                data_states[node_idx]["all"][:, -1],
                                                data_states[node_idx + 1]["all"][:, 0],
                                            )
                                        ),
                                        np.hstack(
                                            (
                                                data_controls[node_idx]["all"][:, -1],
                                                data_controls[node_idx + 1]["all"][:, 0],
                                            )
                                        ),
                                        data_params_in_dyn,
                                        **self.plot_func[key][i].parameters,
                                    )
                                else:
                                    if (
                                        self.plot_func[key][i].label == "CONTINUITY"
                                        and nlp.ode_solver.is_direct_collocation
                                    ):
                                        states = state[:, node_idx * (step_size) : (node_idx + 1) * (step_size) + 1]
                                    else:
                                        states = state[
                                            :, node_idx * step_size : (node_idx + 1) * step_size + x_mod : step_size
                                        ]
                                    control_tp = control[:, node_idx : node_idx + 1 + u_mod]
                                    if np.isnan(control_tp).any():
                                        control_tp = np.array(())
                                    val = self.plot_func[key][i].function(
                                        node_idx,
                                        states,
                                        control_tp,
                                        data_params_in_dyn,
                                        **self.plot_func[key][i].parameters,
                                    )
                                y[i_node] = val[i_var]
                        self.ydata.append(y)

                else:
                    y = np.empty((self.variable_sizes[i][key], len(self.t[i])))
                    y.fill(np.nan)
                    if self.plot_func[key][i].compute_derivative:
                        for i_node, node_idx in enumerate(self.plot_func[key][i].node_idx):
                            val = self.plot_func[key][i].function(
                                node_idx,
                                state[:, node_idx * step_size : (node_idx + 1) * step_size + 1 : step_size],
                                control[:, node_idx : node_idx + 1 + 1],
                                data_params_in_dyn,
                                **self.plot_func[key][i].parameters,
                            )
                            y[:, i_node] = val
                    else:
                        nodes = self.plot_func[key][i].node_idx
                        if nodes and len(nodes) > 1 and len(nodes) == round(state.shape[1] / step_size):
                            # Assume we are integrating but did not specify plot as such.
                            # Therefore the arrival point is missing
                            nodes += [nodes[-1] + 1]

                        val = self.plot_func[key][i].function(
                            nodes,
                            state[:, ::step_size],
                            control,
                            data_params_in_dyn,
                            **self.plot_func[key][i].parameters,
                        )
                        if val.shape != y.shape:
                            raise RuntimeError(
                                f"Wrong dimensions for plot {key}. Got {val.shape}, but expected {y.shape}"
                            )
                        y[:, :] = val
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
            elif plot[0] == PlotType.POINT:
                plot[2].set_xdata(self.t[phase_idx][np.array(self.plot_func[plot[3]][phase_idx].node_idx)])
                ax = plot[2].axes
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

    def __append_to_ydata(self, data: list | np.ndarray):
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
            if (not self.show_bounds) or (self.axes[key][0].bounds is None):
                for i, ax in enumerate(self.axes[key][1]):
                    if not self.axes[key][0].ylim:
                        y_max = -np.inf
                        y_min = np.inf
                        for p in ax.get_children():
                            if isinstance(p, lines.Line2D):
                                y_min = min(y_min, np.nanmin(p.get_ydata()))
                                y_max = max(y_max, np.nanmax(p.get_ydata()))
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

        for fig in self.all_figures:
            fig.set_tight_layout(True)

    @staticmethod
    def __compute_ylim(min_val: np.ndarray | DM, max_val: np.ndarray | DM, factor: float) -> tuple:
        """
        Dynamically find the ylim
        Parameters
        ----------
        min_val: np.ndarray | DM
            The minimal value of the y axis
        max_val: np.ndarray | DM
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
    eval(self, arg: list | tuple) -> list[int]
        Send the current data to the plotter
    """

    def __init__(self, ocp, opts: dict = None, show_options: dict = None):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to show
        opts: dict
            Option to AnimateCallback method of CasADi
        show_options: dict
            The options to pass to PlotOcp
        """
        if opts is None:
            opts = {}

        Callback.__init__(self)
        self.ocp = ocp
        self.nx = self.ocp.v.vector.shape[0]
        self.ng = 0
        self.construct("AnimateCallback", opts)

        self.queue = mp.Queue()
        self.plotter = self.ProcessPlotter(self.ocp)
        self.plot_process = mp.Process(target=self.plotter, args=(self.queue, show_options), daemon=True)
        self.plot_process.start()

    def close(self):
        self.plot_process.kill()

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

    def eval(self, arg: list | tuple) -> list:
        """
        Send the current data to the plotter

        Parameters
        ----------
        arg: list | tuple
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

        def __call__(self, pipe: mp.Queue, show_options: dict):
            """
            Parameters
            ----------
            pipe: mp.Queue
                The multiprocessing queue to evaluate
            show_options: dict
                The option to pass to PlotOcp
            """

            if show_options is None:
                show_options = {}
            self.pipe = pipe
            self.plot = PlotOcp(self.ocp, **show_options)
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
                v = self.pipe.get()
                self.plot.update_data(v)

            for i, fig in enumerate(self.plot.all_figures):
                fig.canvas.draw()
            return True
