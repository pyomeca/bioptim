from typing import Callable, Any
import multiprocessing as mp
from copy import copy
import tkinter
from itertools import accumulate

import numpy as np
from matplotlib import pyplot as plt, lines
from matplotlib.ticker import StrMethodFormatter
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity, DM, Function

from ..limits.path_conditions import Bounds
from ..limits.penalty_helpers import PenaltyHelpers
from ..limits.multinode_constraint import MultinodeConstraint
from ..misc.enums import (
    PlotType,
    ControlType,
    InterpolationType,
    Shooting,
    SolutionIntegrator,
    QuadratureRule,
    PhaseDynamics,
)
from ..misc.mapping import Mapping, BiMapping
from ..optimization.solution.solution import Solution
from ..dynamics.ode_solver import OdeSolver
from ..optimization.optimization_vector import OptimizationVectorHelper


class CustomPlot:
    """
    Interface to create/add plots of the simulation

    Attributes
    ----------
    function: Callable[time, states, controls, parameters, stochastic_variables]
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
        axes_idx: BiMapping | tuple | list = None,
        legend: tuple | list = None,
        combine_to: str = None,
        color: str = None,
        linestyle: str = None,
        ylim: tuple | list = None,
        bounds: Bounds = None,
        node_idx: list | slice | range = None,
        label: list = None,
        compute_derivative: bool = False,
        integration_rule: QuadratureRule = QuadratureRule.RECTANGLE_LEFT,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        update_function: Callable[time, states, controls, parameters, stochastic_variables]
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
            self.phase_mappings = BiMapping(to_second=Mapping(axes_idx), to_first=Mapping(axes_idx))
        elif isinstance(axes_idx, BiMapping):
            self.phase_mappings = axes_idx
        else:
            raise RuntimeError("phase_mapping must be a list or a Mapping")
        self.legend = legend if legend is not None else ()
        self.combine_to = combine_to
        self.color = color
        self.linestyle = linestyle
        self.ylim = ylim
        self.bounds = bounds
        self.node_idx = node_idx  # If this is None, it is all nodes and will be initialize when we know the dimension of the problem
        self.label = label
        self.compute_derivative = compute_derivative
        if integration_rule == QuadratureRule.MIDPOINT or integration_rule == QuadratureRule.RECTANGLE_RIGHT:
            raise NotImplementedError(f"{integration_rule} has not been implemented yet.")
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
    nodes: int
        The total number of nodes points
    n_vertical_windows: int
        The number of figure rows
    ocp: OptimalControlProgram
        A reference to the full ocp
    plots: list
        The list of handlers to the matplotlib plots for the ydata
    plots_bounds: list
        The list of handlers to the matplotlib plots for the bounds
    custom_plots: dict
        The dictionary of all the CustomPlot
    plot_options: dict
        matplotlib options template for specified PlotType
    plots_vertical_lines: list
        The list of handlers to the matplotlib plots for the phase separators
    shooting_type: Shooting
        The type of integration method
    t: list[float]
        The time vector
    t_integrated: list[float]
        The time vector integrated
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
    _update_time_vector(self)
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
    _append_to_ydata(self, data: list)
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
        self.n_nodes = 0

        self.t = []
        self.t_integrated = []
        self.integrator = integrator

        # Emulate the time from Solution.time, this is just to give the size anyway
        dummy_phase_times = OptimizationVectorHelper.extract_step_times(ocp, DM(np.ones(ocp.n_phases)))
        self._update_time_vector(dummy_phase_times)

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

        self.custom_plots = {}
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

    def _update_time_vector(self, phase_times):
        """
        Setup the time and time integrated vector, which is the x-axes of the graphs
        """

        self.t = []
        self.t_integrated = phase_times
        last_t = 0
        for nlp, time in zip(self.ocp.nlp, self.t_integrated):
            self.n_nodes += nlp.n_states_nodes
            time_phase = np.linspace(last_t, last_t + float(time[-1][-1]), nlp.n_states_nodes)
            last_t += float(time[-1][-1])
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

                    # This is the point where we can safely define node_idx of the plot
                    if nlp.plot[key].node_idx is None:
                        nlp.plot[key].node_idx = range(nlp.n_states_nodes)

                    if nlp.plot[key].phase_mappings is None:
                        node_index = nlp.plot[key].node_idx[0]
                        nlp.states.node_index = node_index
                        nlp.states_dot.node_index = node_index
                        nlp.controls.node_index = node_index
                        nlp.stochastic_variables.node_index = node_index

                        # If multi-node penalties = None, stays zero
                        size_x = nlp.states.shape
                        size_u = nlp.controls.shape
                        size_p = nlp.parameters.shape
                        size_s = nlp.stochastic_variables.shape
                        if "penalty" in nlp.plot[key].parameters:
                            penalty = nlp.plot[key].parameters["penalty"]
                            casadi_function = penalty.weighted_function_non_threaded[0]
                            
                            if casadi_function is not None:
                                size_x = casadi_function.nnz_in(2)
                                size_u = casadi_function.nnz_in(3)
                                size_p = casadi_function.nnz_in(4)
                                size_s = casadi_function.nnz_in(5)

                        size = (
                            nlp.plot[key].function(
                                0,  # t0
                                np.zeros(len(self.ocp.nlp)),  # phases_dt
                                node_index,  # node_idx
                                np.zeros((size_x, 1)),  # states
                                np.zeros((size_u, 1)),  # controls
                                np.zeros((size_p, 1)),  # parameters
                                np.zeros((size_s, 1)),  # stochastic_variables
                                **nlp.plot[key].parameters,  # parameters
                            )
                            .shape[0]
                        )
                        nlp.plot[key].phase_mappings = BiMapping(to_first=range(size), to_second=range(size))
                    else:
                        size = max(nlp.plot[key].phase_mappings.to_second.map_idx) + 1
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
        self.custom_plots = {}
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
                            max(
                                len(nlp.plot[variable].phase_mappings.to_second.map_idx),
                                max(nlp.plot[variable].phase_mappings.to_second.map_idx) + 1,
                            )
                            if variable in nlp.plot
                            else 0
                            for nlp in self.ocp.nlp
                        ]
                    )
                    n_cols, n_rows = PlotOcp._generate_windows_size(nb)
                    axes = self.__add_new_axis(variable, nb, n_rows, n_cols)
                    self.axes[variable] = [nlp.plot[variable], axes]
                    if not y_min_all[var_idx]:
                        y_min_all[var_idx] = [np.inf] * nb
                        y_max_all[var_idx] = [-np.inf] * nb

                if variable not in self.custom_plots:
                    self.custom_plots[variable] = [
                        nlp_tp.plot[variable] if variable in nlp_tp.plot else None for nlp_tp in self.ocp.nlp
                    ]
                if not self.custom_plots[variable][i]:
                    continue

                mapping_to_first_index = nlp.plot[variable].phase_mappings.to_first.map_idx
                mapping_range_index = list(range(max(nlp.plot[variable].phase_mappings.to_second.map_idx) + 1))
                for ctr in mapping_range_index:
                    ax = axes[ctr]
                    if ctr in mapping_to_first_index:
                        index_legend = mapping_to_first_index.index(ctr)
                        if len(nlp.plot[variable].legend) > index_legend:
                            ax.set_title(nlp.plot[variable].legend[index_legend])
                    ax.grid(**self.plot_options["grid"])
                    ax.set_xlim(0, self.t[-1][-1])
                    if ctr in mapping_to_first_index:
                        if nlp.plot[variable].ylim:
                            ax.set_ylim(nlp.plot[variable].ylim)
                        elif self.show_bounds and nlp.plot[variable].bounds:
                            if nlp.plot[variable].bounds.type != InterpolationType.CUSTOM:
                                y_min = nlp.plot[variable].bounds.min[mapping_to_first_index.index(ctr), :].min()
                                y_max = nlp.plot[variable].bounds.max[mapping_to_first_index.index(ctr), :].max()
                            else:
                                repeat = 1
                                if isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
                                    repeat = nlp.ode_solver.polynomial_degree + 1
                                nlp.plot[variable].bounds.check_and_adjust_dimensions(
                                    len(mapping_to_first_index), nlp.ns
                                )
                                y_min = min(
                                    [
                                        nlp.plot[variable].bounds.min.evaluate_at(j)[mapping_to_first_index.index(ctr)]
                                        for j in range(nlp.ns * repeat)
                                    ]
                                )
                                y_max = max(
                                    [
                                        nlp.plot[variable].bounds.max.evaluate_at(j)[mapping_to_first_index.index(ctr)]
                                        for j in range(nlp.ns * repeat)
                                    ]
                                )
                            if y_min.__array__()[0] < y_min_all[var_idx][mapping_to_first_index.index(ctr)]:
                                y_min_all[var_idx][mapping_to_first_index.index(ctr)] = y_min
                            if y_max.__array__()[0] > y_max_all[var_idx][mapping_to_first_index.index(ctr)]:
                                y_max_all[var_idx][mapping_to_first_index.index(ctr)] = y_max

                            y_range, _ = self.__compute_ylim(
                                y_min_all[var_idx][mapping_to_first_index.index(ctr)],
                                y_max_all[var_idx][mapping_to_first_index.index(ctr)],
                                1.25,
                            )
                            ax.set_ylim(y_range)

                    plot_type = self.custom_plots[variable][i].type
                    t = self.t[i][nlp.plot[variable].node_idx] if plot_type == PlotType.POINT else self.t[i]
                    if self.custom_plots[variable][i].label:
                        label = self.custom_plots[variable][i].label
                    else:
                        label = None

                    if plot_type == PlotType.PLOT:
                        zero = np.zeros((t.shape[0], 1))
                        color = self.custom_plots[variable][i].color if self.custom_plots[variable][i].color else "tab:green"
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
                        color = self.custom_plots[variable][i].color if self.custom_plots[variable][i].color else "tab:brown"
                        for cmp in range(nlp.ns):
                            plots_integrated.append(
                                ax.plot(
                                    self.t_integrated[i][cmp],
                                    np.zeros((self.t_integrated[i][cmp].shape[0], 1)),
                                    color=color,
                                    label=label,
                                    **self.plot_options["integrated_plots"],
                                )[0]
                            )
                        self.plots.append([plot_type, i, plots_integrated])

                    elif plot_type == PlotType.STEP:
                        zero = np.zeros((t.shape[0], 1))
                        color = self.custom_plots[variable][i].color if self.custom_plots[variable][i].color else "tab:orange"
                        linestyle = (
                            self.custom_plots[variable][i].linestyle if self.custom_plots[variable][i].linestyle else "-"
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
                        color = self.custom_plots[variable][i].color if self.custom_plots[variable][i].color else "tab:purple"
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

                for ctr, ax in enumerate(axes):
                    if ctr in mapping_to_first_index:
                        intersections_time = self.find_phases_intersections()
                        for time in intersections_time:
                            self.plots_vertical_lines.append(ax.axvline(time, **self.plot_options["vertical_lines"]))

                        if nlp.plot[variable].bounds and self.show_bounds:
                            if nlp.plot[variable].bounds.type == InterpolationType.EACH_FRAME:
                                ns = nlp.plot[variable].bounds.min.shape[1] - 1
                            else:
                                ns = nlp.ns

                            # TODO: introduce repeat for the COLLOCATIONS min/max_bounds only for states graphs.
                            # For now the plots in COLLOCATIONS with LINEAR are not giving the right values
                            nlp.plot[variable].bounds.check_and_adjust_dimensions(
                                n_elements=len(mapping_to_first_index), n_shooting=ns
                            )
                            bounds_min = np.array(
                                [
                                    nlp.plot[variable].bounds.min.evaluate_at(k)[mapping_to_first_index.index(ctr)]
                                    for k in range(ns + 1)
                                ]
                            )
                            bounds_max = np.array(
                                [
                                    nlp.plot[variable].bounds.max.evaluate_at(k)[mapping_to_first_index.index(ctr)]
                                    for k in range(ns + 1)
                                ]
                            )
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

        return list(accumulate([t[-1][-1] for t in self.t_integrated]))[:-1]

    @staticmethod
    def show():
        """
        Force the show of the graphs. This is a blocking function
        """

        plt.show()

    def update_data(self, v: np.ndarray):
        """
        Update ydata from the variable a solution structure

        Parameters
        ----------
        v: np.ndarray
            The data to parse
        """

        self.ydata = []

        sol = Solution.from_vector(self.ocp, v)
        data_states_decision = sol.decision_states(scaled=True, concatenate_keys=True)
        data_states_stepwise = sol.stepwise_states(scaled=True, concatenate_keys=True)

        data_controls = sol.controls(scaled=True, concatenate_keys=True)
        p = sol.parameters(scaled=True, concatenate_keys=True)
        data_stochastic = sol.stochastic(scaled=True, concatenate_keys=True)

        if len(self.ocp.nlp) == 1:
            # This is automatically removed in the Solution, but to keep things clean we put them back in a list
            data_states_decision = [data_states_decision]
            data_states_stepwise = [data_states_stepwise]
            data_controls = [data_controls]
            data_stochastic = [data_stochastic]

        time_stepwise = sol.time()
        phases_dt = sol.phases_dt
        self._update_xdata(time_stepwise)

        for nlp in self.ocp.nlp:
            phase_idx = nlp.phase_idx
            x_decision = data_states_decision[phase_idx]
            x_stepwise = data_states_stepwise[phase_idx]
            u = data_controls[phase_idx]
            s = data_stochastic[phase_idx]

            for key in self.variable_sizes[phase_idx]:
                y_data = self._compute_y_from_plot_func(self.custom_plots[key][phase_idx], time_stepwise, phases_dt, x_decision, x_stepwise, u, s, p)
                if y_data is None:
                    continue
                self._append_to_ydata(y_data)

        self.__update_axes()

    @staticmethod
    def _compute_y_from_plot_func(custom_plot: CustomPlot, time_stepwise, dt, x_decision, x_stepwise, u, s, p):
        """
        Compute the y data from the plot function

        Parameters
        ----------
        custom_plot: CustomPlot
            The custom plot to compute
        time_stepwise: list[list[DM], ...]
        dt
            The delta times of the current phase
        x
            The states of the current phase (stepwise)
        u
            The controls of the current phase
        s
            The stochastic of the current phase
        p
            The parameters of the current phase

        Returns
        -------
        The y data
        """

        if not custom_plot:
            return None
        if custom_plot.label:
            if custom_plot.label[:16] == "PHASE_TRANSITION":
                return np.zeros(np.shape(x)[0])

        x = x_stepwise if custom_plot.type == PlotType.INTEGRATED else x_decision

        # Compute the values of the plot at each node
        all_y = []
        for idx in custom_plot.node_idx:
            if "penalty" in custom_plot.parameters:
                penalty = custom_plot.parameters["penalty"]
                t0 = PenaltyHelpers.t0(penalty, idx, lambda p_idx, n_idx: time_stepwise[p_idx][n_idx])
                
                x_node = PenaltyHelpers.states(penalty, idx, lambda p_idx, n_idx: x[n_idx])
                u_node = PenaltyHelpers.controls(penalty, idx, lambda p_idx, n_idx: u[n_idx])
                p_node = PenaltyHelpers.parameters(penalty, lambda: np.array(p))
                s_node = PenaltyHelpers.stochastic(penalty, idx, lambda p_idx, n_idx: s[n_idx])
                
            else:
                t0 = idx

                x_node = x[idx]
                u_node = u[idx]
                p_node = p
                s_node = s[idx]

            tp = custom_plot.function(t0, dt, idx, x_node, u_node, p_node, s_node, **custom_plot.parameters)

            y_tp = np.ndarray((len(custom_plot.phase_mappings.to_first.map_idx), tp.shape[1])) * np.nan
            for ctr, axe_index in enumerate(custom_plot.phase_mappings.to_first.map_idx):
                y_tp[axe_index, :] = tp[ctr, :]
            all_y.append(y_tp)

        # Dispatch the values so they will by properly dispatched to the correct axes later
        if custom_plot.type == PlotType.INTEGRATED:
            out = []
            for idx in range(max(custom_plot.phase_mappings.to_second.map_idx) + 1):
                y_tp = []
                if idx in custom_plot.phase_mappings.to_second.map_idx:
                    for y in all_y:
                        y_tp.append(y[idx, :])
                else:
                    y_tp = None
                out.append(y_tp)
            return out

        elif custom_plot.type in (PlotType.PLOT, PlotType.STEP, PlotType.POINT):
            all_y = np.concatenate(all_y, axis=1)
            y = []
            for i in range(all_y.shape[0]):
                y.append(all_y[i, :])
            return y
        else:
            raise RuntimeError(f"Plot type {custom_plot.type} not implemented yet")

    def _update_xdata(self, phase_times):
        """
        Update of the time axes in plots
        """

        self._update_time_vector(phase_times)
        for plot in self.plots:
            phase_idx = plot[1]
            if plot[0] == PlotType.INTEGRATED:
                for cmp, p in enumerate(plot[2]):
                    p.set_xdata(np.array(self.t_integrated[phase_idx][cmp]))
                ax = plot[2][-1].axes
            elif plot[0] == PlotType.POINT:
                plot[2].set_xdata(self.t[phase_idx][np.array(self.custom_plots[plot[3]][phase_idx].node_idx)])
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

    def _append_to_ydata(self, data: list | np.ndarray):
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
            if y is None:
                # Jump the plots which are empty
                y = (np.nan,) * len(plot[2])

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
            # TODO:  set_tight_layout function will be deprecated. Use set_layout_engine instead.

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
        self.nx = self.ocp.variables_vector.shape[0]
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
