import tkinter
from typing import Callable, Any

import numpy as np
from casadi import DM
from matplotlib import pyplot as plt, lines
from matplotlib.ticker import FuncFormatter

from .serializable_class import OcpSerializable
from ..dynamics.ode_solver import OdeSolver
from ..limits.path_conditions import Bounds
from ..limits.penalty_helpers import PenaltyHelpers
from ..misc.enums import PlotType, Shooting, SolutionIntegrator, QuadratureRule, InterpolationType
from ..misc.mapping import Mapping, BiMapping
from ..optimization.solution.solution import Solution
from ..optimization.solution.solution_data import SolutionMerge


DEFAULT_COLORS = {
    PlotType.PLOT: "tab:green",
    PlotType.INTEGRATED: "tab:brown",
    PlotType.STEP: "tab:orange",
    PlotType.POINT: "tab:purple",
}

DEFAULT_LINESTYLES = {PlotType.PLOT: "-", PlotType.INTEGRATED: None, PlotType.STEP: "-", PlotType.POINT: None}


class CustomPlot:
    """
    Interface to create/add plots of the simulation

    Attributes
    ----------
    function: Callable[time, states, controls, parameters, algebraic_states]
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
        all_variables_in_one_subplot: bool = False,
        **parameters: Any,
    ):
        """
        Parameters
        ----------
        update_function: Callable[time, states, controls, parameters, algebraic_states]
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
        all_variables_in_one_subplot: bool
            If all indices of the variables should be put on the same graph. This is not cute, but allows to display variables with a lot of entries.
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
        self.all_variables_in_one_subplot = all_variables_in_one_subplot


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
    _create_plots(self)
        Setup the plots
    _add_new_axis(self, variable: str, nb: int, n_rows: int, n_cols: int)
        Add a new axis to the axes pool
    _organize_windows(self, n_windows: int)
        Automatically organize the figure across the screen.
    find_phases_intersections(self)
        Finds the intersection between the phases
    show()
        Force the show of the graphs. This is a blocking function
    update_data(self, v: dict)
        Update ydata from the variable a solution structure
    _update_xdata(self)
        Update of the time axes in plots
    _update_axes(self)
        Update the plotted data from ydata
    _compute_ylim(min_val: np.ndarray | DM, max_val: np.ndarray | DM, factor: float) -> tuple:
        Dynamically find the ylim
    _generate_windows_size(nb: int) -> tuple[int, int]
        Defines the number of column and rows of subplots from the number of variables to plot.
    """

    def __init__(
        self,
        ocp: OcpSerializable,
        automatically_organize: bool = True,
        show_bounds: bool = False,
        shooting_type: Shooting = Shooting.MULTIPLE,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
        dummy_phase_times: list[list[float]] = None,
        only_initialize_variables: bool = False,
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
        dummy_phase_times: list[list[float]]
            The time of each phase
        only_initialize_variables: bool
            If the plots should be initialized but not shown (this is useful for the online plot which must be declared
            on the server side and on the client side)
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

        self.n_nodes = 0

        self.t = []
        self.t_integrated = []
        self.integrator = integrator

        # Emulate the time from Solution.time, this is just to give the size anyway
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
        if not only_initialize_variables:
            self._organize_windows(len(self.ocp.nlp[0].states) + len(self.ocp.nlp[0].controls))

        self.custom_plots = {}
        self.variable_sizes = []
        self.show_bounds = show_bounds
        self._create_plots(only_initialize_variables)
        self.shooting_type = shooting_type

        if not only_initialize_variables:
            self._spread_figures_on_screen()

        if self.ocp.plot_ipopt_outputs:
            from ..gui.ipopt_output_plot import create_ipopt_output_plot
            from ..interfaces.ipopt_interface import IpoptInterface

            interface = IpoptInterface(self.ocp)
            create_ipopt_output_plot(ocp, interface)

        if self.ocp.plot_check_conditioning:
            from ..gui.check_conditioning import create_conditioning_plots

            create_conditioning_plots(ocp)

    def _update_time_vector(self, phase_times):
        """
        Setup the time and time integrated vector, which is the x-axes of the graphs
        """

        self.t = []
        self.t_integrated = []
        for nlp, time in zip(self.ocp.nlp, phase_times):
            self.n_nodes += nlp.n_states_nodes

            self.t_integrated.append(time)
            self.t.append(np.linspace(float(time[0][0]), float(time[-1][-1]), nlp.n_states_nodes))

    def _create_plots(self, only_initialize_variables: bool):
        """
        Setup the plots

        Parameters
        ----------
        only_initialize_variables: bool
            If the plots should be initialized but not shown (this is useful for the online plot which must be declared
            on the server side and on the client side)
        """

        def legend_without_duplicate_labels(ax):
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            if unique:
                ax.legend(*zip(*unique))

        variable_sizes = []

        self.ocp.finalize_plot_phase_mappings()
        for i, nlp in enumerate(self.ocp.nlp):

            variable_sizes.append({})
            if nlp.plot:
                for key in nlp.plot:
                    if isinstance(nlp.plot[key], tuple):
                        nlp.plot[key] = nlp.plot[key][0]

                    # This is the point where we can safely define node_idx of the plot
                    if nlp.plot[key].node_idx is None:
                        nlp.plot[key].node_idx = range(nlp.n_states_nodes)

                    n_subplots = max(nlp.plot[key].phase_mappings.to_second.map_idx) + 1

                    if key not in variable_sizes[i]:
                        variable_sizes[i][key] = n_subplots
                    else:
                        variable_sizes[i][key] = max(variable_sizes[i][key], n_subplots)

        self.variable_sizes = variable_sizes
        if not variable_sizes:
            # No graph was setup in problem_type
            return

        all_keys_across_phases = []
        for variable_sizes in self.variable_sizes:
            keys_not_in_previous_phases = [
                key for key in list(variable_sizes.keys()) if key not in all_keys_across_phases
            ]
            all_keys_across_phases += keys_not_in_previous_phases

        y_min_all = [None for _ in all_keys_across_phases]
        y_max_all = [None for _ in all_keys_across_phases]

        self.custom_plots = {}
        for i, nlp in enumerate(self.ocp.nlp):

            for var_idx, variable in enumerate(self.variable_sizes[i]):
                y_range_var_idx = all_keys_across_phases.index(variable)

                if not only_initialize_variables:
                    if nlp.plot[variable].combine_to:
                        self.axes[variable] = self.axes[nlp.plot[variable].combine_to]
                        axes = self.axes[variable][1]
                    elif i > 0 and variable in self.axes:
                        axes = self.axes[variable][1]
                    else:
                        nb_subplots = max(
                            [
                                (
                                    max(
                                        len(nlp.plot[variable].phase_mappings.to_first.map_idx),
                                        max(nlp.plot[variable].phase_mappings.to_first.map_idx) + 1,
                                    )
                                    if variable in nlp.plot
                                    else 0
                                )
                                for nlp in self.ocp.nlp
                            ]
                        )

                        # TODO: get rid of all_variables_in_one_subplot by fixing the mapping appropriately
                        if not nlp.plot[variable].all_variables_in_one_subplot:
                            n_cols, n_rows = PlotOcp._generate_windows_size(nb_subplots)
                        else:
                            n_cols = 1
                            n_rows = 1
                        axes = self._add_new_axis(variable, nb_subplots, n_rows, n_cols)
                        self.axes[variable] = [nlp.plot[variable], axes]

                        if not y_min_all[y_range_var_idx]:
                            y_min_all[y_range_var_idx] = [np.inf] * nb_subplots
                            y_max_all[y_range_var_idx] = [-np.inf] * nb_subplots

                if variable not in self.custom_plots:
                    self.custom_plots[variable] = [
                        nlp_tp.plot[variable] if variable in nlp_tp.plot else None for nlp_tp in self.ocp.nlp
                    ]
                if not self.custom_plots[variable][i] or only_initialize_variables:
                    continue

                mapping_to_first_index = nlp.plot[variable].phase_mappings.to_first.map_idx
                for ctr in mapping_to_first_index:
                    if not nlp.plot[variable].all_variables_in_one_subplot:
                        ax = axes[ctr]
                        if ctr in mapping_to_first_index:
                            index_legend = mapping_to_first_index.index(ctr)
                            if len(nlp.plot[variable].legend) > index_legend:
                                ax.set_title(nlp.plot[variable].legend[index_legend])
                    else:
                        ax = axes[0]
                    ax.grid(**self.plot_options["grid"])
                    ax.set_xlim(self.t[-1][[0, -1]])

                    if nlp.plot[variable].ylim:
                        ax.set_ylim(nlp.plot[variable].ylim)
                    elif (
                        self.show_bounds
                        and nlp.plot[variable].bounds
                        and not nlp.plot[variable].all_variables_in_one_subplot
                    ):
                        if nlp.plot[variable].bounds.type != InterpolationType.CUSTOM:
                            y_min = nlp.plot[variable].bounds.min[mapping_to_first_index.index(ctr), :].min()
                            y_max = nlp.plot[variable].bounds.max[mapping_to_first_index.index(ctr), :].max()
                        else:
                            repeat = 1
                            if isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
                                repeat = nlp.ode_solver.polynomial_degree + 1
                            nlp.plot[variable].bounds.check_and_adjust_dimensions(len(mapping_to_first_index), nlp.ns)
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

                        if y_min.__array__()[0] < y_min_all[y_range_var_idx][mapping_to_first_index.index(ctr)]:
                            y_min_all[y_range_var_idx][mapping_to_first_index.index(ctr)] = y_min

                        if y_max.__array__()[0] > y_max_all[y_range_var_idx][mapping_to_first_index.index(ctr)]:
                            y_max_all[y_range_var_idx][mapping_to_first_index.index(ctr)] = y_max

                        y_range = self._compute_ylim(
                            y_min_all[y_range_var_idx][mapping_to_first_index.index(ctr)],
                            y_max_all[y_range_var_idx][mapping_to_first_index.index(ctr)],
                            1.25,
                        )
                        ax.set_ylim(y_range)

                    plot_type = self.custom_plots[variable][i].type
                    t = self.t[i][nlp.plot[variable].node_idx] if plot_type == PlotType.POINT else self.t[i]
                    if self.custom_plots[variable][i].label:
                        label = self.custom_plots[variable][i].label
                    else:
                        label = None

                    color = (
                        self.custom_plots[variable][i].color
                        if self.custom_plots[variable][i].color
                        else DEFAULT_COLORS[plot_type]
                    )

                    if plot_type == PlotType.PLOT:
                        zero = np.zeros((t.shape[0], 1))
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
                        linestyle = (
                            self.custom_plots[variable][i].linestyle
                            if self.custom_plots[variable][i].linestyle
                            else "-"
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
                            self.plots_vertical_lines.append(
                                ax.axvline(float(time), **self.plot_options["vertical_lines"])
                            )

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

    def _add_new_axis(self, variable: str, nb: int, n_rows: int, n_cols: int):
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
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, tick_value: f"{value:.2f}"))

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

    def _spread_figures_on_screen(self):
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

    def find_phases_intersections(self):
        """
        Finds the intersection between the phases
        """

        return list([t[-1][-1] for t in self.t_integrated])[:-1]

    @staticmethod
    def show():
        """
        Force the show of the graphs. This is a blocking function
        """

        plt.show()

    def parse_data(self, **args) -> tuple[list, list]:
        """
        Parse the data to be plotted, the return of this method can be passed to update_data to update the plots

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the full ocp
        variable_sizes: list[int]
            The size of all variables. This is the reference to the PlotOcp.variable_sizes (which can't be accessed
            from this static method)
        custom_plots: dict
            The dictionary of all the CustomPlot. This is the reference to the PlotOcp.custom_plots (which can't be
            accessed from this static method)
        """
        from ..interfaces.interface_utils import get_numerical_timeseries

        ydata = []

        sol = Solution.from_vector(self.ocp, args["x"])
        data_states_decision = sol.decision_states(scaled=True, to_merge=SolutionMerge.KEYS)
        data_states_stepwise = sol.stepwise_states(scaled=True, to_merge=SolutionMerge.KEYS)

        data_controls = sol.stepwise_controls(scaled=True, to_merge=SolutionMerge.KEYS)
        p = sol.decision_parameters(scaled=True, to_merge=SolutionMerge.KEYS)
        data_algebraic_states = sol.decision_algebraic_states(scaled=True, to_merge=SolutionMerge.KEYS)

        if len(self.ocp.nlp) == 1:
            # This is automatically removed in the Solution, but to keep things clean we put them back in a list
            data_states_decision = [data_states_decision]
            data_states_stepwise = [data_states_stepwise]
            data_controls = [data_controls]
            data_algebraic_states = [data_algebraic_states]

        time_stepwise = sol.stepwise_time(continuous=True)
        if self.ocp.n_phases == 1:
            time_stepwise = [time_stepwise]
        phases_dt = sol.phases_dt
        xdata = time_stepwise

        for nlp in self.ocp.nlp:

            phase_idx = nlp.phase_idx
            x_decision = data_states_decision[phase_idx]
            x_stepwise = data_states_stepwise[phase_idx]
            u = data_controls[phase_idx]
            a = data_algebraic_states[phase_idx]
            d = []
            for n_idx in range(nlp.ns + 1):
                d_tp = get_numerical_timeseries(self.ocp, phase_idx, n_idx, 0)
                if d_tp.shape == (0, 0):
                    d += [np.array([])]
                else:
                    d += [np.array(d_tp)]

            for key in self.variable_sizes[phase_idx]:
                y_data = self._compute_y_from_plot_func(
                    self.custom_plots[key][phase_idx],
                    phase_idx,
                    time_stepwise,
                    phases_dt,
                    x_decision,
                    x_stepwise,
                    u,
                    p,
                    a,
                    d,
                )
                if y_data is None:
                    continue
                mapped_y_data = []
                for i in nlp.plot[key].phase_mappings.to_first.map_idx:
                    mapped_y_data.append(y_data[i])
                for y in mapped_y_data:
                    ydata.append(y)

        return xdata, ydata

    def update_data(
        self,
        xdata: list,
        ydata: list,
        **args: dict,
    ):
        """
        Update xdata and ydata. The input are the output of the parse_data method

        Parameters
        ----------
        xdata: list
            The time vector
        ydata: list
            The actual current data to be plotted
        args: dict
            The same args as the parse_data method (that is so ipopt outputs can be plotted, this should be done properly
            in the future, when ready, remove this parameter)
        """

        self._update_xdata(xdata)
        self._update_ydata(ydata)

        if self.ocp.plot_ipopt_outputs:
            from ..gui.ipopt_output_plot import update_ipopt_output_plot

            update_ipopt_output_plot(args, self.ocp)

        if self.ocp.save_ipopt_iterations_info is not None:
            from ..gui.ipopt_output_plot import save_ipopt_output

            save_ipopt_output(args, self.ocp.save_ipopt_iterations_info)

        if self.ocp.plot_check_conditioning:
            from ..gui.check_conditioning import update_conditioning_plots

            update_conditioning_plots(args["x"], self.ocp)

    def _compute_y_from_plot_func(
        self, custom_plot: CustomPlot, phase_idx, time_stepwise, dt, x_decision, x_stepwise, u, p, a, d
    ) -> list[np.ndarray | list]:
        """
        Compute the y data from the plot function

        Parameters
        ----------
        custom_plot: CustomPlot
            The custom plot to compute
        phase_idx: int
            The index of the current phase
        time_stepwise: list[list[DM]]
            The time vector of each phase
        dt
            The delta times of the current phase
        x_decision
            The states of the current phase (decision)
        x_stepwise
            The states of the current phase (stepwise)
        u
            The controls of the current phase
        p
            The parameters of the current phase
        a
            The algebraic states of the current phase
        d
            The numerical timeseries of the current phase

        Returns
        -------
        list[np.ndarray | list, ...]
            The y data, list of len number of axes per figure, the sublist is empty when the plot is not to be shown.
        """
        from ..interfaces.interface_utils import get_numerical_timeseries

        if not custom_plot:
            return None

        x = x_stepwise if custom_plot.type == PlotType.INTEGRATED else x_decision

        # Compute the values of the plot at each node
        all_y = []
        for idx in range(len(custom_plot.node_idx)):
            node_idx = custom_plot.node_idx[idx]
            if "penalty" in custom_plot.parameters:
                penalty = custom_plot.parameters["penalty"]
                t0 = PenaltyHelpers.t0(penalty, idx, lambda p_idx, n_idx: time_stepwise[p_idx][n_idx][0])

                x_node = PenaltyHelpers.states(
                    penalty,
                    idx,
                    lambda p_idx, n_idx, sn_idx: x[n_idx][:, sn_idx] if n_idx < len(x) else np.ndarray((0, 1)),
                )
                u_node = PenaltyHelpers.controls(
                    penalty,
                    idx,
                    lambda p_idx, n_idx, sn_idx: u[n_idx][:, sn_idx] if n_idx < len(u) else np.ndarray((0, 1)),
                )
                p_node = PenaltyHelpers.parameters(penalty, 0, lambda p_idx, n_idx, sn_idx: np.array(p))
                a_node = PenaltyHelpers.states(
                    penalty,
                    idx,
                    lambda p_idx, n_idx, sn_idx: a[n_idx][:, sn_idx] if n_idx < len(a) else np.ndarray((0, 1)),
                )
                d_node = PenaltyHelpers.numerical_timeseries(
                    penalty,
                    idx,
                    lambda p_idx, n_idx, sn_idx: get_numerical_timeseries(self.ocp, p_idx, n_idx, sn_idx),
                )
                if d_node.shape == (0, 0):
                    d_node = DM(0, 1)
            else:
                t0 = time_stepwise[phase_idx][node_idx][0]

                x_node = x[node_idx]
                u_node = u[node_idx] if node_idx < len(u) else np.ndarray((0, 1))
                p_node = p
                a_node = a[node_idx]
                d_node = d[node_idx]

            tp = custom_plot.function(
                t0, dt, node_idx, x_node, u_node, p_node, a_node, d_node, **custom_plot.parameters
            )

            map_idx = custom_plot.phase_mappings.to_first.map_idx
            y_tp = np.ndarray((max(map_idx) + 1, tp.shape[1])) * np.nan
            for ctr, axe_index in enumerate(map_idx):
                y_tp[axe_index, :] = tp[ctr, :]
            all_y.append(y_tp)

        # Dispatch the values so they will by properly dispatched to the correct axes later
        if custom_plot.type == PlotType.INTEGRATED:
            out = [[] for _ in range(max(np.abs(custom_plot.phase_mappings.to_first.map_idx)) + 1)]
            for idx in custom_plot.phase_mappings.to_first.map_idx:
                y_tp = []
                for y in all_y:
                    y_tp.append(y[custom_plot.phase_mappings.to_first.map_idx.index(idx), :])
                out[idx] = y_tp
            return out

        elif custom_plot.type in (PlotType.PLOT, PlotType.STEP, PlotType.POINT):
            all_y = np.concatenate([tp[:, 0:1] for tp in all_y], axis=1)
            out = [[] for _ in range(max(np.abs(custom_plot.phase_mappings.to_first.map_idx)) + 1)]
            for idx in custom_plot.phase_mappings.to_first.map_idx:
                out[idx] = all_y[idx, :]
            return out
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
                    self.plots_vertical_lines[p * n + i].set_xdata([float(time), float(time)])

    def _update_ydata(self, ydata):
        """
        Update the plotted data from ydata
        """

        assert len(self.plots) == len(ydata)
        for i, plot in enumerate(self.plots):
            y = ydata[i]
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
                        ax.set_ylim(self._compute_ylim(y_min, y_max, 1.25))

        for p in self.plots_vertical_lines:
            p.set_ydata((0, 1))

        for fig in self.all_figures:
            fig.set_tight_layout(True)
            # TODO:  set_tight_layout function will be deprecated. Use set_layout_engine instead.

    @staticmethod
    def _compute_ylim(min_val: np.ndarray | DM, max_val: np.ndarray | DM, factor: float) -> tuple:
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
        return data_mean - y_range, data_mean + y_range

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
