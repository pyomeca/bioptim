import multiprocessing as mp
from copy import copy
import numpy as np
import tkinter
import pickle
import os
from itertools import accumulate

from matplotlib import pyplot as plt, lines
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity

from ..misc.data import Data
from ..misc.enums import PlotType, ControlType
from ..misc.mapping import Mapping
from ..misc.utils import check_version


class CustomPlot:
    def __init__(
        self, update_function, plot_type=PlotType.PLOT, axes_idx=None, legend=(), combine_to=None, color=None, ylim=None
    ):
        """
        Initializes the plot.
        :param update_function: Function to plot.
        :param plot_type: Type of plot. (PLOT = 0, INTEGRATED = 1 or STEP = 2)
        :param axes_idx: Index of the axis to be mapped. (integer)
        :param legend: Legend of the graphs. (?)
        :param combine_to: Plot in which to add the graph. ??
        :param color: Color of the graphs. (?)
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
        self.ylim = ylim


class PlotOcp:
    def __init__(self, ocp, automatically_organize=True):
        """Prepares the figure"""
        for i in range(1, ocp.nb_phases):
            if ocp.nlp[0]["nbQ"] != ocp.nlp[i]["nbQ"]:
                raise RuntimeError("Graphs with nbQ different at each phase is not implemented yet")

        self.ocp = ocp
        self.ydata = []
        self.ns = 0

        self.t = []
        self.t_integrated = []
        if isinstance(self.ocp.initial_phase_time, (int, float)):
            self.tf = [self.ocp.initial_phase_time]
        else:
            self.tf = list(self.ocp.initial_phase_time)
        self.t_idx_to_optimize = []
        for i, nlp in enumerate(self.ocp.nlp):
            if isinstance(nlp["tf"], self.ocp.CX):
                self.t_idx_to_optimize.append(i)
        self.__update_time_vector()

        self.axes = {}
        self.plots = []
        self.plots_vertical_lines = []
        self.all_figures = []

        self.automatically_organize = automatically_organize
        self._organize_windows(len(self.ocp.nlp[0]["var_states"]) + len(self.ocp.nlp[0]["var_controls"]))

        self.plot_func = {}
        self.variable_sizes = []
        self.__create_plots()

        horz = 0
        vert = 1 if len(self.all_figures) < self.nb_vertical_windows * self.nb_horizontal_windows else 0
        for i, fig in enumerate(self.all_figures):
            if self.automatically_organize:
                try:
                    fig.canvas.manager.window.move(
                        int(vert * self.width_step), int(self.top_margin + horz * self.height_step)
                    )
                    vert += 1
                    if vert >= self.nb_vertical_windows:
                        horz += 1
                        vert = 0
                except AttributeError:
                    pass
            fig.canvas.draw()

    def __update_time_vector(self):
        """Sets x-axis array"""
        self.t = []
        self.t_integrated = []
        last_t = 0
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            nb_int_steps = nlp["nb_integration_steps"]
            dt_ns = self.tf[phase_idx] / nlp["ns"]
            time_phase_integrated = []
            last_t_int = copy(last_t)
            for _ in range(nlp["ns"]):
                time_phase_integrated.append(np.linspace(last_t_int, last_t_int + dt_ns, nb_int_steps + 1))
                last_t_int += dt_ns
            self.t_integrated.append(time_phase_integrated)

            self.ns += nlp["ns"] + 1
            time_phase = np.linspace(last_t, last_t + self.tf[phase_idx], nlp["ns"] + 1)
            last_t += self.tf[phase_idx]
            self.t.append(time_phase)

    def __create_plots(self):
        """Actually plots"""
        variable_sizes = []
        for i, nlp in enumerate(self.ocp.nlp):
            variable_sizes.append({})
            if "plot" in nlp:
                for key in nlp["plot"]:
                    if nlp["plot"][key].phase_mappings is None:
                        size = (
                            nlp["plot"][key]
                            .function(np.zeros((nlp["nx"], 1)), np.zeros((nlp["nu"], 1)), np.zeros((nlp["np"], 1)))
                            .shape[0]
                        )
                        nlp["plot"][key].phase_mappings = Mapping(range(size))
                    else:
                        size = len(nlp["plot"][key].phase_mappings.map_idx)
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
                nb = max(nlp["plot"][variable].phase_mappings.map_idx) + 1
                nb_cols, nb_rows = PlotOcp._generate_windows_size(nb)
                if nlp["plot"][variable].combine_to:
                    self.axes[variable] = self.axes[nlp["plot"][variable].combine_to]
                    axes = self.axes[variable][1]
                elif i > 0 and variable in self.axes:
                    axes = self.axes[variable][1]
                else:
                    axes = self.__add_new_axis(variable, nb, nb_rows, nb_cols)
                    self.axes[variable] = [nlp["plot"][variable], axes]

                t = self.t[i]
                if variable not in self.plot_func:
                    self.plot_func[variable] = [None] * self.ocp.nb_phases
                self.plot_func[variable][i] = nlp["plot"][variable]

                mapping = self.plot_func[variable][i].phase_mappings.map_idx
                for k in mapping:
                    ax = axes[k]
                    if k < len(self.plot_func[variable][i].legend):
                        axes[k].set_title(self.plot_func[variable][i].legend[k])
                    ax.grid(color="k", linestyle="--", linewidth=0.5)
                    ax.set_xlim(0, self.t[-1][-1])
                    if nlp["plot"][variable].ylim:
                        ax.set_ylim(nlp["plot"][variable].ylim)

                    zero = np.zeros((t.shape[0], 1))
                    plot_type = self.plot_func[variable][i].type
                    if plot_type == PlotType.PLOT:
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:green"
                        self.plots.append(
                            [plot_type, i, ax.plot(t, zero, ".-", color=color, markersize=3, zorder=0)[0]]
                        )
                    elif plot_type == PlotType.INTEGRATED:
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:brown"
                        plots_integrated = []
                        nb_int_steps = nlp["nb_integration_steps"]
                        for cmp in range(nlp["ns"]):
                            plots_integrated.append(
                                ax.plot(
                                    self.t_integrated[i][cmp],
                                    np.zeros(nb_int_steps + 1),
                                    "-",
                                    color=color,
                                    markersize=3,
                                    linewidth=0.8,
                                )[0]
                            )
                        self.plots.append([plot_type, i, plots_integrated])

                    elif plot_type == PlotType.STEP:
                        color = self.plot_func[variable][i].color if self.plot_func[variable][i].color else "tab:orange"
                        self.plots.append([plot_type, i, ax.step(t, zero, where="post", color=color, zorder=0)[0]])
                    else:
                        raise RuntimeError(f"{plot_type} is not implemented yet")

                for ax in axes:
                    intersections_time = self.find_phases_intersections()
                    for time in intersections_time:
                        self.plots_vertical_lines.append(ax.axvline(time, linestyle="--", linewidth=1.2, c="k"))

    def __add_new_axis(self, variable, nb, nb_rows, nb_cols):
        """
        Sets the axis of the plots.
        :param variable: Variable to plot (integer)
        :param nb: Number of the figure. ?? (integer)
        :param nb_rows: Number of rows of plots in subplots. (integer)
        :param nb_cols: Number of columns of plots in subplots. (integer)
        :return: axes: Axes of the plots. (instance of subplot class)
        """
        if self.automatically_organize:
            self.all_figures.append(plt.figure(variable, figsize=(self.width_step / 100, self.height_step / 131)))
        else:
            self.all_figures.append(plt.figure(variable))
        axes = self.all_figures[-1].subplots(nb_rows, nb_cols)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for i in range(nb, len(axes)):
            axes[i].remove()
        axes = axes[:nb]

        idx_center = nb_rows * nb_cols - int(nb_cols / 2) - 1
        if idx_center >= len(axes):
            idx_center = len(axes) - 1
        axes[idx_center].set_xlabel("time (s)")

        self.all_figures[-1].tight_layout()
        return axes

    def _organize_windows(self, nb_windows):
        """
        Organizes esthetically the figure.
        :param nb_windows: Number of variables to plot. (integer)
        """
        self.nb_vertical_windows, self.nb_horizontal_windows = PlotOcp._generate_windows_size(nb_windows)
        if self.automatically_organize:
            height = tkinter.Tk().winfo_screenheight()
            width = tkinter.Tk().winfo_screenwidth()
            self.top_margin = height / 15
            self.height_step = (height - self.top_margin) / self.nb_horizontal_windows
            self.width_step = width / self.nb_vertical_windows
        else:
            self.top_margin = None
            self.height_step = None
            self.width_step = None

    def find_phases_intersections(self):
        """Finds the intersection between phases"""
        return list(accumulate(self.tf))[:-1]

    @staticmethod
    def show():
        plt.show()

    def update_data(self, V):
        """Update of the variable V to plot (dependent axis)"""
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
            step_size = nlp["nb_integration_steps"] + 1
            nb_elements = nlp["ns"] * step_size + 1

            state = np.ndarray((0, nb_elements))
            for s in nlp["var_states"]:
                if isinstance(data_states_per_phase[s], (list, tuple)):
                    state = np.concatenate((state, data_states_per_phase[s][i]))
                else:
                    state = np.concatenate((state, data_states_per_phase[s]))
            control = np.ndarray((0, nlp["ns"] + 1))
            for s in nlp["var_controls"]:
                if isinstance(data_controls_per_phase[s], (list, tuple)):
                    control = np.concatenate((control, data_controls_per_phase[s][i]))
                else:
                    control = np.concatenate((control, data_controls_per_phase[s]))

            if nlp["control_type"] == ControlType.CONSTANT:
                u_mod = 1
            elif nlp["control_type"] == ControlType.LINEAR_CONTINUOUS:
                u_mod = 2
            else:
                raise NotImplementedError(f"Plotting {nlp['control_type']} is not implemented yet")

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
                    y[:, :] = self.plot_func[key][i].function(state[:, ::step_size], control, data_param_in_dyn)
                    self.__append_to_ydata(y)
        self.__update_axes()

    def __update_xdata(self):
        """Update of the time in plots (independent axis)"""
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

        intersections_time = self.find_phases_intersections()
        n = len(intersections_time)
        if n > 0:
            for p in range(int(len(self.plots_vertical_lines) / n)):
                for i, time in enumerate(intersections_time):
                    self.plots_vertical_lines[p * n + i].set_xdata([time, time])

    def __append_to_ydata(self, data):
        for y in data:
            self.ydata.append(y)

    def __update_axes(self):
        """Updates axes ranges"""
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
            for i, ax in enumerate(self.axes[key][1]):
                if not self.axes[key][0].ylim:
                    y_max = -np.inf
                    y_min = np.inf
                    for p in ax.get_children():
                        if isinstance(p, lines.Line2D):
                            y_min = min(y_min, np.min(p.get_ydata()))
                            y_max = max(y_max, np.max(p.get_ydata()))
                    if np.isnan(y_min) or np.isinf(y_min):
                        y_min = 0
                    if np.isnan(y_max) or np.isinf(y_max):
                        y_max = 1
                    data_mean = np.mean((y_min, y_max))
                    data_range = y_max - y_min
                    if np.abs(data_range) < 0.8:
                        data_range = 0.8
                    y_range = (1.25 * data_range) / 2
                    y_range = data_mean - y_range, data_mean + y_range
                    ax.set_ylim(y_range)
                    ax.set_yticks(np.arange(y_range[0], y_range[1], step=data_range / 4))

        for p in self.plots_vertical_lines:
            p.set_ydata((0, 1))

    @staticmethod
    def _generate_windows_size(nb):
        """
        Defines the number of column and rows of subplots in function of the number of variables to plot.
        :param nb: Number of variables to plot. (integer)
        :return: nb_rows: Number of rows of subplot. (integer)
        """
        nb_rows = int(round(np.sqrt(nb)))
        return nb_rows + 1 if nb_rows * nb_rows < nb else nb_rows, nb_rows


class ShowResult:
    def __init__(self, ocp, sol):
        self.ocp = ocp
        self.sol = sol

    def graphs(self, automatically_organize=True):
        plot_ocp = PlotOcp(self.ocp, automatically_organize=automatically_organize)
        plot_ocp.update_data(self.sol["x"])
        plt.show()

    def animate(self, nb_frames=80, **kwargs):
        """
        Animate solution with BiorbdViz
        :param nb_frames: Number of frames in the animation. (integer)
        """
        try:
            import BiorbdViz
        except ModuleNotFoundError:
            raise RuntimeError("BiorbdViz must be install to animate the model")
        check_version(BiorbdViz, "1.3.3", "1.4.0")
        data_interpolate, data_control = Data.get_data(
            self.ocp, self.sol["x"], integrate=False, interpolate_nb_frames=nb_frames
        )
        if not isinstance(data_interpolate["q"], (list, tuple)):
            data_interpolate["q"] = [data_interpolate["q"]]

        all_bioviz = []
        for idx_phase, data in enumerate(data_interpolate["q"]):
            all_bioviz.append(BiorbdViz.BiorbdViz(loaded_model=self.ocp.nlp[idx_phase]["model"], **kwargs))
            all_bioviz[-1].load_movement(self.ocp.nlp[idx_phase]["q_mapping"].expand.map(data))

        b_is_visible = [True] * len(all_bioviz)
        while sum(b_is_visible):
            for i, b in enumerate(all_bioviz):
                if b.vtk_window.is_active:
                    b.update()
                else:
                    b_is_visible[i] = False

    @staticmethod
    def keep_matplotlib():
        plt.figure(figsize=(0.01, 0.01)).canvas.manager.window.move(1000, 100)
        plt.show()


class OnlineCallback(Callback):
    def __init__(self, ocp, opts={}):
        Callback.__init__(self)
        self.nlp = ocp
        self.nx = ocp.V.rows()
        self.ng = 0
        self.construct("AnimateCallback", opts)

        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = self.ProcessPlotter(ocp)
        self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    @staticmethod
    def get_n_in():
        return nlpsol_n_out()

    @staticmethod
    def get_n_out():
        return 1

    @staticmethod
    def get_name_in(i):
        return nlpsol_out(i)

    @staticmethod
    def get_name_out(_):
        return "ret"

    def get_sparsity_in(self, i):
        n = nlpsol_out(i)
        if n == "f":
            return Sparsity.scalar()
        elif n in ("x", "lam_x"):
            return Sparsity.dense(self.nx)
        elif n in ("g", "lam_g"):
            return Sparsity.dense(self.ng)
        else:
            return Sparsity(0, 0)

    def eval(self, arg):
        send = self.plot_pipe.send
        send(arg[0])
        return [0]

    class ProcessPlotter(object):
        def __init__(self, ocp):
            self.ocp = ocp

        def __call__(self, pipe):
            self.pipe = pipe
            self.plot = PlotOcp(self.ocp)
            timer = self.plot.all_figures[0].canvas.new_timer(interval=100)
            timer.add_callback(self.callback)
            timer.start()

            plt.show()

        def callback(self):
            while self.pipe.poll():
                V = self.pipe.recv()
                self.plot.update_data(V)
                Iterations.save(V)
            for i, fig in enumerate(self.plot.all_figures):
                fig.canvas.draw()
            return True


class Iterations:
    @staticmethod
    def save(V):
        file_path = ".__tmp_biorbd_optim/temp_save_iter.bobo"
        if os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                previews_iterations = pickle.load(file)
            previews_iterations.append(np.array(V))
            with open(file_path, "wb") as file:
                pickle.dump(previews_iterations, file)
