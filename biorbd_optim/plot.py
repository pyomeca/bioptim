import multiprocessing as mp
import numpy as np
import tkinter
from itertools import accumulate

from matplotlib import pyplot as plt, lines
from casadi import MX, Callback, nlpsol_out, nlpsol_n_out, Sparsity

from .variable_optimization import Data
from .mapping import Mapping
from .enums import PlotType


class CustomPlot:
    def __init__(self, update_function, plot_type=PlotType.PLOT, axes_idx=None, legend=(), combine_to=None, color=None):
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


class PlotOcp:
    def __init__(self, ocp, automatically_organize=True):
        for i in range(1, ocp.nb_phases):
            if ocp.nlp[0]["nbQ"] != ocp.nlp[i]["nbQ"]:
                raise RuntimeError("Graphs with nbQ different at each phase is not implemented yet")

        self.ocp = ocp
        self.ydata = []
        self.ns = 0

        self.t = []
        if isinstance(self.ocp.initial_phase_time, (int, float)):
            self.tf = [self.ocp.initial_phase_time]
        else:
            self.tf = list(self.ocp.initial_phase_time)
        self.t_idx_to_optimize = []
        for i, nlp in enumerate(self.ocp.nlp):
            if isinstance(nlp["tf"], MX):
                self.t_idx_to_optimize.append(i)
        self.__init_time_vector()

        self.axes = {}
        self.plots = []
        self.plots_vertical_lines = []
        self.all_figures = []

        running_cmp = 0
        self.automatically_organize = automatically_organize
        self._organize_windows(len(self.ocp.nlp[0]["var_states"]) + len(self.ocp.nlp[0]["var_controls"]),)

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

    def __init_time_vector(self):
        self.t = []
        last_t = 0
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            self.ns += nlp["ns"] + 1
            time_phase = np.linspace(last_t, last_t + self.tf[phase_idx], nlp["ns"] + 1)
            last_t = last_t + self.tf[phase_idx]
            self.t.append(time_phase)

    def __create_plots(self):
        variable_sizes = []
        for i, nlp in enumerate(self.ocp.nlp):
            variable_sizes.append({})
            if "plot" in nlp:
                for key in nlp["plot"]:
                    if nlp["plot"][key].phase_mappings is None:
                        size = nlp["plot"][key].function(np.zeros((nlp["nx"], 1)), np.zeros((nlp["nu"], 1))).shape[0]
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
                    axes = self.axes[variable]
                elif i > 0:
                    axes = self.axes[variable]
                else:
                    axes = self.__add_new_axis(variable, nb, nb_rows, nb_cols)

                t = self.t[i]
                if variable not in self.plot_func:
                    self.plot_func[variable] = [nlp["plot"][variable]]
                else:
                    self.plot_func[variable].append(nlp["plot"][variable])

                mapping = self.plot_func[variable][-1].phase_mappings.map_idx
                for k in mapping:
                    ax = axes[k]
                    if k < len(self.plot_func[variable][-1].legend):
                        axes[k].set_title(self.plot_func[variable][-1].legend[mapping[k]])
                    ax.grid(color="k", linestyle="--", linewidth=0.5)
                    ax.set_xlim(0, self.t[-1][-1])
                    if "muscles" in variable:
                        ax.set_ylim(0, 1)

                    zero = np.zeros((t.shape[0], 1))
                    plot_type = self.plot_func[variable][0].type
                    if plot_type == PlotType.PLOT:
                        color = self.plot_func[variable][0].color if self.plot_func[variable][0].color else "tab:green"
                        self.plots.append(
                            [plot_type, i, ax.plot(t, zero, ".-", color=color, markersize=3, zorder=0)[0]]
                        )
                    elif plot_type == PlotType.INTEGRATED:
                        color = self.plot_func[variable][0].color if self.plot_func[variable][0].color else "tab:brown"
                        plots_integrated = []
                        for cmp in range(nlp["ns"]):
                            plots_integrated.append(
                                ax.plot(
                                    self.t[i][[cmp, cmp + 1]], (0, 0), ".-", color=color, markersize=3, linewidth=0.8,
                                )[0]
                            )
                        self.plots.append([plot_type, i, plots_integrated])

                    elif plot_type == PlotType.STEP:
                        color = self.plot_func[variable][0].color if self.plot_func[variable][0].color else "tab:orange"
                        self.plots.append([plot_type, i, ax.step(t, zero, where="post", color=color, zorder=0)[0]])
                    else:
                        raise RuntimeError(f"{plot_type} is not implemented yet")

            for ax in axes:
                intersections_time = self.find_phases_intersections()
                for time in intersections_time:
                    self.plots_vertical_lines.append(ax.axvline(time, linestyle="--", linewidth=1.2, c="k"))

    def __add_new_axis(self, variable, nb, nb_rows, nb_cols):
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

        self.axes[variable] = axes
        self.all_figures[-1].tight_layout()
        return axes

    def _organize_windows(self, nb_windows):
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
        return list(accumulate(self.tf))[:-1]

    @staticmethod
    def show():
        plt.show()

    def update_data(self, V):
        self.ydata = []

        data_states, data_controls, data_param = Data.get_data(
            self.ocp, V, get_parameters=True, integrate=True, concatenate=False
        )

        for _ in self.ocp.nlp:
            if self.t_idx_to_optimize:
                for i_in_time, i_in_tf in enumerate(self.t_idx_to_optimize):
                    self.tf[i_in_tf] = data_param["time"][i_in_time]
            self.__update_xdata()

        data_states_per_phase, data_controls_per_phase = Data.get_data(self.ocp, V, concatenate=False)
        for i, nlp in enumerate(self.ocp.nlp):
            state = np.ndarray((0, nlp["ns"] + 1))
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
            for key in self.variable_sizes[i]:
                y = np.empty((self.variable_sizes[i][key], len(self.t[i])))
                y.fill(np.nan)
                y[:, :] = self.plot_func[key][i].function(state, control)
                self.__append_to_ydata(y)
        self.__update_axes()

    def __update_xdata(self):
        self.__init_time_vector()
        for plot in self.plots:
            phase_idx = plot[1]
            if plot[0] == PlotType.INTEGRATED:
                for cmp, p in enumerate(plot[2]):
                    p.set_xdata(self.t[phase_idx][[cmp, cmp + 1]])
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
        for i, plot in enumerate(self.plots):
            y = self.ydata[i]

            if plot[0] == PlotType.INTEGRATED:
                for cmp, p in enumerate(plot[2]):
                    p.set_ydata(y[[cmp, cmp + 1]])
            else:
                plot[2].set_ydata(y)

        for p in self.plots_vertical_lines:
            p.set_ydata((np.nan, np.nan))

        for key in self.axes:
            for i, ax in enumerate(self.axes[key]):
                if "muscles" in key:
                    y_range = [0, 1]
                    ax.set_ylim(y_range)
                    ax.set_yticks(np.arange(y_range[0], y_range[1], step=1/5, ))
                else:
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
                    ax.set_yticks(np.arange(y_range[0], y_range[1], step=data_range / 4,))

        for p in self.plots_vertical_lines:
            p.set_ydata((0, 1))

    @staticmethod
    def _generate_windows_size(nb):
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
        try:
            from BiorbdViz import BiorbdViz
        except ModuleNotFoundError:
            raise RuntimeError("BiorbdViz must be install to animate the model")
        data_interpolate, data_control = Data.get_data(
            self.ocp, self.sol["x"], integrate=False, interpolate_nb_frames=nb_frames
        )
        if not isinstance(data_interpolate["q"], (list, tuple)):
            data_interpolate["q"] = [data_interpolate["q"]]

        all_bioviz = []
        for idx_phase, data in enumerate(data_interpolate["q"]):
            all_bioviz.append(BiorbdViz(loaded_model=self.ocp.nlp[idx_phase]["model"], **kwargs))
            all_bioviz[-1].load_movement(self.ocp.nlp[idx_phase]["q_mapping"].expand.map(data).T)

        b_is_visible = [True] * len(all_bioviz)
        while sum(b_is_visible):
            for i, b in enumerate(all_bioviz):
                if b.vtk_window.is_active:
                    if b.show_analyses_panel and b.is_animating:
                        b.movement_slider[0].setValue(
                            (b.movement_slider[0].value() + 1) % b.movement_slider[0].maximum()
                        )
                    b.refresh_window()
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

            for i, fig in enumerate(self.plot.all_figures):
                fig.canvas.draw()
            return True
