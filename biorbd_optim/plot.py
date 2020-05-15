import os
import multiprocessing as mp
import numpy as np
import tkinter
from itertools import accumulate

import matplotlib

if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using non-interactive Agg backend")
    matplotlib.use("Agg")
from matplotlib import pyplot as plt, lines
from casadi import MX, Callback, nlpsol_out, nlpsol_n_out, Sparsity

from .variable_optimization import Data
from .mapping import Mapping


class CustomPlot:
    def __init__(self, size, update_function, phase_mappings=None, legend=()):
        self.size = size
        self.function = update_function
        self.phase_mappings = Mapping(range(size)) if phase_mappings is None else phase_mappings
        self.legend = legend


class PlotOcp:
    def __init__(self, ocp):
        for i in range(1, ocp.nb_phases):
            if ocp.nlp[0]["nbQ"] != ocp.nlp[i]["nbQ"]:
                raise RuntimeError("Graphs with nbQ different at each phase is not implemented yet")

        self.ocp = ocp
        self.ydata = []
        self.ns = 0

        self.t = []
        self.t_per_phase = []
        self.t_integrated = []
        if isinstance(self.ocp.initial_phase_time, (int, float)):
            self.tf = [self.ocp.initial_phase_time]
        else:
            self.tf = list(self.ocp.initial_phase_time)
        self.t_idx_to_optimize = []
        for i, nlp in enumerate(self.ocp.nlp):
            if isinstance(nlp["tf"], MX):
                self.t_idx_to_optimize.append(i)
        self.__init_time_vector()

        self.axes = []
        self.plots = []
        self.plots_vertical_lines = []
        self.all_figures = []

        running_cmp = 0
        self.matching_mapping = dict()
        for state in ocp.nlp[0]["var_states"]:
            if state in ocp.nlp[0]["var_controls"]:
                self.matching_mapping[state] = running_cmp
            running_cmp += ocp.nlp[0]["var_states"][state]
        self._organize_windows(
            len(self.ocp.nlp[0]["var_states"]) + len(self.ocp.nlp[0]["var_controls"]) - len(self.matching_mapping)
        )

        self.plot_func = {}
        self.variable_sizes = {}
        self.__create_plots(ocp.nlp[0]["var_states"], "state")
        self.__create_plots(ocp.nlp[0]["var_controls"], "control")
        self.__create_custom_plots(ocp.nlp, "custom_plots")

        horz, vert = 0, 0
        for i, fig in enumerate(self.all_figures):
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
        self.t = [0]
        self.t_per_phase = []
        self.t_integrated = []
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            self.ns += nlp["ns"] + 1
            time_phase = np.linspace(self.t[-1], self.t[-1] + self.tf[phase_idx], nlp["ns"] + 1)
            self.t_per_phase.append(time_phase)
            self.t = np.append(self.t, time_phase)
            self.t_integrated = np.append(self.t_integrated, PlotOcp.generate_integrated_time(time_phase))
        self.t = self.t[1:]

    def __create_plots(self, has, var_type):
        for variable in has:
            nb = has[variable]

            nb_cols, nb_rows = PlotOcp._generate_windows_size(nb)
            if var_type == "control" and variable in self.matching_mapping:
                axes = self.axes[self.matching_mapping[variable] : self.matching_mapping[variable] + nb]
            else:
                axes = self.__add_new_axis(variable, nb, nb_rows, nb_cols)

            for i, ax in enumerate(axes):
                if var_type == "state":
                    cmp = 0
                    plots = []
                    for idx_phase in range(self.ocp.nb_phases):
                        for _ in range(self.ocp.nlp[idx_phase]["ns"]):
                            plots.append(
                                ax.plot(
                                    self.t_integrated[2 * cmp + idx_phase : 2 * (cmp + 1) + idx_phase],
                                    np.zeros(2),
                                    color="tab:brown",
                                    linewidth=0.8,
                                )[0]
                            )
                            plots.append(
                                ax.plot(
                                    self.t_integrated[2 * cmp + idx_phase],
                                    np.zeros(1),
                                    color="tab:brown",
                                    marker=".",
                                    markersize=6,
                                )[0]
                            )
                            cmp += 1
                    self.plots.append(plots)
                elif var_type == "control":
                    self.plots.append(
                        ax.step(self.t, np.zeros((self.ns, 1)), where="post", color="tab:orange", zorder=0)
                    )
                else:
                    raise RuntimeError("Plot of parameters is not supported yet")

                intersections_time = self.find_phases_intersections()
                for time in intersections_time:
                    self.plots_vertical_lines.append(ax.axvline(time, linestyle="--", linewidth=1.2, c="k"))
                ax.grid(color="k", linestyle="--", linewidth=0.5)
                ax.set_xlim(0, self.t[-1])

    def __create_custom_plots(self, all_nlp, var_type):
        variable_sizes = {}
        for nlp in all_nlp:
            if var_type in nlp:
                for key in nlp[var_type]:
                    if key not in variable_sizes:
                        variable_sizes[key] = nlp[var_type][key].size
                    else:
                        variable_sizes[key] = max(variable_sizes[key], nlp[var_type][key].size)
        self.variable_sizes[var_type] = variable_sizes
        if not variable_sizes:
            return

        self.plot_func[var_type] = []
        for variable in variable_sizes:
            nb = variable_sizes[variable]
            nb_cols, nb_rows = PlotOcp._generate_windows_size(nb)
            axes = self.__add_new_axis(variable, nb, nb_rows, nb_cols)

            plots = []
            for i, nlp in enumerate(all_nlp):
                t = self.t_per_phase[i]
                if variable in all_nlp[i][var_type]:
                    self.plot_func[var_type].append([variable, all_nlp[i][var_type][variable]])
                else:
                    pass

                for k, ax in enumerate(axes):
                    mapping = self.plot_func[var_type][-1][1].phase_mappings.map_idx
                    if k < len(mapping):
                        axes[k].set_title(self.plot_func[var_type][-1][1].legend[mapping[k]])
                    ax.grid(color="k", linestyle="--", linewidth=0.5)
                    ax.set_xlim(0, self.t[-1])
                    if var_type == "custom_plots":
                        plots.append(ax.plot(t, np.zeros((t.shape[0], 1)), ".-", color="tab:green", zorder=0))

            for ax in axes:
                intersections_time = self.find_phases_intersections()
                for time in intersections_time:
                    self.plots_vertical_lines.append(ax.axvline(time, linestyle="--", linewidth=1.2, c="k"))

            self.plots.extend(plots)

    def __add_new_axis(self, variable, nb, nb_rows, nb_cols):
        self.all_figures.append(plt.figure(variable, figsize=(self.width_step / 100, self.height_step / 131)))
        axes = self.all_figures[-1].subplots(nb_rows, nb_cols)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for i in range(nb, len(axes)):
            axes[i].remove()
        axes = axes[:nb]

        for k in range(nb):
            if "q" in variable or "q_dot" in variable or "tau" in variable:
                mapping = self.ocp.nlp[0][f"{variable}_mapping"].expand.map_idx
                axes[k].set_title(self.ocp.nlp[0]["model"].nameDof()[mapping[k]].to_string())
            elif "muscles" in variable:
                axes[k].set_title(self.ocp.nlp[0]["model"].muscleNames()[k].to_string())
        idx_center = nb_rows * nb_cols - int(nb_cols / 2) - 1
        if idx_center >= len(axes):
            idx_center = len(axes) - 1
        axes[idx_center].set_xlabel("time (s)")

        self.axes.extend(axes)
        self.all_figures[-1].tight_layout()
        return axes

    def _organize_windows(self, nb_windows):
        height = tkinter.Tk().winfo_screenheight()
        width = tkinter.Tk().winfo_screenwidth()
        self.nb_vertical_windows, nb_horizontal_windows = PlotOcp._generate_windows_size(nb_windows)
        self.top_margin = height / 15
        self.height_step = (height - self.top_margin) / nb_horizontal_windows
        self.width_step = width / self.nb_vertical_windows

    @staticmethod
    def generate_integrated_time(t):
        for i in range(len(t) - 1, 0, -1):
            t = np.insert(t, i, t[i])
        return t

    def find_phases_intersections(self):
        return list(accumulate(self.tf))[:-1]

    @staticmethod
    def show():
        plt.show()

    def update_data(self, V):
        self.ydata = [[] for _ in range(self.ocp.nb_phases)]

        data_states, data_controls, data_param = Data.get_data(
            self.ocp, V, get_parameters=True, integrate=True, concatenate=False
        )

        for i, nlp in enumerate(self.ocp.nlp):
            if self.t_idx_to_optimize:
                for i_in_time, i_in_tf in enumerate(self.t_idx_to_optimize):
                    self.tf[i_in_tf] = data_param["time"][i_in_time]
                self.__update_xdata()
            self.__update_ydata(data_states, i)
            self.__update_ydata(data_controls, i)

        if "custom_plots" in self.plot_func:
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
                plot = self.plot_func["custom_plots"][i]
                y = {"y": np.empty((self.variable_sizes["custom_plots"][plot[0]], len(self.t_per_phase[i])))}
                y["y"].fill(np.nan)
                y["y"][plot[1].phase_mappings.map_idx, :] = plot[1].function(state, control)
                self.__update_ydata(y, 0)
        self.__update_axes()

    def __update_xdata(self):
        self.__init_time_vector()
        for i, p in enumerate(self.plots):
            if i < self.ocp.nlp[0]["nx"]:
                for j in range(int(len(p) / 2)):
                    p[2 * j].set_xdata(self.t_integrated[j * 2 : 2 * j + 2])
                    p[2 * j + 1].set_xdata(self.t_integrated[j * 2])
            else:
                p[0].set_xdata(self.t)
            ax = p[0].axes
            ax.set_xlim(0, self.t[-1])

        intersections_time = self.find_phases_intersections()
        n = len(intersections_time)
        if n > 0:
            for p in range(int(len(self.plots_vertical_lines) / n)):
                for i, time in enumerate(intersections_time):
                    self.plots_vertical_lines[p * n + i].set_xdata([time, time])

    def __update_ydata(self, data, phase_idx):
        for key in data:
            y_data = data[key]
            if not isinstance(y_data, (tuple, list)):
                y_data = [y_data]

            for y in y_data[phase_idx]:
                self.ydata[phase_idx].append(y)

    def __update_axes(self):
        for i, p in enumerate(self.plots):
            ax = p[0].axes
            y = np.array([])
            y_per_phase = []
            for phase in self.ydata:
                # TODO: To be removed when phases are directly in ydata (as for custom_plots)
                if i < len(phase):
                    y = np.append(y, phase[i])
                    y_per_phase.append(phase[i])
            if not y.any():
                continue

            if i < self.ocp.nlp[0]["nx"]:
                cmp = 0
                for idx_phase in range(self.ocp.nb_phases):
                    for _ in range(self.ocp.nlp[idx_phase]["ns"]):
                        p[2 * cmp].set_ydata(y[2 * cmp + idx_phase : 2 * (cmp + 1) + idx_phase])
                        p[2 * cmp + 1].set_ydata(y[2 * cmp + idx_phase])
                        cmp += 1
            elif i >= self.ocp.nlp[0]["nx"] + self.ocp.nlp[0]["nu"]:
                for idx_phase in range(len(self.t_per_phase)):
                    # TODO: To be removed when phases are directly in ydata (as for custom_plots)
                    if idx_phase < len(y_per_phase):
                        p[idx_phase].set_ydata(y_per_phase[idx_phase])

            else:
                p[0].set_ydata(y)

        for p in self.plots_vertical_lines:
            p.set_ydata((np.nan, np.nan))

        for i, ax in enumerate(self.axes):
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
            data_range = y_max - y_min
            if data_range == 0:
                data_range = 1
            mean = data_range / 2 + y_min
            y_range = (1.3 * data_range) / 2
            y_range = mean - y_range, mean + y_range
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

    def graphs(self):
        plot_ocp = PlotOcp(self.ocp)
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
        self.ng = ocp.g.rows()
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
