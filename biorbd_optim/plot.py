import multiprocessing as mp
import numpy as np
import tkinter

from matplotlib import pyplot as plt
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity

from .variable_optimization import Data


class PlotOcp:
    def __init__(self, ocp):
        for i in range(1, ocp.nb_phases):
            if ocp.nlp[0]["nbQ"] != ocp.nlp[i]["nbQ"]:
                raise RuntimeError("Graphs with nbQ different at each phase is not implemented yet")

        self.ocp = ocp
        self.ydata = []
        self.ns = 0

        self.t = [0]
        self.t_integrated = []
        for nlp in self.ocp.nlp:
            self.ns += nlp["ns"] + 1
            time_phase = np.linspace(self.t[-1], self.t[-1] + nlp["tf"], nlp["ns"] + 1)
            self.t = np.append(self.t, time_phase)
            self.t_integrated = np.append(self.t_integrated, PlotOcp.generate_integrated_time(time_phase))
        self.t = self.t[1:]

        self.axes = []
        self.plots = []
        self.all_figures = []

        running_cmp = 0
        self.matching_variables = dict()
        for state in ocp.nlp[0]["has_states"]:
            if state in ocp.nlp[0]["has_controls"]:
                self.matching_variables[state] = running_cmp
            running_cmp += ocp.nlp[0]["has_states"][state]
        self._organize_windows(
            len(self.ocp.nlp[0]["has_states"]) + len(self.ocp.nlp[0]["has_controls"]) - len(self.matching_variables)
        )

        self.create_plots(ocp.nlp[0]["has_states"], "state")
        self.create_plots(ocp.nlp[0]["has_controls"], "control")

        horz, vert = 0, 0
        for i, fig in enumerate(self.all_figures):
            fig.canvas.manager.window.move(int(vert * self.width_step), int(self.top_margin + horz * self.height_step))
            vert += 1
            if vert >= self.nb_vertical_windows:
                horz += 1
                vert = 0
            fig.canvas.draw()

    def create_plots(self, has, var_type):
        for variable in has:
            nb = has[variable]
            nb_cols, nb_rows = PlotOcp._generate_windows_size(nb)
            if var_type == "control" and variable in self.matching_variables:
                axes = self.axes[self.matching_variables[variable] : self.matching_variables[variable] + nb]
            else:
                self.all_figures.append(plt.figure(variable, figsize=(self.width_step / 100, self.height_step / 131)))
                axes = self.all_figures[-1].subplots(nb_rows, nb_cols).flatten()
                for i in range(nb, len(axes)):
                    axes[i].remove()
                axes = axes[:nb]

                for k in range(nb):
                    if "q" in variable or "q_dot" in variable or "tau" in variable:
                        axes[k].set_title(self.ocp.nlp[0]["model"].nameDof()[k].to_string())
                    elif "muscles" in variable:
                        axes[k].set_title(self.ocp.nlp[0]["model"].muscleNames()[k].to_string())
                axes[nb_rows * nb_cols - int(nb_cols / 2) - 1].set_xlabel("time (s)")

                self.axes.extend(axes)
                self.all_figures[-1].tight_layout()

            intersections_time = PlotOcp.find_phases_intersections(self.ocp)
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
                                    color="g",
                                    linewidth=0.8,
                                )[0]
                            )
                            plots.append(
                                ax.plot(
                                    self.t_integrated[2 * cmp + idx_phase],
                                    np.zeros(1),
                                    color="g",
                                    marker=".",
                                    markersize=6,
                                )[0]
                            )
                            cmp += 1
                    self.plots.append(plots)
                elif var_type == "control":
                    self.plots.append(ax.step(self.t, np.zeros((self.ns, 1)), where="post", color="r"))
                else:
                    raise RuntimeError("Plot of parameters is not supported yet")

                for time in intersections_time:
                    ax.axvline(time, linestyle="--", linewidth=1.2, c="k")
                ax.grid(color="k", linestyle="--", linewidth=0.5)
                ax.set_xlim(0, self.t[-1])

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

    @staticmethod
    def find_phases_intersections(ocp):
        intersections_time = []
        time = 0
        for i in range(len(ocp.nlp) - 1):
            time += ocp.nlp[i]["tf"]
            intersections_time.append(time)
        return intersections_time

    @staticmethod
    def show():
        plt.show()

    def update_data(self, V):
        self.ydata = [[] for _ in range(self.ocp.nb_phases)]
        for i, nlp in enumerate(self.ocp.nlp):
            data_states, data_controls = Data.get_data_from_V(self.ocp, V, integrate=True)
            for key in data_states:
                self.__update_ydata(data_states[key], i)
            for key in data_controls:
                self.__update_ydata(data_controls[key], i)
        self.__update_axes()

    def __update_ydata(self, data, phase_idx):
        for i in range(data.nb_elements):
            d = data.to_matrix(idx=i, phases=phase_idx)
            self.ydata[phase_idx].append(d)

    def __update_axes(self):
        for i, p in enumerate(self.plots):
            ax = p[0].axes
            y = np.array([])
            for phase in self.ydata:
                y = np.append(y, phase[i])

            y_range = np.max([np.max(y) - np.min(y), 0.5])
            mean = y_range / 2 + np.min(y)
            axe_range = (1.1 * y_range) / 2
            ax.set_ylim(mean - axe_range, mean + axe_range)
            ax.set_yticks(
                np.arange(
                    np.round(mean - axe_range, 1),
                    np.round(mean + axe_range, 1),
                    step=np.round((mean + axe_range - (mean - axe_range)) / 4, 1),
                )
            )
            if i < self.ocp.nlp[0]["nx"]:
                cmp = 0
                for idx_phase in range(self.ocp.nb_phases):
                    for _ in range(self.ocp.nlp[idx_phase]["ns"]):
                        p[2 * cmp].set_ydata(y[2 * cmp + idx_phase : 2 * (cmp + 1) + idx_phase])
                        p[2 * cmp + 1].set_ydata(y[2 * cmp + idx_phase])
                        cmp += 1

            else:
                p[0].set_ydata(y)

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
        data_interpolate, data_control = Data.get_data_from_V(
            self.ocp, self.sol["x"], integrate=False, interpolate_nb_frames=nb_frames
        )

        all_bioviz = []
        for idx_phase, d in enumerate(data_interpolate["q"].phase):
            all_bioviz.append(BiorbdViz(loaded_model=self.ocp.nlp[idx_phase]["model"], **kwargs))
            all_bioviz[-1].load_movement(d.T)

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
