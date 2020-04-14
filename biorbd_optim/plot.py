import multiprocessing as mp

from matplotlib import pyplot as plt
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity
import numpy as np

from .problem_type import ProblemType


class PlotOcp:
    def __init__(self, ocp):
        self.ocp = ocp
        self.ns_per_phase = [nlp["ns"] + 1 for nlp in ocp.nlp]
        self.ydata = []
        self.ns = 0

        self.problem_type = None
        for i, nlp in enumerate(self.ocp.nlp):
            if self.problem_type is None:
                self.problem_type = nlp["problem_type"]

            if i == 0:
                self.t = np.linspace(0, nlp["tf"], nlp["ns"] + 1)
            else:
                self.t = np.append(
                    self.t,
                    np.linspace(self.t[-1], self.t[-1] + nlp["tf"], nlp["ns"] + 1),
                )
            self.ns += nlp["ns"] + 1

        if self.problem_type == ProblemType.torque_driven_no_contact:
            for i in range(self.ocp.nb_phases):
                if self.ocp.nlp[0]["nbQ"] != self.ocp.nlp[i]["nbQ"]:
                    raise RuntimeError(
                        "Graphs with nbQ different at each phase is not implemented yet"
                    )
            nlp = self.ocp.nlp[0]

            self.fig_state, self.axes = plt.subplots(3, nlp["nbQ"], figsize=(10, 6))
            self.axes = self.axes.flatten()
            mid_column_idx = int(nlp["nbQ"] / 2)
            self.axes[mid_column_idx].set_title("q")
            self.axes[nlp["nbQ"] + mid_column_idx].set_title("q_dot")
            self.axes[nlp["nbQ"] + nlp["nbQdot"] + mid_column_idx].set_title("tau")
            self.axes[nlp["nbQ"] + nlp["nbQdot"] + mid_column_idx].set_xlabel(
                "time (s)"
            )
        else:
            raise RuntimeError("Plot is not ready for this type of problem")

        for i, ax in enumerate(self.axes):
            if i < self.ocp.nlp[0]["nx"]:
                ax.plot(self.t, np.zeros((self.ns, 1)))
            else:
                ax.step(self.t, np.zeros((self.ns, 1)), where="post")

            intersections_time = PlotOcp.find_phases_intersections(ocp)
            for time in intersections_time:
                ax.axvline(time, linestyle="--", linewidth=1.2, c="k")
            ax.grid(color="k", linestyle="--", linewidth=0.5)
            ax.set_xlim(0, self.t[-1])

        plt.tight_layout()

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
            if (
                self.problem_type == ProblemType.torque_driven_no_contact
                or self.problem_type == ProblemType.muscles_and_torque_driven
            ):
                if self.problem_type == ProblemType.torque_driven_no_contact:
                    q, q_dot, tau = ProblemType.get_data_from_V(self.ocp, V, i)

                elif self.problem_type == ProblemType.muscles_and_torque_driven:
                    q, q_dot, tau, muscle = ProblemType.get_data_from_V(self.ocp, V, i)
                    self.__update_ydata(muscle, nlp["nbMuscle"], i)

                self.__update_ydata(q, nlp["nbQ"], i)
                self.__update_ydata(q_dot, nlp["nbQdot"], i)
                self.__update_ydata(tau, nlp["nbTau"], i)
        self.__update_axes()

    def __update_ydata(self, array, nb_variables, phase_idx):
        for i in range(nb_variables):
            self.ydata[phase_idx].append(array[i, :])

    def __update_axes(self):
        for i in range(len(self.ydata[0])):
            y = np.array([])
            for p in self.ydata:
                y = np.append(y, p[i])

            y_range = np.max([np.max(y) - np.min(y), 0.5])
            mean = y_range / 2 + np.min(y)
            axe_range = (1.1 * y_range) / 2
            self.axes[i].set_ylim(mean - axe_range, mean + axe_range)
            self.axes[i].set_yticks(
                np.arange(
                    np.round(mean - axe_range, 1),
                    np.round(mean + axe_range, 1),
                    step=np.round((mean + axe_range - (mean - axe_range)) / 4, 1),
                )
            )
            self.axes[i].get_lines()[0].set_ydata(y)


class AnimateCallback(Callback):
    def __init__(self, ocp, opts={}):
        Callback.__init__(self)
        self.nlp = ocp
        self.nx = ocp.V.rows()
        self.ng = ocp.g.rows()
        self.construct("AnimateCallback", opts)

        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = self.ProcessPlotter(ocp)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True
        )
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

            timer = self.plot.fig_state.canvas.new_timer(interval=100)
            timer.add_callback(self.callback)
            timer.start()
            plt.show()

        def callback(self):
            while self.pipe.poll():
                V = self.pipe.recv()
                self.plot.update_data(V)

            self.plot.fig_state.canvas.draw()
            return True
