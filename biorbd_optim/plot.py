import multiprocessing as mp

from matplotlib import pyplot as plt
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity
import numpy as np

from .problem_type import ProblemType


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
            self.ns = 0

            self.problem_type = None
            for i, nlp in enumerate(self.ocp.nlp):
                if self.problem_type is None:
                    self.problem_type = nlp["problem_type"]

                if i == 0:
                    self.t = np.linspace(0, nlp["tf"], nlp["ns"] + 1)
                else:
                    self.t = np.append(self.t, np.linspace(self.t[-1], self.t[-1] + nlp["tf"], nlp["ns"] + 1))
                self.ns += nlp["ns"] + 1

            if self.problem_type == ProblemType.torque_driven:
                for i in range(self.ocp.nb_phases):
                    if self.ocp.nlp[0]["nbQ"] != self.ocp.nlp[i]["nbQ"]:
                        raise RuntimeError("Graphs with nbQ different at each phase is not implemented yet")
                nlp = self.ocp.nlp[0]
                self.nx = nlp["nx"]
                self.nu = nlp["nu"]
                self.nbQ = nlp["nbQ"]
                self.nbQdot = nlp["nbQdot"]
                self.nbTau = nlp["nbTau"]
                self.nbMuscle = nlp["nbMuscle"]

                self.fig_state, self.axes = plt.subplots(
                    3, self.nbQ, figsize=(10, 6)
                )
                self.axes = self.axes.flatten()
                mid_column_idx = int(self.nbQ / 2)
                self.axes[mid_column_idx].set_title("q")
                self.axes[self.nbQ + mid_column_idx].set_title("q_dot")
                self.axes[self.nbQ + self.nbQdot + mid_column_idx].set_title(
                    "tau"
                )
                self.axes[self.nbQ + self.nbQdot + mid_column_idx].set_xlabel(
                    "time (s)"
                )
            else:
                raise RuntimeError("Plot is not ready for this type of problem")

            for i, ax in enumerate(self.axes):
                if i < self.nx:
                    ax.plot(self.t, np.zeros((self.ns, 1)))
                else:
                    ax.step(self.t, np.zeros((self.ns, 1)), where="post")
                ax.grid(color="k", linestyle="--", linewidth=0.5)
                ax.set_xlim(0, self.t[-1])

            timer = self.fig_state.canvas.new_timer(interval=100)
            timer.add_callback(self.call_back)
            timer.start()

            plt.tight_layout()
            plt.show()

        def call_back(self):
            while self.pipe.poll():
                arg = self.pipe.recv()
                if self.problem_type == ProblemType.torque_driven:
                    q, q_dot, tau = self.__get_data(arg)
                    for i in range(self.nlp.nbQ):
                        self.__update_plot(i, q)
                        self.__update_plot(i + self.nlp.nbQ, q_dot)
                        self.__update_plot(i + self.nlp.nbQ + self.nlp.nbQdot, tau)
                elif self.nlp.problem_type == ProblemType.muscles_and_torque_driven:
                    q, q_dot, tau, muscle = self.__get_data(arg)
                    for i in range(self.nlp.nbQ):
                        self.__update_plot(i, q)
                        self.__update_plot(i + self.nlp.nbQ, q_dot)
                        self.__update_plot(i + self.nlp.nbQ + self.nlp.nbQdot, tau)
                        # self.__update_plot(i + self.nlp.nbQ + self.nlp.nbQdot + self.nlp.nbTau, muscle)
            self.fig_state.canvas.draw()
            return True

        def __get_data(self, V):
            if (
                self.problem_type == ProblemType.torque_driven
                or self.problem_type == ProblemType.muscles_and_torque_driven
            ):
                q = np.ndarray((self.ns, self.nbQ))
                q_dot = np.ndarray((self.ns, self.nbQdot))
                tau = np.ndarray((self.ns, self.nbTau))
                for idx in range(self.nbQ):
                    q[:, idx] = np.array(V[idx :: self.nx + self.nu]).squeeze()
                    q_dot[:, idx] = np.array(V[self.nbQ + idx :: self.nx + self.nu]).squeeze()
                    tau[: self.ns, idx] = np.array(
                        V[self.ns + idx :: self.nx + self.nu]
                    )
                tau[-1, :] = tau[-2, :]
                if self.problem_type == ProblemType.muscles_and_torque_driven:
                    muscle = np.ndarray((self.ns + self.ocp.nb_phases, self.nbMuscle))
                    for idx in range(self.nbMuscle):
                        muscle[: self.ns, :] = np.array(
                            V[self.ns + self.nbTau + idx :: self.nx + self.nu]
                        )
                    muscle[-1, :] = muscle[-2, :]
                    return q, q_dot, tau, muscle
                else:
                    return q, q_dot, tau
            else:
                raise RuntimeError(
                    "plot.__get_data not implemented yet for this problem_type"
                )

        def __update_plot(self, i, y):
            y_range = np.max(y) - np.min(y)
            y_mean = y_range / 2 + np.min(y)
            y_range = (np.max([y_range, np.pi / 2]) + y_range * 0.1) / 2
            self.axes[i].set_ylim(y_mean - y_range, y_mean + y_range)
            self.axes[i].set_yticks(
                np.arange(
                    np.round(y_mean - y_range, 1),
                    np.round(y_mean + y_range, 1),
                    step=np.round((y_mean + y_range - (y_mean - y_range)) / 4, 1),
                )
            )
            self.axes[i].get_lines()[0].set_ydata(np.array(y))
