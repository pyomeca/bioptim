import multiprocessing as mp

from matplotlib import pyplot as plt
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity
import numpy as np

from .variable import Variable


class AnimateCallback(Callback):
    def __init__(self, nlp, opts={}):
        Callback.__init__(self)
        self.nlp = nlp
        self.nx = nlp.V.rows()
        self.ng = nlp.g.rows()
        self.construct("AnimateCallback", opts)

        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = self.ProcessPlotter(nlp)
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
        def __init__(self, nlp):
            self.nlp = nlp

        def __call__(self, pipe):
            self.pipe = pipe

            self.t = np.linspace(0, self.nlp.tf, self.nlp.ns + 1)
            if self.nlp.variable_type == Variable.torque_driven:
                self.fig_state, self.axes = plt.subplots(
                    3, self.nlp.nbQ, figsize=(10, 6)
                )
                self.axes = self.axes.flatten()
                mid_column_idx = int(self.nlp.nbQ / 2)
                self.axes[mid_column_idx].set_title("q")
                self.axes[self.nlp.nbQ + mid_column_idx].set_title("q_dot")
                self.axes[self.nlp.nbQ + self.nlp.nbQdot + mid_column_idx].set_title(
                    "tau"
                )
            else:
                raise RuntimeError("Plot is not ready for this type of variable")

            for i, ax in enumerate(self.axes):
                if i < self.nlp.nbQ + self.nlp.nbQdot:
                    ax.plot(self.t, np.zeros((self.nlp.ns + 1, 1)))
                else:
                    ax.step(self.t, np.zeros((self.nlp.ns + 1, 1)), where="post")
                ax.grid(color="k", linestyle="--", linewidth=0.5)
                ax.set_xlim(0, self.nlp.tf)

            if self.nlp.variable_type == Variable.torque_driven:
                self.axes[self.nlp.nbQ + self.nlp.nbQdot + mid_column_idx].set_xlabel(
                    "time (s)"
                )

            timer = self.fig_state.canvas.new_timer(interval=100)
            timer.add_callback(self.call_back)
            timer.start()

            plt.tight_layout()
            plt.show()

        def call_back(self):
            while self.pipe.poll():
                arg = self.pipe.recv()
                if self.nlp.variable_type == Variable.torque_driven:
                    for i in range(self.nlp.nbQ):
                        q, q_dot, tau = self.__get_data(arg, i)
                        self.__update_plot(i, q)
                        self.__update_plot(i + self.nlp.nbQ, q_dot)
                        self.__update_plot(i + self.nlp.nbQ + self.nlp.nbQdot, tau)
            self.fig_state.canvas.draw()
            return True

        def __get_data(self, V, idx):
            if self.nlp.variable_type == Variable.torque_driven:
                q = np.array(V[0 * self.nlp.nbQ + idx :: 3 * self.nlp.nbQ])
                q_dot = np.array(V[1 * self.nlp.nbQdot + idx :: 3 * self.nlp.nbQdot])
                tau = np.ndarray((self.nlp.ns + 1, 1))
                tau[0 : self.nlp.ns, :] = np.array(
                    V[2 * self.nlp.nbTau + idx :: 3 * self.nlp.nbTau]
                )
                tau[-1, :] = tau[-2, :]
            return q, q_dot, tau

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
