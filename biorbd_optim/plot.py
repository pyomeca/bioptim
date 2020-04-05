import multiprocessing as mp

from matplotlib import pyplot as plt
from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity
import numpy as np


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
            print("starting plotter...")

            self.pipe = pipe
            self.fig_state, self.axes_state = plt.subplots(
                3, self.nlp.model.nbQ(), figsize=(10, 6)
            )
            self.axes_state = self.axes_state.flatten()
            self.axes_state[1].set_title("q")
            self.axes_state[self.nlp.model.nbQ() + 1].set_title("q_dot")
            self.axes_state[2 * self.nlp.model.nbQ() + 1].set_title("tau")

            self.t = np.linspace(0, self.nlp.tf, self.nlp.ns + 1)

            for i in range(self.nlp.model.nbQ()):
                self.axes_state[i].plot(self.t, np.zeros((self.nlp.ns + 1, 1)))
                self.axes_state[i].grid(color="k", linestyle="--", linewidth=0.5)
                self.axes_state[i].set_xlim(0, self.nlp.tf)

                self.axes_state[self.nlp.model.nbQ() + i].plot(
                    self.t, np.zeros((self.nlp.ns + 1, 1))
                )
                self.axes_state[self.nlp.model.nbQ() + i].grid(
                    color="k", linestyle="--", linewidth=0.5
                )
                self.axes_state[self.nlp.model.nbQ() + i].set_xlim(0, self.nlp.tf)

                self.plot_control(
                    self.axes_state[2 * self.nlp.model.nbQ() + i],
                    np.zeros((self.nlp.ns, 1)),
                )
                self.axes_state[2 * self.nlp.model.nbQ() + i].grid(
                    color="k", linestyle="--", linewidth=0.5
                )
                self.axes_state[2 * self.nlp.model.nbQ() + i].set_xlim(0, self.nlp.tf)
                self.axes_state[2 * self.nlp.model.nbQ() + i].set_xlabel("time (s)")

            timer = self.fig_state.canvas.new_timer(interval=100)
            timer.add_callback(self.call_back)
            timer.start()

            print("...plot is ready")
            plt.tight_layout()
            plt.show()

        def call_back(self):
            while self.pipe.poll():
                arg = self.pipe.recv()
                for i in range(self.nlp.model.nbQ()):
                    q, q_dot, u = self.get_states(arg, i)

                    y_range = np.max(q) - np.min(q)
                    y_mean = y_range / 2 + np.min(q)
                    y_range = (np.max([y_range, np.pi / 2]) + y_range * 0.1) / 2
                    self.axes_state[i].set_ylim(y_mean - y_range, y_mean + y_range)
                    self.axes_state[i].set_yticks(
                        np.arange(
                            np.round(y_mean - y_range, 1),
                            np.round(y_mean + y_range, 1),
                            step=np.round(
                                (y_mean + y_range - (y_mean - y_range)) / 4, 1
                            ),
                        )
                    )
                    lines_state = self.axes_state[i].get_lines()
                    lines_state[0].set_ydata(np.array(q))

                    y_range = np.max(q_dot) - np.min(q_dot)
                    y_mean = y_range / 2 + np.min(q_dot)
                    y_range = (np.max([y_range, np.pi]) + y_range * 0.1) / 2
                    self.axes_state[self.nlp.model.nbQ() + i].set_ylim(
                        y_mean - y_range, y_mean + y_range
                    )
                    self.axes_state[self.nlp.model.nbQ() + i].set_yticks(
                        np.arange(
                            np.round(y_mean - y_range, 1),
                            np.round(y_mean + y_range, 1),
                            step=np.round(
                                (y_mean + y_range - (y_mean - y_range)) / 4, 1
                            ),
                        )
                    )
                    lines_state_dot = self.axes_state[
                        self.nlp.model.nbQ() + i
                    ].get_lines()
                    lines_state_dot[0].set_ydata(np.array(q_dot))

                    y_range = np.max(u) - np.min(u)
                    y_mean = y_range / 2 + np.min(u)
                    y_range = (np.max([y_range, 2]) + y_range * 0.1) / 2
                    self.axes_state[2 * self.nlp.model.nbQ() + i].set_ylim(
                        y_mean - y_range, y_mean + y_range
                    )
                    self.axes_state[2 * self.nlp.model.nbQ() + i].set_yticks(
                        np.arange(
                            np.round(y_mean - y_range, 1),
                            np.round(y_mean + y_range, 1),
                            step=np.round(
                                (y_mean + y_range - (y_mean - y_range)) / 4, 1
                            ),
                        )
                    )
                    self.plot_control(
                        self.axes_state[2 * self.nlp.model.nbQ() + i], np.array(u)
                    )
            self.fig_state.canvas.draw()
            return True

        def get_states(self, V, idx):
            q = V[0 * self.nlp.model.nbQ() + idx :: 3 * self.nlp.model.nbQ()]
            q_dot = V[1 * self.nlp.model.nbQ() + idx :: 3 * self.nlp.model.nbQ()]
            u = V[2 * self.nlp.model.nbQ() + idx :: 3 * self.nlp.model.nbQ()]
            return q, q_dot, u

        def plot_control(self, ax, x):
            lines = ax.get_lines()
            if len(lines) == 0:
                for n in range(self.nlp.ns - 1):
                    ax.plot(
                        [self.t[n], self.t[n + 1], self.t[n + 1]],
                        [x[n], x[n], x[n + 1]],
                        "r-",
                    )
            else:
                for n in range(self.nlp.ns - 1):
                    lines[n].set_xdata([self.t[n], self.t[n + 1], self.t[n + 1]])
                    lines[n].set_ydata([x[n], x[n], x[n + 1]])
