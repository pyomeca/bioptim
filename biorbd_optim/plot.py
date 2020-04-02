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
            target=self.plotter, args=(plotter_pipe,), daemon=True)
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
        if n == 'f':
            return Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
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
            print('starting plotter...')

            self.pipe = pipe
            self.fig, self.ax = plt.subplots()
            self.plt = self.ax.plot(np.linspace(0, self.nlp.tf, self.nlp.ns + 1), np.zeros((self.nlp.ns + 1, 1)))
            self.ax.set_title('hope it works')
            timer = self.fig.canvas.new_timer(interval=100)
            timer.add_callback(self.call_back)
            timer.start()

            print('...plot is ready')
            plt.show()

        def call_back(self):
            while self.pipe.poll():
                arg = self.pipe.recv()
                i = 0
                q, q_dot, u = self.get_states(arg, i)
                self.plt[i].set_ydata(np.array(q))

                y_range = np.max(q) - np.min(q)
                y_mean = y_range/2 + np.min(q)
                y_range = (np.max([y_range, np.pi/2]) + y_range*0.1)/2
                self.ax.set_ylim(y_mean - y_range, y_mean + y_range)
            self.fig.canvas.draw()
            return True

        def get_states(self, V, idx):
            q = V[0 * self.nlp.model.nbQ() + idx :: 3 * self.nlp.model.nbQ()]
            q_dot = V[1 * self.nlp.model.nbQ() + idx :: 3 * self.nlp.model.nbQ()]
            u = V[2 * self.nlp.model.nbQ() + idx :: 3 * self.nlp.model.nbQ()]
            return q, q_dot, u
