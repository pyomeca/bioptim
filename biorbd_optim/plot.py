import threading

from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity

class OnlinePlot():
    def plot_callback(callback_data):
        print('NEW THREAD plot thread')
        # fig, ax = plt.subplots()
        # ax.plot(np.linspace(0, 10, 10), np.zeros((10)))
        # ax.set_title('hope it works')
        # plt.show(block=False)
        # plt.pause(0.01)

        while True:
            if callback_data.update_sol:
                print('NEW DATA plot\n')
                data = callback_data.sol_data
                print(str(data[0]) + '\n')
                # ax.set_ydata(np.random.rand(10))
                callback_data.update_sol = False
            # plt.draw()
            # plt.pause(.001)
            time.sleep(0.001)

    # _thread.start_new_thread(print_callback, (mycallback,))
    plot_thread = threading.Thread(name = 'plot_data', target = plot_callback, args = (mycallback, ))                      # new thread
    plot_thread.start()                                                                                                     # start new thread

    class AnimateCallback(Callback):
        def __init__(self, name, nx, ng, np, opts={}):
            Callback.__init__(self)
            self.nx     = nx                   # optimized value number
            self.ng     = ng                   # constraints number
            self.nP     = np

            self.sol_data   = None             # optimized variables
            self.update_sol = True             # first iteration

            self.construct(name, opts)

        def get_n_in(self): return nlpsol_n_out()
        def get_n_out(self): return 1
        def get_name_in(self, i): return nlpsol_out(i)
        def get_name_out(self, i): return "ret"

        def get_sparsity_in(self, i):
            n = nlpsol_out(i)
            if n == 'f': return Sparsity.scalar()
            elif n in ('x', 'lam_x'): return Sparsity.dense(self.nx)
            elif n in ('g', 'lam_g'): return Sparsity.dense(self.ng)
            else: return Sparsity(0, 0)

        def eval(self, arg):
            darg = {}
            for (i, s) in enumerate(nlpsol_out()):
                darg[s] = arg[i]

            self.sol_data   = darg["x"]
            self.update_sol = True
            return [0]