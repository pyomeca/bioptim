import multiprocessing as mp
import threading

from casadi import nlpsol_out, DM
from matplotlib import pyplot as plt
import numpy as np

from .plot import PlotOcp, OcpSerializable
from ..optimization.optimization_vector import OptimizationVectorHelper
from .online_callback_abstract import OnlineCallbackAbstract


class OnlineCallbackMultiprocess(OnlineCallbackAbstract):
    """
    Multiprocessing implementation of the online callback

    Attributes
    ----------
    queue: mp.Queue
        The multiprocessing queue
    plotter: ProcessPlotter
        The callback for plotting for the multiprocessing
    plot_process: mp.Process
        The multiprocessing placeholder
    """

    def __init__(self, ocp, opts: dict = None, show_options: dict = None):
        super(OnlineCallbackMultiprocess, self).__init__(ocp, opts, show_options)

        self.queue = mp.Queue()
        self.plotter = self.ProcessPlotter(self.ocp)
        self.plot_process = mp.Process(target=self.plotter, args=(self.queue, show_options), daemon=True)
        self.plot_process.start()

    def close(self):
        self.plot_process.kill()

    def eval(self, arg: list | tuple, force: bool = False) -> list:
        # Dequeuing the data by removing previous not useful data
        while not self.queue.empty():
            self.queue.get_nowait()

        args_dict = {}
        for i, s in enumerate(nlpsol_out()):
            args_dict[s] = arg[i]
        self.queue.put_nowait(args_dict)
        return [0]

    class ProcessPlotter(object):
        """
        The plotter that interface PlotOcp and the multiprocessing

        Attributes
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to show
        pipe: mp.Queue
            The multiprocessing queue to evaluate
        plot: PlotOcp
            The handler on all the figures

        Methods
        -------
        callback(self) -> bool
            The callback to update the graphs
        """

        def __init__(self, ocp):
            """
            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp to show
            """

            self._ocp: OcpSerializable = ocp
            self._plotter: PlotOcp = None
            self._update_time = 0.001

        def __call__(self, pipe: mp.Queue, show_options: dict | None):
            """
            Parameters
            ----------
            pipe: mp.Queue
                The multiprocessing queue to evaluate
            show_options: dict
                The option to pass to PlotOcp
            """

            if show_options is None:
                show_options = {}
            self._pipe = pipe

            dummy_phase_times = OptimizationVectorHelper.extract_step_times(self._ocp, DM(np.ones(self._ocp.n_phases)))
            self._plotter = PlotOcp(self._ocp, dummy_phase_times=dummy_phase_times, **show_options)
            threading.Timer(self._update_time, self.plot_update).start()
            plt.show()

        def plot_update(self) -> bool:
            """
            The callback to update the graphs

            Returns
            -------
            True if everything went well
            """

            args = {}
            while not self._pipe.empty():
                args = self._pipe.get()

            if args:
                self._plotter.update_data(*self._plotter.parse_data(**args), **args)

            # We want to redraw here to actually consume a bit of time, otherwise it goes to fast and pipe remains empty
            for fig in self._plotter.all_figures:
                fig.canvas.draw()
            if [plt.fignum_exists(fig.number) for fig in self._plotter.all_figures].count(True) > 0:
                # If there are still figures, we keep updating
                threading.Timer(self._update_time, self.plot_update).start()

            return True
