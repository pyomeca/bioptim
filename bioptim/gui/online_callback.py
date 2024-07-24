from abc import ABC, abstractmethod
from enum import Enum
import multiprocessing as mp
import socket

from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity
from matplotlib import pyplot as plt

from .plot import PlotOcp


class OnlineCallbackType(Enum):
    """
    The type of callback

    Attributes
    ----------
    MULTIPROCESS: int
        Using multiprocessing
    SERVER: int
        Using a server to communicate with the client
    """

    MULTIPROCESS = 0
    SERVER = 1


class OnlineCallbackAbstract(Callback, ABC):
    """
    CasADi interface of Ipopt callbacks

    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp to show
    nx: int
        The number of optimization variables
    ng: int
        The number of constraints

    Methods
    -------
    get_n_in() -> int
        Get the number of variables in
    get_n_out() -> int
        Get the number of variables out
    get_name_in(i: int) -> int
        Get the name of a variable
    get_name_out(_) -> str
        Get the name of the output variable
    get_sparsity_in(self, i: int) -> tuple[int]
        Get the sparsity of a specific variable
    eval(self, arg: list | tuple) -> list[int]
        Send the current data to the plotter
    """

    def __init__(self, ocp, opts: dict = None, show_options: dict = None):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to show
        opts: dict
            Option to AnimateCallback method of CasADi
        show_options: dict
            The options to pass to PlotOcp
        """
        if opts is None:
            opts = {}

        Callback.__init__(self)
        self.ocp = ocp
        self.nx = self.ocp.variables_vector.shape[0]

        # There must be an option to add an if here
        from ..interfaces.ipopt_interface import IpoptInterface

        interface = IpoptInterface(ocp)
        all_g, _ = interface.dispatch_bounds()
        self.ng = all_g.shape[0]

        self.construct("AnimateCallback", opts)

    @abstractmethod
    def close(self):
        """
        Close the callback
        """

    @staticmethod
    def get_n_in() -> int:
        """
        Get the number of variables in

        Returns
        -------
        The number of variables in
        """

        return nlpsol_n_out()

    @staticmethod
    def get_n_out() -> int:
        """
        Get the number of variables out

        Returns
        -------
        The number of variables out
        """

        return 1

    @staticmethod
    def get_name_in(i: int) -> int:
        """
        Get the name of a variable

        Parameters
        ----------
        i: int
            The index of the variable

        Returns
        -------
        The name of the variable
        """

        return nlpsol_out(i)

    @staticmethod
    def get_name_out(_) -> str:
        """
        Get the name of the output variable

        Returns
        -------
        The name of the output variable
        """

        return "ret"

    def get_sparsity_in(self, i: int) -> tuple:
        """
        Get the sparsity of a specific variable

        Parameters
        ----------
        i: int
            The index of the variable

        Returns
        -------
        The sparsity of the variable
        """

        n = nlpsol_out(i)
        if n == "f":
            return Sparsity.scalar()
        elif n in ("x", "lam_x"):
            return Sparsity.dense(self.nx)
        elif n in ("g", "lam_g"):
            return Sparsity.dense(self.ng)
        else:
            return Sparsity(0, 0)

    @abstractmethod
    def eval(self, arg: list | tuple) -> list:
        """
        Send the current data to the plotter

        Parameters
        ----------
        arg: list | tuple
            The data to send

        Returns
        -------
        A list of error index
        """


class OnlineCallback(OnlineCallbackAbstract):
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
        super(OnlineCallback, self).__init__(ocp, opts, show_options)

        self.queue = mp.Queue()
        self.plotter = self.ProcessPlotter(self.ocp)
        self.plot_process = mp.Process(target=self.plotter, args=(self.queue, show_options), daemon=True)
        self.plot_process.start()

    def close(self):
        self.plot_process.kill()

    def eval(self, arg: list | tuple) -> list:
        send = self.queue.put
        args_dict = {}
        for i, s in enumerate(nlpsol_out()):
            args_dict[s] = arg[i]
        send(args_dict)
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

            self.ocp = ocp

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
            self.pipe = pipe
            self.plot = PlotOcp(self.ocp, **show_options)
            timer = self.plot.all_figures[0].canvas.new_timer(interval=10)
            timer.add_callback(self.callback)
            timer.start()
            plt.show()

        def callback(self) -> bool:
            """
            The callback to update the graphs

            Returns
            -------
            True if everything went well
            """

            while not self.pipe.empty():
                args = self.pipe.get()
                self.plot.update_data(args)

            for i, fig in enumerate(self.plot.all_figures):
                fig.canvas.draw()
            return True


class OnlineCallbackServer:
    class _ServerMessages(Enum):
        INITIATE_CONNEXION = 0
        NEW_DATA = 1
        CLOSE_CONNEXION = 2

    def __init__(self):
        # Define the host and port
        self._host = "localhost"
        self._port = 3050
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._initialize_connexion()

        self._plotter: PlotOcp = None

    def _initialize_connexion(self):
        # Start listening to the server
        self._socket.bind((self._host, self._port))
        self._socket.listen(5)
        print(f"Server started on {self._host}:{self._port}")

        while True:
            client_socket, addr = self._socket.accept()
            print(f"Connection from {addr}")

            # Receive the actual data
            data = b""
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                data += chunk
            data_as_list = data.decode().split("\n")

            if data_as_list[0] == OnlineCallbackServer._ServerMessages.INITIATE_CONNEXION:
                print(f"Received from client: {data_as_list[1]}")

                response = "Hello from server!"
                client_socket.send(response.encode())
                # TODO Get the OCP and show_options from the client
                # ocp = data_as_list[1]
                # show_options = data_as_list[2]
                # self._initialize_plotter(ocp, show_options=show_options)
            elif data_as_list[0] == OnlineCallbackServer._ServerMessages.NEW_DATA:
                print(f"Received from client: {data_as_list[1]}")

                response = "Hello from server!"
                client_socket.send(response.encode())
                # self._plotter.update_data(data_as_list[1])
            elif data_as_list[0] == OnlineCallbackServer._ServerMessages.CLOSE_CONNEXION:
                print("Closing the server")
                client_socket.close()
                continue
            else:
                print("Unknown message received")
                continue

    def _initialize_plotter(self, ocp, **show_options):
        self._plotter = PlotOcp(ocp, **show_options)


class OnlineCallbackTcp(OnlineCallbackAbstract, OnlineCallbackServer):
    def __init__(self, ocp, opts: dict = None, show_options: dict = None):
        super(OnlineCallbackAbstract, self).__init__(ocp, opts, show_options)
        super(OnlineCallbackServer, self).__init__()

    def _initialize_connexion(self):
        # Start the client
        try:
            self._socket.connect((self._host, self._port))
        except ConnectionError:
            print("Could not connect to the server, make sure it is running")
        print(f"Connected to {self._host}:{self._port}")

        message = f"{OnlineCallbackServer._ServerMessages.INITIATE_CONNEXION}\nHello from client!"
        self._socket.send(message.encode())
        data = self._socket.recv(1024).decode()
        print(f"Received from server: {data}")

        self.close()

    def close(self):
        self._socket.send(f"{OnlineCallbackServer._ServerMessages.CLOSE_CONNEXION}\nGoodbye from client!".encode())
        self._socket.close()

    def eval(self, arg: list | tuple) -> list:
        self._socket.send(f"{OnlineCallbackServer._ServerMessages.NEW_DATA}\n{arg}".encode())
