from abc import ABC, abstractmethod
from enum import Enum
import json
import logging
import multiprocessing as mp
import socket
import struct
from typing import Callable


from casadi import Callback, nlpsol_out, nlpsol_n_out, Sparsity, DM
from matplotlib import pyplot as plt
import numpy as np

from .plot import PlotOcp, OcpSerializable
from ..optimization.optimization_vector import OptimizationVectorHelper


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

            self.ocp: OcpSerializable = ocp
            self._plotter: PlotOcp = None

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

            dummy_phase_times = OptimizationVectorHelper.extract_step_times(self.ocp, DM(np.ones(self.ocp.n_phases)))
            self._plotter = PlotOcp(self.ocp, dummy_phase_times=dummy_phase_times, **show_options)
            timer = self._plotter.all_figures[0].canvas.new_timer(interval=10)
            timer.add_callback(self.plot_update)
            timer.start()
            plt.show()

        def plot_update(self) -> bool:
            """
            The callback to update the graphs

            Returns
            -------
            True if everything went well
            """

            while not self.pipe.empty():
                args = self.pipe.get()
                self._plotter.update_data(args)

            for i, fig in enumerate(self._plotter.all_figures):
                fig.canvas.draw()
            return True


_default_host = "localhost"
_default_port = 3050


class OnlineCallbackServer:
    class _ServerMessages(Enum):
        INITIATE_CONNEXION = 0
        NEW_DATA = 1
        CLOSE_CONNEXION = 2

    def _prepare_logger(self):
        name = "OnlineCallbackServer"
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "{asctime} - {name}:{levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )
        console_handler.setFormatter(formatter)

        self._logger = logging.getLogger(name)
        self._logger.addHandler(console_handler)
        self._logger.setLevel(logging.DEBUG)

    def __init__(self, host: str = None, port: int = None):
        self._prepare_logger()

        # Define the host and port
        self._host = host if host else _default_host
        self._port = port if port else _default_port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._plotter: PlotOcp = None

    def run(self):
        # Start listening to the server
        self._socket.bind((self._host, self._port))
        self._socket.listen(1)
        self._logger.info(f"Server started on {self._host}:{self._port}")

        try:
            while True:
                self._logger.info("Waiting for a new connexion")
                client_socket, addr = self._socket.accept()
                self._logger.info(f"Connection from {addr}")
                self._handle_client(client_socket, addr)
        except Exception as e:
            self._logger.error(f"Error while running the server: {e}")
        finally:
            self._socket.close()

    def _handle_client(self, client_socket: socket.socket, addr: tuple):
        while True:
            # Receive the actual data
            try:
                self._logger.debug("Waiting for data from client")
                data = client_socket.recv(1024)
            except:
                self._logger.warning("Error while receiving data from client, closing connexion")
                return

            data_as_list = data.decode().split("\n")
            self._logger.debug(f"Received from client: {data_as_list}")

            if not data:
                self._logger.info("The client closed the connexion")
                plt.close()
                return

            try:
                message_type = OnlineCallbackServer._ServerMessages(int(data_as_list[0]))
            except ValueError:
                self._logger.warning("Unknown message type received")
                continue

            if message_type == OnlineCallbackServer._ServerMessages.INITIATE_CONNEXION:
                self._initiate_connexion(client_socket, data_as_list)
                continue

            elif message_type == OnlineCallbackServer._ServerMessages.NEW_DATA:
                try:
                    self._update_data(client_socket, data_as_list)
                except:
                    self._logger.warning("Error while updating data from client, closing connexion")
                    plt.close()
                    client_socket.close()
                    return
                continue

            elif message_type == OnlineCallbackServer._ServerMessages.CLOSE_CONNEXION:
                self._logger.info("Received close connexion from client")
                client_socket.close()
                plt.close()
                return
            else:
                self._logger.warning("Unknown message received")
                continue

    def _initiate_connexion(self, client_socket: socket.socket, data_as_list: list):
        self._logger.debug(f"Received hand shake from client, len of OCP: {data_as_list[1]}")
        ocp_len = data_as_list[1]
        try:
            ocp_data = client_socket.recv(int(ocp_len))
        except:
            self._logger.warning("Error while receiving OCP data from client, closing connexion")
            return

        data_json = json.loads(ocp_data)

        try:
            dummy_time_vector = []
            for phase_times in data_json["dummy_phase_times"]:
                dummy_time_vector.append([DM(v) for v in phase_times])
            del data_json["dummy_phase_times"]
        except:
            self._logger.warning("Error while extracting dummy time vector from OCP data, closing connexion")
            return

        try:
            self.ocp = OcpSerializable.deserialize(data_json)
        except:
            self._logger.warning("Error while deserializing OCP data from client, closing connexion")
            return

        show_options = {}
        self._plotter = PlotOcp(self.ocp, dummy_phase_times=dummy_time_vector, **show_options)
        plt.ion()
        plt.draw()  # TODO HERE!

    def _update_data(self, client_socket: socket.socket, data_as_list: list):
        n_bytes = [int(d) for d in data_as_list[1][1:-1].split(",")]
        n_points = [int(d / 8) for d in n_bytes]
        all_data = []
        for n_byte, n_point in zip(n_bytes, n_points):
            data = client_socket.recv(n_byte)
            data_tp = struct.unpack("d" * n_point, data)
            all_data.append(DM(data_tp))

        self._logger.debug(f"Received new data from client")
        # self._plotter.update_data(data_as_list[1])


class OnlineCallbackTcp(OnlineCallbackAbstract):
    def __init__(self, ocp, opts: dict = None, show_options: dict = None, host: str = None, port: int = None):
        super().__init__(ocp, opts, show_options)

        self._host = host if host else _default_host
        self._port = port if port else _default_port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._initialize_connexion()

    def _initialize_connexion(self):
        # Start the client
        try:
            self._socket.connect((self._host, self._port))
        except ConnectionError:
            raise RuntimeError(
                "Could not connect to the plotter server, make sure it is running "
                "by calling 'OnlineCallbackServer().start()' on another python instance)"
            )

        ocp_plot = OcpSerializable.from_ocp(self.ocp).serialize()
        ocp_plot["dummy_phase_times"] = []
        for phase_times in OptimizationVectorHelper.extract_step_times(self.ocp, DM(np.ones(self.ocp.n_phases))):
            ocp_plot["dummy_phase_times"].append([np.array(v)[:, 0].tolist() for v in phase_times])
        serialized_ocp = json.dumps(ocp_plot).encode()
        self._socket.send(
            f"{OnlineCallbackServer._ServerMessages.INITIATE_CONNEXION.value}\n{len(serialized_ocp)}".encode()
        )

        # TODO ADD SHOW OPTIONS to the send
        self._socket.send(serialized_ocp)

    def close(self):
        self._socket.send(
            f"{OnlineCallbackServer._ServerMessages.CLOSE_CONNEXION.value}\nGoodbye from client!".encode()
        )
        self._socket.close()

    def eval(self, arg: list | tuple) -> list:
        arg_as_bytes = []
        for a in arg:
            to_pack = np.array(a).T.tolist()
            if len(to_pack) == 1:
                to_pack = to_pack[0]
            arg_as_bytes.append(struct.pack("d" * len(to_pack), *to_pack))

        self._socket.send(
            f"{OnlineCallbackServer._ServerMessages.NEW_DATA.value}\n{[len(a) for a in arg_as_bytes]}".encode()
        )
        for a in arg_as_bytes:
            self._socket.sendall(a)
        return [0]
