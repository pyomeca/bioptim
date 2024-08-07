from enum import IntEnum, auto
import json
import logging
import platform
import socket
import struct
import time
import threading

from casadi import nlpsol_out, DM
from matplotlib import pyplot as plt
import numpy as np

from .online_callback_abstract import OnlineCallbackAbstract
from .plot import PlotOcp, OcpSerializable
from ..optimization.optimization_vector import OptimizationVectorHelper


_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 3050


def _serialize_show_options(show_options: dict) -> bytes:
    return json.dumps(show_options).encode()


def _deserialize_show_options(show_options: bytes) -> dict:
    return json.loads(show_options.decode())


class _ServerMessages(IntEnum):
    INITIATE_CONNEXION = auto()
    NEW_DATA = auto()
    CLOSE_CONNEXION = auto()
    EMPTY = auto()
    TOO_SOON = auto()
    UNKNOWN = auto()


class PlottingServer:
    def __init__(self, host: str = None, port: int = None, log_level: int | None = logging.INFO):
        """
        Initializes the server

        Parameters
        ----------
        host: str
            The host to listen to, by default "localhost"
        port: int
            The port to listen to, by default 3050
        log_level: int
            The log level (see logging), by default logging.INFO
        """

        if log_level is None:
            log_level = logging.INFO

        self._prepare_logger(log_level)
        self._get_data_interval = 1.0
        self._update_plot_interval = 10
        self._force_redraw = False

        # Define the host and port
        self._host = host if host else _DEFAULT_HOST
        self._port = port if port else _DEFAULT_PORT
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._plotter: PlotOcp = None

        self._should_send_ok_to_client_on_new_data = False

        self._run()

    def _prepare_logger(self, log_level: int) -> None:
        """
        Prepares the logger

        Parameters
        ----------
        log_level: int
            The log level
        """

        name = "PlottingServer"
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "{asctime} - {name}:{levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S.%03d",
        )
        console_handler.setFormatter(formatter)

        self._logger = logging.getLogger(name)
        self._logger.addHandler(console_handler)
        self._logger.setLevel(log_level)

    def _run(self) -> None:
        """
        Starts the server, this method is blocking
        """
        # Start listening to the server
        self._socket.bind((self._host, self._port))
        self._socket.listen(1)
        self._logger.info(f"Server started on {self._host}:{self._port}")

        try:
            while True:
                self._logger.info("Waiting for a new connexion")
                client_socket, addr = self._socket.accept()
                self._logger.info(f"Connexion from {addr}")
                self._wait_for_new_connexion(client_socket)
        except Exception as e:
            self._logger.error(
                f"Fatal error while running the server"
                f"{''if self._logger.level == logging.DEBUG else ', for more information set log_level to DEBUG'}"
            )
            self._logger.debug(f"Error: {e}")
        finally:
            self._socket.close()

    def _wait_for_new_connexion(self, client_socket: socket.socket) -> None:
        """
        Waits for a new connexion

        Parameters
        ----------
        client_socket: socket.socket
            The client socket
        """

        message_type, data = self._recv_data(client_socket=client_socket, send_confirmation=True)
        if message_type == _ServerMessages.INITIATE_CONNEXION:
            self._logger.debug(f"Received hand shake from client")
            self._initialize_plotter(client_socket, data)

    def _recv_data(self, client_socket: socket.socket, send_confirmation: bool) -> tuple[_ServerMessages, list]:
        """
        Waits for data from the client

        Parameters
        ----------
        client_socket: socket.socket
            The client socket
        send_confirmation: bool
            If True, the server will send a "OK" confirmation to the client after receiving the data, otherwise it will
            not send anything. This is part of the communication protocol

        Returns
        -------
        The message type and the data
        """
        self._logger.debug("Waiting for data from client")
        message_type, data_len = self._recv_message_type_and_data_len(client_socket, send_confirmation)
        if data_len is None:
            return message_type, None

        data = self._recv_serialize_data(client_socket, send_confirmation, data_len)
        return message_type, data

    def _recv_message_type_and_data_len(
        self, client_socket: socket.socket, send_confirmation: bool
    ) -> tuple[_ServerMessages, list]:
        """
        Waits for data len from the client (first part of the protocol)

        Parameters
        ----------
        client_socket: socket.socket
            The client socket
        send_confirmation: bool
            If True, the server will send a "OK" confirmation to the client after receiving the data, otherwise it will
            not send anything. This is part of the communication protocol

        Returns
        -------
        The message type and the data
        """

        # Receive the actual data
        try:
            data = client_socket.recv(1024)
            if not data:
                return _ServerMessages.EMPTY, None
        except:
            self._logger.info("Client closed connexion")
            client_socket.close()
            return _ServerMessages.CLOSE_CONNEXION, None

        data_as_list = data.decode().split("\n")
        try:
            message_type = _ServerMessages(int(data_as_list[0]))
        except ValueError:
            self._logger.error("Unknown message type received")
            # Sends failure
            if send_confirmation:
                client_socket.sendall("NOK".encode())
            return _ServerMessages.UNKNOWN, None

        if message_type == _ServerMessages.CLOSE_CONNEXION:
            self._logger.info("Received close connexion from client")
            client_socket.close()
            return _ServerMessages.CLOSE_CONNEXION, None

        try:
            len_all_data = [int(len_data) for len_data in data_as_list[1][1:-1].split(",")]
        except Exception as e:
            self._logger.error("Length of data could not be extracted")
            self._logger.debug(f"Error: {e}")
            # Sends failure
            if send_confirmation:
                client_socket.sendall("NOK".encode())
            return _ServerMessages.UNKNOWN, None

        # If we are here, everything went well, so send confirmation
        self._logger.debug(f"Received from client: {message_type} ({len_all_data} bytes)")
        if send_confirmation:
            client_socket.sendall("OK".encode())

        return message_type, len_all_data

    def _recv_serialize_data(self, client_socket: socket.socket, send_confirmation: bool, len_all_data: list) -> tuple:
        """
        Receives the data from the client (second part of the protocol)

        Parameters
        ----------
        client_socket: socket.socket
            The client socket
        send_confirmation: bool
            If True, the server will send a "OK" confirmation to the client after receiving the data, otherwise it will
            not send anything. This is part of the communication protocol
        len_all_data: list
            The length of the data to receive

        Returns
        -------
        The unparsed serialized data
        """

        data_out = []
        try:
            for len_data in len_all_data:
                self._logger.debug(f"Waiting for {len_data} bytes from client")
                data_tp = b""
                while len(data_tp) != len_data:
                    data_tp += client_socket.recv(len_data - len(data_tp))
                data_out.append(data_tp)
        except Exception as e:
            self._logger.error("Unknown message type received")
            self._logger.debug(f"Error: {e}")
            # Sends failure
            if send_confirmation:
                client_socket.sendall("NOK".encode())
            return None

        # If we are here, everything went well, so send confirmation
        if send_confirmation:
            client_socket.sendall("OK".encode())

        self._logger.debug(f"Received data from client: {[len(d) for d in data_out]} bytes")
        return data_out

    def _initialize_plotter(self, client_socket: socket.socket, ocp_raw: list) -> None:
        """
        Initializes the plotter

        Parameters
        ----------
        client_socket: socket.socket
            The client socket
        ocp_raw: list
            The serialized raw data from the client
        """

        try:
            data_json = json.loads(ocp_raw[0])
        except Exception as e:
            self._logger.error("Error while converting data to json format, closing connexion")
            client_socket.sendall("NOK".encode())
            raise e

        try:
            self._should_send_ok_to_client_on_new_data = data_json["request_confirmation_on_new_data"]
        except Exception as e:
            self._logger.error("Did not receive if confirmation should be sent, closing connexion")
            client_socket.sendall("NOK".encode())
            raise e

        try:
            dummy_time_vector = []
            for phase_times in data_json["dummy_phase_times"]:
                dummy_time_vector.append([DM(v) for v in phase_times])
            del data_json["dummy_phase_times"]
        except Exception as e:
            self._logger.error("Error while extracting dummy time vector from OCP data, closing connexion")
            client_socket.sendall("NOK".encode())
            raise e

        try:
            self.ocp = OcpSerializable.deserialize(data_json)
        except Exception as e:
            self._logger.error("Error while deserializing OCP data from client, closing connexion")
            client_socket.sendall("NOK".encode())
            raise e

        try:
            show_options = _deserialize_show_options(ocp_raw[1])
        except Exception as e:
            self._logger.error("Error while extracting show options, closing connexion")
            client_socket.sendall("NOK".encode())
            raise e

        try:
            self._plotter = PlotOcp(self.ocp, dummy_phase_times=dummy_time_vector, **show_options)
        except Exception as e:
            self._logger.error("Error while initializing the plotter, closing connexion")
            client_socket.sendall("NOK".encode())
            raise e

        # Send the confirmation to the client
        client_socket.sendall("PLOT_READY".encode())

        # Start the callbacks
        threading.Timer(self._get_data_interval, self._wait_for_new_data_to_plot, (client_socket,)).start()

        # Use the canvas timer for _redraw as threading won't work for updating the graphs on Macos
        timer = self._plotter.all_figures[0].canvas.new_timer(self._update_plot_interval)
        timer.add_callback(self._redraw)
        timer.start()

        plt.show()

    @property
    def has_at_least_one_active_figure(self) -> bool:
        """
        If at least one figure is active

        Returns
        -------
        If at least one figure is active
        """

        return [plt.fignum_exists(fig.number) for fig in self._plotter.all_figures].count(True) > 0

    def _redraw(self) -> None:
        """
        Redraws the plot, this method is called periodically as long as at least one figure is open
        """

        self._logger.debug("Updating plot")
        for fig in self._plotter.all_figures:
            fig.canvas.draw()
            if platform.system() != "Darwin":
                fig.canvas.flush_events()
        self._force_redraw = False

    def _wait_for_new_data_to_plot(self, client_socket: socket.socket) -> None:
        """
        Waits for new data from the client, sends a "READY_FOR_NEXT_DATA" message to the client to signal that the server
        is ready to receive new data. If the client sends new data, the server will update the plot, if client disconnects
        the connexion will be closed

        Parameters
        ----------
        client_socket: socket.socket
            The client socket
        """

        self._logger.debug(f"Waiting for new data from client")

        if self._force_redraw and platform.system() != "Darwin":
            time.sleep(self._update_plot_interval)

        try:
            client_socket.sendall("READY_FOR_NEXT_DATA".encode())
        except Exception as e:
            self._logger.error("Error while sending READY_FOR_NEXT_DATA to client, closing connexion")
            self._logger.debug(f"Error: {e}")
            client_socket.close()
            return

        should_continue = False
        message_type, data = self._recv_data(
            client_socket=client_socket, send_confirmation=self._should_send_ok_to_client_on_new_data
        )
        if message_type == _ServerMessages.NEW_DATA:
            try:
                self._update_plot(data)
                should_continue = True
            except Exception as e:
                self._logger.error("Error while updating data from client, closing connexion")
                self._logger.debug(f"Error: {e}")
                client_socket.close()
                return

        elif message_type in (_ServerMessages.EMPTY, _ServerMessages.CLOSE_CONNEXION):
            self._logger.debug("Received empty data from client (end of stream), closing connexion")

        if should_continue:
            timer_get_data = threading.Timer(self._get_data_interval, self._wait_for_new_data_to_plot, (client_socket,))
            timer_get_data.start()

    def _update_plot(self, serialized_raw_data: list) -> None:
        """
        This method parses the data from the client

        Parameters
        ----------
        serialized_raw_data: list
            The serialized raw data from the client, see `xydata_encoding` below
        """
        self._logger.debug(f"Received new data from client")
        xdata, ydata = _deserialize_xydata(serialized_raw_data)
        self._plotter.update_data(xdata, ydata)

        self._force_redraw = True


class OnlineCallbackServer(OnlineCallbackAbstract):
    def __init__(self, ocp, opts: dict = None, show_options: dict = None, host: str = None, port: int = None):
        """
        Initializes the client. This is not supposed to be called directly by the user, but by the solver. During the
        initialization, we need to perform some tasks that are not possible to do in server side. Then the results of
        these initialization are passed to the server

        Parameters
        ----------
        ocp: OptimalControlProgram
            The ocp
        opts: dict
            The options for the solver
        show_options: dict
            The options for the plot
        host: str
            The host to connect to, by default "localhost"
        port: int
            The port to connect to, by default 3050
        """

        super().__init__(ocp, opts, show_options)

        self._host = host if host else _DEFAULT_HOST
        self._port = port if port else _DEFAULT_PORT
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._should_wait_ok_to_client_on_new_data = platform.system() == "Darwin"

        if self.ocp.plot_ipopt_outputs:
            raise NotImplementedError("The online callback with TCP does not support the plot_ipopt_outputs option")
        if self.ocp.save_ipopt_iterations_info:
            raise NotImplementedError(
                "The online callback with TCP does not support the save_ipopt_iterations_info option"
            )
        if self.ocp.plot_check_conditioning:
            raise NotImplementedError(
                "The online callback with TCP does not support the plot_check_conditioning option"
            )

        self._initialize_connexion(**show_options)

    def _initialize_connexion(self, retries: int = 0, **show_options) -> None:
        """
        Initializes the connexion to the server

        Parameters
        ----------
        retries: int
            The number of retries to connect to the server (retry 5 times with 1s sleep between each retry, then raises
            an error if it still cannot connect)
        show_options: dict
            The options to pass to PlotOcp
        """

        # Start the client
        try:
            self._socket.connect((self._host, self._port))
        except:
            if retries > 5:
                raise RuntimeError(
                    "Could not connect to the plotter server, make sure it is running by calling 'PlottingServer()' on "
                    "another python instance or allowing for automatic start of the server by calling "
                    "'PlottingServer.as_multiprocess()' in the main script"
                )
            else:
                time.sleep(1)
                return self._initialize_connexion(retries + 1, **show_options)

        ocp_plot = OcpSerializable.from_ocp(self.ocp).serialize()
        dummy_phase_times = OptimizationVectorHelper.extract_step_times(self.ocp, DM(np.ones(self.ocp.n_phases)))
        ocp_plot["dummy_phase_times"] = []
        for phase_times in dummy_phase_times:
            ocp_plot["dummy_phase_times"].append([np.array(v)[:, 0].tolist() for v in phase_times])

        ocp_plot["request_confirmation_on_new_data"] = self._should_wait_ok_to_client_on_new_data
        serialized_ocp = json.dumps(ocp_plot).encode()

        serialized_show_options = _serialize_show_options(show_options)

        # Sends message type and dimensions
        self._socket.sendall(
            f"{_ServerMessages.INITIATE_CONNEXION.value}\n{[len(serialized_ocp), len(serialized_show_options)]}".encode()
        )
        if self._socket.recv(1024).decode() != "OK":
            raise RuntimeError("The server did not acknowledge the connexion")

        self._socket.sendall(serialized_ocp)
        self._socket.sendall(serialized_show_options)
        if self._socket.recv(1024).decode() != "OK":
            raise RuntimeError("The server did not acknowledge the connexion")

        # Wait for the server to be ready
        data = self._socket.recv(1024).decode().split("\n")
        if data[0] != "PLOT_READY":
            raise RuntimeError("The server did not acknowledge the OCP data, this should not happen, please report")

        self._plotter = PlotOcp(
            self.ocp, only_initialize_variables=True, dummy_phase_times=dummy_phase_times, **show_options
        )

    def close(self) -> None:
        """
        Closes the connexion
        """

        self._socket.sendall(f"{_ServerMessages.CLOSE_CONNEXION.value}\nGoodbye from client!".encode())
        self._socket.close()

    def eval(self, arg: list | tuple, enforce: bool = False) -> list[int]:
        """
        Sends the current data to the plotter, this method is automatically called by the solver

        Parameters
        ----------
        arg: list | tuple
            The current data
        enforce: bool
            If True, the client will block until the server is ready to receive new data. This is useful at the end of
            the optimization to make sure the data are plot (and not discarded)

        Returns
        -------
        A mandatory [0] to respect the CasADi callback signature
        """

        if not enforce:
            self._socket.setblocking(False)

        try:
            data = self._socket.recv(1024).decode()
            if data != "READY_FOR_NEXT_DATA":
                return [0]
        except BlockingIOError:
            # This is to prevent the solving to be blocked by the server if it is not ready to update the plots
            return [0]
        finally:
            self._socket.setblocking(True)

        args_dict = {}
        for i, s in enumerate(nlpsol_out()):
            args_dict[s] = arg[i]
        xdata, ydata = self._plotter.parse_data(**args_dict)
        header, data_serialized = _serialize_xydata(xdata, ydata)

        self._socket.sendall(f"{_ServerMessages.NEW_DATA.value}\n{[len(header), len(data_serialized)]}".encode())
        if self._should_wait_ok_to_client_on_new_data and self._socket.recv(1024).decode() != "OK":
            raise RuntimeError("The server did not acknowledge the connexion")

        self._socket.sendall(header)
        self._socket.sendall(data_serialized)
        if self._should_wait_ok_to_client_on_new_data and self._socket.recv(1024).decode() != "OK":
            raise RuntimeError("The server did not acknowledge the connexion")

        return [0]


def _serialize_xydata(xdata: list, ydata: list) -> tuple:
    """
    Serialize the data to send to the server, it will be deserialized by `_deserialize_xydata`

    Parameters
    ----------
    xdata: list
        The X data to serialize from PlotOcp.parse_data
    ydata: list
        The Y data to serialize from PlotOcp.parse_data

    Returns
    -------
    The serialized data as expected by the server (header, serialized_data)
    """

    header = f"{len(xdata)}"
    data_serialized = b""
    for x_nodes in xdata:
        header += f",{len(x_nodes)}"
        for x_steps in x_nodes:
            header += f",{x_steps.shape[0]}"
            x_steps_tp = np.array(x_steps)[:, 0].tolist()
            data_serialized += struct.pack("d" * len(x_steps_tp), *x_steps_tp)

    header += f",{len(ydata)}"
    for y_nodes_variable in ydata:
        if isinstance(y_nodes_variable, np.ndarray):
            header += f",0"
            y_nodes_variable = [y_nodes_variable]
        else:
            header += f",{len(y_nodes_variable)}"

        for y_steps in y_nodes_variable:
            header += f",{y_steps.shape[0]}"
            y_steps_tp = y_steps.tolist()
            data_serialized += struct.pack("d" * len(y_steps_tp), *y_steps_tp)

    header = header.encode()
    return header, data_serialized


def _deserialize_xydata(serialized_raw_data: list) -> tuple:
    """
    Deserialize the data from the client, based on the serialization used in _serialize_xydata`

    Parameters
    ----------
    serialized_raw_data: list
        The serialized raw data from the client

    Returns
    -------
    The deserialized data as expected by PlotOcp.update_data
    """

    # Header is made of ints comma separated from the first line
    header = [int(v) for v in serialized_raw_data[0].decode().split(",")]

    # Data is made of doubles (d) from the second line, the length of which is 8 bytes each
    data = serialized_raw_data[1]
    all_data = np.array(struct.unpack("d" * (len(data) // 8), data))

    # Based on the header, we can now parse the data, assuming the number of phases, nodes and steps from the header
    header_cmp = 0
    all_data_cmp = 0
    xdata = []
    n_phases = header[header_cmp]  # Number of phases
    header_cmp += 1
    for _ in range(n_phases):
        n_nodes = header[header_cmp]  # Number of nodes in the phase
        header_cmp += 1
        x_phases = []
        for _ in range(n_nodes):
            n_steps = header[header_cmp]  # Number of steps in the node
            header_cmp += 1

            x_phases.append(all_data[all_data_cmp : all_data_cmp + n_steps])  # The X data of the node
            all_data_cmp += n_steps
        xdata.append(x_phases)

    ydata = []
    n_variables = header[header_cmp]  # Number of variables (states, controls, etc.)
    header_cmp += 1
    for _ in range(n_variables):
        n_nodes = header[header_cmp]  # Number of nodes for the variable
        header_cmp += 1
        if n_nodes == 0:
            n_nodes = 1

        y_variables = []
        for _ in range(n_nodes):
            n_steps = header[header_cmp]  # Number of steps in the node for the variable
            header_cmp += 1

            y_variables.append(all_data[all_data_cmp : all_data_cmp + n_steps])  # The Y data of the node
            all_data_cmp += n_steps
        ydata.append(y_variables)

    return xdata, ydata
