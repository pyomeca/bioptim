from enum import Enum
import json
import logging
import socket
import struct
import threading

from casadi import nlpsol_out, DM
from matplotlib import pyplot as plt
import numpy as np

from .online_callback_abstract import OnlineCallbackAbstract
from .plot import PlotOcp, OcpSerializable
from ..optimization.optimization_vector import OptimizationVectorHelper


_default_host = "localhost"
_default_port = 3050


def _serialize_show_options(show_options: dict) -> bytes:
    return json.dumps(show_options).encode()


def _deserialize_show_options(show_options: bytes) -> dict:
    return json.loads(show_options.decode())


class _ServerMessages(Enum):
    INITIATE_CONNEXION = 0
    NEW_DATA = 1
    CLOSE_CONNEXION = 2
    EMPTY = 3
    TOO_SOON = 4
    UNKNOWN = 5


class OnlineCallbackServerBackend:
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
        self._get_data_interval = 1.0
        self._update_plot_interval = 0.01

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
                self._wait_for_new_connexion(client_socket)
        except Exception as e:
            self._logger.error(f"Error while running the server: {e}")
        finally:
            self._socket.close()

    def _wait_for_data(self, client_socket: socket.socket, send_confirmation: bool = True):
        # Receive the actual data
        try:
            self._logger.debug("Waiting for data from client")
            data = client_socket.recv(1024)
            if not data:
                return _ServerMessages.EMPTY, None
        except:
            self._logger.warning("Client closed connexion")
            client_socket.close()
            return _ServerMessages.CLOSE_CONNEXION, None

        data_as_list = data.decode().split("\n")
        try:
            message_type = _ServerMessages(int(data_as_list[0]))
            len_all_data = [int(len_data) for len_data in data_as_list[1][1:-1].split(",")]
            # Sends confirmation and waits for the next message
            if send_confirmation:
                client_socket.sendall("OK".encode())
            self._logger.debug(f"Received from client: {message_type} ({len_all_data} bytes)")
            data_out = []
            for len_data in len_all_data:
                data_out.append(client_socket.recv(len_data))
                if len(data_out[-1]) != len_data:
                    data_out[-1] += client_socket.recv(len_data - len(data_out[-1]))
            if send_confirmation:
                client_socket.sendall("OK".encode())
        except ValueError:
            self._logger.warning("Unknown message type received")
            message_type = _ServerMessages.UNKNOWN
            # Sends failure
            if send_confirmation:
                client_socket.sendall("NOK".encode())
            data_out = []

        if message_type == _ServerMessages.CLOSE_CONNEXION:
            self._logger.info("Received close connexion from client")
            client_socket.close()
            plt.close()
            return _ServerMessages.CLOSE_CONNEXION, None

        return message_type, data_out

    def _wait_for_new_connexion(self, client_socket: socket.socket):
        message_type, data = self._wait_for_data(client_socket=client_socket)
        if message_type == _ServerMessages.INITIATE_CONNEXION:
            self._logger.debug(f"Received hand shake from client")
            self._initialize_plotter(client_socket, data)

    def _initialize_plotter(self, client_socket: socket.socket, ocp_raw: list):
        try:
            data_json = json.loads(ocp_raw[0])
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
            client_socket.sendall("FAILED".encode())
            self._logger.warning("Error while deserializing OCP data from client, closing connexion")
            return

        try:
            show_options = _deserialize_show_options(ocp_raw[1])
        except:
            self._logger.warning("Error while extracting show options, closing connexion")
            return

        self._plotter = PlotOcp(self.ocp, dummy_phase_times=dummy_time_vector, **show_options)

        # Send the confirmation to the client
        client_socket.sendall("PLOT_READY".encode())

        # Start the callbacks
        threading.Timer(self._get_data_interval, self._wait_for_new_data, (client_socket,)).start()
        threading.Timer(self._update_plot_interval, self._redraw).start()
        plt.show()

    def _redraw(self):
        self._logger.debug("Updating plot")
        for _, fig in enumerate(self._plotter.all_figures):
            fig.canvas.draw()

        if [plt.fignum_exists(fig.number) for fig in self._plotter.all_figures].count(True) > 0:
            threading.Timer(self._update_plot_interval, self._redraw).start()
        else:
            self._logger.info("All figures have been closed, stop updating the plots")

    def _wait_for_new_data(self, client_socket: socket.socket) -> bool:
        """
        The callback to update the graphs

        Returns
        -------
        True if everything went well
        """
        self._logger.debug(f"Waiting for new data from client")
        client_socket.sendall("READY_FOR_NEXT_DATA".encode())

        should_continue = False
        message_type, data = self._wait_for_data(client_socket=client_socket, send_confirmation=False)
        if message_type == _ServerMessages.NEW_DATA:
            try:
                self._update_data(data)
                should_continue = True
            except:
                self._logger.warning("Error while updating data from client, closing connexion")
                plt.close()
                client_socket.close()
        elif message_type == _ServerMessages.EMPTY or message_type == _ServerMessages.CLOSE_CONNEXION:
            self._logger.debug("Received empty data from client (end of stream), closing connexion")

        if should_continue:
            timer_get_data = threading.Timer(self._get_data_interval, self._wait_for_new_data, (client_socket,))
            timer_get_data.start()

    def _update_data(self, data_raw: list):
        header = [int(v) for v in data_raw[0].decode().split(",")]

        data = data_raw[1]
        all_data = np.array(struct.unpack("d" * (len(data) // 8), data))

        header_cmp = 0
        all_data_cmp = 0
        xdata = []
        n_phases = header[header_cmp]
        header_cmp += 1
        for _ in range(n_phases):
            n_nodes = header[header_cmp]
            header_cmp += 1
            x_phases = []
            for _ in range(n_nodes):
                n_steps = header[header_cmp]
                header_cmp += 1

                x_phases.append(all_data[all_data_cmp : all_data_cmp + n_steps])
                all_data_cmp += n_steps
            xdata.append(x_phases)

        ydata = []
        n_variables = header[header_cmp]
        header_cmp += 1
        for _ in range(n_variables):
            n_nodes = header[header_cmp]
            header_cmp += 1
            if n_nodes == 0:
                n_nodes = 1

            y_variables = []
            for _ in range(n_nodes):
                n_steps = header[header_cmp]
                header_cmp += 1

                y_variables.append(all_data[all_data_cmp : all_data_cmp + n_steps])
                all_data_cmp += n_steps
            ydata.append(y_variables)

        self._logger.debug(f"Received new data from client")
        self._plotter.update_data(xdata, ydata)


class OnlineCallbackServer(OnlineCallbackAbstract):
    def __init__(self, ocp, opts: dict = None, show_options: dict = None, host: str = None, port: int = None):
        super().__init__(ocp, opts, show_options)

        self._host = host if host else _default_host
        self._port = port if port else _default_port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

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

    def _initialize_connexion(self, **show_options):
        # Start the client
        try:
            self._socket.connect((self._host, self._port))
        except ConnectionError:
            raise RuntimeError(
                "Could not connect to the plotter server, make sure it is running "
                "by calling 'OnlineCallbackServer().start()' on another python instance)"
            )

        ocp_plot = OcpSerializable.from_ocp(self.ocp).serialize()
        dummy_phase_times = OptimizationVectorHelper.extract_step_times(self.ocp, DM(np.ones(self.ocp.n_phases)))
        ocp_plot["dummy_phase_times"] = []
        for phase_times in dummy_phase_times:
            ocp_plot["dummy_phase_times"].append([np.array(v)[:, 0].tolist() for v in phase_times])
        serialized_ocp = json.dumps(ocp_plot).encode()

        serialized_show_options = _serialize_show_options(show_options)

        # Sends message type and dimensions
        self._socket.sendall(
            f"{_ServerMessages.INITIATE_CONNEXION.value}\n{[len(serialized_ocp), len(serialized_show_options)]}".encode()
        )
        if self._socket.recv(1024).decode() != "OK":
            raise RuntimeError("The server did not acknowledge the connexion")

        # TODO ADD SHOW OPTIONS to the send
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

    def close(self):
        self._socket.sendall(f"{_ServerMessages.CLOSE_CONNEXION.value}\nGoodbye from client!".encode())
        self._socket.close()

    def eval(self, arg: list | tuple, force: bool = False) -> list:
        if not force:
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
        xdata_raw, ydata_raw = self._plotter.parse_data(**args_dict)

        header = f"{len(xdata_raw)}"
        data_serialized = b""
        for x_nodes in xdata_raw:
            header += f",{len(x_nodes)}"
            for x_steps in x_nodes:
                header += f",{x_steps.shape[0]}"
                x_steps_tp = np.array(x_steps)[:, 0].tolist()
                data_serialized += struct.pack("d" * len(x_steps_tp), *x_steps_tp)

        header += f",{len(ydata_raw)}"
        for y_nodes_variable in ydata_raw:
            if isinstance(y_nodes_variable, np.ndarray):
                header += f",0"
                y_nodes_variable = [y_nodes_variable]
            else:
                header += f",{len(y_nodes_variable)}"

            for y_steps in y_nodes_variable:
                header += f",{y_steps.shape[0]}"
                y_steps_tp = y_steps.tolist()
                data_serialized += struct.pack("d" * len(y_steps_tp), *y_steps_tp)

        self._socket.sendall(f"{_ServerMessages.NEW_DATA.value}\n{[len(header), len(data_serialized)]}".encode())
        # If send_confirmation is True, we should wait for the server to acknowledge the data here (sends OK)
        self._socket.sendall(header.encode())
        self._socket.sendall(data_serialized)
        # Again, if send_confirmation is True, we should wait for the server to acknowledge the data here (sends OK)
        return [0]
