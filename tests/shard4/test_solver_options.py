import pytest

from bioptim import Solver
from bioptim.gui.online_callback_server import OnlineCallbackServer, _ResponseHeader
from bioptim.gui.online_callback_multiprocess_server import OnlineCallbackMultiprocessServer
from bioptim.interfaces.interface_utils import generic_online_optim
from bioptim.misc.enums import SolverType, OnlineOptim


class FakeSolver:
    def __init__(
        self,
        options_common: dict = None,
    ):
        self.options_common = options_common


class FakeInterface:
    def __init__(self, online_optim):
        self.opts = type("Opts", (), {"online_optim": online_optim})()
        self.options_common = {}


def test_ipopt_solver_options():
    solver = Solver.IPOPT()
    assert solver.type == SolverType.IPOPT
    assert solver.show_online_optim is None
    assert solver.online_optim is None
    assert solver.show_options is None
    assert solver.tol == 1e-6
    assert solver.dual_inf_tol == 1.0
    assert solver.constr_viol_tol == 0.0001
    assert solver.compl_inf_tol == 0.0001
    assert solver.acceptable_tol == 1e-6
    assert solver.acceptable_dual_inf_tol == 1e10
    assert solver.acceptable_constr_viol_tol == 1e-2
    assert solver.acceptable_compl_inf_tol == 1e-2
    assert solver.max_iter == 1000
    assert solver.hessian_approximation == "exact"
    assert solver.limited_memory_max_history == 50
    assert solver.linear_solver == "mumps"
    assert solver.nlp_scaling_method == "gradient-based"
    assert solver.mu_init == 0.1
    assert solver.warm_start_init_point == "no"
    assert solver.warm_start_mult_bound_push == 0.001
    assert solver.warm_start_slack_bound_push == 0.001
    assert solver.warm_start_bound_push == 0.001
    assert solver.warm_start_slack_bound_frac == 0.001
    assert solver.warm_start_bound_frac == 0.001
    assert solver.bound_push == 0.01
    assert solver.bound_frac == 0.01
    assert solver.print_level == 5
    assert solver.c_compile is False
    assert solver.check_derivatives_for_naninf == "no"

    solver.set_linear_solver("ma57")
    assert solver.linear_solver == "ma57"
    solver.set_tol(2)
    assert solver.tol == 2
    solver.set_dual_inf_tol(3)
    assert solver.dual_inf_tol == 3
    solver.set_constr_viol_tol(4)
    assert solver.constr_viol_tol == 4
    solver.set_compl_inf_tol(5)
    assert solver.compl_inf_tol == 5
    solver.set_acceptable_tol(6)
    assert solver.acceptable_tol == 6
    solver.set_acceptable_dual_inf_tol(7)
    assert solver.acceptable_dual_inf_tol == 7
    solver.set_acceptable_constr_viol_tol(8)
    assert solver.acceptable_constr_viol_tol == 8
    solver.set_acceptable_compl_inf_tol(9)
    assert solver.acceptable_compl_inf_tol == 9
    solver.set_maximum_iterations(10)
    assert solver.max_iter == 10
    solver.set_hessian_approximation("hello bioptim")
    assert solver.hessian_approximation == "hello bioptim"
    solver.set_nlp_scaling_method("how are you?")
    assert solver.nlp_scaling_method == "how are you?"
    solver.set_limited_memory_max_history(11)
    assert solver.limited_memory_max_history == 11
    solver.set_mu_init(12)
    assert solver.mu_init == 12
    solver.set_warm_start_init_point("super!")
    assert solver.warm_start_init_point == "super!"
    solver.set_warm_start_mult_bound_push(13)
    assert solver.warm_start_mult_bound_push == 13
    solver.set_warm_start_slack_bound_push(14)
    assert solver.warm_start_slack_bound_push == 14
    solver.set_warm_start_bound_push(15)
    assert solver.warm_start_bound_push == 15
    solver.set_warm_start_slack_bound_frac(16)
    assert solver.warm_start_slack_bound_frac == 16
    solver.set_warm_start_bound_frac(17)
    assert solver.warm_start_bound_frac == 17
    solver.set_bound_push(18)
    assert solver.bound_push == 18
    solver.set_bound_frac(19)
    assert solver.bound_frac == 19
    solver.set_print_level(20)
    assert solver.print_level == 20
    solver.set_c_compile(True)
    assert solver.c_compile is True
    solver.set_check_derivatives_for_naninf(True)
    assert solver.check_derivatives_for_naninf == "yes"

    solver.set_convergence_tolerance(21)
    assert solver.tol == 21
    assert solver.acceptable_tol == 21
    assert solver.compl_inf_tol == 21
    assert solver.acceptable_compl_inf_tol == 21

    solver.set_constraint_tolerance(22)
    assert solver.constr_viol_tol == 22
    assert solver.acceptable_constr_viol_tol == 22

    solver.set_warm_start_options(42)
    assert solver.warm_start_init_point == "yes"
    assert solver.mu_init == 42
    assert solver.warm_start_mult_bound_push == 42
    assert solver.warm_start_slack_bound_push == 42
    assert solver.warm_start_bound_push == 42
    assert solver.warm_start_slack_bound_frac == 42
    assert solver.warm_start_bound_frac == 42

    solver.set_initialization_options(44)
    assert solver.bound_push == 44
    assert solver.bound_frac == 44

    solver.set_option_unsafe(666, "mysterious option")
    assert solver.__dict__["_mysterious option"] == 666

    fake_solver = FakeSolver(options_common={"ipopt.casino_gain": 777})
    solver_dict = solver.as_dict(fake_solver)
    assert solver_dict["ipopt.casino_gain"] == 777
    assert solver_dict["ipopt.tol"] == 21
    assert not "_c_compile" in solver_dict
    assert not "type" in solver_dict
    assert not "show_online_optim" in solver_dict
    assert not "online_optim" in solver_dict
    assert not "show_options" in solver_dict

    solver.set_nlp_scaling_method("gradient-fiesta")
    assert solver.nlp_scaling_method == "gradient-fiesta"


def test_generic_online_optim_skips_when_default_is_unavailable(monkeypatch):
    interface = FakeInterface(OnlineOptim.DEFAULT)

    monkeypatch.setattr(OnlineOptim, "get_default", lambda self: None)

    generic_online_optim(interface, ocp=None)

    assert interface.options_common == {}


def test_generic_online_optim_uses_multiprocess_on_linux(monkeypatch):
    interface = FakeInterface(OnlineOptim.DEFAULT)
    callback = object()

    monkeypatch.setattr("bioptim.misc.enums.platform.system", lambda: "Linux")
    monkeypatch.setattr("bioptim.interfaces.interface_utils.OnlineCallbackMultiprocess", lambda *args, **kwargs: callback)

    generic_online_optim(interface, ocp=None)

    assert interface.options_common["iteration_callback"] is callback


def test_generic_online_optim_uses_multiprocess_server_on_windows(monkeypatch):
    interface = FakeInterface(OnlineOptim.DEFAULT)
    callback = object()

    monkeypatch.setattr("bioptim.misc.enums.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "bioptim.interfaces.interface_utils.OnlineCallbackMultiprocessServer", lambda *args, **kwargs: callback
    )

    generic_online_optim(interface, ocp=None)

    assert interface.options_common["iteration_callback"] is callback


def test_generic_online_optim_uses_multiprocess_server_on_macos(monkeypatch):
    interface = FakeInterface(OnlineOptim.DEFAULT)
    callback = object()

    monkeypatch.setattr("bioptim.misc.enums.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "bioptim.interfaces.interface_utils.OnlineCallbackMultiprocessServer", lambda *args, **kwargs: callback
    )

    generic_online_optim(interface, ocp=None)

    assert interface.options_common["iteration_callback"] is callback


def test_generic_online_optim_rejects_multiprocess_on_macos(monkeypatch):
    interface = FakeInterface(OnlineOptim.MULTIPROCESS)

    monkeypatch.setattr("bioptim.interfaces.interface_utils.platform.system", lambda: "Darwin")

    with pytest.raises(NotImplementedError, match="MULTIPROCESS is not available on macOS"):
        generic_online_optim(interface, ocp=None)


def test_generic_online_optim_allows_server_on_macos(monkeypatch):
    interface = FakeInterface(OnlineOptim.SERVER)
    callback = object()

    monkeypatch.setattr("bioptim.interfaces.interface_utils.platform.system", lambda: "Darwin")
    monkeypatch.setattr("bioptim.interfaces.interface_utils.OnlineCallbackServer", lambda *args, **kwargs: callback)

    generic_online_optim(interface, ocp=None)

    assert interface.options_common["iteration_callback"] is callback


def test_online_callback_server_recreates_socket_between_retries(monkeypatch):
    class FakeSocket:
        def __init__(self, should_fail=False):
            self.should_fail = should_fail
            self.closed = False
            self.sent = []

        def connect(self, _):
            if self.should_fail:
                raise ConnectionRefusedError("server not ready")

        def close(self):
            self.closed = True

        def sendall(self, data):
            self.sent.append(data)

        def recv(self, _):
            return _ResponseHeader.PLOT_READY.encode()

    sockets = [FakeSocket(should_fail=True), FakeSocket(should_fail=False)]

    monkeypatch.setattr("bioptim.gui.online_callback_server.socket.socket", lambda *args, **kwargs: sockets.pop(0))
    monkeypatch.setattr("bioptim.gui.online_callback_server.time.sleep", lambda *_: None)
    monkeypatch.setattr(
        "bioptim.gui.online_callback_server.OcpSerializable.from_ocp",
        lambda ocp: type("FakeSerializable", (), {"serialize": lambda self: {}})(),
    )
    monkeypatch.setattr(
        "bioptim.gui.online_callback_server.OptimizationVectorHelper.extract_step_times", lambda ocp, _: []
    )
    monkeypatch.setattr("bioptim.gui.online_callback_server.PlotOcp", lambda *args, **kwargs: "plotter")

    callback = OnlineCallbackServer.__new__(OnlineCallbackServer)
    callback.ocp = object()
    callback._host = "localhost"
    callback._port = 3050
    callback._socket = None
    callback._should_wait_ok_to_client_on_new_data = False
    callback._has_received_ok = lambda: True

    callback._reset_client_socket()
    first_socket = callback._socket

    callback._initialize_connexion()

    assert first_socket.closed is True
    assert callback._socket.sent
    assert callback._plotter == "plotter"


def test_online_callback_multiprocess_server_uses_free_port_when_missing(monkeypatch):
    recorded = {}

    class FakeProcess:
        def __init__(self, target, kwargs):
            recorded["process_target"] = target
            recorded["process_kwargs"] = kwargs

        def start(self):
            recorded["started"] = True

    def fake_super_init(self, *args, **kwargs):
        recorded["super_kwargs"] = kwargs

    monkeypatch.setattr("bioptim.gui.online_callback_multiprocess_server._find_available_tcp_port", lambda host: 4242)
    monkeypatch.setattr("bioptim.gui.online_callback_multiprocess_server.Process", FakeProcess)
    monkeypatch.setattr(OnlineCallbackServer, "__init__", fake_super_init)

    OnlineCallbackMultiprocessServer(ocp=None)

    assert recorded["process_kwargs"]["port"] == 4242
    assert recorded["super_kwargs"]["port"] == 4242
    assert recorded["started"] is True
