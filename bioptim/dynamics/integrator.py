import numpy as np
from casadi import Function, vertcat, horzcat, collocation_points, rootfinder, DM, MX, SX, linspace

from .lagrange_interpolation import LagrangeInterpolation
from ..misc.enums import ControlType
from ..models.protocols.biomodel import BioModel
from ..misc.parameters_types import (
    Float,
    DoubleIntTuple,
    StrList,
    AnyDict,
    AnyTuple,
    NpArray,
    CX,
)


class Integrator:
    """
    Abstract class for CasADi-based integrator

    Attributes
    ----------
    model: BioModel
        The biorbd model to integrate
    time_integration_grid = tuple[float, ...]
        The time integration grid
    cx: MX | SX
        The CasADi type the integration should be built from
    x_sym: MX | SX
        The state variables
    u_sym: MX | SX
        The control variables
    param_sym: MX | SX
        The parameters variables
    a_sym: MX | SX
        The algebraic states variables
    numerical_timeseries_sym: MX | SX
        The numerical timeseries variables
    fun: Callable
        The dynamic function which provides the derivative of the states
    implicit_fun: Callable
        The implicit dynamic function which provides the defects of the dynamics
    control_type: ControlType
        The type of the controls
    function = casadi.Function
        The CasADi graph of the integration

    Methods
    -------
    __call__(self, *args, **kwargs)
        Interface to self.function
    map(self, *args, **kwargs) -> Function
        Get the multithreaded CasADi graph of the integration
    get_u(self, u: np.ndarray, t: float) -> np.ndarray
        Get the control at a given time
    compute_states_end(self, h: float, time: float | MX | SX, states: MX | SX, controls: MX | SX, params: MX | SX, algebraic_states: MX | SX, numerical_timeseries: MX | SX) -> tuple[SX, list[SX]]
        Integrate dxdt over one complete shooting interval to get the states at the end of the interval.
    """

    # Todo change ode and ode_opt into class
    def __init__(self, ode: AnyDict, ode_opt: AnyDict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        self.model = ode_opt["model"]
        self.ode_idx = ode_opt["ode_index"]
        self.cx = ode_opt["cx"]
        self.t_span_sym = ode["t"]
        self.x_sym = ode["x"]
        self.u_sym = ode["u"]
        self.param_sym = ode["param"]
        self.a_sym = ode["a"]
        self.numerical_timeseries_sym = ode["d"]
        self.fun = ode["ode"]
        self.implicit_fun = ode["implicit_ode"]
        self.defects_type = ode_opt["defects_type"]
        self.control_type = ode_opt["control_type"]
        self.function = None
        self.duplicate_starting_point = ode_opt["duplicate_starting_point"]

        # Initialize is expected to set step_time
        self._initialize(ode, ode_opt)

        self.step_times_from_dt = self._time_xall_from_dt_func
        self.function = Function(
            "integrator",
            [
                self.t_span_sym,
                self._x_sym_modified,
                self.u_sym,
                self.param_sym,
                self._a_sym_modified,
                self.numerical_timeseries_sym,
            ],
            self.compute_states_end(
                states=self.x_sym,
                controls=self.u_sym,
                params=self.param_sym,
                algebraic_states=self.a_sym,
                numerical_timeseries=self.numerical_timeseries_sym,
            ),
            self._input_names,
            self._output_names,
        )

    @property
    def shape_xf(self) -> DoubleIntTuple:
        """
        Returns the expected shape of xf
        """
        raise NotImplementedError("This method should be implemented for a given integrator")

    @property
    def shape_xall(self) -> DoubleIntTuple:
        """
        Returns the expected shape of xall
        """
        raise NotImplementedError("This method should be implemented for a given integrator")

    @property
    def time_xall(self) -> DM:
        """
        Returns the time vector of xall
        """
        raise NotImplementedError("This method should be implemented for a given integrator")

    @property
    def _time_xall_from_dt_func(self) -> Function:
        raise NotImplementedError("This method should be implemented for a given integrator")

    @property
    def _x_sym_modified(self) -> CX:
        return self.x_sym

    @property
    def _a_sym_modified(self) -> CX:
        return self.a_sym

    @property
    def _input_names(self) -> StrList:
        return ["t_span", "x0", "u", "p", "a", "d"]

    @property
    def _output_names(self) -> StrList:
        return ["xf", "xall"]

    def _initialize(self, ode: AnyDict, ode_opt: AnyDict):
        """
        This method is called by the constructor to initialize the integrator right before
        creating the CasADi function from compute_states_end
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Interface to self.function
        """

        return self.function(*args, **kwargs)

    def map(self, *args) -> Function:
        """
        Get the multithreaded CasADi graph of the integration

        Returns
        -------
        The multithreaded CasADi graph of the integration
        """
        return self.function.map(*args)

    @property
    def _integration_time(self):
        raise NotImplementedError("This method should be implemented for a given integrator")

    def get_u(self, u: NpArray, t: Float) -> NpArray:
        """
        Get the control at a given time

        Parameters
        ----------
        u: np.ndarray
            The control matrix
        t: float
            The time a which control should be computed

        Returns
        -------
        The control at a given time
        """

        if self.control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
            return u
        elif self.control_type == ControlType.LINEAR_CONTINUOUS:
            dt_norm = (t - self.t_span_sym[0]) / self.t_span_sym[1]
            return u[:, 0] + (u[:, 1] - u[:, 0]) * dt_norm
        else:
            raise RuntimeError(f"{self.control_type} ControlType not implemented yet")

    def compute_states_end(
        self,
        states: CX,
        controls: CX,
        params: CX,
        algebraic_states: CX,
        numerical_timeseries: CX,
    ) -> AnyTuple:
        """
        The dynamics of the system

        Parameters
        ----------
        states: MX | SX
            The states of the system
        controls: MX | SX
            The controls of the system
        params: MX | SX
            The parameters of the system
        algebraic_states: MX | SX
            The algebraic states of the system
        numerical_timeseries: MX | SX
            The numerical timeseries of the system

        Returns
        -------
        The derivative of the states
        """

        raise RuntimeError("Integrator is abstract, please specify a proper one")


class RK(Integrator):
    """
    Abstract class for Runge-Kutta integrators

    Attributes
    ----------
    _n_step: int
        Number of finite element during the integration
    """

    def __init__(self, ode: AnyDict, ode_opt: AnyDict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """
        self._n_step = ode_opt["number_of_finite_elements"]
        super(RK, self).__init__(ode, ode_opt)

    @property
    def _integration_time(self):
        return self.t_span_sym[1] / self._n_step

    @property
    def shape_xf(self) -> DoubleIntTuple:
        return (self.x_sym.shape[0], 1)

    @property
    def shape_xall(self) -> DoubleIntTuple:
        return (self.x_sym.shape[0], self._n_step + 1)

    @property
    def _time_xall_from_dt_func(self) -> Function:
        return Function(
            "step_time",
            [self.t_span_sym],
            [linspace(self.t_span_sym[0], self.t_span_sym[0] + self.t_span_sym[1], self.shape_xall[1])],
        )

    @property
    def h(self):
        return self.t_span_sym[1] / self._n_step

    @property
    def dt(self):
        return self.t_span_sym[1]

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, a: MX | SX, d: MX | SX) -> MX | SX:
        """
        Compute the next integrated state (abstract)

        Parameters
        ----------
        t0: float | MX | SX
            The initial time of the integration
        x_prev: MX | SX
            The current state of the system
        u: MX | SX
            The control of the system
        p: MX | SX
            The parameters of the system
        a: MX | SX
            The algebraic states of the system
        d: MX | SX
            The numerical timeseries of the system

        Returns
        -------
        The next integrate states
        """

        raise RuntimeError("RK is abstract, please select a specific RK")

    def compute_states_end(
        self,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        algebraic_states: MX | SX,
        numerical_timeseries: MX | SX,
    ) -> tuple:
        u = controls
        x = self.cx(states.shape[0], self._n_step + 1)
        p = params
        x[:, 0] = states
        a = algebraic_states
        d = numerical_timeseries

        for i in range(1, self._n_step + 1):
            t = self.t_span_sym[0] + self._integration_time * (i - 1)
            x[:, i] = self.next_x(t, x[:, i - 1], u, p, a, d)
            if self.model.nb_quaternions > 0:
                x[: self.model.nb_q, i] = self.model.normalize_state_quaternions()(x[: self.model.nb_q, i])

        return x[:, -1], x


class RK1(RK):
    """
    Numerical integration using first order Runge-Kutta 1 Method (Forward Euler Method).
    """

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, a: MX | SX, d: MX | SX) -> MX | SX:
        return x_prev + self.h * self.fun(vertcat(t0, self.dt), x_prev, self.get_u(u, t0), p, a, d)[:, self.ode_idx]


class RK2(RK):
    """
    Numerical integration using second order Runge-Kutta Method (Midpoint Method).
    """

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, a: MX | SX, d: MX | SX) -> MX | SX:
        h = self.h
        dt = self.dt

        k1 = self.fun(vertcat(t0, dt), x_prev, self.get_u(u, t0), p, a, d)[:, self.ode_idx]
        return (
            x_prev
            + h
            * self.fun(vertcat(t0 + h / 2, dt), x_prev + h / 2 * k1, self.get_u(u, t0 + h / 2), p, a, d)[
                :, self.ode_idx
            ]
        )


class RK4(RK):
    """
    Numerical integration using fourth order Runge-Kutta method.
    """

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, a: MX | SX, d: MX | SX) -> MX | SX:
        h = self.h
        dt = self.dt

        k1 = self.fun(vertcat(t0, dt), x_prev, self.get_u(u, t0), p, a, d)[:, self.ode_idx]
        k2 = self.fun(vertcat(t0 + h / 2, dt), x_prev + h / 2 * k1, self.get_u(u, t0 + h / 2), p, a, d)[:, self.ode_idx]
        k3 = self.fun(vertcat(t0 + h / 2, dt), x_prev + h / 2 * k2, self.get_u(u, t0 + h / 2), p, a, d)[:, self.ode_idx]
        k4 = self.fun(vertcat(t0 + h, dt), x_prev + h * k3, self.get_u(u, t0 + h), p, a, d)[:, self.ode_idx]
        return x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class RK8(RK4):
    """
    Numerical integration using eighth order Runge-Kutta method.
    """

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, a: MX | SX, d: MX | SX) -> MX | SX:
        h = self.h
        dt = self.dt

        k1 = self.fun(vertcat(t0, dt), x_prev, self.get_u(u, t0), p, a, d)[:, self.ode_idx]
        k2 = self.fun(
            vertcat(t0 + h * 4 / 27, dt),
            x_prev + (h * 4 / 27) * k1,
            self.get_u(u, t0 + h * (4 / 27)),
            p,
            a,
            d,
        )[:, self.ode_idx]
        k3 = self.fun(
            vertcat(t0 + h / 18, dt),
            x_prev + (h / 18) * (k1 + 3 * k2),
            self.get_u(u, t0 + h * (2 / 9)),
            p,
            a,
            d,
        )[:, self.ode_idx]
        k4 = self.fun(
            vertcat(t0 + h / 12, dt),
            x_prev + (h / 12) * (k1 + 3 * k3),
            self.get_u(u, t0 + h * (1 / 3)),
            p,
            a,
            d,
        )[:, self.ode_idx]
        k5 = self.fun(
            vertcat(t0 + h / 8, dt),
            x_prev + (h / 8) * (k1 + 3 * k4),
            self.get_u(u, t0 + h * (1 / 2)),
            p,
            a,
            d,
        )[:, self.ode_idx]
        k6 = self.fun(
            vertcat(t0 + h / 54, dt),
            x_prev + (h / 54) * (13 * k1 - 27 * k3 + 42 * k4 + 8 * k5),
            self.get_u(u, t0 + h * (2 / 3)),
            p,
            a,
            d,
        )[:, self.ode_idx]
        k7 = self.fun(
            vertcat(t0 + h / 4320, dt),
            x_prev + (h / 4320) * (389 * k1 - 54 * k3 + 966 * k4 - 824 * k5 + 243 * k6),
            self.get_u(u, t0 + h * (1 / 6)),
            p,
            a,
            d,
        )[:, self.ode_idx]
        k8 = self.fun(
            vertcat(t0 + h / 20, dt),
            x_prev + (h / 20) * (-234 * k1 + 81 * k3 - 1164 * k4 + 656 * k5 - 122 * k6 + 800 * k7),
            self.get_u(u, t0 + h),
            p,
            a,
            d,
        )[:, self.ode_idx]
        k9 = self.fun(
            vertcat(t0 + h / 288, dt),
            x_prev + (h / 288) * (-127 * k1 + 18 * k3 - 678 * k4 + 456 * k5 - 9 * k6 + 576 * k7 + 4 * k8),
            self.get_u(u, t0 + h * (5 / 6)),
            p,
            a,
            d,
        )[:, self.ode_idx]
        k10 = self.fun(
            vertcat(t0 + h / 820, dt),
            x_prev
            + (h / 820) * (1481 * k1 - 81 * k3 + 7104 * k4 - 3376 * k5 + 72 * k6 - 5040 * k7 - 60 * k8 + 720 * k9),
            self.get_u(u, t0 + h),
            p,
            a,
            d,
        )[:, self.ode_idx]
        return x_prev + h / 840 * (41 * k1 + 27 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 216 * k9 + 41 * k10)


class TRAPEZOIDAL(Integrator):
    """
    Numerical integration using trapezoidal method.
    Not that it is only possible to have one step using trapezoidal.
    It behaves like an order 1 collocation method meaning that the integration is implicit (but since the polynomial is
    of order 1, it is not possible to put a constraint on the slopes).
    """

    def __init__(self, ode: dict, ode_opt: dict):
        self._n_step = 1
        super().__init__(ode, ode_opt)

    def next_x(
        self,
        t0: float | MX | SX,
        x_prev: MX | SX,
        x_next: MX | SX,
        u_prev: MX | SX,
        u_next: MX | SX,
        p: MX | SX,
        a_prev: MX | SX,
        a_next: MX | SX,
        d_prev: MX | SX,
        d_next: MX | SX,
    ):
        dx = self.fun(vertcat(t0, self.dt), x_prev, u_prev, p, a_prev, d_prev)[:, self.ode_idx]
        dx_next = self.fun(vertcat(t0 + self.h, self.dt), x_next, u_next, p, a_next, d_next)[:, self.ode_idx]
        return x_prev + (dx + dx_next) * self.h / 2

    @property
    def _time_xall_from_dt_func(self) -> Function:
        return Function(
            "step_time",
            [self.t_span_sym],
            [linspace(self.t_span_sym[0], self.t_span_sym[0] + self.t_span_sym[1], self.shape_xall[1])],
        )

    @property
    def shape_xf(self):
        return [self.x_sym.shape[0], 1]

    @property
    def shape_xall(self):
        return [self.x_sym.shape[0], self._n_step + 1]

    @property
    def h(self):
        return self.t_span_sym[1]

    @property
    def dt(self):
        return self.t_span_sym[1]

    def compute_states_end(
        self,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        algebraic_states: MX | SX,
        numerical_timeseries: MX | SX,
    ) -> tuple:

        x_prev = self.cx(states.shape[0], 2)

        states_next = states[:, 1]
        controls_prev = controls[:, 0]
        controls_next = controls[:, 1]
        if algebraic_states.shape != (0, 0):
            algebraic_states_prev = algebraic_states[:, 0]
            algebraic_states_next = algebraic_states[:, 1]
        else:
            algebraic_states_prev = algebraic_states
            algebraic_states_next = algebraic_states
        if numerical_timeseries.shape != (0, 0):
            numerical_timeseries_prev = numerical_timeseries[:, 0]
            numerical_timeseries_next = numerical_timeseries[:, 1]
        else:
            numerical_timeseries_prev = numerical_timeseries
            numerical_timeseries_next = numerical_timeseries

        x_prev[:, 0] = states[:, 0]

        x_prev[:, 1] = self.next_x(
            self.t_span_sym[0],
            x_prev[:, 0],
            states_next,
            controls_prev,
            controls_next,
            params,
            algebraic_states_prev,
            algebraic_states_next,
            numerical_timeseries_prev,
            numerical_timeseries_next,
        )

        if self.model.nb_quaternions > 0:
            x_prev[: self.model.nb_q, 1] = self.model.normalize_state_quaternions(x_prev[: self.model.nb_q, 1])

        return x_prev[:, 1], x_prev


class COLLOCATION(Integrator):
    """
    Numerical integration using implicit Runge-Kutta method.

    Attributes
    ----------
    degree: int
        The interpolation order of the polynomial approximation

    Methods
    -------
    get_u(self, u: np.ndarray, t: float | MX | SX) -> np.ndarray
        Get the control at a given time
    """

    def _initialize(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """
        self.method = ode_opt["method"]
        self.degree = ode_opt["irk_polynomial_interpolation_degree"]
        self.lagrange_interpolation = LagrangeInterpolation(time_grid=self._integration_time)

    @property
    def _x_sym_modified(self):
        return horzcat(*self.x_sym) if self.duplicate_starting_point else horzcat(*self.x_sym[1:])

    @property
    def _a_sym_modified(self):
        return horzcat(*self.a_sym) if self.duplicate_starting_point else horzcat(*self.a_sym[1:])

    @property
    def _output_names(self):
        return ["xf", "xall", "defects"]

    @property
    def h(self):
        return self.t_span_sym[1]

    @property
    def dt(self):
        return self.t_span_sym[1]

    @property
    def _integration_time(self):
        return [0] + collocation_points(self.degree, self.method)

    @property
    def shape_xf(self) -> DoubleIntTuple:
        return (self._x_sym_modified.shape[0], self.degree + 1)

    @property
    def shape_xall(self) -> DoubleIntTuple:
        return (self._x_sym_modified.shape[0], self.degree + 2)

    @property
    def _time_xall_from_dt_func(self) -> Function:
        return Function("step_time", [self.t_span_sym], [self.t_span_sym[0] + (self._integration_time + [1]) * self.h])

    def get_u(self, u: NpArray, t: Float | CX) -> NpArray:
        """
        Get the control at a given time

        Parameters
        ----------
        u: np.ndarray
            The control matrix
        t: float | MX | SX
            The time a which control should be computed (t is computed using _integration_time and is in [0, 1])

        Returns
        -------
        The control at a given time
        """

        if self.control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
            return u
        elif self.control_type == ControlType.LINEAR_CONTINUOUS:
            return u[:, 0] + (u[:, 1] - u[:, 0]) * t
        else:
            raise RuntimeError(f"{self.control_type} ControlType not implemented yet")

    def compute_states_end(
        self,
        states: CX,
        controls: CX,
        params: CX,
        algebraic_states: CX,
        numerical_timeseries: CX,
    ) -> AnyTuple:

        if self.method == "radau":
            # For Radau, the last collocation point is the same as the final point of the interval
            states_end = states[-1]
        else:
            # For Legendre, the final point is obtained by interpolation
            states_end = self.lagrange_interpolation.interpolate(states[1:], 1.0)

        defects = []
        for j in range(1, self.degree + 1):
            t = vertcat(self.t_span_sym[0] + self._integration_time[j] * self.h, self.h)

            xp_j = self.lagrange_interpolation.interpolate_first_derivative(
                states[0:1] + states[2:], self._integration_time[j]
            )

            defects.append(
                self.implicit_fun(
                    t,
                    states[j + 1],  # +1 instead of 0 since the first subnode is duplicated
                    self.get_u(controls, self._integration_time[j]),
                    params,
                    algebraic_states[j + 1],  # +1 instead of 0 since the first subnode is duplicated
                    numerical_timeseries,
                    xp_j / self.h,
                )
            )

        # Concatenate constraints
        defects = vertcat(*defects)
        collocation_states = horzcat(*states) if self.duplicate_starting_point else horzcat(*states[1:])
        return states_end, collocation_states, defects


class IRK(COLLOCATION):
    """
    Numerical integration using implicit Runge-Kutta method.

    Methods
    -------
    get_u(self, u: np.ndarray, t: float) -> np.ndarray
        Get the control at a given time
    """

    @property
    def _x_sym_modified(self):
        return self.x_sym[0]

    @property
    def _output_names(self) -> StrList:
        return ["xf", "xall"]

    @property
    def shape_xf(self) -> DoubleIntTuple:
        return (self._x_sym_modified.shape[0], 1)

    @property
    def shape_xall(self) -> DoubleIntTuple:
        return (self._x_sym_modified.shape[0], 2)

    @property
    def _time_xall_from_dt_func(self) -> Function:
        return Function(
            "step_time", [self.t_span_sym], [vertcat(*[self.t_span_sym[0], self.t_span_sym[0] + self.t_span_sym[1]])]
        )

    def compute_states_end(
        self,
        states: CX,
        controls: CX,
        params: CX,
        algebraic_states: CX,
        numerical_timeseries: CX,
    ) -> AnyTuple:
        nx = states[0].shape[0]
        _, _, defect = super(IRK, self).compute_states_end(
            states=states,
            controls=controls,
            params=params,
            algebraic_states=algebraic_states,
            numerical_timeseries=numerical_timeseries,
        )

        # Root-finding function, implicitly defines x_collocation_points as a function of x0 and p
        collocation_states = vertcat(*states[1:]) if self.duplicate_starting_point else vertcat(*states[2:])
        algebraic_states = (
            vertcat(*algebraic_states[1:]) if self.duplicate_starting_point else vertcat(*algebraic_states[2:])
        )
        vfcn = Function(
            "vfcn",
            [collocation_states, self.t_span_sym, states[0], controls, params, algebraic_states, numerical_timeseries],
            [defect],
        ).expand()

        # Create an implicit function instance to solve the system of equations
        ifcn = rootfinder("ifcn", "newton", vfcn, {"error_on_fail": False})
        t = vertcat(self.t_span_sym)  # We should not subtract here as it is already formally done in COLLOCATION
        x_irk_points = ifcn(self.cx(), t, states[0], controls, params, algebraic_states, numerical_timeseries)
        x = [states[0] if r == 0 else x_irk_points[(r - 1) * nx : r * nx] for r in range(self.degree + 1)]

        xf = self.cx.zeros(nx, self.degree + 1)
        for r in range(self.degree + 1):
            xf[:, r] = self.lagrange_interpolation.interpolate(x, self._integration_time[r])

        if self.method == "radau":
            # For Radau, the last collocation point is the same as the final point of the interval
            states_end = x[-1]
        else:
            # For Legendre, the final point is obtained by interpolation
            states_end = self.lagrange_interpolation.interpolate(x, 1.0)

        return states_end, horzcat(states[0], states_end)


class CVODES(Integrator):
    """
    Class for CVODES integrators

    """


class VARIATIONAL(RK4):
    """
    Fake class for variational integrator.
    The real work is done in VariationalOptimalControlProgram.
    TODO: The implementation of the variational integrator could be moved here (see issue #962).
    """
