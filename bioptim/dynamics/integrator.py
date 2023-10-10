from casadi import (
    Function,
    vertcat,
    horzcat,
    collocation_points,
    tangent,
    rootfinder,
    MX,
    SX,
    symvar,
)
import numpy as np

from ..misc.enums import ControlType, DefectType
from ..interfaces.biomodel import BioModel


class Integrator:
    """
    Abstract class for CasADi-based integrator

    Attributes
    ----------
    model: BioModel
        The biorbd model to integrate
    time_integration_grid = tuple[float, ...]
        The time integration grid
    idx: int
        The index of the degrees of freedom to integrate
    cx: MX | SX
        The CasADi type the integration should be built from
    x_sym: MX | SX
        The state variables
    u_sym: MX | SX
        The control variables
    param_sym: MX | SX
        The parameters variables
    param_scaling: MX | SX
        The parameters variables scaling factor
    s_sym: MX | SX
        The stochastic variables
    fun: Callable
        The dynamic function which provides the derivative of the states
    implicit_fun: Callable
        The implicit dynamic function which provides the defects of the dynamics
    control_type: ControlType
        The type of the controls
    step_time: float
        The time of the full integration
    h: float
        The time of the integration step
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
    dxdt(self, h: float, time: float | MX | SX, states: MX | SX, controls: MX | SX, params: MX | SX, stochastic_variables: MX | SX) -> tuple[SX, list[SX]]
        The dynamics of the system
    _finish_init(self)
        Prepare the CasADi function from dxdt
    """

    # Todo change ode and ode_opt into class
    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        self.model = ode_opt["model"]
        self.time_integration_grid = ode_opt["time_integration_grid"]
        self.tf = ode_opt["tf"]
        self.idx = ode_opt["idx"]
        self.cx = ode_opt["cx"]
        self.x_sym = ode["x_scaled"]
        self.u_sym = [] if ode_opt["control_type"] is ControlType.NONE else ode["p_scaled"]
        self.param_sym = ode_opt["param"].cx
        self.param_scaling = ode_opt["param"].scaling
        self.s_sym = ode["s_scaled"]
        self.fun = ode["ode"]
        self.implicit_fun = ode["implicit_ode"]
        self.defects_type = ode_opt["defects_type"]
        self.control_type = ode_opt["control_type"]
        self.step_time = ode_opt["tf"] - ode_opt["t0"]
        self.h = self.step_time
        self.function = None

    def __call__(self, *args, **kwargs):
        """
        Interface to self.function
        """

        return self.function(*args, **kwargs)

    def map(self, *args, **kwargs) -> Function:
        """
        Get the multithreaded CasADi graph of the integration

        Returns
        -------
        The multithreaded CasADi graph of the integration
        """
        return self.function.map(*args, **kwargs)

    def get_u(self, u: np.ndarray, t: float) -> np.ndarray:
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

        if self.control_type == ControlType.CONSTANT or self.control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            return u
        elif self.control_type == ControlType.LINEAR_CONTINUOUS:
            dt_norm = 1 - (self.tf - t) / self.step_time
            return u[:, 0] + (u[:, 1] - u[:, 0]) * dt_norm
        elif self.control_type == ControlType.NONE:
            return np.ndarray((0,))
        else:
            raise RuntimeError(f"{self.control_type} ControlType not implemented yet")

    def dxdt(
        self,
        h: float,
        time: float | MX | SX,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        param_scaling,
        stochastic_variables: MX | SX,
    ) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        time: float | MX | SX
            The time of the system
        states: MX | SX
            The states of the system
        controls: MX | SX
            The controls of the system
        params: MX | SX
            The parameters of the system
        param_scaling
            The parameters scaling factor
        stochastic_variables: MX | SX
            The stochastic variables of the system

        Returns
        -------
        The derivative of the states
        """

        raise RuntimeError("Integrator is abstract, please specify a proper one")

    def _finish_init(self):
        """
        Prepare the CasADi function from dxdt
        """

        self.function = Function(
            "integrator",
            [
                self.x_sym,
                self.u_sym,
                self.param_sym,
                self.s_sym,
            ],
            self.dxdt(
                h=self.h,
                time=self.time_integration_grid[0],
                states=self.x_sym,
                controls=self.u_sym,
                params=self.param_sym,
                param_scaling=self.param_scaling,
                stochastic_variables=self.s_sym,
            ),
            ["x0", "u", "p", "s"],
            ["xf", "xall"],
        )


class RK(Integrator):
    """
    Abstract class for Runge-Kutta integrators

    Attributes
    ----------
    n_step: int
        Number of finite element during the integration
    h_norm: float
        Normalized time step
    h: float
        Length of steps

    Methods
    -------
    next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
        Compute the next integrated state (abstract)
    dxdt(self, h: float, time: float | MX | SX, states: MX | SX, controls: MX | SX, params: MX | SX, stochastic_variables: MX | SX) -> tuple[SX, list[SX]]
        The dynamics of the system
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """
        super(RK, self).__init__(ode, ode_opt)
        self.n_step = ode_opt["number_of_finite_elements"]
        self.h_norm = 1 / self.n_step
        self.h = self.step_time * self.h_norm

    def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX) -> MX | SX:
        """
        Compute the next integrated state (abstract)

        Parameters
        ----------
        h: float
            The time step
        t0: float | MX | SX
            The initial time of the integration
        x_prev: MX | SX
            The current state of the system
        u: MX | SX
            The control of the system
        p: MX | SX
            The parameters of the system
        s: MX | SX
            The stochastic variables of the system

        Returns
        -------
        The next integrate states
        """

        raise RuntimeError("RK is abstract, please select a specific RK")

    def dxdt(
        self,
        h: float,
        time: float | MX | SX,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        param_scaling,
        stochastic_variables: MX | SX,
    ) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        time: float | MX | SX
            The time of the system
        states: MX | SX
            The states of the system
        controls: MX | SX
            The controls of the system
        params: MX | SX
            The parameters of the system
        param_scaling
            The parameters scaling factor
        stochastic_variables: MX | SX
            The stochastic variables of the system

        Returns
        -------
        The derivative of the states
        """
        u = controls
        x = self.cx(states.shape[0], self.n_step + 1)
        p = params * param_scaling
        x[:, 0] = states
        s = stochastic_variables

        for i in range(1, self.n_step + 1):
            t = self.time_integration_grid[i - 1]
            x[:, i] = self.next_x(h, t, x[:, i - 1], u, p, s)
            if self.model.nb_quaternions > 0:
                x[:, i] = self.model.normalize_state_quaternions(x[:, i])

        return x[:, -1], x


class RK1(RK):
    """
    Numerical integration using first order Runge-Kutta 1 Method (Forward Euler Method).

    Methods
    -------
    next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
        Compute the next integrated state (abstract)
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        super(RK1, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX) -> MX | SX:
        return x_prev + h * self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.idx]


class RK2(RK):
    """
    Numerical integration using second order Runge-Kutta Method (Midpoint Method).

    Methods
    -------
    next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
        Compute the next integrated state (abstract)
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        super(RK2, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
        k1 = self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.idx]
        return x_prev + h * self.fun(t0, x_prev + h / 2 * k1, self.get_u(u, t0 + self.h / 2), p, s)[:, self.idx]


class RK4(RK):
    """
    Numerical integration using fourth order Runge-Kutta method.

    Methods
    -------
    next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
        Compute the next integrated state (abstract)
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        super(RK4, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
        k1 = self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.idx]
        k2 = self.fun(t0 + self.h / 2, x_prev + h / 2 * k1, self.get_u(u, t0 + self.h / 2), p, s)[:, self.idx]
        k3 = self.fun(t0 + self.h / 2, x_prev + h / 2 * k2, self.get_u(u, t0 + self.h / 2), p, s)[:, self.idx]
        k4 = self.fun(t0 + self.h, x_prev + h * k3, self.get_u(u, t0 + self.h), p, s)[:, self.idx]
        return x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class RK8(RK4):
    """
    Numerical integration using eighth order Runge-Kutta method.

    Methods
    -------
    next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX)
        Compute the next integrated state (abstract)
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        super(RK8, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h: float, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
        k1 = self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.idx]
        k2 = self.fun(t0, x_prev + (h * 4 / 27) * k1, self.get_u(u, t0 + self.h * (4 / 27)), p, s)[:, self.idx]
        k3 = self.fun(t0, x_prev + (h / 18) * (k1 + 3 * k2), self.get_u(u, t0 + self.h * (2 / 9)), p, s)[:, self.idx]
        k4 = self.fun(t0, x_prev + (h / 12) * (k1 + 3 * k3), self.get_u(u, t0 + self.h * (1 / 3)), p, s)[:, self.idx]
        k5 = self.fun(t0, x_prev + (h / 8) * (k1 + 3 * k4), self.get_u(u, t0 + self.h * (1 / 2)), p, s)[:, self.idx]
        k6 = self.fun(
            t0, x_prev + (h / 54) * (13 * k1 - 27 * k3 + 42 * k4 + 8 * k5), self.get_u(u, t0 + self.h * (2 / 3)), p, s
        )[:, self.idx]
        k7 = self.fun(
            t0,
            x_prev + (h / 4320) * (389 * k1 - 54 * k3 + 966 * k4 - 824 * k5 + 243 * k6),
            self.get_u(u, t0 + self.h * (1 / 6)),
            p,
            s,
        )[:, self.idx]
        k8 = self.fun(
            t0,
            x_prev + (h / 20) * (-234 * k1 + 81 * k3 - 1164 * k4 + 656 * k5 - 122 * k6 + 800 * k7),
            self.get_u(u, t0 + self.h),
            p,
            s,
        )[:, self.idx]
        k9 = self.fun(
            t0,
            x_prev + (h / 288) * (-127 * k1 + 18 * k3 - 678 * k4 + 456 * k5 - 9 * k6 + 576 * k7 + 4 * k8),
            self.get_u(u, t0 + self.h * (5 / 6)),
            p,
            s,
        )[:, self.idx]
        k10 = self.fun(
            t0,
            x_prev
            + (h / 820) * (1481 * k1 - 81 * k3 + 7104 * k4 - 3376 * k5 + 72 * k6 - 5040 * k7 - 60 * k8 + 720 * k9),
            self.get_u(u, t0 + self.h),
            p,
            s,
        )[:, self.idx]
        return x_prev + h / 840 * (41 * k1 + 27 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 216 * k9 + 41 * k10)


class TRAPEZOIDAL(Integrator):
    """
    Numerical integration using trapezoidal method.
    Not that it is only possible to have one step using trapezoidal.
    It behaves like an order 1 collocation method meaning that the integration is implicit (but since the polynomial is
    of order 1, it is not possible to put a constraint on the slopes).

    Methods
    -------
    next_x(self, h: float, t: float | MX | SX, x_prev: MX | SX, x_next: MX | SX, u: MX | SX, u_next: MX | SX, p: MX | SX, s: MX | SX)
        Compute the next integrated state
    dxdt(self, h: float, states: MX | SX, controls: MX | SX, params: MX | SX, stochastic_variables: MX | SX) -> tuple[SX, list[SX]]
        The dynamics of the system
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        super(TRAPEZOIDAL, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(
        self,
        h: float,
        t0: float | MX | SX,
        x_prev: MX | SX,
        x_next: MX | SX,
        u_prev: MX | SX,
        u_next: MX | SX,
        p: MX | SX,
        s_prev: MX | SX,
        s_next: MX | SX,
    ):
        """
        Compute the next integrated state

        Parameters
        ----------
        h: float
            The time step
        t0: float | MX | SX
            The initial time of the integration
        x_prev: MX | SX
            The current state of the system
        x_next: MX | SX
            The state of the system at the next shooting node
        u_prev: MX | SX
            The current control of the system
        u_next: MX | SX
            The control of the system at the next shooting node
        p: MX | SX
            The parameters of the system
        s_prev: MX | SX
            The current stochastic variables of the system
        s_next: MX | SX
            The stochastic variables of the system at the next shooting node

        Returns
        -------
        The next integrate states
        """
        dx = self.fun(t0, x_prev, u_prev, p, s_prev)[:, self.idx]
        dx_next = self.fun(t0, x_next, u_next, p, s_next)[:, self.idx]
        return x_prev + (dx + dx_next) * h / 2

    def dxdt(
        self,
        h: float,
        time: float | MX | SX,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        param_scaling,
        stochastic_variables: MX | SX,
    ) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        time: float | MX | SX
            The time of the system
        states: MX | SX
            The states of the system
        controls: MX | SX
            The controls of the system
        params: MX | SX
            The parameters of the system
        param_scaling
            The parameters scaling factor
        stochastic_variables: MX | SX
            The stochastic variables of the system

        Returns
        -------
        The derivative of the states
        """

        x_prev = self.cx(states.shape[0], 2)
        p = params * param_scaling

        states_next = states[:, 1]
        controls_prev = controls[:, 0]
        controls_next = controls[:, 1]
        if stochastic_variables.shape != (0, 0):
            stochastic_variables_prev = stochastic_variables[:, 0]
            stochastic_variables_next = stochastic_variables[:, 1]
        else:
            stochastic_variables_prev = stochastic_variables
            stochastic_variables_next = stochastic_variables

        x_prev[:, 0] = states[:, 0]

        x_prev[:, 1] = self.next_x(
            h,
            time,
            x_prev[:, 0],
            states_next,
            controls_prev,
            controls_next,
            p,
            stochastic_variables_prev,
            stochastic_variables_next,
        )

        if self.model.nb_quaternions > 0:
            x_prev[:, 1] = self.model.normalize_state_quaternions(x_prev[:, 1])

        return x_prev[:, 1], x_prev

    def _finish_init(self):
        """
        Prepare the CasADi function from dxdt
        """

        self.function = Function(
            "integrator",
            [
                self.x_sym,
                self.u_sym,
                self.param_sym,
                self.s_sym,
            ],
            self.dxdt(
                self.h,
                self.time_integration_grid[0],
                self.x_sym,
                self.u_sym,
                self.param_sym,
                self.param_scaling,
                self.s_sym,
            ),
            ["x0", "u", "p", "s"],
            ["xf", "xall"],
        )


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
    dxdt(self, h: float, time: float | MX | SX, states: MX | SX, controls: MX | SX, params: MX | SX, stochastic_variables: MX | SX) -> tuple[SX, list[SX]]
        The dynamics of the system
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        super(COLLOCATION, self).__init__(ode, ode_opt)

        self.method = ode_opt["method"]
        self.degree = ode_opt["irk_polynomial_interpolation_degree"]

        # Coefficients of the collocation equation
        self._c = self.cx.zeros((self.degree + 1, self.degree + 1))

        # Coefficients of the continuity equation
        self._d = self.cx.zeros(self.degree + 1)

        # Choose collocation points
        self.step_time = [0] + collocation_points(self.degree, self.method)

        # Dimensionless time inside one control interval
        time_control_interval = self.cx.sym("time_control_interval")

        # For all collocation points
        for j in range(self.degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            _l = 1
            for r in range(self.degree + 1):
                if r != j:
                    _l *= (time_control_interval - self.step_time[r]) / (self.step_time[j] - self.step_time[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            if self.method == "radau":
                self._d[j] = 1 if j == self.degree else 0
            else:
                lfcn = Function("lfcn", [time_control_interval], [_l])
                self._d[j] = lfcn(1.0)

            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            _l = 1
            for r in range(self.degree + 1):
                if r != j:
                    _l *= (time_control_interval - self.step_time[r]) / (self.step_time[j] - self.step_time[r])

            # Evaluate the time derivative of the polynomial at all collocation points to get
            # the coefficients of the continuity equation
            tfcn = Function("tfcn", [time_control_interval], [tangent(_l, time_control_interval)])
            for r in range(self.degree + 1):
                self._c[j, r] = tfcn(self.step_time[r])

        self._finish_init()

    def get_u(self, u: np.ndarray, t: float | MX | SX) -> np.ndarray:
        """
        Get the control at a given time

        Parameters
        ----------
        u: np.ndarray
            The control matrix
        t: float | MX | SX
            The time a which control should be computed

        Returns
        -------
        The control at a given time
        """

        if self.control_type == ControlType.CONSTANT or self.control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            return super(COLLOCATION, self).get_u(u, t)
        else:
            raise NotImplementedError(f"{self.control_type} ControlType not implemented yet with COLLOCATION")

    def dxdt(
        self,
        h: float,
        time: float | MX | SX,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        param_scaling,
        stochastic_variables: MX | SX,
    ) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        time: float | MX | SX
            The time of the system
        states: MX | SX
            The states of the system
        controls: MX | SX
            The controls of the system
        params: MX | SX
            The parameters of the system
        param_scaling: MX | SX
            The parameters scaling of the system
        stochastic_variables: MX | SX
            The stochastic variables of the system

        Returns
        -------
        states_end: MX | SX
            The evaluation of the polynomial at the end of the interval (states integrated)
        horzcat(states[0], states_end): MX | SX
            The states at each collocation point
        defects: list[MX | SX] (shape = degree)
            The constraints insuring that the polynomial has the right derivative at each collocation point
        """

        # Total number of variables for one finite element
        states_end = self._d[0] * states[0]
        defects = []
        for j in range(1, self.degree + 1):
            # Expression for the state derivative at the collocation point
            xp_j = 0
            for r in range(self.degree + 1):
                xp_j += self._c[r, j] * states[r]

            if self.defects_type == DefectType.EXPLICIT:
                f_j = self.fun(
                    time,
                    states[j],
                    self.get_u(controls, self.step_time[j]),
                    params * param_scaling,
                    stochastic_variables,
                )[:, self.idx]
                defects.append(h * f_j - xp_j)
            elif self.defects_type == DefectType.IMPLICIT:
                defects.append(
                    self.implicit_fun(
                        time,
                        states[j],
                        self.get_u(controls, time),
                        params * param_scaling,
                        stochastic_variables,
                        xp_j / h,
                    )
                )
            else:
                raise ValueError("Unknown defects type. Please use 'explicit' or 'implicit'")

            # Add contribution to the end state
            states_end += self._d[j] * states[j]

        # Concatenate constraints
        defects = vertcat(*defects)
        return states_end, horzcat(states[0], states_end), defects

    def _finish_init(self):
        """
        Prepare the CasADi function from dxdt
        """

        self.function = Function(
            "integrator",
            [
                horzcat(*self.x_sym),
                self.u_sym,
                self.param_sym,
                self.s_sym,
            ],
            self.dxdt(
                h=self.h,
                time=self.time_integration_grid[0],
                states=self.x_sym,
                controls=self.u_sym,
                params=self.param_sym,
                param_scaling=self.param_scaling,
                stochastic_variables=self.s_sym,
            ),
            ["x0", "u", "p", "s"],
            ["xf", "xall", "defects"],
        )


class IRK(COLLOCATION):
    """
    Numerical integration using implicit Runge-Kutta method.

    Methods
    -------
    get_u(self, u: np.ndarray, t: float) -> np.ndarray
        Get the control at a given time
    dxdt(self, h: float, t: float | MX | SX, states: MX | SX, controls: MX | SX, params: MX | SX, stochastic_variables: MX | SX) -> tuple[SX, list[SX]]
        The dynamics of the system
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        super(IRK, self).__init__(ode, ode_opt)

    def dxdt(
        self,
        h: float,
        time: float | MX | SX,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        param_scaling,
        stochastic_variables: MX | SX,
    ) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        time: float | MX | SX
            The time of the system
        states: MX | SX
            The states of the system
        controls: MX | SX
            The controls of the system
        params: MX | SX
            The parameters of the system
        param_scaling
            The parameters scaling of the system
        stochastic_variables: MX | SX
            The stochastic variables of the system

        Returns
        -------
        The derivative of the states
        """

        nx = states[0].shape[0]
        _, _, defect = super(IRK, self).dxdt(
            h=h,
            time=time,
            states=states,
            controls=controls,
            params=params,
            param_scaling=param_scaling,
            stochastic_variables=stochastic_variables,
        )

        # Root-finding function, implicitly defines x_collocation_points as a function of x0 and p
        time_sym = []
        vfcn = Function(
            "vfcn",
            [vertcat(*states[1:]), time_sym, states[0], controls, params, stochastic_variables],
            [defect],
        ).expand()

        # Create a implicit function instance to solve the system of equations
        ifcn = rootfinder("ifcn", "newton", vfcn)
        x_irk_points = ifcn(self.cx(), time, states[0], controls, params, stochastic_variables)
        x = [states[0] if r == 0 else x_irk_points[(r - 1) * nx : r * nx] for r in range(self.degree + 1)]

        # Get an expression for the state at the end of the finite element
        xf = self.cx.zeros(nx, self.degree + 1)  # 0 #
        for r in range(self.degree + 1):
            xf[:, r] = xf[:, r - 1] + self._d[r] * x[r]

        return xf[:, -1], horzcat(states[0], xf[:, -1])

    def _finish_init(self):
        """
        Prepare the CasADi function from dxdt
        """

        self.function = Function(
            "integrator",
            [
                self.x_sym[0],
                self.u_sym,
                self.param_sym,
                self.s_sym,
            ],
            self.dxdt(
                h=self.h,
                time=self.time_integration_grid[0],
                states=self.x_sym,
                controls=self.u_sym,
                params=self.param_sym,
                param_scaling=self.param_scaling,
                stochastic_variables=self.s_sym,
            ),
            ["x0", "u", "p", "s"],
            ["xf", "xall"],
        )


class CVODES(Integrator):
    """
    Class for CVODES integrators

    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        super(CVODES, self).__init__(ode, ode_opt)
