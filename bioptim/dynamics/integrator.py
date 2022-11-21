from typing import Union

from casadi import Function, vertcat, horzcat, norm_fro, collocation_points, tangent, rootfinder, MX, SX
import numpy as np

from ..misc.enums import ControlType, DefectType


class Integrator:
    """
    Abstract class for CasADi-based integrator

    Attributes
    ----------
    model: biorbd.Model
        The biorbd model to integrate
    t_span = tuple[float, float]
        The initial and final time
    idx: int
        The index of the degrees of freedom to integrate
    cx: Union[MX, SX]
        The CasADi type the integration should be built from
    x_sym: Union[MX, SX]
        The state variables
    u_sym: Union[MX, SX]
        The control variables
    param_sym: Union[MX, SX]
        The parameters variables
    param_scaling: Union[MX, SX]
        The parameters variables scaling factor
    fun: Callable
        The dynamic function which provides the derivative of the states
    implicit_fun: Callable
        The implicit dynamic function which provides the defects of the dynamics
    control_type: ControlType
        The type of the controls
    step_time: float
        The time of the full integration
    h: float
        The time of the an integration step
    function = casadi.Function
        The CasADi graph of the integration

    Methods
    -------
    __call__(self, *args, **kwargs)
        Interface to self.function
    map(self, *args, **kwargs) -> Function
        Get the multithreaded CasADi graph of the integration
    get_u(self, u: np.ndarray, dt_norm: float) -> np.ndarray
        Get the control at a given time
    dxdt(self, h: float, states: Union[MX, SX], controls: Union[MX, SX], params: Union[MX, SX]) -> tuple[SX, list[SX]]
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
        self.t_span = ode_opt["t0"], ode_opt["tf"]
        self.idx = ode_opt["idx"]
        self.cx = ode_opt["cx"]
        self.x_sym = ode["x"]
        self.u_sym = ode["p"]
        self.param_sym = ode_opt["param"].cx
        self.param_scaling = ode_opt["param"].scaling
        self.fun = ode["ode"]
        self.implicit_fun = ode["implicit_ode"]
        self.defects_type = ode_opt["defects_type"]
        self.control_type = ode_opt["control_type"]
        self.step_time = self.t_span[1] - self.t_span[0]
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

    def get_u(self, u: np.ndarray, dt_norm: float) -> np.ndarray:
        """
        Get the control at a given time

        Parameters
        ----------
        u: np.ndarray
            The control matrix
        dt_norm: float
            The time a which control should be computed

        Returns
        -------
        The control at a given time
        """

        if self.control_type == ControlType.CONSTANT:
            return u
        elif self.control_type == ControlType.LINEAR_CONTINUOUS:
            return u[:, 0] + (u[:, 1] - u[:, 0]) * dt_norm
        else:
            raise RuntimeError(f"{self.control_type} ControlType not implemented yet")

    def dxdt(self, h: float, states: Union[MX, SX], controls: Union[MX, SX], params: Union[MX, SX]) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        states: Union[MX, SX]
            The states of the system
        controls: Union[MX, SX]
            The controls of the system
        params: Union[MX, SX]
            The parameters of the system

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
            [self.x_sym, self.u_sym, self.param_sym],
            self.dxdt(self.h, self.x_sym, self.u_sym, self.param_sym * self.param_scaling),
            ["x0", "p", "params"],
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
    next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX])
        Compute the next integrated state (abstract)
    dxdt(self, h: float, states: Union[MX, SX], controls: Union[MX, SX], params: Union[MX, SX]) -> tuple[SX, list[SX]]
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

    def next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX]):
        """
        Compute the next integrated state (abstract)

        Parameters
        ----------
        h: float
            The time step
        t: float
            The initial time of the integration
        x_prev: Union[MX, SX]
            The current state of the system
        u: Union[MX, SX]
            The control of the system
        p: Union[MX, SX]
            The parameters of the system

        Returns
        -------
        The next integrate states
        """

        raise RuntimeError("RK is abstract, please select a specific RK")

    def dxdt(self, h: float, states: Union[MX, SX], controls: Union[MX, SX], params: Union[MX, SX]) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        states: Union[MX, SX]
            The states of the system
        controls: Union[MX, SX]
            The controls of the system
        params: Union[MX, SX]
            The parameters of the system

        Returns
        -------
        The derivative of the states
        """

        u = controls
        x = self.cx(states.shape[0], self.n_step + 1)
        p = params
        x[:, 0] = states

        n_dof = 0
        quat_idx = []
        quat_number = 0
        for j in range(self.model.nbSegment()):
            if self.model.segment(j).isRotationAQuaternion():
                quat_idx.append([n_dof, n_dof + 1, n_dof + 2, self.model.nbDof() + quat_number])
                quat_number += 1
            n_dof += self.model.segment(j).nbDof()

        for i in range(1, self.n_step + 1):
            t_norm_init = (i - 1) / self.n_step  # normalized time
            x[:, i] = self.next_x(h, t_norm_init, x[:, i - 1], u, p)

            for j in range(self.model.nbQuat()):
                quaternion = vertcat(
                    x[quat_idx[j][3], i], x[quat_idx[j][0], i], x[quat_idx[j][1], i], x[quat_idx[j][2], i]
                )
                quaternion /= norm_fro(quaternion)
                x[quat_idx[j][0]: quat_idx[j][2] + 1, i] = quaternion[1:4]
                x[quat_idx[j][3], i] = quaternion[0]

        return x[:, -1], x


class RK1(RK):
    """
    Numerical integration using first order Runge-Kutta 1 Method (Forward Euler Method).

    Methods
    -------
    next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX])
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

    def next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX]):
        """
        Compute the next integrated state

        Parameters
        ----------
        h: float
            The time step
        t: float
            The initial time of the integration
        x_prev: Union[MX, SX]
            The current state of the system
        u: Union[MX, SX]
            The control of the system
        p: Union[MX, SX]
            The parameters of the system

        Returns
        -------
        The next integrate states
        """

        return x_prev + h * self.fun(x_prev, self.get_u(u, t), p)[:, self.idx]


class RK2(RK):
    """
    Numerical integration using second order Runge-Kutta Method (Midpoint Method).

    Methods
    -------
    next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX])
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

    def next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX]):
        """
        Compute the next integrated state

        Parameters
        ----------
        h: float
            The time step
        t: float
            The initial time of the integration
        x_prev: Union[MX, SX]
            The current state of the system
        u: Union[MX, SX]
            The control of the system
        p: Union[MX, SX]
            The parameters of the system

        Returns
        -------
        The next integrate states
        """
        k1 = self.fun(x_prev, self.get_u(u, t), p)[:, self.idx]
        return x_prev + h * self.fun(x_prev + h / 2 * k1, self.get_u(u, t + self.h_norm / 2), p)[:, self.idx]


class RK4(RK):
    """
    Numerical integration using fourth order Runge-Kutta method.

    Methods
    -------
    next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX])
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

    def next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX]):
        """
        Compute the next integrated state

        Parameters
        ----------
        h: float
            The time step
        t: float
            The initial time of the integration
        x_prev: Union[MX, SX]
            The current state of the system
        u: Union[MX, SX]
            The control of the system
        p: Union[MX, SX]
            The parameters of the system

        Returns
        -------
        The next integrate states
        """

        k1 = self.fun(x_prev, self.get_u(u, t), p)[:, self.idx]
        k2 = self.fun(x_prev + h / 2 * k1, self.get_u(u, t + self.h_norm / 2), p)[:, self.idx]
        k3 = self.fun(x_prev + h / 2 * k2, self.get_u(u, t + self.h_norm / 2), p)[:, self.idx]
        k4 = self.fun(x_prev + h * k3, self.get_u(u, t + self.h_norm), p)[:, self.idx]
        return x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class LF(RK):
    """
    Numerical integration using LeapFrog method.

    Methods
    -------
    next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX])
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

        super(LF, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX]):
        """
        Implements the 4th-order Yoshida integrator to compute the predicted next state.

        Parameters
        ----------
        u_k : array
            Current state vector u_k
        delta_t : float_like
            Time step size where delta_t = t_{k+1} - t_k
        n : integer
            # of bodies in the simulation

        Returns
        -------
        u_kplus1 : array
            1 x N array of the predicted next state vector u_{k+1}

        """

        q0 = x_prev[0:self.model.nbQ()]
        qdot0 = x_prev[self.model.nbQ():]

        q1 = q0 + h * qdot0 + 0.5 * h**2 * self.fun(x_prev, self.get_u(u, t), p)[self.model.nbQ():, self.idx]
        qddot1 = self.fun(vertcat(q1, qdot0), self.get_u(u, t), p)[self.model.nbQ():, self.idx]
        qdot1 = qdot0 + h * 0.5 * (self.fun(x_prev, self.get_u(u, t), p)[self.model.nbQ():, self.idx] + qddot1)

        return vertcat(q1, qdot1)


class YO4(RK):
    """
    Numerical integration using fourth order Runge-Kutta method.

    Methods
    -------
    next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX])
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

        super(YO4, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX]):
        """
        Implements the 4th-order Yoshida integrator to compute the predicted next state.

        Parameters
        ----------
        u_k : array
            Current state vector u_k
        delta_t : float_like
            Time step size where delta_t = t_{k+1} - t_k
        n : integer
            # of bodies in the simulation

        Returns
        -------
        u_kplus1 : array
            1 x N array of the predicted next state vector u_{k+1}

        """

        # gather initial position and velocity vectors
        q_prev = x_prev[0:self.model.nbQ()]
        qdot_prev = x_prev[self.model.nbQ():]

        # define constants
        w0 = -2 ** (1 / 3) / (2 - 2 ** (1 / 3))
        w1 = 1 / (2 - 2 ** (1 / 3))
        c1 = w1 / 2
        c4 = c1
        c2 = (w0 + w1) / 2
        c3 = c2
        d1 = w1
        d3 = d1
        d2 = w0

        # compute first position equation
        q_k1 = q_prev + c1 * qdot_prev * h  # TODO check if tspan or step_time

        # compute acceleration at q_k1
        xdot_k1 = self.fun(vertcat(q_k1, qdot_prev), self.get_u(u, t + c1*self.h_norm), p)[:, self.idx]  # TODO check index : was with qdot index

        # compute first velocity equation
        qdot_k1 = qdot_prev + d1 * xdot_k1[self.model.nbQ():] * h

        # compute second position equation
        q_k2 = q_k1 + c2 * qdot_k1 * h

        # compute acceleration at q_k2
        a_k2 = self.fun(vertcat(q_k2, qdot_prev), self.get_u(u, t + (c1+c2)*self.h_norm), p)[self.model.nbQ():, self.idx]
        # TODO check qdot: qdot_prev or qdot_k1

        # compute second velocity equation
        qdot_k2 = qdot_k1 + d2 * a_k2 * h

        # compute third position equation
        q_k3 = q_k2 + c3 * qdot_k2 * h

        # compute acceleration at q_k3
        x_k3_temp = vertcat(q_k3, qdot_prev)
        a_k3 = self.fun(x_k3_temp, self.get_u(u, t + (c1+c2+c3)*self.h_norm), p)[self.model.nbQ():, self.idx]  # TODO check index : was with qdot index

        # compute third velocity equation
        qdot_k3 = qdot_k2 + d3 * a_k3 * h

        q_k4 = q_k3 + c4 * qdot_k3 * h

        return vertcat(q_k4, qdot_k3)


class YO(RK):
    """
    Numerical integration using fourth order Runge-Kutta method.

    Methods
    -------
    next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX])
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

        super(YO, self).__init__(ode, ode_opt)
        self._finish_init()

    def next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX]):
        """
        Implements the 4th-order Yoshida integrator to compute the predicted next state.
https://github.com/ajaguirreguzman/netSim/blob/f7fa08700d78e5998d7ccbba177f797a8117ddd5/iterative_methods.py

        Parameters
        ----------
        u_k : array
            Current state vector u_k
        delta_t : float_like
            Time step size where delta_t = t_{k+1} - t_k
        n : integer
            # of bodies in the simulation

        Returns
        -------
        u_kplus1 : array
            1 x N array of the predicted next state vector u_{k+1}

        """

        q0 = x_prev[0:self.model.nbQ()]
        qdot0 = x_prev[self.model.nbQ():]

        # define constants
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w0 = -cbrt2 / (2 - cbrt2)
        w1 = 1 / (2 - cbrt2)
        c1 = w1 / 2.0
        c4 = c1
        c2 = (w0 + w1) / 2
        c3 = c2
        d1 = w1
        d3 = d1
        d2 = w0

        # compute first state equations
        qddot0 = self.fun(vertcat(q0, qdot0), self.get_u(u, t), p)[self.model.nbQ():, self.idx]
        q1 = q0 + c1 * h * qdot0
        qdot1 = qdot0 + c1 * h * qddot0
        qddot1 = self.fun(vertcat(q1, qdot1), self.get_u(u, t + c1 * self.h_norm), p)[self.model.nbQ():, self.idx]
        qdot1 = qdot0 + d1 * h * qddot1


        q2 = q1 + c2 * h * qdot1
        qdot2 = qdot1 + c2 * h * qddot1
        qddot2 = self.fun(vertcat(q2, qdot2), self.get_u(u, t + (c1+c2) * self.h_norm), p)[self.model.nbQ():, self.idx]
        qdot2 = qdot1 + d2 * h * qddot2

        q3 = q2 + c3 * h * qdot2
        qdot3 = qdot2 + c3 * h * qddot2
        qddot3 = self.fun(vertcat(q3, qdot3), self.get_u(u, t + (c1+c2+c3) * self.h_norm), p)[self.model.nbQ():, self.idx]
        qdot3 = qdot2 + d3 * h * qddot3

        q4 = q3 + c4 * h * qdot3

        return vertcat(q4, qdot3)


class RK8(RK4):
    """
    Numerical integration using eighth order Runge-Kutta method.

    Methods
    -------
    next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX])
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

    def next_x(self, h: float, t: float, x_prev: Union[MX, SX], u: Union[MX, SX], p: Union[MX, SX]):
        """
        Compute the next integrated state

        Parameters
        ----------
        h: float
            The time step
        t: float
            The initial time of the integration
        x_prev: Union[MX, SX]
            The current state of the system
        u: Union[MX, SX]
            The control of the system
        p: Union[MX, SX]
            The parameters of the system

        Returns
        -------
        The next integrate states
        """

        k1 = self.fun(x_prev, self.get_u(u, t), p)[:, self.idx]
        k2 = self.fun(x_prev + (h * 4 / 27) * k1, self.get_u(u, t + self.h_norm * (4 / 27)), p)[:, self.idx]
        k3 = self.fun(x_prev + (h / 18) * (k1 + 3 * k2), self.get_u(u, t + self.h_norm * (2 / 9)), p)[:, self.idx]
        k4 = self.fun(x_prev + (h / 12) * (k1 + 3 * k3), self.get_u(u, t + self.h_norm * (1 / 3)), p)[:, self.idx]
        k5 = self.fun(x_prev + (h / 8) * (k1 + 3 * k4), self.get_u(u, t + self.h_norm * (1 / 2)), p)[:, self.idx]
        k6 = self.fun(
            x_prev + (h / 54) * (13 * k1 - 27 * k3 + 42 * k4 + 8 * k5), self.get_u(u, t + self.h_norm * (2 / 3)), p
        )[:, self.idx]
        k7 = self.fun(
            x_prev + (h / 4320) * (389 * k1 - 54 * k3 + 966 * k4 - 824 * k5 + 243 * k6),
            self.get_u(u, t + self.h_norm * (1 / 6)),
            p,
        )[:, self.idx]
        k8 = self.fun(
            x_prev + (h / 20) * (-234 * k1 + 81 * k3 - 1164 * k4 + 656 * k5 - 122 * k6 + 800 * k7),
            self.get_u(u, t + self.h_norm),
            p,
        )[:, self.idx]
        k9 = self.fun(
            x_prev + (h / 288) * (-127 * k1 + 18 * k3 - 678 * k4 + 456 * k5 - 9 * k6 + 576 * k7 + 4 * k8),
            self.get_u(u, t + self.h_norm * (5 / 6)),
            p,
        )[:, self.idx]
        k10 = self.fun(
            x_prev
            + (h / 820) * (1481 * k1 - 81 * k3 + 7104 * k4 - 3376 * k5 + 72 * k6 - 5040 * k7 - 60 * k8 + 720 * k9),
            self.get_u(u, t + self.h_norm),
            p,
        )[:, self.idx]

        return x_prev + h / 840 * (41 * k1 + 27 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 216 * k9 + 41 * k10)


class COLLOCATION(Integrator):
    """
    Numerical integration using implicit Runge-Kutta method.

    Attributes
    ----------
    degree: int
        The interpolation order of the polynomial approximation

    Methods
    -------
    get_u(self, u: np.ndarray, dt_norm: float) -> np.ndarray
        Get the control at a given time
    dxdt(self, h: float, states: Union[MX, SX], controls: Union[MX, SX], params: Union[MX, SX]) -> tuple[SX, list[SX]]
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

    def get_u(self, u: np.ndarray, dt_norm: float) -> np.ndarray:
        """
        Get the control at a given time

        Parameters
        ----------
        u: np.ndarray
            The control matrix
        dt_norm: float
            The time a which control should be computed

        Returns
        -------
        The control at a given time
        """

        if self.control_type == ControlType.CONSTANT:
            return super(COLLOCATION, self).get_u(u, dt_norm)
        else:
            raise NotImplementedError(f"{self.control_type} ControlType not implemented yet with COLLOCATION")

    def dxdt(self, h: float, states: Union[MX, SX], controls: Union[MX, SX], params: Union[MX, SX]) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        states: Union[MX, SX]
            The states of the system
        controls: Union[MX, SX]
            The controls of the system
        params: Union[MX, SX]
            The parameters of the system

        Returns
        -------
        The derivative of the states
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
                f_j = self.fun(states[j], self.get_u(controls, self.step_time[j]), params)[:, self.idx]
                defects.append(h * f_j - xp_j)
            elif self.defects_type == DefectType.IMPLICIT:
                defects.append(self.implicit_fun(states[j], self.get_u(controls, self.step_time[j]), params, xp_j / h))
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
            [horzcat(*self.x_sym), self.u_sym, self.param_sym],
            self.dxdt(self.h, self.x_sym, self.u_sym, self.param_sym * self.param_scaling),
            ["x0", "p", "params"],
            ["xf", "xall", "defects"],
        )


class IRK(COLLOCATION):
    """
    Numerical integration using implicit Runge-Kutta method.

    Methods
    -------
    get_u(self, u: np.ndarray, dt_norm: float) -> np.ndarray
        Get the control at a given time
    dxdt(self, h: float, states: Union[MX, SX], controls: Union[MX, SX], params: Union[MX, SX]) -> tuple[SX, list[SX]]
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

    def dxdt(self, h: float, states: Union[MX, SX], controls: Union[MX, SX], params: Union[MX, SX]) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        h: float
            The time step
        states: Union[MX, SX]
            The states of the system
        controls: Union[MX, SX]
            The controls of the system
        params: Union[MX, SX]
            The parameters of the system

        Returns
        -------
        The derivative of the states
        """

        nx = states[0].shape[0]
        _, _, defect = super(IRK, self).dxdt(h, states, controls, params)

        # Root-finding function, implicitly defines x_collocation_points as a function of x0 and p
        vfcn = Function("vfcn", [vertcat(*states[1:]), states[0], controls, params], [defect]).expand()

        # Create a implicit function instance to solve the system of equations
        ifcn = rootfinder("ifcn", "newton", vfcn)
        x_irk_points = ifcn(self.cx(), states[0], controls, params)
        x = [states[0] if r == 0 else x_irk_points[(r - 1) * nx: r * nx] for r in range(self.degree + 1)]

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
            [self.x_sym[0], self.u_sym, self.param_sym],
            self.dxdt(self.h, self.x_sym, self.u_sym, self.param_sym * self.param_scaling),
            ["x0", "p", "params"],
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
