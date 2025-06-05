from typing import Callable, Any

import numpy as np
from casadi import vertcat
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from ..optimization.non_linear_program import NonLinearProgram
from ..misc.enums import Shooting, ControlType, SolutionIntegrator
from ..misc.parameters_types import (
    NpArrayList,
    NpArray,
    Float,
)


def solve_ivp_interface(
    list_of_dynamics: list[Callable],
    shooting_type: Shooting,
    nlp: NonLinearProgram,
    t: NpArrayList,
    x: NpArrayList,
    u: NpArrayList,
    p: NpArrayList,
    a: NpArrayList,
    d: NpArrayList,
    method: SolutionIntegrator = SolutionIntegrator.SCIPY_RK45,
):
    """
    This function solves the initial value problem with the dynamics_func built by bioptim

    Parameters
    ----------
    nlp: NonLinearProgram
        The current instance of the NonLinearProgram
    t : np.ndarray
        array of time
    x : np.ndarray
        array of initial conditions
    u : np.ndarray
        arrays of controls u evaluated at t_eval
    p : np.ndarray
        array of parameters
    a : np.ndarray
        array of the algebraic states of the system
    d : np.ndarray
        array of the numerical timeseries
    shooting_type : Shooting
        The way we integrate the solution such as SINGLE, SINGLE_CONTINUOUS, MULTIPLE
    method: SolutionIntegrator
        The integrator to use to solve the OCP

    Returns
    -------
    y: np.ndarray
        array of the solution of the system at the times t_eval
    """

    y = []
    control_type = nlp.control_type

    for node in range(nlp.ns):
        if method == SolutionIntegrator.OCP:
            t_span = vertcat(t[node][0], t[node][1] - t[node][0])
        else:
            t_span = t[node]
        t_eval = np.linspace(
            float(t_span[0]), float(t_span[1]), nlp.n_states_stepwise_steps(node, nlp.dynamics_type.ode_solver)
        )

        # If multiple shooting, we need to set the first x0, otherwise use the previous answer
        x0i = np.array(x[node] if node == 0 or shooting_type == Shooting.MULTIPLE else y[-1][:, -1])

        if method == SolutionIntegrator.OCP:
            result = _solve_ivp_bioptim_interface(
                lambda t, x: nlp.dynamics[node](t, x, u[node], p, a[node], d[node])[1], x0=x0i, t_span=np.array(t_span)
            )

        elif method in (
            SolutionIntegrator.SCIPY_RK45,
            SolutionIntegrator.SCIPY_RK23,
            SolutionIntegrator.SCIPY_DOP853,
            SolutionIntegrator.SCIPY_BDF,
            SolutionIntegrator.SCIPY_LSODA,
        ):
            # Prevent from integrating collocation points
            if len(x0i.shape) > 1:
                x0i = x0i[:, 0]

            result = _solve_ivp_scipy_interface(
                lambda t, x: np.array(
                    list_of_dynamics[node](
                        t, x, _control_function(control_type, t, t_span, u[node]), p, a[node], d[node]
                    )
                )[:, 0],
                x0=x0i,
                t_span=np.array(t_span),
                t_eval=t_eval,
                method=method.value,
            )

        else:
            raise NotImplementedError(f"{method} is not implemented yet")

        y.append(result)

    y.append(x[-1] if shooting_type == Shooting.MULTIPLE else y[-1][:, -1][:, np.newaxis])

    return y


def _solve_ivp_scipy_interface(
    dynamics: Callable,
    t_span: NpArray,
    x0: NpArray,
    t_eval: NpArray,
    method: SolutionIntegrator = SolutionIntegrator.SCIPY_RK45,
):
    result: Any = solve_ivp(dynamics, y0=x0, t_span=np.array(t_span), t_eval=t_eval, method=method)
    return result.y


def _solve_ivp_bioptim_interface(
    dynamics: Callable,
    t_span: NpArray,
    x0: NpArray,
):
    # y always contains [x0, xf] of the interval
    return np.array(dynamics(t_span, x0))


def _control_function(control_type: ControlType, t: Float, t_span: NpArray, u: NpArray) -> NpArray:
    """
    This function is used to wrap the control function in a way that solve_ivp can use it

    Parameters
    ----------
    control_type: ControlType
        The type of control
    t: float
        The time at which the control is evaluated
    t_span: np.ndarray
        The time span
    u: np.ndarray
        The control value

    Returns
    -------
    np.ndarray
        The control value
    """

    if control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
        return u
    elif control_type == ControlType.LINEAR_CONTINUOUS:
        return interp1d(np.array(t_span)[:, 0], u, kind="linear", axis=1)(t)
    else:
        raise NotImplementedError("Control type not implemented in integration")
