from typing import Union, List, Callable, Any
import numpy as np
from scipy.integrate import solve_ivp


def solve_ivp_interface(
    dynamics_func: Callable,
    t_eval: Union[np.ndarray, List[float]],
    x0: np.ndarray,
    u: np.ndarray,
    params: np.ndarray,
    method: str = "RK45",
    keep_intermediate_points: bool = False,
):
    """
    This function solves the initial value problem with scipy.integrate.solve_ivp

    Parameters
    ----------
    dynamics_func : Callable
        function that computes the dynamics of the system
    t_eval : Union[np.ndarray, List[float]]
        array of times t the controls u are evaluated at
    x0 : np.ndarray
        array of initial conditions
    u : np.ndarray
        arrays of controls u evaluated at t_eval
    params : np.ndarray
        array of parameters
    method : str, optional
        method to use for the integration, by default "RK45"
    keep_intermediate_points : bool
        whether to keep the intermediate points or not, by default False

    Returns
    -------
    y: np.ndarray
        array of the solution of the system at the times t_eval

    """
    if keep_intermediate_points:
        # repeat values of u to match the size of t_eval
        u = np.concatenate((
            np.repeat(u[:, :-1], t_eval[:-1].shape[0] / u[: , :-1].shape[1], axis=1),
            u[:, -1:]),
            axis=1,
        )

    t_span = [t_eval[0], t_eval[-1]]
    integrated_sol = solve_ivp(
        lambda t, x: np.array(dynamics_func(x, piecewise_constant_u(t, t_eval, u), params))[:, 0],
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method=method,
    )

    return integrated_sol.y


def piecewise_constant_u(t: float, t_eval: Union[np.ndarray, List[float]], u: np.ndarray) -> float:
    """
    This function computes the values of u at any time t as piecewise constant function.
    As the last element is an open end point, we need to use the previous element.

    Parameters
    ----------
    t : float
        time to evaluate the piecewise constant function
    t_eval : Union[np.ndarray, List[float]]
        array of times t the controls u are evaluated at
    u : np.ndarray
        arrays of controls u over the tspans of t_eval

    Returns
    -------
    u_t: float
        value of u at time t
    """

    def previous_t(t: float, t_eval: Union[np.ndarray, List[float]]) -> int:
        """
        find the closest time in t_eval to t

        Parameters
        ----------
        t : float
            time to compare to t_eval
        t_eval : Union[np.ndarray, List[float]]
            array of times to compare to t

        Returns
        -------
        idx: int
            index of the closest previous time in t_eval to t
        """
        diff = t_eval - t
        # keep only positive values
        diff = diff[diff <= 0]
        return int(np.argmin(np.abs(diff)))

    def previous_t_except_the_last_one(t: float, t_eval: Union[np.ndarray, List[float]]) -> int:
        """
        find the closest time in t_eval to t

        Parameters
        ----------
        t : float
            time to compare to t_eval
        t_eval : Union[np.ndarray, List[float]]
            array of times to compare to t

        Returns
        -------
        idx: int
            index of the closest previous time in t_eval to t
        """
        return previous_t(t, t_eval) - 1 if previous_t(t, t_eval) == len(t_eval) - 1 else previous_t(t, t_eval)

    return u[:, previous_t_except_the_last_one(t, t_eval)]
