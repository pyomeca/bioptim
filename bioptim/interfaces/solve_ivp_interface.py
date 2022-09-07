from typing import Union, List, Callable, Any
import numpy as np
from scipy.integrate import solve_ivp
from ..misc.enums import Shooting


def solve_ivp_interface(
        dynamics_func: Callable,
        t_eval: Union[np.ndarray, List[float]],
        x0: np.ndarray,
        u: np.ndarray,
        params: np.ndarray,
        method: str = "RK45",
        keep_intermediate_points: bool = False,
        continuous: bool = False,
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
    continuous : bool
        whether the problem is continuous or not, by default False

    Returns
    -------
    y: np.ndarray
        array of the solution of the system at the times t_eval

    """
    if isinstance(t_eval[0], np.ndarray):  # Direct multiple shooting

        y_final = np.array([], dtype=np.float).reshape(x0.shape[0], 0)

        for s, (t_eval_step, ui) in enumerate(zip(t_eval, u[:, -1].T)):
            # determine the initial values
            if continuous:  # direct multiple shooting
                x0i = y[:, -1] if s > 0 else x0[:, 0]
            else:
                x0i = x0[:, s]

            y = solve_ivp_interface(
                dynamics_func=dynamics_func,
                t_eval=t_eval_step,
                x0=x0i,
                u=np.repeat(ui[:, np.newaxis], t_eval_step.shape[0], axis=1),
                params=params,
                method=method,
                keep_intermediate_points=False,
                continuous=continuous,
            )

            if continuous:
                y_final = np.hstack((y_final, y[:, :-1]))
            else:
                y_final = np.hstack((y_final, y))

        return y_final

    else: # Single shooting
        if keep_intermediate_points:
            # repeat values of u to match the size of t_eval
            u = np.concatenate(
                (np.repeat(u[:, :-1], t_eval[:-1].shape[0] / u[:, :-1].shape[1], axis=1), u[:, -1:]),
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

    if t_eval.shape[0] != u.shape[1]:
        raise ValueError("t_eval and u must have the same length, please report the bug to the developers")

    return u[:, previous_t_except_the_last_one(t, t_eval)]


def solve_ivp_bioptim_interface(
        dynamics_func: list[Callable],
        keep_intermediate_points: bool,
        continuous: bool,
        x0: np.ndarray,
        u: np.ndarray,
        params: np.ndarray,
        param_scaling: np.ndarray,
        shooting_type: Shooting,
):
    """
    This function solves the initial value problem with scipy.integrate.solve_ivp

    Parameters
    ----------
    dynamics_func : list[Callable]
        function that computes the dynamics of the system
    keep_intermediate_points : bool
        whether to keep the intermediate points or not
    continuous : bool
        whether to keep the last node of the interval or not, if continuous is True, the last node is not kept
    x0 : np.ndarray
        array of initial conditions
    u : np.ndarray
        arrays of controls u evaluated at t_eval
    params : np.ndarray
        array of parameters
    param_scaling : np.ndarray
        array of scaling factors for the parameters
    shooting_type : Shooting
        The way we integrate the solution such as SINGLE, SINGLE_CONTINUOUS, MULTIPLE

    Returns
    -------
    y: np.ndarray
        array of the solution of the system at the times t_eval

    """
    if len(x0.shape) != len(u.shape):
        x0 = x0[:, np.newaxis]
    # if multiple shooting, we need to set the first x0
    x0i = x0[:, 0] if x0.shape[1] > 1 else x0

    dynamics_output = "xall" if keep_intermediate_points else "xf"

    # todo: begin with an empty array for all cases
    if continuous and keep_intermediate_points:
        y_final = np.array([], dtype=np.float).reshape(x0i.shape[0], 0)
    else:
        y_final = x0i

    for s, func in enumerate(dynamics_func):
        y = np.array(func(x0=x0i, p=u[:, s], params=params / param_scaling)[dynamics_output])
        # select the output of the integrated solution
        if continuous and keep_intermediate_points:
            concatenated_y = y[:, 0:-1]
        elif not continuous and keep_intermediate_points:
            concatenated_y = y[:, 1:]
        else:
            concatenated_y = y
        y_final = np.concatenate((y_final, concatenated_y), axis=1)

        # update x0 for the next step
        if shooting_type == Shooting.MULTIPLE and continuous is False:
            x0i = x0[:, s + 1]
        else:
            x0i = y[:, -1:]

    if continuous and keep_intermediate_points:
        y_final = np.concatenate((y_final, x0i), axis=1)

    return y_final
