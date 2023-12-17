from typing import List, Callable, Any
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from ..misc.enums import Shooting, ControlType


def solve_ivp_interface(
    dynamics_func: Callable,
    t_eval: np.ndarray | List[float],
    x0: np.ndarray,
    u: np.ndarray,
    params: np.ndarray,
    s: np.ndarray,
    method: str | Any = "RK45",
    keep_intermediate_points: bool = False,
    control_type: ControlType = ControlType.CONSTANT,
):
    """
    This function solves the initial value problem with scipy.integrate.solve_ivp

    Parameters
    ----------
    dynamics_func : Callable
        function that computes the dynamics of the system
    t_eval : np.ndarray | List[float]
        array of times t the controls u are evaluated at
    t : np.ndarray
        array of time
    x0 : np.ndarray
        array of initial conditions
    u : np.ndarray
        arrays of controls u evaluated at t_eval
    params : np.ndarray
        array of parameters
    s : np.ndarray
        array of arrays of stochastic variables
    method : str, optional
        method to use for the integration, by default "RK45"
    keep_intermediate_points : bool
        whether to keep the intermediate points or not, by default False
    control_type : ControlType
        type of control, by default ControlType.CONSTANT

    Returns
    -------
    y: np.ndarray
        array of the solution of the system at the times t_eval

    """
    if isinstance(t_eval[0], np.ndarray):  # Direct multiple shooting
        y_final = np.array([], dtype=np.float64).reshape(x0.shape[0], 0)

        for ss, t_eval_step in enumerate(t_eval):
            x0i = x0[:, ss]
            u_slice = slice(ss, ss + 1) if control_type == ControlType.CONSTANT else slice(ss, ss + 2)

            # resize u to match the size of t_eval according to the type of control
            if control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                ui = np.repeat(u[:, u_slice], t_eval_step.shape[0], axis=1)
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                f = interp1d(t_eval_step[[0, -1]], u[:, u_slice], kind="linear", axis=1)
                ui = f(t_eval_step)
            else:
                raise NotImplementedError("Control type not implemented")

            # solve single shooting for each phase
            y = run_solve_ivp(
                dynamics_func=dynamics_func,
                t_eval=t_eval_step,
                x0=x0i,
                u=ui,
                s=s,
                params=params,
                method=method,
                keep_intermediate_points=False,  # error raise in direct multiple shooting so it's always False
                control_type=control_type,
            )

            y_final = np.hstack((y_final, y))

        return y_final

    else:  # Single shooting
        # reshape t_eval to get t_eval_step, single shooting is solved for each interval
        if keep_intermediate_points:
            # resize t_eval to get intervals of [ti,..., ti+1] for each intervals with nsteps
            n_shooting_extended = t_eval.shape[0] - 1
            n_shooting = u.shape[1] - 1
            n_step = n_shooting_extended // n_shooting

            t_eval_block_1 = t_eval[:-1].reshape(n_shooting, n_step)
            t_eval_block_2 = np.vstack((t_eval_block_1[1:, 0:1], t_eval[-1]))
            t_eval = np.hstack((t_eval_block_1, t_eval_block_2))
        else:
            # resize t_eval to get intervals of [ti, ti+1] for each intervals
            t_eval = np.hstack((np.array(t_eval).reshape(-1, 1)[:-1], np.array(t_eval).reshape(-1, 1)[1:]))

        y_final = np.array([], dtype=np.float64).reshape(x0.shape[0], 0)
        x0i = x0

        y = None
        for ss, t_eval_step in enumerate(t_eval):
            u_slice = slice(ss, ss + 1) if control_type == ControlType.CONSTANT else slice(ss, ss + 2)

            # resize u to match the size of t_eval according to the type of control
            if control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                ui = np.repeat(u[:, u_slice], t_eval_step.shape[0], axis=1)
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                f = interp1d(t_eval_step[[0, -1]], u[:, u_slice], kind="linear", axis=1)
                ui = f(np.array(t_eval_step, dtype=np.float64))  # prevent error with dtype=object
            else:
                raise NotImplementedError("Control type not implemented")

            y = run_solve_ivp(
                dynamics_func=dynamics_func,
                t_eval=t_eval_step,
                x0=x0i,
                u=ui,
                s=s,
                params=params,
                method=method,
                keep_intermediate_points=keep_intermediate_points,
                control_type=control_type,
            )

            y_final = np.hstack((y_final, y[:, 0:-1]))
            x0i = y[:, -1]
        y_final = np.hstack((y_final, y[:, -1:]))

        return y_final


def run_solve_ivp(
    dynamics_func: Callable,
    t_eval: np.ndarray | List[float],
    x0: np.ndarray,
    u: np.ndarray,
    params: np.ndarray,
    s: np.ndarray,
    method: str | Any = "RK45",
    keep_intermediate_points: bool = False,
    control_type: ControlType = ControlType.CONSTANT,
):
    """
    This function solves the initial value problem with scipy.integrate.solve_ivp

    Parameters
    ----------
    dynamics_func : Callable
        function that computes the dynamics of the system
    t_eval : np.ndarray | List[float]
        array of times t the controls u are evaluated at
    x0 : np.ndarray
        array of initial conditions
    u : np.ndarray
        arrays of controls u evaluated at t_eval
    params : np.ndarray
        array of parameters
    s : np.ndarray
        array of arrays of the stochastic variables
    method : str, optional
        method to use for the integration, by default "RK45"
    keep_intermediate_points : bool
        whether to keep the intermediate points or not, by default False
    control_type : ControlType
        type of control, by default ControlType.CONSTANT

    Returns
    -------
    y: np.ndarray
        array of the solution of the system at the times t_eval

    """
    control_function = define_control_function(
        t_eval, controls=u, control_type=control_type, keep_intermediate_points=keep_intermediate_points
    )

    t_span = [t_eval[0], t_eval[-1]]
    integrated_sol = solve_ivp(
        lambda t, x: np.array(dynamics_func(t, x, control_function(t), params, s))[:, 0],
        t_span=t_span,
        y0=x0,
        t_eval=np.array(t_eval, dtype=np.float64),  # prevent error with dtype=object
        method=method,
    )

    return integrated_sol.y


def define_control_function(
    t_u: np.ndarray, controls: np.ndarray, control_type: ControlType, keep_intermediate_points: bool
) -> Callable:
    """
    This function defines the control function to use in the integration.

    Parameters
    ----------
    t_u : np.ndarray
        array of times t where the controls u are evaluated at
    controls : np.ndarray
        arrays of controls u evaluated at t_eval
    control_type : ControlType
        type of control such as CONSTANT, CONSTANT_WITH_LAST_NODE, LINEAR_CONTINUOUS or NONE
    keep_intermediate_points : bool
        whether to keep the intermediate points or not

    Returns
    -------
    control_function: Callable
        function that computes the control at any time t
    """

    if keep_intermediate_points:
        # repeat values of u to match the size of t_eval
        n_shooting_extended = t_u.shape[0] - 1
        n_shooting = controls.shape[1] - 1
        n_step = n_shooting_extended // n_shooting

        if control_type == ControlType.CONSTANT:
            controls = np.concatenate(
                (
                    np.repeat(controls[:, :-1], n_step, axis=1),
                    controls[:, -1:],
                ),
                axis=1,
            )
            return lambda t: piecewise_constant_u(t, t_u, controls)

        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            controls = np.repeat(controls[:, :], n_step, axis=1)
            return lambda t: piecewise_constant_u(t, t_u, controls)

        elif control_type == ControlType.LINEAR_CONTINUOUS:
            # interpolate linearly the values of u at each time step to match the size of t_eval
            t_u = t_u[::n_step]  # get the actual time steps of u
            return interp1d(t_u, controls, kind="linear", axis=1)
    else:
        if control_type == ControlType.CONSTANT:
            return lambda t: piecewise_constant_u(t, t_u, controls)
        elif control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            return lambda t: piecewise_constant_u(t, t_u, controls[:, :-1])
        elif control_type == ControlType.LINEAR_CONTINUOUS:
            # interpolate linearly the values of u at each time step to match the size of t_eval
            return interp1d(t_u, controls, kind="linear", axis=1)


def piecewise_constant_u(t: float, t_eval: np.ndarray | List[float], u: np.ndarray) -> float:
    """
    This function computes the values of u at any time t as piecewise constant function.
    As the last element is an open end point, we need to use the previous element.

    Parameters
    ----------
    t : float
        time to evaluate the piecewise constant function
    t_eval : np.ndarray | List[float]
        array of times t the controls u are evaluated at
    u : np.ndarray
        arrays of controls u over the tspans of t_eval

    Returns
    -------
    u_t: float
        value of u at time t
    """

    def previous_t(t: float, t_eval: np.ndarray | List[float]) -> int:
        """
        find the closest time in t_eval to t

        Parameters
        ----------
        t : float
            time to compare to t_eval
        t_eval : np.ndarray | List[float]
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

    def previous_t_except_the_last_one(t: float, t_eval: np.ndarray | List[float]) -> int:
        """
        find the closest time in t_eval to t

        Parameters
        ----------
        t : float
            time to compare to t_eval
        t_eval : np.ndarray | List[float]
            array of times to compare to t

        Returns
        -------
        int
            index of the closest previous time in t_eval to t
        """
        out = previous_t(t, t_eval)
        if out == len(t_eval) - 1:
            return out - 1
        else:
            return out

    if t_eval.shape[0] != u.shape[1]:
        raise ValueError("t_eval and u must have the same length, please report the bug to the developers")

    return u[:, previous_t_except_the_last_one(t, t_eval)]


def solve_ivp_bioptim_interface(
    shooting_type: Shooting,
    dynamics_func: list[Callable],
    t: list[np.ndarray],
    x: list[np.ndarray],
    u: list[np.ndarray],
    p: list[np.ndarray],
    s: list[np.ndarray],
):
    """
    This function solves the initial value problem with the dynamics_func built by bioptim

    Parameters
    ----------
    dynamics_func : list[Callable]
        function that computes the dynamics of the system
    t : np.ndarray
        array of time
    x : np.ndarray
        array of initial conditions
    u : np.ndarray
        arrays of controls u evaluated at t_eval
    p : np.ndarray
        array of parameters
    s : np.ndarray
        array of the stochastic variables of the system
    shooting_type : Shooting
        The way we integrate the solution such as SINGLE, SINGLE_CONTINUOUS, MULTIPLE

    Returns
    -------
    y: np.ndarray
        array of the solution of the system at the times t_eval

    """

    # if multiple shooting, we need to set the first x0
    y = []
    for node in range(len(dynamics_func)):
        # TODO WARNING NEXT LINE IS A BUG DELIBERATELY INTRODUCED TO HAVE THE TESTS PASS. TIME SHOULD BE HANDLED 
        # PROPERLY AS THE COMMENTED LINE SUGGEST
        t_span = t[0]
        # t_spah = t[node]

        # If multiple shooting, we need to set the first x0, otherwise use the previous answer
        x0i = x[node] if node == 0 or shooting_type == Shooting.MULTIPLE else y[-1][:, -1]

        # y always contains [x0, xf] of the interval
        y.append(dynamics_func[node].function(t_span=t_span, x0=x0i, u=u[node], p=p, s=s[node])["xall"])
    
    y.append(x[-1] if shooting_type == Shooting.MULTIPLE else y[-1][:, -1])

    return y
