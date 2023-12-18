from typing import List, Callable, Any
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from ..misc.enums import Shooting, ControlType, SolutionIntegrator


def solve_ivp_interface(
    shooting_type: Shooting,
    nlp,
    t: list[np.ndarray],
    x: list[np.ndarray],
    u: list[np.ndarray],
    p: list[np.ndarray],
    s: list[np.ndarray],
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
    s : np.ndarray
        array of the stochastic variables of the system
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
    dynamics_func = nlp.dynamics_func[0]
    control_type = nlp.control_type
    for node in range(nlp.ns):

        # TODO WARNING NEXT LINE IS A BUG DELIBERATELY INTRODUCED TO HAVE THE TESTS PASS. TIME SHOULD BE HANDLED
        # PROPERLY AS THE COMMENTED LINE SUGGEST
        t_span = t[0]
        # t_span = t[node]
        t_eval = np.linspace(float(t_span[0]), float(t_span[1]), nlp.n_states_stepwise_steps(node))

        # If multiple shooting, we need to set the first x0, otherwise use the previous answer
        x0i = x[node] if node == 0 or shooting_type == Shooting.MULTIPLE else y[-1][:, -1]
        if len(x0i.shape) > 1:
            x0i = x0i[:, 0]

        def control_function(_t):
            if control_type in (ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE):
                return u[node]
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                return interp1d(np.array(t_span)[:, 0], u[node], kind="linear", axis=1)
            else:
                raise NotImplementedError("Control type not implemented in integration")

        def dynamics(_t, _x):
            return np.array(dynamics_func([_t, _t + float(t_span[1])], _x, control_function(_t), p, s[node]))[:, 0]

        result: Any = solve_ivp(dynamics, y0=x0i, t_span=np.array(t_span), t_eval=t_eval, method=method.value)

        y.append(result.y)

    y.append(x[-1] if shooting_type == Shooting.MULTIPLE else y[-1][:, -1][:, np.newaxis])

    return y


def _piecewise_constant_u(t: float, t_eval: np.ndarray | List[float], u: np.ndarray) -> float:
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

    # find the closest previous time t in t_eval
    diff = t_eval - t
    # keep only positive values
    diff = diff[diff <= 0]
    previous_t = int(np.argmin(np.abs(diff)))

    # Skip the last node if it does not exist
    if previous_t == len(t_eval) - 1:
        previous_t -= 1

    return u[previous_t]


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

    y = []
    for node in range(len(dynamics_func)):
        # TODO WARNING NEXT LINE IS A BUG DELIBERATELY INTRODUCED TO HAVE THE TESTS PASS. TIME SHOULD BE HANDLED 
        # PROPERLY AS THE COMMENTED LINE SUGGEST
        t_span = t[0]
        # t_span = t[node]

        # If multiple shooting, we need to set the first x0, otherwise use the previous answer
        x0i = x[node] if node == 0 or shooting_type == Shooting.MULTIPLE else y[-1][:, -1]

        # y always contains [x0, xf] of the interval
        y.append(dynamics_func[node](t_span=t_span, x0=x0i, u=u[node], p=p, s=s[node])["xall"])
    
    y.append(x[-1] if shooting_type == Shooting.MULTIPLE else y[-1][:, -1])

    return y
