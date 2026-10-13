from types import SimpleNamespace

from casadi import Function, SX
import numpy as np
import numpy.testing as npt

from bioptim.dynamics.integrator import RK8
from bioptim.misc.enums import ControlType, DefectType


def _rk8_integrator(number_of_finite_elements: int, rhs):
    cx = SX
    t_span = cx.sym("t_span", 2, 1)
    states = cx.sym("states", 1, 1)
    controls = cx.sym("controls", 1, 1)
    parameters = cx.sym("parameters", 0, 1)
    algebraic_states = cx.sym("algebraic_states", 0, 1)
    numerical_timeseries = cx.sym("numerical_timeseries", 0, 1)

    dynamics = Function(
        "non_autonomous_dynamics",
        [t_span, states, controls, parameters, algebraic_states, numerical_timeseries],
        [rhs(t_span, states)],
        ["t_span", "x", "u", "p", "a", "d"],
        ["xdot"],
    )
    ode = {
        "t": t_span,
        "x": states,
        "u": controls,
        "param": parameters,
        "a": algebraic_states,
        "d": numerical_timeseries,
        "ode": dynamics,
        "implicit_ode": None,
    }
    ode_options = {
        "model": SimpleNamespace(nb_quaternions=0),
        "ode_index": 0,
        "cx": cx,
        "defects_type": DefectType.NOT_APPLICABLE,
        "control_type": ControlType.CONSTANT,
        "duplicate_starting_point": False,
        "number_of_finite_elements": number_of_finite_elements,
    }
    return RK8(ode, ode_options)


def _integrate(integrator: RK8, initial_state: float):
    return float(integrator(t_span=[0, 1], x0=[initial_state], u=[0], p=[], a=[], d=[])["xf"])


def test_rk8_non_autonomous_dynamics_uses_butcher_abscissas():
    """The RK8 stages integrate xdot=t**8 with the expected eighth-order quadrature."""

    exact = 1 / 9
    errors = []
    for number_of_finite_elements in (1, 2, 4, 8):
        integrator = _rk8_integrator(number_of_finite_elements, lambda time, _: time[0] ** 8)
        errors.append(abs(_integrate(integrator, initial_state=0) - exact))

    npt.assert_allclose(errors[0], 2.5720164609053497e-05)
    npt.assert_array_less(np.asarray(errors[1:]) * 200, np.asarray(errors[:-1]))


def test_rk8_autonomous_dynamics_uses_the_reference_k8_coefficient():
    """The -231/20 coefficient preserves the expected RK8 accuracy for xdot=x."""

    integrator = _rk8_integrator(1, lambda _, states: states)

    assert abs(_integrate(integrator, initial_state=1) - np.e) < 3e-6
