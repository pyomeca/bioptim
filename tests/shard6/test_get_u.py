import pytest
from bioptim import ControlType
from bioptim.limits.penalty_option import _get_u
from casadi import MX, transpose

from ..utils import TestUtils


def test_constant_control():
    control_type = ControlType.CONSTANT
    u = MX([10, 20])
    dt = MX(0.5)
    result = _get_u(control_type, u, dt)
    expected = u
    assert (result - expected).is_zero()


def test_constant_with_last_node_control():
    control_type = ControlType.CONSTANT_WITH_LAST_NODE
    u = MX([10, 20])
    dt = MX(0.5)
    result = _get_u(control_type, u, dt)
    expected = u
    assert (result - expected).is_zero()


def test_linear_continuous_control():
    control_type = ControlType.LINEAR_CONTINUOUS
    u = transpose(MX([10, 20]))
    dt = MX(0.5)
    result = _get_u(control_type, u, dt)
    expected = u[:, 0] + (u[:, 1] - u[:, 0]) * dt
    array_expected = TestUtils.mx_to_array(expected)
    array_result = TestUtils.mx_to_array(result)
    assert array_expected - array_result == 0


def test_unimplemented_control_type():
    with pytest.raises(RuntimeError, match="ControlType not implemented yet"):
        _get_u("SomeRandomControlType", MX([10, 20]), MX(0.5))
