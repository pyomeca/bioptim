import pytest
from casadi import SX, MX, vertcat, Function
import numpy as np

from bioptim import NonLinearProgram, PhaseDynamics
from bioptim.interfaces.interface_utils import get_padded_array, get_node_control_info, get_padded_control_array


@pytest.fixture
def nlp_sx():
    # Create a dummy NonLinearProgram object with necessary attributes
    nlp = NonLinearProgram(None)
    nlp.X = [SX([[1], [2], [3]])]
    nlp.X_scaled = [SX([[4], [5], [6]])]
    # Add more attributes as needed
    return nlp


@pytest.fixture
def nlp_mx():
    # Create a dummy NonLinearProgram object with necessary attributes
    nlp = NonLinearProgram(None)
    nlp.X = [MX(np.array([[1], [2], [3]]))]
    nlp.X_scaled = [MX(np.array([[4], [5], [6]]))]
    # Add more attributes as needed
    return nlp


def test_valid_input(nlp_sx):
    result = get_padded_array(nlp_sx, "X", 0, SX, 5)
    expected = vertcat(SX([[1], [2], [3]]), SX(2, 1))
    assert (result - expected).is_zero()


def test_no_padding(nlp_sx):
    result = get_padded_array(nlp_sx, "X", 0, SX)
    expected = SX([[1], [2], [3]])
    assert (result - expected).is_zero()


def test_custom_target_length(nlp_sx):
    result = get_padded_array(nlp_sx, "X", 0, SX, 4)
    expected = vertcat(SX([[1], [2], [3]]), SX(1, 1))
    assert (result - expected).is_zero()


def test_invalid_attribute(nlp_sx):
    with pytest.raises(AttributeError):
        get_padded_array(nlp_sx, "invalid_attribute", 0, SX)


def test_invalid_node_idx(nlp_sx):
    with pytest.raises(IndexError):
        get_padded_array(nlp_sx, "X", 10, SX)


def test_sx(nlp_sx):
    result_sx = get_padded_array(nlp_sx, "X", 0, SX, 5)
    expected = vertcat(SX([[1], [2], [3]]), SX(2, 1))
    assert (result_sx - expected).is_zero()


def test_mx(nlp_mx):
    result_mx = get_padded_array(nlp_mx, "X", 0, MX, 5)
    expected = vertcat(MX(np.array([[1], [2], [3]])), MX(2, 1))
    res = Function("test", [], [result_mx - expected])
    assert res()["o0"].is_zero()


@pytest.fixture
def nlp_control_sx():
    nlp = NonLinearProgram(PhaseDynamics.SHARED_DURING_THE_PHASE)
    nlp.U_scaled = [SX([[1], [2], [3]])]
    return nlp


@pytest.fixture
def nlp_control_mx():
    nlp = NonLinearProgram(PhaseDynamics.SHARED_DURING_THE_PHASE)
    nlp.U_scaled = [MX(np.array([[1], [2], [3]]))]
    return nlp


def test_get_node_control_info(nlp_control_sx):
    is_shared_dynamics, is_within_control_limit, len_u = get_node_control_info(nlp_control_sx, 0, "U_scaled")
    assert is_shared_dynamics
    assert is_within_control_limit
    assert len_u == 3


def test_get_padded_control_array_sx(nlp_control_sx):
    _u_sym = get_padded_control_array(nlp_control_sx, 0, 0, "U_scaled", 5, True, True, SX)
    expected = vertcat(SX([[1], [2], [3]]), SX(2, 1))
    assert (_u_sym - expected).is_zero()


def test_get_padded_control_array_mx(nlp_control_mx):
    _u_sym = get_padded_control_array(nlp_control_mx, 0, 0, "U_scaled", 5, True, True, MX)
    expected = vertcat(MX(np.array([[1], [2], [3]])), MX(2, 1))
    res = Function("test", [], [_u_sym - expected])
    assert res()["o0"].is_zero()


def test_get_node_control_info_not_shared(nlp_control_sx):
    nlp_control_sx.phase_dynamics = PhaseDynamics.ONE_PER_NODE
    is_shared_dynamics, _, _ = get_node_control_info(nlp_control_sx, 0, "U_scaled")
    assert not is_shared_dynamics


def test_get_node_control_info_outside_limit(nlp_control_sx):
    is_shared_dynamics, is_within_control_limit, _ = get_node_control_info(nlp_control_sx, 10, "U_scaled")
    assert not is_within_control_limit


def test_get_padded_control_array_no_padding_sx(nlp_control_sx):
    _u_sym = get_padded_control_array(nlp_control_sx, 0, 0, "U_scaled", 3, True, True, SX)
    expected = SX([[1], [2], [3]])
    assert (_u_sym - expected).is_zero()


def test_get_padded_control_array_different_u_mode_sx(nlp_control_sx):
    _u_sym = get_padded_control_array(nlp_control_sx, 1, 1, "U_scaled", 5, True, True, SX)
    expected = vertcat(SX([[1], [2], [3]]), SX(2, 1))
    assert (_u_sym - expected).is_zero()


def test_get_padded_control_array_no_shared_dynamics_sx(nlp_control_sx):
    nlp_control_sx.phase_dynamics = PhaseDynamics.ONE_PER_NODE
    _u_sym = get_padded_control_array(nlp_control_sx, 0, 0, "U_scaled", 5, True, False, SX)
    expected = vertcat(SX([[1], [2], [3]]), SX(2, 1))
    assert (_u_sym - expected).is_zero()


def test_get_padded_control_array_outside_control_limit_sx(nlp_control_sx):
    with pytest.raises(IndexError):
        get_padded_control_array(nlp_control_sx, 10, 0, "U_scaled", 5, False, True, SX)

    _u_sym = get_padded_control_array(nlp_control_sx, 0, 0, "U_scaled", 5, False, False, SX)
    assert (_u_sym - SX([[1], [2], [3]])).is_zero()
