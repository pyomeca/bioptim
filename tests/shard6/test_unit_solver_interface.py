import pytest
from casadi import SX, MX, vertcat, Function
import numpy as np

from bioptim import NonLinearProgram
from bioptim.interfaces.interface_utils import get_padded_array


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
    result = get_padded_array(nlp_sx, 'X', 0, SX, 5)
    expected = vertcat(SX([[1], [2], [3]]), SX(2, 1))
    assert (result - expected).is_zero()


def test_no_padding(nlp_sx):
    result = get_padded_array(nlp_sx, 'X', 0, SX)
    expected = SX([[1], [2], [3]])
    assert (result - expected).is_zero()


def test_custom_target_length(nlp_sx):
    result = get_padded_array(nlp_sx, 'X', 0, SX, 4)
    expected = vertcat(SX([[1], [2], [3]]), SX(1, 1))
    assert (result - expected).is_zero()


def test_invalid_attribute(nlp_sx):
    with pytest.raises(AttributeError):
        get_padded_array(nlp_sx, 'invalid_attribute', 0, SX)


def test_invalid_node_idx(nlp_sx):
    with pytest.raises(IndexError):
        get_padded_array(nlp_sx, 'X', 10, SX)


def test_sx(nlp_sx):
    result_sx = get_padded_array(nlp_sx, 'X', 0, SX, 5)
    expected = vertcat(SX([[1], [2], [3]]), SX(2, 1))
    assert (result_sx - expected).is_zero()


def test_mx(nlp_mx):
    result_mx = get_padded_array(nlp_mx, 'X', 0, MX, 5)
    expected = vertcat(
        MX(np.array([[1], [2], [3]])),
        MX(2,1)
    )
    res = Function('test', [], [result_mx-expected])
    assert res()["o0"].is_zero()
