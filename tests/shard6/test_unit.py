from casadi import Function, MX
import numpy as np
import pytest


from bioptim.optimization.optimal_control_program import (
    _reshape_to_column,
    _get_time_step,
    _get_target_values,
    _scale_values,
)

from bioptim.optimization.solution.utils import (
    concatenate_optimization_variables_dict,
    concatenate_optimization_variables,
)


def test_reshape_to_column_1d():
    array = np.array([1, 2, 3, 4])
    reshaped = _reshape_to_column(array)

    assert reshaped.shape == (4, 1)
    assert np.all(reshaped == np.array([[1], [2], [3], [4]]))


def test_reshape_to_column_already_2d():
    array = np.array([[1, 2], [3, 4], [5, 6]])
    reshaped = _reshape_to_column(array)

    assert reshaped.shape == (3, 2)
    assert np.all(reshaped == array)


def test_reshape_to_column_already_column():
    array = np.array([[1], [2], [3], [4]])
    reshaped = _reshape_to_column(array)

    assert reshaped.shape == (4, 1)
    assert np.all(reshaped == array)


def test_reshape_to_column_empty():
    array = np.array([])
    reshaped = _reshape_to_column(array)

    assert reshaped.shape == (0, 1)
    assert len(reshaped) == 0


@pytest.mark.parametrize(
    "input_array, expected_shape",
    [
        (np.array([1, 2, 3]), (3, 1)),
        (np.array([[1, 2], [3, 4]]), (2, 2)),
        (np.array([[]]), (1, 0)),
        (np.array([]), (0, 1)),
    ],
)
def test_reshape_to_column_parametrized(input_array, expected_shape):
    reshaped = _reshape_to_column(input_array)
    assert reshaped.shape == expected_shape


class MockPenalty:
    def __init__(self, dt, target=None, node_idx=0, multinode_penalty=False):
        self.dt = dt
        self.target = target
        self.node_idx = node_idx
        self.multinode_penalty = multinode_penalty


def test_function_dt():
    p = MX.sym("p", 1)
    dt = Function("fake_dt", [p], [2.0])
    p = 1  # This doesn't matter since our mocked Function doesn't use it
    x = np.array([[0, 1], [1, 2]])
    penalty = MockPenalty(1.0)
    penalty_phase = 0  # This doesn't matter in this test

    result = _get_time_step(dt, p, x, penalty, penalty_phase)
    assert result == 2.0


def test_normal_dt():
    dt = 2.0
    p = None
    x = np.array([[0, 1, 2], [1, 2, 3]])
    penalty = MockPenalty(1.0)
    penalty_phase = 0

    result = _get_time_step(dt, p, x, penalty, penalty_phase)
    assert result == 1.0  # Since x has three columns


def test_penalty_dt_array():
    dt = np.array([2.0, 3.0])
    p = None
    x = np.array([[0, 1, 2], [1, 2, 3]])
    penalty = MockPenalty(np.array([0.5, 1.0]))
    penalty_phase = 1

    result = _get_time_step(dt, p, x, penalty, penalty_phase)
    assert result == 1.5  # Since penalty_phase is 1 and dt is [2.0, 3.0]


def test_mayer_term():
    dt = 2.0
    p = None
    x = np.array([[0]])
    penalty = MockPenalty(1.0)
    penalty_phase = 0

    result = _get_time_step(dt, p, x, penalty, penalty_phase)
    assert result == 2.0  # Since x has one column, we just return dt


def test_get_target_values():
    penalty = MockPenalty(
        dt=1.0, target=[np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])], node_idx=[10, 20, 30]
    )

    result = _get_target_values(20, penalty)
    assert np.all(result == np.array([2, 5, 8, 11]))  # Because 20 corresponds to the 1-index


def test_non_integer_t():
    penalty = MockPenalty(
        dt=1.0, target=[np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])], node_idx=[10, 20, 30]
    )

    result = _get_target_values(20.5, penalty)
    assert result == []


def test_no_target():
    penalty = MockPenalty(dt=1.0, target=None, node_idx=[10, 20, 30])

    result = _get_target_values(20, penalty)
    assert result == []


class ScalingEntity:
    def __init__(self, shape):
        self.shape = shape


class ScalingData:
    def __init__(self, scaling):
        self.scaling = scaling


def test_scale_values_no_multinode():
    values = np.array([[2, 4], [3, 6], [2, 4], [3, 8]])
    scaling_entities = {"a": ScalingEntity((2,)), "b": ScalingEntity((2,))}
    scaling_data = {"a": ScalingData(np.array([2, 3])), "b": ScalingData(np.array([1, 2]))}
    penalty = MockPenalty(1, multinode_penalty=False)
    result = _scale_values(values, scaling_entities, penalty, scaling_data)
    expected = np.array([[1, 2], [1, 2], [2, 4], [1.5, 4]])
    assert np.allclose(result, expected)


def test_scale_values_with_multinode():
    values = np.array([[2, 4], [3, 6], [2, 4], [3, 8]])
    scaling_entities = {"a": ScalingEntity(2), "b": ScalingEntity(2)}
    scaling_data = {"a": ScalingData(np.array([2, 3])), "b": ScalingData(np.array([1, 2]))}
    penalty = MockPenalty(1, multinode_penalty=True)
    result = _scale_values(values, scaling_entities, penalty, scaling_data)
    expected = np.array([[1, 2], [1, 2], [2, 4], [1.5, 4]])
    assert np.allclose(result, expected)


def test_concatenate_optimization_variables_dict():
    variables = [
        {"a": np.array([[1, 2, 3], [4, 5, 6]]), "b": np.array([[7, 8, 9], [10, 11, 12]])},
        {"a": np.array([[13, 14, 15], [16, 17, 18]]), "b": np.array([[19, 20, 21], [22, 23, 24]])},
    ]
    result = concatenate_optimization_variables_dict(variables)
    expected_a = np.array([[1, 2, 13, 14, 15], [4, 5, 16, 17, 18]])
    expected_b = np.array([[7, 8, 19, 20, 21], [10, 11, 22, 23, 24]])
    assert np.array_equal(result[0]["a"], expected_a)
    assert np.array_equal(result[0]["b"], expected_b)

    with pytest.raises(ValueError):
        concatenate_optimization_variables_dict({"a": np.array([1, 2, 3])})


def test_concatenate_optimization_variables_simple():
    variables = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    result = concatenate_optimization_variables(variables)
    assert np.array_equal(result, np.array([1, 2, 4, 5, 7, 8, 9]))


def test_concatenate_optimization_variables_flags():
    variables = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
    result = concatenate_optimization_variables(variables, continuous_phase=False)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6, 7, 8]))
