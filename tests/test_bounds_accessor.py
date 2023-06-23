import numpy as np
from bioptim import BoundsList, InterpolationType


def test_accessors_on_bounds_option():
    x_min = [-100] * 6
    x_max = [100] * 6
    x_bounds = BoundsList()
    x_bounds.add("my_key", min_bound=x_min, max_bound=x_max, interpolation=InterpolationType.CONSTANT)
    x_bounds["my_key"][:3] = 0
    x_bounds["my_key"].min[3:] = -10
    x_bounds["my_key"].max[1:3] = 10

    # Check min and max have the right value
    np.testing.assert_almost_equal(x_bounds["my_key"].min[:], np.array([[0], [0], [0], [-10], [-10], [-10]]))
    np.testing.assert_almost_equal(x_bounds["my_key"].max[:], np.array([[0], [10], [10], [100], [100], [100]]))


def test_accessors_on_bounds_option_multidimensional():
    x_min = [[-100, -50, 0] for i in range(6)]
    x_max = [[100, 150, 200] for i in range(6)]
    x_bounds = BoundsList()
    x_bounds.add(
        "my_key",
        min_bound=x_min,
        max_bound=x_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds["my_key"][:3, 0] = 0
    x_bounds["my_key"].min[1:5, 1:] = -10
    x_bounds["my_key"].max[1:5, 1:] = 10

    # Check min and max have the right value
    np.testing.assert_almost_equal(
        x_bounds["my_key"].min[:],
        np.array([[0, -50, 0], [0, -10, -10], [0, -10, -10], [-100, -10, -10], [-100, -10, -10], [-100, -50, 0]]),
    )
    np.testing.assert_almost_equal(
        x_bounds["my_key"].max[:],
        np.array([[0, 150, 200], [0, 10, 10], [0, 10, 10], [100, 10, 10], [100, 10, 10], [100, 150, 200]]),
    )
