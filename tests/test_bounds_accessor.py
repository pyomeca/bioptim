import pytest

import numpy as np
import biorbd

from bioptim import (
    BoundsOption,
    InterpolationType,
)


def test_accessors_on_bounds_option():
    x_min = [-100] * 6
    x_max = [100] * 6
    x_bounds = BoundsOption([x_min, x_max], interpolation=InterpolationType.CONSTANT)
    x_bounds[:3] = 0
    x_bounds.min[3:] = -10
    x_bounds.max[1:3] = 10

    # Check accessor and min/max values to be equal
    np.testing.assert_almost_equal(x_bounds[:], (x_bounds.min[:], x_bounds.max[:]))

    # Check min and max have the right value
    np.testing.assert_almost_equal(x_bounds.min[:], np.array([[0], [0], [0], [-10], [-10], [-10]]))
    np.testing.assert_almost_equal(x_bounds.max[:], np.array([[0], [10], [10], [100], [100], [100]]))


def test_accessors_on_bounds_option_multidimensional():
    x_min = [[-100, -50, 0] for i in range(6)]
    x_max = [[100, 150, 200] for i in range(6)]
    x_bounds = BoundsOption([x_min, x_max], interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds[:3, 0] = 0
    x_bounds.min[1:5, 1:] = -10
    x_bounds.max[1:5, 1:] = 10

    # Check accessor and min/max values to be equal
    np.testing.assert_almost_equal(x_bounds[:], (x_bounds.min[:], x_bounds.max[:]))

    # Check min and max have the right value
    np.testing.assert_almost_equal(
        x_bounds.min[:],
        np.array([[0, -50, 0], [0, -10, -10], [0, -10, -10], [-100, -10, -10], [-100, -10, -10], [-100, -50, 0]]),
    )
    np.testing.assert_almost_equal(
        x_bounds.max[:],
        np.array([[0, 150, 200], [0, 10, 10], [0, 10, 10], [100, 10, 10], [100, 10, 10], [100, 150, 200]]),
    )
