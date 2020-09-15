import pytest

import numpy as np
import biorbd

from biorbd_optim import (
    BoundsOption,
    InterpolationType,
)

def test_accessors_on_bounds_option():
    x_min = [-100] * 6
    x_max = [100] * 6
    x_bounds = BoundsOption([x_min, x_max], interpolation=InterpolationType.CONSTANT)
    x_bounds[:3] = 0
    x_bounds.min[3:] = -10
    x_bounds.max[:3] = 10

    # Check accessor and min/max values to be equal
    np.testing.assert_almost_equal(x_bounds[:], (x_bounds.min[:], x_bounds.max[:]))

    # Check min and max have the right value
    np.testing.assert_almost_equal(x_bounds.min[:], np.array([[0], [0], [0], [-10], [-10], [-10]]))
    np.testing.assert_almost_equal(x_bounds.max[:], np.array([[10], [10], [10], [100], [100], [100]]))
