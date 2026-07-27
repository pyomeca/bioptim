from types import SimpleNamespace

import numpy as np
import pytest

from bioptim.interfaces.acados_utils import scaled_control_bounds
from bioptim.limits.path_conditions import Bounds
from bioptim.misc.enums import InterpolationType


class _Variables(dict):
    @property
    def shape(self):
        return sum(len(variable.index) for variable in self.values())


def test_scaled_control_bounds_keep_lower_upper_order_and_variable_indices():
    controls = _Variables(
        pulse=SimpleNamespace(index=[1]),
        tau=SimpleNamespace(index=[0, 2]),
    )
    nlp = SimpleNamespace(
        controls=controls,
        u_bounds={
            "pulse": Bounds(
                "pulse", [20.0], [60.0], interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            ),
            "tau": Bounds(
                "tau",
                [-10.0, -30.0],
                [10.0, 50.0],
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            ),
        },
        u_scaling={
            "pulse": SimpleNamespace(scaling=np.array([[10.0]])),
            "tau": SimpleNamespace(scaling=np.array([[2.0], [10.0]])),
        },
    )

    lower, upper = scaled_control_bounds(nlp)

    np.testing.assert_allclose(lower, [-5.0, 2.0, -3.0])
    np.testing.assert_allclose(upper, [5.0, 6.0, 5.0])
    assert np.all(lower <= upper)


def test_scaled_control_bounds_reject_inverted_bounds():
    controls = _Variables(u=SimpleNamespace(index=[0]))
    nlp = SimpleNamespace(
        controls=controls,
        u_bounds={
            "u": Bounds("u", [2.0], [1.0], interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
        },
        u_scaling={"u": SimpleNamespace(scaling=np.array([[1.0]]))},
    )

    with pytest.raises(ValueError, match="inconsistent"):
        scaled_control_bounds(nlp)
