"""
Test for file IO.
It tests that a model path with another type than string or biorbdmodel return an error
"""

import os
import pytest
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    BiorbdModel,
)


def test_biorbd_model_import():
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    model_path = "/models/pendulum.bioMod"
    BiorbdModel(bioptim_folder + model_path)

    BiorbdModel(biorbd.Model(bioptim_folder + model_path))

    with pytest.raises(ValueError, match="The model should be of type 'str' or 'biorbd.Model'"):
        BiorbdModel(1)

    with pytest.raises(ValueError, match="The model should be of type 'str' or 'biorbd.Model'"):
        BiorbdModel([])


@pytest.mark.parametrize(
    "my_keys",
    [
        ["q"],
        ["qdot"],
        ["qddot"],
        ["q", "qdot"],
        ["qdot", "q"],
        ["q", "qdot", "qddot"],
        ["qddot", "q", "qdot"],
        ["qdot", "qddot", "q"],
        ["q", "qddot", "qdot"],
        ["qddot", "qdot", "q"],
        ["qdot", "q", "qddot"],
    ],
)
def test_bounds_from_ranges(my_keys):
    from bioptim.examples.getting_started import pendulum as ocp_module

    x_min_q = [[-1.0, -1.0, -1.0], [-6.28318531, -6.28318531, -6.28318531]]
    x_min_qdot = [[-31.41592654, -31.41592654, -31.41592654], [-31.41592654, -31.41592654, -31.41592654]]
    x_min_qddot = [[-314.15926536, -314.15926536, -314.15926536], [-314.15926536, -314.15926536, -314.15926536]]
    x_max_q = [[5.0, 5.0, 5.0], [6.28318531, 6.28318531, 6.28318531]]
    x_max_qdot = [[31.41592654, 31.41592654, 31.41592654], [31.41592654, 31.41592654, 31.41592654]]
    x_max_qddot = [[314.15926536, 314.15926536, 314.15926536], [314.15926536, 314.15926536, 314.15926536]]

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    model_path = "/models/pendulum.bioMod"
    bio_model = BiorbdModel(bioptim_folder + model_path)

    for key in my_keys:
        x_bounds = bio_model.bounds_from_ranges(key)

        if key == "q":
            x_min = np.array(x_min_q)
            x_max = np.array(x_max_q)

        elif key == "qdot":
            x_min = np.array(x_min_qdot)
            x_max = np.array(x_max_qdot)

        elif key == "qddot":
            x_min = np.array(x_min_qddot)
            x_max = np.array(x_max_qddot)

        else:
            raise ValueError("Wrong key")

        # Check min and max have the right value
        np.testing.assert_almost_equal(x_bounds.min, x_min)
        np.testing.assert_almost_equal(x_bounds.max, x_max)
