"""
Test for file IO.
It tests that a model path with another type than string or biorbdmodel return an error
"""

import os

import biorbd_casadi as biorbd
from bioptim import BiorbdModel
import pytest
import numpy as np
import numpy.testing as npt

from ..utils import TestUtils


def test_biorbd_model_import():
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)
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

    bioptim_folder = TestUtils.module_folder(ocp_module)
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
        npt.assert_almost_equal(x_bounds.min, x_min)
        npt.assert_almost_equal(x_bounds.max, x_max)


def test_function_cached():
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)
    model_path = "/models/pendulum.bioMod"
    bio_model = BiorbdModel(bioptim_folder + model_path)

    # No cached function
    assert len(bio_model._cached_functions.keys()) == 0

    # CoM (no argument in the first parenthesis)
    # First call create the function and cache it
    com = bio_model.center_of_mass()(bio_model.q, bio_model.parameters)
    com_id = id(bio_model._cached_functions[("center_of_mass", (), frozenset())])
    assert len(bio_model._cached_functions.keys()) == 1

    # Second call use the cached function
    com = bio_model.center_of_mass()(bio_model.q, bio_model.parameters)
    assert com_id == id(bio_model._cached_functions[("center_of_mass", (), frozenset())])
    assert len(bio_model._cached_functions.keys()) == 1

    # marker (with an argument in the first parenthesis)
    # First call create the function and cache it
    marker = bio_model.marker(index=0)(bio_model.q, bio_model.parameters)
    marker_id = id(bio_model._cached_functions[("marker", (), frozenset({("index", 0)}))])
    assert len(bio_model._cached_functions.keys()) == 2

    # Second call use the cached function
    marker = bio_model.center_of_mass()(bio_model.q, bio_model.parameters)
    assert marker_id == id(bio_model._cached_functions[("marker", (), frozenset({("index", 0)}))])
    assert len(bio_model._cached_functions.keys()) == 2

    # First call with other argument creates the function again and cache it
    marker2 = bio_model.marker(index=1)(bio_model.q, bio_model.parameters)
    marker_id2 = id(bio_model._cached_functions[("marker", (), frozenset({("index", 1)}))])
    assert len(bio_model._cached_functions.keys()) == 3
    assert marker_id != marker_id2

    # Second call use the cached function
    marker2 = bio_model.center_of_mass()(bio_model.q, bio_model.parameters)
    assert marker_id2 == id(bio_model._cached_functions[("marker", (), frozenset({("index", 1)}))])
    assert len(bio_model._cached_functions.keys()) == 3
