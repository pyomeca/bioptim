"""
Test for file IO.
It tests that a model path with another type than string or biorbdmodel return an error
"""

import os
import pytest
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

    with pytest.raises(RuntimeError, match="Type must be a 'str' or a 'biorbd.Model'"):
        BiorbdModel(1)

    with pytest.raises(RuntimeError, match="Type must be a 'str' or a 'biorbd.Model'"):
        BiorbdModel([])
