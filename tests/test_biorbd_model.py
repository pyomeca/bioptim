"""
Test for file IO.
It tests that a model path with another type than string or biorbdmodel return an error
"""

import os
import pytest

import numpy as np
from bioptim import (
    OdeSolver,
)

from .utils import TestUtils


@pytest.mark.parametrize("model", ["/models/pendulum.bioMod"])
def test_biorbd_model_fail(model):
    # Load pendulum_min_time_Mayer
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    with pytest.raises(RuntimeError, match="Type must be a string or a biorbdmodel"):
        ocp = ocp_module.prepare_ocp(
            biorbd_model_path=[bioptim_folder + model],
            final_time=2,
            n_shooting=10,
            ode_solver=OdeSolver.COLLOCATION(),
        )
    with pytest.raises(RuntimeError, match="Type must be a string or a biorbdmodel"):
        ocp = ocp_module.prepare_ocp(
            biorbd_model_path=np.array(bioptim_folder + model),
            final_time=2,
            n_shooting=10,
            ode_solver=OdeSolver.COLLOCATION(),
        )
    with pytest.raises(RuntimeError, match="Type must be a string or a biorbdmodel"):
        ocp = ocp_module.prepare_ocp(
            biorbd_model_path=1,
            final_time=2,
            n_shooting=10,
            ode_solver=OdeSolver.COLLOCATION(),
        )
