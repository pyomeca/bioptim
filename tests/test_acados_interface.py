"""
Test for file IO.
It tests the results of an optimal control problem with acados regarding the proper functioning of :
- the handling of mayer and lagrange obj
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from bioptim import Data
from .utils import TestUtils

def test_mayer_and_lagrange():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    ocp = cube.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.7592028279017864)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0.1189651, -0.0904378, -0.7999996, 0.7999996)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((1.2636414, -1.3010929, -3.6274687, 3.6274687)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-22.1218282)))
    np.testing.assert_almost_equal(tau[:, -1], np.array(0.2653957))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)