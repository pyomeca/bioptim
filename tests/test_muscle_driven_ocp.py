"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np
import biorbd

from biorbd_optim import ProblemType

# Load eocarSym
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "static_arm", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/static_arm.py",
)
static_arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_arm)


def test_muscle_driven_ocp():
    ocp = static_arm.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/emg_and_marker_tracking/arm26.bioMod"
    )
    sol = ocp.solve()

    #
    # # Check objective function value
    # f = np.array(sol["f"])
    # np.testing.assert_equal(f.shape, (1, 1))
    # np.testing.assert_almost_equal(f[0, 0], 1.9859178242635677e-15)
    #
    # # Check constraints
    # g = np.array(sol["g"])
    # np.testing.assert_equal(g.shape, (120, 1))
    # np.testing.assert_almost_equal(g, np.zeros((120, 1)), decimal=6)
    #
    # # Check some of the results
    # q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])
    #
    # # initial and final position
    # np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    # np.testing.assert_almost_equal(q[:, -1], np.array([-0.06756487, -0.02846171]))
    # # initial and final velocities
    # np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
    # np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.03906034, -1.22301875]))
    # # initial and final controls
    # np.testing.assert_almost_equal(tau[:, 0], np.array([1.31437879e-07, -3.16564845e-07]))
    # np.testing.assert_almost_equal(tau[:, -1], np.array([-7.51833328e-10,  1.73856102e-09]))
    #



    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 8.178189067845232e-06)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g, np.zeros((120, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-0.26440157, -0.19031678]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([ 0.09919984, -1.39985429]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([ 1.22439952e-07, -2.96837414e-07]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-4.77842946e-10,  1.18882743e-09]))
    np.testing.assert_almost_equal(
        mus[:, 0], np.array([0.00319779, 0.00319391, 0.00319484, 0.00319917, 0.00319912,
       0.0031946]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1], np.array([0.00319755, 0.00319755, 0.00319755, 0.00319754, 0.00319754,
        0.00319754]),
    )
