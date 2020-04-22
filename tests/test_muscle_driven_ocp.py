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
    np.testing.assert_almost_equal(f[0, 0], 9.188543388500895e-14)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g, np.zeros((120, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-0.65298015, -0.47821798]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([ 1.94992424, -3.97761726]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([ 8.45793845e-07, -2.36825540e-06]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-1.90083503e-09,  3.90260915e-09]))
    np.testing.assert_almost_equal(
        mus[:, 0], np.array([0.58563093, 0.1440457 , 0.18375457, 0.74112925, 0.7369461 ,
       0.17317304]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1], np.array([0.50046396, 0.49982784, 0.49968305, 0.50004153, 0.50003796,
       0.49956926]),
    )
