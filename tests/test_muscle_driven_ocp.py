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


    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 38.36734873968106)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g, np.zeros((120, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-0.07264118,  0.52329762]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.03963854, -0.8312761]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([0.99999652, 0.99999979]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([0.99995636, 0.99998742]))



    #
    # # Check objective function value
    # f = np.array(sol["f"])
    # np.testing.assert_equal(f.shape, (1, 1))
    # np.testing.assert_almost_equal(f[0, 0], 0.09990643702945518)
    #
    # # Check constraints
    # g = np.array(sol["g"])
    # np.testing.assert_equal(g.shape, (120, 1))
    # np.testing.assert_almost_equal(g, np.zeros((120, 1)), decimal=6)
    #
    # # Check some of the results
    # q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])
    #
    # # initial and final position
    # np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    # np.testing.assert_almost_equal(q[:, -1], np.array([1.25897393, -3.93236422]))
    # # initial and final velocities
    # np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
    # np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.18678309, -5.93159108]))
    # # initial and final controls
    # np.testing.assert_almost_equal(tau[:, 0], np.array([-0.99946814,  0.99920787]))
    # np.testing.assert_almost_equal(tau[:, -1], np.array([0.9990183 , -0.99911126]))
    # np.testing.assert_almost_equal(
    #     mus[:, 0], np.array([3.27792898e-01, 1.72406097e-01, 2.14413672e-01, 7.25272142e-05,
    #    7.45552332e-05, 2.56485014e-01]),
    # )
    # np.testing.assert_almost_equal(
    #     mus[:, -1], np.array([1.92162446e-04, 2.45779020e-04, 5.46807443e-05, 1.98144086e-02,
    #    1.97345933e-02, 8.29861803e-05]),
    # )
