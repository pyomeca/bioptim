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
    np.testing.assert_almost_equal(f[0, 0], 0.05695142924014181)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g, np.zeros((120, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.81764019, -3.76397092]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.28161585, -4.41094598]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-0.01102078,  0.0013934 ]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([ 0.00408093, -0.00458276]))
    np.testing.assert_almost_equal(
        mus[:, 0], np.array([2.59841012e-01, 1.37678764e-04, 1.17411900e-02, 6.44958524e-04,
       6.61972924e-04, 2.80298348e-02]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1], np.array([0.00023481, 0.00375259, 0.00023331, 0.00566022, 0.00565047,
       0.0002493 ]),
    )
