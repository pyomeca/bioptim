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
    np.testing.assert_almost_equal(f[0, 0], 0.03928217227963457)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g, np.zeros((120, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-3.2961726,  4.0696519]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-1.04474827,  3.81391233]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-0.00488432, -0.00645102]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00300375,  0.0016473]))
    np.testing.assert_almost_equal(
        mus[:, 0], np.array([2.00151754e-01, 4.32317180e-05, 7.97613093e-05, 7.06526090e-02,
       6.87228618e-02, 7.97498498e-05]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1], np.array([0.0004186 , 0.0005029 , 0.00035645, 0.01507679, 0.01483208,
       0.00058814]),
    )
