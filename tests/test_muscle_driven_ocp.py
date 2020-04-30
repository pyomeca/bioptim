"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np
import biorbd

from biorbd_optim import ProblemType

# Load static_arm
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "static_arm", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/static_arm.py",
)
static_arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_arm)

# Load static_arm_with_contact
spec = importlib.util.spec_from_file_location(
    "static_arm_with_contact", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/static_arm_with_contact.py",
)
static_arm_with_contact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_arm_with_contact)


def test_muscle_driven_ocp():
    ocp = static_arm.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod", final_time=2, number_shooting_points=10
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.12145862454010191)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-0.95746379, 3.09503322]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.34608528, -0.42902204]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([0.00458156, 0.00423722]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00134814, 0.00312362]))
    np.testing.assert_almost_equal(
        mus[:, 0],
        np.array([6.86190724e-06, 1.68975654e-01, 8.70683950e-02, 2.47147957e-05, 2.53934274e-05, 8.45479390e-02]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1],
        np.array([1.96717613e-02, 3.42406388e-05, 3.29456098e-05, 8.61728932e-03, 8.57918458e-03, 7.07096066e-03]),
    )


def test_muscle_with_contact_driven_ocp():
    ocp = static_arm_with_contact.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26_with_contact.bioMod",
        final_time=2,
        number_shooting_points=10,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.12145850795787627)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (60, 1))
    np.testing.assert_almost_equal(g, np.zeros((60, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0, 0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.00806796, -0.95744612, 3.09501746]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0.0, 0.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.00060479, 0.34617159, -0.42910107]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-4.67191123e-05, 4.58080354e-03, 4.23701953e-03]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-1.19345958e-05, -1.34810157e-03, 3.12378179e-03]))
    np.testing.assert_almost_equal(
        mus[:, 0],
        np.array([6.86280305e-06, 1.68961894e-01, 8.70635867e-02, 2.47160155e-05, 2.53946780e-05, 8.45438966e-02]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1],
        np.array([1.96721725e-02, 3.42398501e-05, 3.29454916e-05, 8.61757459e-03, 8.57946846e-03, 7.07152302e-03]),
    )
