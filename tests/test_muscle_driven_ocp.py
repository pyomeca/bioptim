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
    ocp = static_arm.prepare_ocp(str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod")
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.11727308482347926)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g, np.zeros((120, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-0.97984634, 3.13167264]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.23176099, -0.20376921]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([0.00849242, 0.00326032]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00131174, 0.00267676]))
    np.testing.assert_almost_equal(
        mus[:, 0],
        np.array([4.05323613e-05, 1.81646158e-01, 7.31903913e-02, 2.84826207e-04, 2.92694493e-04, 6.49166115e-02]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1], np.array([0.01868069, 0.00034234, 0.00032111, 0.0086349, 0.00860537, 0.00717714]),
    )


def test_muscle_with_contact_driven_ocp():
    ocp = static_arm_with_contact.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26_with_contact.bioMod"
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.1172732247020957)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (180, 1))
    np.testing.assert_almost_equal(g, np.zeros((180, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0, 0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.00793319, -0.97982547, 3.13165934]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0.0, 0.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.98997678e-05, 2.31862799e-01, -2.03868695e-01]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-0.00010404, 0.00849093, 0.00326059]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-9.39470683e-06, -1.31166477e-03, 2.67681508e-03]))
    np.testing.assert_almost_equal(
        mus[:, 0],
        np.array([4.05378143e-05, 1.81638593e-01, 7.31934951e-02, 2.84802413e-04, 2.92670032e-04, 6.49220139e-02]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1], np.array([0.01868095, 0.00034235, 0.00032112, 0.00863491, 0.00860538, 0.0071776]),
    )
