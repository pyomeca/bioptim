"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np

from biorbd_optim import ProblemType

# Load eocarSym
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "eocarSym",
    str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/eocarSym.py",
)
eocarSym = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eocarSym)


def test_eocarSym():
    ocp = eocarSym.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER)
        + "/examples/symmetrical_torque_driven_ocp/eocarSym.bioMod"
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 114.77652947720107)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((4.3548387, 0, 2.27903225)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-4.35483872, 0, -2.27903227)))
