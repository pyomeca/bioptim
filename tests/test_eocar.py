"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np

# Load eocar
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "eocar", str(PROJECT_FOLDER) + "/examples/eocar/eocar.py"
)
eocar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eocar)


def test_oecar():
    nlp = eocar.prepare_nlp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/eocar/eocar.bioMod"
    )
    sol = nlp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 1317.835541713015)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q = []
    q_dot = []
    u = []
    for idx in range(nlp.model.nbQ()):
        q.append(np.array(sol["x"][0 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]))
        q_dot.append(
            np.array(sol["x"][1 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()])
        )
        u.append(np.array(sol["x"][2 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]))
    # initial and final position
    np.testing.assert_almost_equal(q[0][0, 0], 1)
    np.testing.assert_almost_equal(q[0][-1, 0], 2)
    np.testing.assert_almost_equal(q[1][0, 0], 0)
    np.testing.assert_almost_equal(q[1][-1, 0], 0)
    np.testing.assert_almost_equal(q[2][0, 0], 0)
    np.testing.assert_almost_equal(q[2][-1, 0], 1.57)
    # initial and final velocities
    np.testing.assert_almost_equal(q_dot[0][0, 0], 0)
    np.testing.assert_almost_equal(q_dot[0][-1, 0], 0)
    np.testing.assert_almost_equal(q_dot[1][0, 0], 0)
    np.testing.assert_almost_equal(q_dot[1][-1, 0], 0)
    np.testing.assert_almost_equal(q_dot[2][0, 0], 0)
    np.testing.assert_almost_equal(q_dot[2][-1, 0], 0)
    # initial and final controls
    np.testing.assert_almost_equal(u[0][0, 0], 1.4516128810214546)
    np.testing.assert_almost_equal(u[0][-1, 0], -1.4516128810214546)
    np.testing.assert_almost_equal(u[1][0, 0], 9.81)
    np.testing.assert_almost_equal(u[1][-1, 0], 9.81)
    np.testing.assert_almost_equal(u[2][0, 0], 2.2790322540381487)
    np.testing.assert_almost_equal(u[2][-1, 0], -2.2790322540381487)
