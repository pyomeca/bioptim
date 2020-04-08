"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np

# Load jumper2contacts
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "jumper", str(PROJECT_FOLDER) + "/examples/jumper/jumper2contacts.py"
)
jumper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jumper)


def test_jumper():
    nlp = jumper.prepare_nlp(
        biorbd_model_path=str(PROJECT_FOLDER)
        + "/examples/jumper/jumper2contacts.bioMod"
    )
    sol = nlp.solve()

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
    print(q[0][0, 0])
    print(q[0][-1, 0])
    print(q[1][0, 0])
    print(q[1][-1, 0])
    print(q[2][0, 0])
    print(q[2][-1, 0])
    print(q[3][0, 0])
    print(q[3][-1, 0])
    print(q[4][0, 0])
    print(q[4][-1, 0])
    print(q[5][0, 0])
    print(q[5][-1, 0])
    print(q[6][0, 0])
    print(q[6][-1, 0])
    print(q[7][0, 0])
    print(q[7][-1, 0])
    print(q[8][0, 0])
    print(q[8][-1, 0])
    print(q[9][0, 0])
    print(q[9][-1, 0])
    print(q[10][0, 0])
    print(q[10][-1, 0])
    print(q[11][0, 0])
    print(q[12][-1, 0])
    print(q[12][0, 0])
    print(q[12][-1, 0])

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 2204.722517381283)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (480, 1))
    np.testing.assert_almost_equal(g, np.zeros((480, 1)))  # ???

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
    np.testing.assert_almost_equal(q[0][0, 0], 0)
    np.testing.assert_almost_equal(q[0][-1, 0], -0.8831848069141922)
    np.testing.assert_almost_equal(q[1][0, 0], 0)
    np.testing.assert_almost_equal(q[1][-1, 0], 0.5086923695214512)
    np.testing.assert_almost_equal(q[2][0, 0], -0.5336)
    np.testing.assert_almost_equal(q[2][-1, 0], -0.23114696706313925)
    # initial and final velocities
    np.testing.assert_almost_equal(q_dot[0][0, 0], 0)
    np.testing.assert_almost_equal(q_dot[0][-1, 0], -0.07645884632825446)
    np.testing.assert_almost_equal(q_dot[1][0, 0], 1.4)
    np.testing.assert_almost_equal(q_dot[1][-1, 0], -0.835052743838774)
    np.testing.assert_almost_equal(q_dot[2][0, 0], 0.8)
    np.testing.assert_almost_equal(q_dot[2][-1, 0], 0)
    # initial and final controls
    np.testing.assert_almost_equal(u[0][0, 0], 1.4516128810214546)
    np.testing.assert_almost_equal(u[0][-1, 0], -1.4516128810214546)
    np.testing.assert_almost_equal(u[1][0, 0], 9.81)
    np.testing.assert_almost_equal(u[1][-1, 0], 9.81)
    np.testing.assert_almost_equal(u[2][0, 0], 2.2790322540381487)
    np.testing.assert_almost_equal(u[2][-1, 0], -2.2790322540381487)
