"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import ProblemType, OdeSolver

# Load eocar
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "eocar", str(PROJECT_FOLDER) + "/examples/multiphase_torque_driven_ocp/eocar.py"
)
eocar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eocar)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_eocar(ode_solver):
    ocp = eocar.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/multiphase_torque_driven_ocp/eocar.bioMod",
        ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 9964.984600659047)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (315, 1))
    np.testing.assert_almost_equal(g, np.zeros((315, 1)))

    # Check some of the results
    q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[0][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[0][:, -1], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[1][:, 0], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[1][:, -1], np.array((1, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[0][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[0][:, -1], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[1][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[1][:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[0][:, 0], np.array((1.42857142, 9.81, 0)))
    np.testing.assert_almost_equal(tau[0][:, -1], np.array((-1.42857144, 9.81, 0)))
    np.testing.assert_almost_equal(tau[1][:, 0], np.array((-0.2322581, 9.81, 0.364645157)))
    np.testing.assert_almost_equal(tau[1][:, -1], np.array((0.23225804, 9.81, -0.364645168)))
