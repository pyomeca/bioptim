"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import ProblemType, OdeSolver

# Load symmetry
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "symmetry_by_construction",
    str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/symmetry_by_construction.py",
)
symmetry_by_construction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symmetry_by_construction)

spec = importlib.util.spec_from_file_location(
    "symmetry_by_constraint", str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/symmetry_by_constraint.py",
)
symmetry_by_constraint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symmetry_by_constraint)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_symmetry_by_construction(ode_solver):
    ocp = symmetry_by_construction.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/cubeSym.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 14.437842666006878)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-0.2, -1.1797959, 0.20135792)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0.2, -0.7797959, -0.20135792)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.16129033, 1.16129033, -0.58458751)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.16129033, -1.16129033, 0.58458751)))


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_symmetry_by_constraint(ode_solver):
    ocp = symmetry_by_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/cubeSym.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 16.0614471616022)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (276, 1))
    np.testing.assert_almost_equal(g, np.zeros((276, 1)))

    # Check some of the results
    q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-0.2, -1.1797959, 0.20135792, -0.20135792)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0.2, -0.7797959, -0.20135792, 0.20135792)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.16129033, 1.16129033, -0.58458751, 0.58458751)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.16129033, -1.16129033, 0.58458751, -0.58458751)))
