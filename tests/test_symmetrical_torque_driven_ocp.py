"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import Data, OdeSolver
from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_symmetry_by_construction(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "symmetry_by_construction",
        str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/symmetry_by_construction.py",
    )
    symmetry_by_construction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(symmetry_by_construction)

    ocp = symmetry_by_construction.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/cubeSym.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 216.56763999010334)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-0.2, -1.1797959, 0.20135792)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0.2, -0.7797959, -0.20135792)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.16129033, 1.16129033, -0.58458751)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.16129033, -1.16129033, 0.58458751)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_symmetry_by_constraint(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "symmetry_by_constraint",
        str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/symmetry_by_constraint.py",
    )
    symmetry_by_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(symmetry_by_constraint)

    ocp = symmetry_by_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/cubeSym.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 240.92170742402698)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (276, 1))
    np.testing.assert_almost_equal(g, np.zeros((276, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-0.2, -1.1797959, 0.20135792, -0.20135792)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0.2, -0.7797959, -0.20135792, 0.20135792)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.16129033, 1.16129033, -0.58458751, 0.58458751)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.16129033, -1.16129033, 0.58458751, -0.58458751)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)
