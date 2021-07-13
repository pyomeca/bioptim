"""
Test for file IO
"""
import pytest
import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_symmetry_by_construction(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    sym = TestUtils.load_module(bioptim_folder + "/examples/symmetrical_torque_driven_ocp/symmetry_by_mapping.py")
    ode_solver = ode_solver()

    ocp = sym.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/symmetrical_torque_driven_ocp/cubeSym.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 216.56763999010334)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-0.2, -1.1797959, 0.20135792)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0.2, -0.7797959, -0.20135792)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.16129033, 1.16129033, -0.58458751)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-1.16129033, -1.16129033, 0.58458751)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_symmetry_by_constraint(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    sym = TestUtils.load_module(bioptim_folder + "/examples/symmetrical_torque_driven_ocp/symmetry_by_constraint.py")
    ode_solver = ode_solver()

    ocp = sym.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/symmetrical_torque_driven_ocp/cubeSym.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 216.56763999010465)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (336, 1))
    np.testing.assert_almost_equal(g, np.zeros((336, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-0.2, -1.1797959, 0, 0.20135792, -0.20135792)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0.2, -0.7797959, 0, -0.20135792, 0.20135792)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.16129033, 1.16129033, 0, -0.58458751, 0.58458751)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-1.16129033, -1.16129033, 0, 0.58458751, -0.58458751)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)
