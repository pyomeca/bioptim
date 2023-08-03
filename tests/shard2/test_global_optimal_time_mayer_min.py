"""
Test for file IO
"""
import os
import platform
import pytest

import numpy as np
from bioptim import OdeSolver

from tests.utils import TestUtils


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_pendulum_min_time_mayer(ode_solver, assume_phase_dynamics):
    # Load pendulum_min_time_Mayer
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if ode_solver == OdeSolver.IRK:
        ft = 2
        ns = 35
    elif ode_solver == OdeSolver.COLLOCATION:
        ft = 2
        ns = 10
    elif ode_solver == OdeSolver.RK4:
        ft = 2
        ns = 30
    else:
        raise ValueError("Test not implemented")

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=ft,
        n_shooting=ns,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (ns * 20, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 20, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (ns * 4, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 4, 1)), decimal=6)

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    tf = sol.parameters["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 0.2855606738489079)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((87.13363409, 0)), decimal=6)
        np.testing.assert_almost_equal(tau[:, -2], np.array((-99.99938226, 0)), decimal=6)

        # optimized time
        np.testing.assert_almost_equal(tf, 0.2855606738489079)

    elif ode_solver == OdeSolver.COLLOCATION:
        pass

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 0.2862324498580764)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((70.46234418, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-99.99964325, 0)))

        # optimized time
        np.testing.assert_almost_equal(tf, 0.2862324498580764)
    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
# @pytest.mark.parametrize("ode_solver", [OdeSolver.COLLOCATION])
def test_pendulum_min_time_mayer_constrained(ode_solver, assume_phase_dynamics):
    if platform.system() != "Linux":
        # This is a long test and CI is already long for Windows and Mac
        return

    # Load pendulum_min_time_Mayer
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if ode_solver == OdeSolver.IRK:
        ft = 2
        ns = 35
        min_ft = 0.5
    elif ode_solver == OdeSolver.COLLOCATION:
        ft = 2
        ns = 10
        min_ft = 0.5
    elif ode_solver == OdeSolver.RK4:
        ft = 2
        ns = 30
        min_ft = 0.5
    else:
        raise ValueError("Test not implemented")

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=ft,
        n_shooting=ns,
        ode_solver=ode_solver(),
        min_time=min_ft,
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (ns * 20, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 20, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (ns * 4, 1))
        np.testing.assert_almost_equal(g, np.zeros((ns * 4, 1)), decimal=6)

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]
    tf = sol.parameters["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 1.1878186850775596)
    else:
        np.testing.assert_almost_equal(f[0, 0], min_ft)

    # optimized time
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(tf, 1.1878186850775596)
    else:
        np.testing.assert_almost_equal(tf, min_ft)

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)
