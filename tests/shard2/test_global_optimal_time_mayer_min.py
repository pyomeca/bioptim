"""
Test for file IO
"""
import os
import platform
import pytest

import numpy as np
from bioptim import OdeSolver, PhaseDynamics, SolutionMerge

from tests.utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_pendulum_min_time_mayer(ode_solver, phase_dynamics):
    # Load pendulum_min_time_Mayer
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

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
        phase_dynamics=phase_dynamics,
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
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    tf = sol.decision_time(to_merge=SolutionMerge.NODES)[-1, 0]

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
        np.testing.assert_almost_equal(tau[:, -1], np.array((-99.99938226, 0)), decimal=6)

        # optimized time
        np.testing.assert_almost_equal(tf, 0.2855606738489079)

    elif ode_solver == OdeSolver.COLLOCATION:
        pass

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 0.2862324498580764)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((70.46224716, 0)), decimal=6)
        np.testing.assert_almost_equal(tau[:, -1], np.array((-99.99964325, 0)), decimal=6)

        # optimized time
        np.testing.assert_almost_equal(tf, 0.2862324498580764)
    else:
        raise ValueError("Test not implemented")

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_pendulum_min_time_mayer_constrained(ode_solver, phase_dynamics):
    # Load pendulum_min_time_Mayer
    from bioptim.examples.optimal_time_ocp import pendulum_min_time_Mayer as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    tf = 1
    ns = 30
    min_tf = 0.5
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=tf,
        n_shooting=ns,
        ode_solver=ode_solver(),
        min_time=min_tf,
        phase_dynamics=phase_dynamics,
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
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q, qdot = states["q"], states["qdot"]
    tf = sol.decision_time(to_merge=SolutionMerge.NODES)[-1, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], min_tf, decimal=4)

    # optimized time
    np.testing.assert_almost_equal(tf, min_tf, decimal=4)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)
