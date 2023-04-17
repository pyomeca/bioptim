"""
Test for file IO
"""
import pytest
import os

import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_segment_on_rt(ode_solver):
    from bioptim.examples.track import track_segment_on_rt as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        final_time=0.5,
        n_shooting=8,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 197120.95524154368)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (91, 1))
    np.testing.assert_almost_equal(g, np.zeros((91, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.30543155, 0, -1.57, -1.57]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.30543155, 0, 1.57, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([0, 9.81, 66.98666900582079, 66.98666424580644]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-0, 9.81, -66.98666900582079, -66.98666424580644]))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_marker_on_segment(ode_solver):
    from bioptim.examples.track import track_marker_on_segment as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        final_time=0.5,
        n_shooting=8,
        initialize_near_solution=True,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 42127.04677760122)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (88, 1))
    np.testing.assert_almost_equal(g, np.zeros((88, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1, 0, 0, 0.46364761]))
    np.testing.assert_almost_equal(q[:, -1], np.array([2, 0, 1.57, 0.78539785]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([23.6216587, 12.2590703, 31.520697, 12.9472294]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-16.659525, 14.5872277, -36.1009998, 4.417834]))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)

def test_track_vector_orientation():
    from bioptim.examples.track import track_vector_orientation as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        final_time=1,
        n_shooting=10,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 1206.2849823554816)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (80, 1))
    np.testing.assert_almost_equal(g, np.zeros((80, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.80000001, -0.68299837, -1.57      , -1.56999931]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.80000001, -0.68299837,  1.57      ,  1.56999966]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((-5.62939992e-16,  4.90499998e+00,  3.14000113e+00,  3.13999784e+00)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-5.62940009e-16, -4.90499998e+00,  3.14000029e+00,  3.13999868e+00)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([1.05886391e-25,  6.28215525e-09, -2.53398134e-06,  2.52582173e-06]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([-6.38733211e-24,  6.28215525e-09,  1.31751960e-06, -1.30936553e-06]))
