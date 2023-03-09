"""
Tests for SQP interface.
"""

import os
import sys
import numpy as np
from bioptim import Solver

from .utils import TestUtils


def test_pendulum():
    from bioptim.examples.sqp_method import pendulum as ocp_module

    if sys.platform == "win32":  # for windows ci
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=30,
    )

    solver = Solver.SQP_METHOD(show_online_optim=False)
    solver.set_tol_du(1e-1)
    solver.set_tol_pr(1e-1)
    solver.set_max_iter_ls(1)
    solver.set_maximum_iterations(1)
    sol = ocp.solve(solver)
    sol.detailed_cost_values()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    np.testing.assert_almost_equal(f[0, 0], 84.61466883234333)
    # detailed cost values
    sol.detailed_cost_values()
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 84.61466883234333)

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((14.67225835, 0)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-17.05129702, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)
