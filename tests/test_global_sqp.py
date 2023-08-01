"""
Tests for SQP interface.
"""

import os
import numpy as np
from bioptim import Solver
import pytest

from .utils import TestUtils


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_pendulum(assume_phase_dynamics):
    from bioptim.examples.sqp_method import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=5,
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=True,
    )

    solver = Solver.SQP_METHOD(show_online_optim=False)
    solver.set_tol_du(1e-1)
    solver.set_tol_pr(1e-1)
    solver.set_max_iter_ls(1)
    solver.set_maximum_iterations(1)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    np.testing.assert_almost_equal(f[0, 0], 124.90212482956895)
    # detailed cost values
    np.testing.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 124.90212482956895)

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
    np.testing.assert_almost_equal(tau[:, 0], np.array((11.75634204, 0)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-16.60785771, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, test_solve_of_loaded=True, solver=solver)
