"""
Tests for SQP interface.
"""

from bioptim import Solver, PhaseDynamics, SolutionMerge
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_pendulum(phase_dynamics):
    from bioptim.examples.toy_examples.sqp_method import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/../../models/pendulum.bioMod",
        final_time=1,
        n_shooting=5,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )

    solver = Solver.SQP_METHOD()
    solver.set_tol_du(1e-1)
    solver.set_tol_pr(1e-1)
    solver.set_max_iter_ls(1)
    solver.set_maximum_iterations(1)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    npt.assert_almost_equal(f[0, 0], 124.90212482956895)
    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 124.90212482956895)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((11.75634204, 0)))
    npt.assert_almost_equal(tau[:, -1], np.array((-16.60785771, 0)))
