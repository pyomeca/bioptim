"""
Test for file IO
"""

from bioptim import Solver, PhaseDynamics, SolutionMerge
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_cyclic_nmpc(phase_dynamics):
    def update_functions(_nmpc, cycle_idx, _sol):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    from bioptim.examples.moving_horizon_estimation import cyclic_nmpc as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_cycles = 3
    cycle_len = 20
    nmpc = ocp_module.prepare_nmpc(
        model_path=bioptim_folder + "/../models/arm2.bioMod",
        cycle_len=cycle_len,
        cycle_duration=1,
        max_torque=50,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
    )
    sol = nmpc.solve(update_functions, solver=Solver.IPOPT())

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_equal(q.shape, (3, n_cycles * cycle_len + 1))
    npt.assert_almost_equal(q[:, 0], np.array((-3.14159265, 0.12979378, 2.77623291)))
    npt.assert_almost_equal(q[:, -1], np.array((3.14159265, 0.12979378, 2.77623291)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((6.28268908, -11.63289399, 0.37215021)))
    npt.assert_almost_equal(qdot[:, -1], np.array((6.28268908, -12.14519356, -0.21986407)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((0.01984925, 17.53758229, -1.92204945)))
    npt.assert_almost_equal(tau[:, -1], np.array((-0.01984925, -6.81104298, -1.80560018)))
