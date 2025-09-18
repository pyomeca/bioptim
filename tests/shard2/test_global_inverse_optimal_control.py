"""
Test for file IO
"""

from bioptim import PhaseDynamics, SolutionMerge
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_double_pendulum_torque_driven_IOCP(phase_dynamics):
    # Load double pendulum ocp
    from bioptim.examples.toy_examples.inverse_optimal_control import double_pendulum_torque_driven_IOCP as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)
    biorbd_model_path = bioptim_folder + "/models/double_pendulum.bioMod"

    ocp = ocp_module.prepare_ocp(
        weights=[0.4, 0.3, 0.3],
        coefficients=[1, 1, 1],
        biorbd_model_path=biorbd_model_path,
        phase_dynamics=phase_dynamics,
        n_threads=4 if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE else 1,
        expand_dynamics=True,
    )

    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    npt.assert_equal(g.shape, (120, 1))
    npt.assert_almost_equal(g, np.zeros((120, 1)))

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 12.0765913088802)

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([-3.14159265, 0.0]))
    npt.assert_almost_equal(q[:, -1], np.array([3.14159265, 0.0]))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array([-3.29089471, 15.70796327]))
    npt.assert_almost_equal(qdot[:, -1], np.array([2.97490708, -2.73024542]))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([-11.9453666]))
    npt.assert_almost_equal(tau[:, -1], np.array([0.02482167]))
