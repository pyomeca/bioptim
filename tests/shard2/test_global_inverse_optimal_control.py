"""
Test for file IO
"""
import os

from bioptim import PhaseDynamics
import numpy as np
import pytest


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_double_pendulum_torque_driven_IOCP(phase_dynamics):
    # Load double pendulum ocp
    from bioptim.examples.inverse_optimal_control import double_pendulum_torque_driven_IOCP as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
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
    states, controls = sol.states, sol.controls
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g, np.zeros((120, 1)))

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 12.0765913088802)

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([-3.14159265, 0.0]))
    np.testing.assert_almost_equal(q[:, -1], np.array([3.14159265, 0.0]))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([-3.29089471, 15.70796327]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([ 2.97490708, -2.73024542]))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-11.9453666]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([0.02482167]))