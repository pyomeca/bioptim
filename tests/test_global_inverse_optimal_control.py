"""
Test for file IO
"""
import os
import numpy as np


def test_double_pendulum_torque_driven_IOCP():
    # Load double pendulum ocp
    from bioptim.examples.inverse_optimal_control import double_pendulum_torque_driven_IOCP as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = bioptim_folder + "/models/double_pendulum.bioMod"

    ocp = ocp_module.prepare_ocp(weights=[0.4, 0.3, 0.3], coefficients=[1, 1, 1], biorbd_model_path=biorbd_model_path)

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
    np.testing.assert_almost_equal(f[0, 0], 13.03787939)

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([-3.14159265, 0.0]))
    np.testing.assert_almost_equal(q[:, -1], np.array([3.14159265, 0.0]))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([-3.32315017, 15.70796327]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([3.0362723, -2.87576071]))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-11.49023683]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([0.04617407]))
