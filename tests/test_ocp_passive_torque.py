import pytest
import numpy as np
from bioptim import RigidBodyDynamics, BiorbdModel, OdeSolver, Solver
import os



@pytest.mark.parametrize(
    "rigidbody_dynamics",
    [
        RigidBodyDynamics.DAE_FORWARD_DYNAMICS,
        RigidBodyDynamics.DAE_INVERSE_DYNAMICS,
    ])
@pytest.mark.parametrize(
    "with_passive_torque", [
        False,
        True,
    ])
def test_pendulum_passive_torque(rigidbody_dynamics, with_passive_torque):
    from bioptim.examples.torque_driven_ocp import pendulum_with_passive_torque as ocp_module
    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/pendulum_with_passive_torque.bioMod"
    final_time = 0.1
    n_shooting = 5
    ode_solver = OdeSolver.RK4()
    use_sx = True
    n_threads = 1

    biorbd_model = BiorbdModel(biorbd_model_path)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path,
        final_time,
        n_shooting,
        ode_solver,
        use_sx,
        n_threads,
        rigidbody_dynamics=RigidBodyDynamics.ODE,
        with_passive_torque=with_passive_torque,
    )
    solver = Solver.IPOPT()

    # solver.set_maximum_iterations(10)
    sol = ocp.solve(solver)

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        if with_passive_torque:

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0., 0.]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0., 3.14]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0., 0.]))
            # initial and final controls
            np.testing.assert_almost_equal(tau[:, 0], np.array([37.2828933,  0.]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-4.9490898,  0.]))

        else:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0., 0.]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0., 3.14]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0., 0.]))
            # initial and final controls
            np.testing.assert_almost_equal(tau[:, 0], np.array([-70.3481693,   0.]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-35.5389507,   0.]))

    else :

        if with_passive_torque:

            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0., 0.]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0., 3.14]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0., 0.]))
            # initial and final controls
            np.testing.assert_almost_equal(tau[:, 0], np.array([37.2828933,  0.]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-4.9490898,  0.]))

        else:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0., 0.]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0., 3.14]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0., 0.]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0., 0.]))
            # initial and final controls
            np.testing.assert_almost_equal(tau[:, 0], np.array([-70.3481693,   0.]))
            np.testing.assert_almost_equal(tau[:, -2], np.array([-35.5389507,   0.]))
