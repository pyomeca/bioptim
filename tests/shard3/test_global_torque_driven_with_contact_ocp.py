"""
Test for file IO.
It tests the results of an optimal control problem with torque_driven_with_contact problem type regarding the proper functioning of :
- the maximize/minimize_predicted_height_CoM objective
- the contact_forces_inequality constraint
- the non_slipping constraint
"""
import os
import pytest

import numpy as np
from bioptim import OdeSolver, RigidBodyDynamics, Solver, PhaseDynamics

from tests.utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize(
    "objective_name", ["MINIMIZE_PREDICTED_COM_HEIGHT", "MINIMIZE_COM_POSITION", "MINIMIZE_COM_VELOCITY"]
)
def test_maximize_predicted_height_CoM(objective_name, phase_dynamics):
    from bioptim.examples.torque_driven_ocp import maximize_predicted_height_CoM as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        n_shooting=5,
        use_actuators=False,
        ode_solver=OdeSolver.RK4(),
        objective_name=objective_name,
        com_constraints=True,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
    )
    sol = ocp.solve()

    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (76, 1))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    # initial velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))

    if objective_name == "MINIMIZE_PREDICTED_COM_HEIGHT":
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 0.7719447987404187)

        # final position
        np.testing.assert_almost_equal(q[:, -1], np.array((0.1189654, -0.0904378, -0.7999996, 0.7999996)))
        # final velocities
        np.testing.assert_almost_equal(qdot[:, -1], np.array((1.2477864, -1.2847726, -3.5819658, 3.5819658)))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-18.6635363)))
        np.testing.assert_almost_equal(tau[:, -2], np.array(-0.5142317))
    elif objective_name == "MINIMIZE_COM_POSITION":
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 0.4652603337905152)

        # final position
        np.testing.assert_almost_equal(q[:, -1], np.array((0.1189654, -0.0904378, -0.7999997, 0.7999997)))
        # final velocities
        np.testing.assert_almost_equal(qdot[:, -1], np.array((1.2302646, -1.2667316, -3.5316666, 3.5316666)))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-18.5531974)))
        np.testing.assert_almost_equal(tau[:, -2], np.array(-0.9187262))
    elif objective_name == "MINIMIZE_COM_VELOCITY":
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 0.46678212036841293)

        # final position
        np.testing.assert_almost_equal(q[:, -1], np.array((0.1189654, -0.0904379, -0.7999998, 0.7999998)))
        # final velocities
        np.testing.assert_almost_equal(qdot[:, -1], np.array((1.2640489, -1.3015177, -3.6286507, 3.6286507)))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-18.7970058)))
        np.testing.assert_almost_equal(tau[:, -2], np.array(-0.1918057))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE])
def test_maximize_predicted_height_CoM_with_actuators(phase_dynamics):
    from bioptim.examples.torque_driven_ocp import maximize_predicted_height_CoM as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        n_shooting=20,
        use_actuators=True,
        ode_solver=OdeSolver.RK4(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.21850679397314332)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.2393758, 0.0612086, -0.0006739, 0.0006739)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        qdot[:, -1], np.array((-4.8768219e-01, 3.2867302e-04, 9.7536459e-01, -9.7536459e-01))
    )
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-0.550905)))
    np.testing.assert_almost_equal(tau[:, -2], np.array(-0.0050623))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE])
@pytest.mark.parametrize(
    "rigidbody_dynamics",
    [RigidBodyDynamics.ODE, RigidBodyDynamics.DAE_FORWARD_DYNAMICS, RigidBodyDynamics.DAE_INVERSE_DYNAMICS],
)
def test_maximize_predicted_height_CoM_rigidbody_dynamics(rigidbody_dynamics, phase_dynamics):
    from bioptim.examples.torque_driven_ocp import maximize_predicted_height_CoM as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = OdeSolver.RK4()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        n_shooting=20,
        use_actuators=False,
        ode_solver=ode_solver,
        rigidbody_dynamics=rigidbody_dynamics,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    sol_opt = Solver.IPOPT(show_online_optim=False)
    sol_opt.set_maximum_iterations(1)
    sol = ocp.solve(sol_opt)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    if rigidbody_dynamics == RigidBodyDynamics.ODE:
        np.testing.assert_almost_equal(f[0, 0], 0.8032447451950947)
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
        np.testing.assert_almost_equal(f[0, 0], 0.9695327421106931)
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        np.testing.assert_almost_equal(f[0, 0], 1.691190510518052)

    # Check constraints
    g = np.array(sol.constraints)
    if rigidbody_dynamics == RigidBodyDynamics.ODE:
        np.testing.assert_equal(g.shape, (160, 1))
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
        np.testing.assert_equal(g.shape, (240, 1))
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        np.testing.assert_equal(g.shape, (300, 1))
