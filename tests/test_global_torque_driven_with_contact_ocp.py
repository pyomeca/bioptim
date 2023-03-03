"""
Test for file IO.
It tests the results of an optimal control problem with torque_driven_with_contact problem type regarding the proper functioning of :
- the maximize/minimize_predicted_height_CoM objective
- the contact_forces_inequality constraint
- the non_slipping constraint
"""
import os
import pytest
import sys

import numpy as np
from bioptim import OdeSolver, RigidBodyDynamics, Solver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK, OdeSolver.COLLOCATION])
@pytest.mark.parametrize(
    "objective_name", ["MINIMIZE_PREDICTED_COM_HEIGHT", "MINIMIZE_COM_POSITION", "MINIMIZE_COM_VELOCITY"]
)
@pytest.mark.parametrize("com_constraints", [False, True])
def test_maximize_predicted_height_CoM(ode_solver, objective_name, com_constraints):
    from bioptim.examples.torque_driven_ocp import maximize_predicted_height_CoM as ocp_module

    # no rk8 on windows
    if sys.platform == "win32" and ode_solver == OdeSolver.RK8:  # it actually works but not with the CI
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        n_shooting=20,
        use_actuators=False,
        ode_solver=ode_solver,
        objective_name=objective_name,
        com_constraints=com_constraints,
    )
    sol = ocp.solve()

    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver.is_direct_shooting:
        if com_constraints:
            np.testing.assert_equal(g.shape, (286, 1))

        else:
            np.testing.assert_equal(g.shape, (160, 1))
            np.testing.assert_almost_equal(g, np.zeros((160, 1)))

        # Check some of the results
        q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

        # initial position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
        # initial velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))

        if objective_name == "MINIMIZE_PREDICTED_COM_HEIGHT":
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 0.7592028279017864)

            # final position
            np.testing.assert_almost_equal(q[:, -1], np.array((0.1189651, -0.0904378, -0.7999996, 0.7999996)))
            # final velocities
            np.testing.assert_almost_equal(qdot[:, -1], np.array((1.2636414, -1.3010929, -3.6274687, 3.6274687)))
            # initial and final controls
            np.testing.assert_almost_equal(tau[:, 0], np.array((-22.1218282)))
            np.testing.assert_almost_equal(tau[:, -2], np.array(0.2653957))
        elif objective_name == "MINIMIZE_COM_POSITION":
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 0.458575464873056)

            # final position
            np.testing.assert_almost_equal(q[:, -1], np.array((0.1189651, -0.0904378, -0.7999996, 0.7999996)))
            # final velocities
            np.testing.assert_almost_equal(qdot[:, -1], np.array((1.24525494, -1.28216182, -3.57468814, 3.57468814)))
            # initial and final controls
            np.testing.assert_almost_equal(tau[:, 0], np.array((-21.96213697)))
            np.testing.assert_almost_equal(tau[:, -2], np.array(-0.22120207))
        elif objective_name == "MINIMIZE_COM_VELOCITY":
            # Check objective function value
            np.testing.assert_almost_equal(f[0, 0], 0.4709888694097001)

            # final position
            np.testing.assert_almost_equal(q[:, -1], np.array((0.1189652, -0.09043785, -0.79999979, 0.79999979)))
            # final velocities
            np.testing.assert_almost_equal(qdot[:, -1], np.array((1.26103572, -1.29841047, -3.61998944, 3.61998944)))
            # initial and final controls
            np.testing.assert_almost_equal(tau[:, 0], np.array((-22.18008227)))
            np.testing.assert_almost_equal(tau[:, -2], np.array(-0.02280469))
    else:
        if com_constraints:
            np.testing.assert_equal(g.shape, (926, 1))

        else:
            np.testing.assert_equal(g.shape, (800, 1))
            np.testing.assert_almost_equal(g, np.zeros((800, 1)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_maximize_predicted_height_CoM_with_actuators(ode_solver):
    from bioptim.examples.torque_driven_ocp import maximize_predicted_height_CoM as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        n_shooting=20,
        use_actuators=True,
        ode_solver=ode_solver,
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

    if isinstance(ode_solver, OdeSolver.IRK):
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
        np.testing.assert_almost_equal(q[:, -1], np.array((-0.2393758, 0.0612086, -0.0006739, 0.0006739)))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-4.87675667e-01, 3.28672149e-04, 9.75351556e-01, -9.75351556e-01))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-0.5509092)))
        np.testing.assert_almost_equal(tau[:, -2], np.array(-0.00506117))

    elif isinstance(ode_solver, OdeSolver.RK8):
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
        np.testing.assert_almost_equal(q[:, -1], np.array((-0.23937581, 0.06120861, -0.00067392, 0.00067392)))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-4.87675528e-01, 3.28670915e-04, 9.75351279e-01, -9.75351279e-01))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-0.55090931)))
        np.testing.assert_almost_equal(tau[:, -2], np.array(-0.00506117))

    else:
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


@pytest.mark.parametrize(
    "rigidbody_dynamics",
    [RigidBodyDynamics.ODE, RigidBodyDynamics.DAE_FORWARD_DYNAMICS, RigidBodyDynamics.DAE_INVERSE_DYNAMICS],
)
def test_maximize_predicted_height_CoM_rigidbody_dynamics(rigidbody_dynamics):
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
        np.testing.assert_almost_equal(f[0, 0], 1.6940665057034097)

    # Check constraints
    g = np.array(sol.constraints)
    if rigidbody_dynamics == RigidBodyDynamics.ODE:
        np.testing.assert_equal(g.shape, (160, 1))
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
        np.testing.assert_equal(g.shape, (240, 1))
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        np.testing.assert_equal(g.shape, (300, 1))
