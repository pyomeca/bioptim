"""
Test for file IO
"""
import pytest
import numpy as np
import biorbd_casadi as biorbd
from bioptim import OdeSolver, ConstraintList, ConstraintFcn, Node

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
@pytest.mark.parametrize("actuator_type", [None, 2])
def test_track_markers(ode_solver, actuator_type):
    # Load track_markers
    bioptim_folder = TestUtils.bioptim_folder()
    track = TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/track_markers_with_torque_actuators.py")
    ode_solver = ode_solver()

    ocp = track.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/torque_driven_ocp/cube.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=actuator_type,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19767.53312569522)

    # Check constraints
    g = np.array(sol.constraints)
    if not actuator_type:
        np.testing.assert_equal(g.shape, (186, 1))
    else:
        np.testing.assert_equal(g.shape, (372, 1))
    np.testing.assert_almost_equal(g[:186], np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_markers_changing_constraints(ode_solver):
    # Load track_markers
    bioptim_folder = TestUtils.bioptim_folder()
    track = TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/track_markers_with_torque_actuators.py")
    ode_solver = ode_solver()

    ocp = track.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/torque_driven_ocp/cube.bioMod",
        n_shooting=30,
        final_time=2,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Add a new constraint and reoptimize
    new_constraints = ConstraintList()
    new_constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="m0", second_marker="m2", list_index=2
    )
    ocp.update_constraints(new_constraints)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 20370.211697123825)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (189, 1))
    np.testing.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((4.2641129, 9.81, 2.27903226)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((1.36088709, 9.81, -2.27903226)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)

    # Replace constraints and reoptimize
    new_constraints = ConstraintList()
    new_constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m2", list_index=0
    )
    new_constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="m0", second_marker="m3", list_index=2
    )
    ocp.update_constraints(new_constraints)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 31670.93770220887)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (189, 1))
    np.testing.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-5.625, 21.06, 2.2790323)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-5.625, 21.06, -2.27903226)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_markers_with_actuators(ode_solver):
    # Load track_markers
    bioptim_folder = TestUtils.bioptim_folder()
    track = TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/track_markers_with_torque_actuators.py")
    ode_solver = ode_solver()

    ocp = track.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/torque_driven_ocp/cube.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=1,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 204.18087334169184)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.2140175, 0.981, 0.3360075)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-0.2196496, 0.981, -0.3448498)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_marker_2D_pendulum(ode_solver):
    # Load muscle_activations_contact_tracker
    bioptim_folder = TestUtils.bioptim_folder()
    track = TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/track_markers_2D_pendulum.py")
    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/examples/getting_started/pendulum.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 3
    n_shooting = 20

    # Generate data to fit
    np.random.seed(42)
    markers_ref = np.random.rand(3, 2, n_shooting + 1)
    tau_ref = np.random.rand(2, n_shooting)

    ocp = track.prepare_ocp(biorbd_model, final_time, n_shooting, markers_ref, tau_ref, ode_solver=ode_solver)
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (80, 1))
    np.testing.assert_almost_equal(g, np.zeros((80, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 537.1178745697684)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0.9764023, 4.21321536)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((0.2645088, 3.66993626)))

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((0.98431048, -13.78108592)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-0.15668869, 0.77410131)))

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 537.1268848704174)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0.97637866, 4.2130049)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((0.26520638, 3.66961378)))

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((0.98436218, -13.78113709)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-0.15666471, 0.77420505)))

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 537.1280965743497)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0.9763794, 4.2129851)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((0.26528337, 3.6696134)))

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((0.98431925, -13.78108143)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-0.15666955, 0.77421434)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


def test_trampo_quaternions():
    # Load trampo_quaternion
    bioptim_folder = TestUtils.bioptim_folder()
    trampo = TestUtils.load_module(bioptim_folder + "/examples/torque_driven_ocp/trampo_quaternions.py")

    # Define the problem
    model_path = bioptim_folder + "/examples/torque_driven_ocp/TruncAnd2Arm_Quaternion.bioMod"
    final_time = 0.25
    n_shooting = 5

    ocp = trampo.prepare_ocp(model_path, n_shooting, final_time)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -41.491609816961535)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (130, 1))
    np.testing.assert_almost_equal(g, np.zeros((130, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(
        q[:, 0],
        np.array(
            [
                1.81406193,
                1.91381625,
                2.01645623,
                -0.82692043,
                0.22763972,
                -0.12271756,
                0.01240349,
                -0.2132477,
                -0.02276448,
                -0.02113187,
                0.25834635,
                0.02304512,
                0.16975019,
                0.28942782,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        q[:, -1],
        np.array(
            [
                3.14159265,
                3.14159265,
                3.14159265,
                -0.78539815,
                0.6154797,
                0.10983897,
                0.03886357,
                -0.7291664,
                0.09804026,
                0.1080609,
                0.77553818,
                0.05670268,
                0.67616132,
                0.61939341,
            ]
        ),
    )

    # initial and final velocities
    np.testing.assert_almost_equal(
        qdot[:, 0],
        np.array(
            [
                5.29217272,
                4.89048559,
                5.70619178,
                0.14746316,
                1.53632472,
                0.99292273,
                0.939669,
                0.93500575,
                0.6618465,
                0.71874771,
                1.1402524,
                0.84452034,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        qdot[:, -1],
        np.array(
            [
                5.33106829,
                4.93221086,
                3.29404834,
                0.19267326,
                1.55927187,
                0.8793173,
                0.9413292,
                0.52604828,
                1.27993251,
                1.25250626,
                1.39280633,
                1.13948993,
            ]
        ),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.zeros((12,)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -2], np.zeros((12,)), decimal=6)

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)
