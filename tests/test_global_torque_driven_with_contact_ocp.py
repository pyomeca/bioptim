"""
Test for file IO.
It tests the results of an optimal control problem with torque_driven_with_contact problem type regarding the proper functioning of :
- the maximize/minimize_predicted_height_CoM objective
- the contact_forces_inequality constraint
- the non_slipping constraint
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from bioptim import Data
from .utils import TestUtils


def test_maximize_predicted_height_CoM():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "maximize_predicted_height_CoM",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/maximize_predicted_height_CoM.py",
    )
    maximize_predicted_height_CoM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maximize_predicted_height_CoM)

    ocp = maximize_predicted_height_CoM.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        number_shooting_points=20,
        use_actuators=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.7592028279017864)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0.1189651, -0.0904378, -0.7999996, 0.7999996)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((1.2636414, -1.3010929, -3.6274687, 3.6274687)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-22.1218282)))
    np.testing.assert_almost_equal(tau[:, -1], np.array(0.2653957))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


def test_maximize_predicted_height_CoM_with_actuators():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "maximize_predicted_height_CoM",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/maximize_predicted_height_CoM.py",
    )
    maximize_predicted_height_CoM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maximize_predicted_height_CoM)

    ocp = maximize_predicted_height_CoM.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        number_shooting_points=20,
        use_actuators=True,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.21850679397314332)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

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
    np.testing.assert_almost_equal(tau[:, -1], np.array(-0.0050623))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


def test_contact_forces_inequality_GREATER_THAN_constraint():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "contact_forces_inequality_constraint",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/contact_forces_inequality_constraint.py",
    )
    contact_forces_inequality_GREATER_THAN_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contact_forces_inequality_GREATER_THAN_constraint)

    boundary = 50
    ocp = contact_forces_inequality_GREATER_THAN_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        number_shooting_points=10,
        direction="GREATER_THAN",
        boundary=boundary,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.14525621569048172)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (100, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
    np.testing.assert_array_less(-g[80:], -boundary)
    expected_pos_g = np.array(
        [
            [50.76491919],
            [51.42493119],
            [57.79007374],
            [64.29551934],
            [67.01905769],
            [68.3225625],
            [67.91793917],
            [65.26700138],
            [59.57311867],
            [50.18463134],
            [160.14834799],
            [141.15361769],
            [85.13345729],
            [56.33535022],
            [53.32684286],
            [52.21679255],
            [51.62923106],
            [51.25728666],
            [50.9871531],
            [50.21972377],
        ]
    )
    np.testing.assert_almost_equal(g[80:], expected_pos_g)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.34054748, 0.1341555, -0.0005438, 0.0005438)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.01097559, 1.09352001e-03, 4.02195175, -4.02195175)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-54.1684018)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-15.69338332)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


def test_contact_forces_inequality_LESSER_THAN_constraint():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "contact_forces_inequality_constraint",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/contact_forces_inequality_constraint.py",
    )
    contact_forces_inequality_LESSER_THAN_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contact_forces_inequality_LESSER_THAN_constraint)

    boundary = 100
    ocp = contact_forces_inequality_LESSER_THAN_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        number_shooting_points=10,
        direction="LESSER_THAN",
        boundary=boundary,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.14525619649247054)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (100, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
    np.testing.assert_array_less(g[80:], boundary)
    expected_non_zero_g = np.array(
        [
            [63.27237842],
            [63.02339946],
            [62.13898369],
            [60.38380769],
            [57.31193141],
            [52.19952395],
            [43.9638679],
            [31.14938032],
            [12.45022537],
            [-6.35179034],
            [99.06328211],
            [98.87711942],
            [98.64440005],
            [98.34550037],
            [97.94667107],
            [97.38505013],
            [96.52820867],
            [95.03979128],
            [91.73734926],
            [77.48803304],
        ]
    )
    np.testing.assert_almost_equal(g[80:], expected_non_zero_g)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(
        q[:, -1], np.array((-3.40655617e-01, 1.34155544e-01, -3.27530886e-04, 3.27530886e-04))
    )
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.86650427, 9.38827988e-04, 5.73300901, -5.73300901)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-32.78862874)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-25.23729156)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


def test_non_slipping_constraint():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "non_slipping_constraint",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/non_slipping_constraint.py",
    )
    non_slipping_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(non_slipping_constraint)

    ocp = non_slipping_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.6,
        number_shooting_points=10,
        mu=0.005,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.23984490846250128)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
    np.testing.assert_array_less(-g[80:], 0)
    expected_pos_g = np.array(
        [
            [8.74337995e01],
            [8.74671258e01],
            [8.75687834e01],
            [8.77422814e01],
            [8.79913157e01],
            [8.83197844e01],
            [8.87318039e01],
            [8.92317298e01],
            [8.98241976e01],
            [9.05145013e01],
            [4.63475930e01],
            [4.63130361e01],
            [4.62075073e01],
            [4.60271956e01],
            [4.57680919e01],
            [4.54259742e01],
            [4.49963909e01],
            [4.44746357e01],
            [4.38556802e01],
            [4.31334141e01],
            [1.33775343e00],
            [6.04899894e-05],
            [1.33773204e00],
            [6.95785950e-05],
            [1.33768173e00],
            [8.11784641e-05],
            [1.33759829e00],
            [9.64764869e-05],
            [1.33747653e00],
            [1.17543301e-04],
            [1.33730923e00],
            [1.48352248e-04],
            [1.33708435e00],
            [1.97600363e-04],
            [1.33677502e00],
            [2.88636453e-04],
            [1.33628619e00],
            [5.12590377e-04],
            [1.33466928e00],
            [1.80987419e-03],
        ]
    )
    np.testing.assert_almost_equal(g[80:], expected_pos_g)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.02364845, 0.01211471, -0.44685185, 0.44685185)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-0.08703131, 0.04170362, 0.1930144, -0.1930144)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-14.33813755)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-13.21317493)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)
