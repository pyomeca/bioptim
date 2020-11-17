"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from bioptim import Data, OdeSolver, ControlType
from .utils import TestUtils


def test_align_and_minimize_marker_displacement_global():
    # Load align_and_minimize_marker_velocity
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_and_minimize_marker_velocity",
        str(PROJECT_FOLDER) + "/examples/align/align_and_minimize_marker_velocity.py",
    )
    align_and_minimize_marker_velocity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_and_minimize_marker_velocity)

    ocp = align_and_minimize_marker_velocity.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod",
        number_shooting_points=5,
        final_time=1,
        marker_velocity_or_displacement="disp",
        marker_in_first_coordinates_system=False,
        control_type=ControlType.CONSTANT,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -143.5854887928483)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.71797344, -0.44573002, -3.00001922, 0.02378758]), decimal=2)
    np.testing.assert_almost_equal(q[:, -1], np.array([1.08530972, -0.3869361, 2.99998083, -0.02378757]), decimal=2)
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.37791617, 3.70167396, 10.0, 10.0]), decimal=2)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.37675299, -3.40771446, 10.0, 10.0]), decimal=2)
    # initial and final controls
    np.testing.assert_almost_equal(
        tau[:, 0], np.array([-4.52595667e-02, 9.25475333e-01, -4.34001849e-08, -9.24667407e01]), decimal=2
    )
    np.testing.assert_almost_equal(
        tau[:, -1], np.array([4.42976253e-02, 1.40077846e00, -7.28864793e-13, 9.24667396e01]), decimal=2
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


def test_align_and_minimize_marker_displacement_RT():
    # Load align_and_minimize_marker_velocity
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_and_minimize_marker_velocity",
        str(PROJECT_FOLDER) + "/examples/align/align_and_minimize_marker_velocity.py",
    )
    align_and_minimize_marker_velocity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_and_minimize_marker_velocity)

    ocp = align_and_minimize_marker_velocity.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod",
        number_shooting_points=5,
        final_time=1,
        marker_velocity_or_displacement="disp",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.CONSTANT,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -200.80194174353494)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.02595694, -0.57073004, -1.00000001, 1.57079633]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.05351275, -0.44696334, 1.00000001, 1.57079633]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([5.03822848, 4.6993718, 10.0, -10.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.76267039, -3.46170478, 10.0, 10.0]))
    # initial and final controls
    np.testing.assert_almost_equal(
        tau[:, 0], np.array([-2.62720589e01, 4.40828815e00, -5.68180291e-08, 2.61513716e-08])
    )
    np.testing.assert_almost_equal(
        tau[:, -1], np.array([-2.62720590e01, 4.40828815e00, 5.68179790e-08, 2.61513677e-08])
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


def test_align_and_minimize_marker_velocity():
    # Load align_and_minimize_marker_velocity
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_and_minimize_marker_velocity",
        str(PROJECT_FOLDER) + "/examples/align/align_and_minimize_marker_velocity.py",
    )
    align_and_minimize_marker_velocity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_and_minimize_marker_velocity)

    ocp = align_and_minimize_marker_velocity.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod",
        number_shooting_points=5,
        final_time=1,
        marker_velocity_or_displacement="velo",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.CONSTANT,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -80.20048585400944)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([7.18708669e-01, -4.45703930e-01, -3.14159262e00, 0]))
    np.testing.assert_almost_equal(q[:, -1], np.array([1.08646846e00, -3.86731175e-01, 3.14159262e00, 0]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([3.78330878e-01, 3.70214281, 10, 0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([3.77168521e-01, -3.40782793, 10, 0]))
    # # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-4.52216174e-02, 9.25170010e-01, 0, 0]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([4.4260355e-02, 1.4004583, 0, 0]))

    # # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


def test_align_and_minimize_marker_velocity_linear_controls():
    # Load align_and_minimize_marker_velocity
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_and_minimize_marker_velocity",
        str(PROJECT_FOLDER) + "/examples/align/align_and_minimize_marker_velocity.py",
    )
    align_and_minimize_marker_velocity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_and_minimize_marker_velocity)

    ocp = align_and_minimize_marker_velocity.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod",
        number_shooting_points=5,
        final_time=1,
        marker_velocity_or_displacement="velo",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.LINEAR_CONTINUOUS,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -80.28869898410233)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[2:, 0], np.array([-3.14159264, 0]))
    np.testing.assert_almost_equal(q[2:, -1], np.array([3.14159264, 0]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[2:, 0], np.array([10, 0]))
    np.testing.assert_almost_equal(qdot[2:, -1], np.array([10, 0]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[2:, 0], np.array([-8.495542, 0]), decimal=5)
    np.testing.assert_almost_equal(tau[2:, -1], np.array([8.495541, 0]), decimal=5)

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp, decimal_value=6)
