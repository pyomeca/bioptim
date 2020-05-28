"""
Test for file IO
"""
from IPython import embed
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import Data, OdeSolver
from .utils import TestUtils

# Load align_and_minimize_marker_velocity
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_and_minimize_marker_velocity", str(PROJECT_FOLDER) + "/examples/align/align_and_minimize_marker_velocity.py"
)
align_and_minimize_marker_velocity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_and_minimize_marker_velocity)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_align_and_minimize_marker_velocity(ode_solver):
    ocp = align_and_minimize_marker_velocity.prepare_ocp(
        biorbd_model_path="../examples/align/cube_and_line.bioMod",
        number_shooting_points=30,
        final_time=2,
        ode_solver=OdeSolver.RK,
        marker_velocity_or_displacement="disp",
        marker_in_first_coordinates_system=True,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -299.32003502999976)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (240, 1))
    np.testing.assert_almost_equal(g, np.zeros((240, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.00532492, 0.60560039, -3.14159265, -1.57079629]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.0205633, -0.39416712, -3.14159265, 1.54031493]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([4.23723834, 0.23820796, 10.0, 10.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.06784366, -0.30153594, -10.0, 10.0]))
    # initial and final controls
    np.testing.assert_almost_equal(
        tau[:, 0], np.array([-1.78820451e01, -5.42045478e01, -3.24087732e-07, -4.92075598e-08])
    )
    np.testing.assert_almost_equal(
        tau[:, -1], np.array([-1.43331297e01, 4.44642900e01, -3.24089534e-07, -1.55472797e-11])
    )

    # # save and load
    TestUtils.save_and_load(sol, ocp, False)


# Load align_and_minimize_marker_velocity
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_and_minimize_marker_velocity", str(PROJECT_FOLDER) + "/examples/align/align_and_minimize_marker_velocity.py"
)
align_and_minimize_marker_velocity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_and_minimize_marker_velocity)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_align_and_minimize_marker_velocity(ode_solver):
    ocp = align_and_minimize_marker_velocity.prepare_ocp(
        biorbd_model_path="../examples/align/cube_and_line.bioMod",
        number_shooting_points=30,
        final_time=2,
        ode_solver=OdeSolver.RK,
        marker_velocity_or_displacement="velo",
        marker_in_first_coordinates_system=True,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -182.9629636223562)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (240, 1))
    np.testing.assert_almost_equal(g, np.zeros((240, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(
        q[:, 0], np.array([8.00000006e-01, -2.87113223e-02, -2.66667399e00, -3.29908037e-18])
    )
    np.testing.assert_almost_equal(
        q[:, -1], np.array([8.00000006e-01, -2.87113207e-02, 2.66665936e00, -3.29908059e-18])
    )
    # initial and final velocities
    # np.testing.assert_almost_equal(qdot[:, 0], np.array([8.18846134e-12, 5.76712958e-01, 1.00000000e+01, -5.11754592e-26]))
    # np.testing.assert_almost_equal(qdot[:, -1], np.array([-7.96628725e-11, -5.76712920e-01, 1.00000000e+01, 4.28606932e-25]))
    # # initial and final controls
    # np.testing.assert_almost_equal(tau[:, 0], np.array([ 1.63036579e-11, 4.48708557e+00, 2.26990399e-12, -3.22449566e-24]))
    # np.testing.assert_almost_equal(tau[:, -1], np.array([1.89632057e-10, 4.48708591e+00, -2.31161450e-12, 1.50952281e-23]))

    # # save and load
    TestUtils.save_and_load(sol, ocp, False)
