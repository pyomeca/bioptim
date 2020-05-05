"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import Data, OdeSolver

# Load align_segment_on_rt
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_segment_on_rt", str(PROJECT_FOLDER) + "/examples/align/align_segment_on_rt.py"
)
align_segment_on_rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_segment_on_rt)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_align_segment_on_rt(ode_solver):
    ocp = align_segment_on_rt.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 12320.059717265229)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (91, 1))
    np.testing.assert_almost_equal(g, np.zeros((91, 1)))

    # Check some of the results
    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.305837645, 6.07684988e-18, -1.57, -1.57]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.305837645, 2.34331392e-17, 1.57, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([1.09038782e-23, 9.81, 66.9866667, 66.9866667]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-1.61910771e-23,  9.81, -66.9866667, -66.9866667]))


# Load align_marker_on_segment
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_marker_on_segment", str(PROJECT_FOLDER) + "/examples/align/align_marker_on_segment.py"
)
align_marker_on_segment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_marker_on_segment)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_align_marker_on_segment(ode_solver):
    ocp = align_marker_on_segment.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 2632.94081023)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (88, 1))
    np.testing.assert_almost_equal(g, np.zeros((88, 1)))

    # Check some of the results
    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1, 0, 0, 0.46364761]))
    np.testing.assert_almost_equal(q[:, -1], np.array([2, 0, 1.57, 0.78539785]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([23.62164829, 12.25908105, 31.52069123, 12.94722946]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-16.65951013,  14.58726024, -36.10090342,   4.41782557]))
