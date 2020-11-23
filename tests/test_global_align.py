"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from bioptim import Data, OdeSolver
from .utils import TestUtils


def test_align_segment_on_rt():
    # Load align_segment_on_rt
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_segment_on_rt", str(PROJECT_FOLDER) + "/examples/align/align_segment_on_rt.py"
    )
    align_segment_on_rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_segment_on_rt)

    ocp = align_segment_on_rt.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod",
        final_time=0.5,
        number_shooting_points=8,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 197120.95524154368)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (91, 1))
    np.testing.assert_almost_equal(g, np.zeros((91, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.30543155, 0, -1.57, -1.57]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.30543155, 0, 1.57, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([0, 9.81, 66.98666900582079, 66.98666424580644]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-0, 9.81, -66.98666900582079, -66.98666424580644]))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


def test_align_marker_on_segment():
    # Load align_marker_on_segment
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_marker_on_segment", str(PROJECT_FOLDER) + "/examples/align/align_marker_on_segment.py"
    )
    align_marker_on_segment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_marker_on_segment)

    ocp = align_marker_on_segment.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod",
        final_time=0.5,
        number_shooting_points=8,
        initialize_near_solution=True,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 42127.04677760122)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (88, 1))
    np.testing.assert_almost_equal(g, np.zeros((88, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1, 0, 0, 0.46364761]))
    np.testing.assert_almost_equal(q[:, -1], np.array([2, 0, 1.57, 0.78539785]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([23.6216587, 12.2590703, 31.520697, 12.9472294]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-16.659525, 14.5872277, -36.1009998, 4.417834]))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)
