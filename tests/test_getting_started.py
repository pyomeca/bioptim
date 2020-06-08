"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import (
    Data,
    OdeSolver,
    InterpolationType,
)
from .utils import TestUtils


@pytest.mark.parametrize("nb_threads", [1, 2])
def test_pendulum(nb_threads):
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.py"
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
        nb_threads=nb_threads,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.0)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((17.4928172, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-24.2842703, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_custom_constraint_align_markers(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "custom_constraint", str(PROJECT_FOLDER) + "/examples/getting_started/custom_constraint.py"
    )
    custom_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_constraint)

    ocp = custom_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19767.533125695223)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))


@pytest.mark.parametrize("interpolation_type", InterpolationType)
def test_initial_guesses(interpolation_type):
    #  Load initial_guess
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "initial_guess", str(PROJECT_FOLDER) + "/examples/getting_started/simple_ocp.py"
    )
    initial_guess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(initial_guess)

    np.random.seed(42)
    ocp = initial_guess.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
        final_time=1,
        number_shooting_points=5,
        initial_guess=interpolation_type,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 13954.735)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (36, 1))
    np.testing.assert_almost_equal(g, np.zeros((36, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1, 0, 0]))
    np.testing.assert_almost_equal(q[:, -1], np.array([2, 0, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([5.0, 9.81, 7.85]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-5.0, 9.81, -7.85]))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


def test_cyclic_objective():
    #  Load initial_guess
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "initial_guess", str(PROJECT_FOLDER) + "/examples/getting_started/cyclic_movement.py"
    )
    cyclic_movement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cyclic_movement)

    np.random.seed(42)
    ocp = cyclic_movement.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
        final_time=1,
        number_shooting_points=10,
        loop_from_constraint=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 56851.881815451816)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (67, 1))
    np.testing.assert_almost_equal(g, np.zeros((67, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1.60205103, -0.01069317, 0.62477988]))
    np.testing.assert_almost_equal(q[:, -1], np.array([1, 0, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0.12902365, 0.09340155, -0.20256713)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([9.89210954, 9.39362112, -15.53061197]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([17.16370432, 9.78643138, -26.94701577]))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


def test_cyclic_constraint():
    #  Load initial_guess
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "initial_guess", str(PROJECT_FOLDER) + "/examples/getting_started/cyclic_movement.py"
    )
    cyclic_movement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cyclic_movement)

    np.random.seed(42)
    ocp = cyclic_movement.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
        final_time=1,
        number_shooting_points=10,
        loop_from_constraint=True,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 78921.61000000016)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (73, 1))
    np.testing.assert_almost_equal(g, np.zeros((73, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([1, 0, 1.57]))
    np.testing.assert_almost_equal(q[:, -1], np.array([1, 0, 1.57]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([20.0, 9.81, -31.4]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([20.0, 9.81, -31.4]))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


def test_phase_transitions():
    # Load phase_transitions
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "phase_transitions", str(PROJECT_FOLDER) + "/examples/getting_started/phase_transitions.py"
    )
    phase_transitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase_transitions)

    ocp = phase_transitions.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 110875.0772043361)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (515, 1))
    np.testing.assert_almost_equal(g, np.zeros((515, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"], concatenate=False)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[0][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[-1][:, -1], np.array((1, 0, 0)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[0][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[-1][:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[0][:, 0], np.array((0.9598672, 9.7085598, -0.0623733)))
    np.testing.assert_almost_equal(tau[-1][:, -1], np.array((0, 1.2717052e01, 1.1487805e00)))

    # cyclic continuity (between phase 3 and phase 0)
    np.testing.assert_almost_equal(q[-1][:, -1], q[0][:, 0])

    # Continuity between phase 0 and phase 1
    np.testing.assert_almost_equal(q[0][:, -1], q[1][:, 0])

    # save and load
    # For some reason, the custom function can't be found from here...
    # The save and load test is therefore skipped
    # TestUtils.save_and_load(sol, ocp, False)


def test_parameter_optimization():
    # Load phase_transitions
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "parameter_optimization", str(PROJECT_FOLDER) + "/examples/getting_started/parameter_optimization.py"
    )
    parameter_optimization = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parameter_optimization)

    ocp = parameter_optimization.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=3,
        number_shooting_points=20,
        min_g=-10,
        max_g=-6,
        target_g=-8,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 853.5406088867209)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (80, 1))
    np.testing.assert_almost_equal(g, np.zeros((80, 1)))

    # Check some of the results
    states, controls, params = Data.get_data(ocp, sol["x"], concatenate=False, get_parameters=True)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]
    gravity = params["gravity_z"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((8.1318336, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-7.91806351, 0)))

    # gravity parameter
    np.testing.assert_almost_equal(gravity, np.array([[-9.09889371]]))

    # save and load
    # TODO: Have a look a this
    # For some reason, the custom function can't be found from here...
    # The save and load test is therefore skipped
    # TestUtils.save_and_load(sol, ocp, True)
