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
    Bounds,
    InitialConditions,
)
from .utils import TestUtils

# import sys
# PROJECT_FOLDER = Path(__file__).parent / ".."
# sys.path.insert(1, str(PROJECT_FOLDER) + "/examples")
# from simple_ocp import custom_bound_func


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
    np.testing.assert_almost_equal(f[0, 0], 6657.974502951726)

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
    np.testing.assert_almost_equal(tau[:, 0], np.array((16.25734477, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-25.59944635, 0)))

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
    # TODO: Have a look a this
    # For some reason, the custom function can't be found from here...
    # The save and load test is therefore skipped
    # TestUtils.save_and_load(sol, ocp, True)


def test_bounds_interpolation():

    # SPLINE
    spline_time = np.hstack((0.0, 1.0, 2.2, 6.0))
    x_init = np.array(
        [
            [0.5, 0.6, 0.2, 0.8],
            [0.4, 0.6, 0.8, 0.2],
            [0.0, 0.3, 0.2, 0.5],
            [0.5, 0.2, 0.9, 0.4],
            [0.5, 0.6, 0.2, 0.8],
            [0.5, 0.6, 0.2, 0.8],
        ]
    )
    x_min = np.array(
        [
            [-4.0, -6.0, -8.0, -2.0],
            [-5.0, -6.0, -2.0, -8.0],
            [-0.0, -3.0, -2.0, -5.0],
            [-5.0, -6.0, -2.0, -8.0],
            [-5.0, -2.0, -9.0, -4.0],
            [-5.0, -6.0, -2.0, -8.0],
        ]
    )
    x_max = np.array(
        [
            [5.0, 6.0, 2.0, 8.0],
            [4.0, 6.0, 8.0, 2.0],
            [0.0, 3.0, 2.0, 5.0],
            [5.0, 2.0, 9.0, 4.0],
            [5.0, 6.0, 2.0, 8.0],
            [5.0, 6.0, 2.0, 8.0],
        ]
    )
    X_bounds = Bounds(x_min, x_max, interpolation_type=InterpolationType.SPLINE)
    X_init = InitialConditions(x_init, interpolation_type=InterpolationType.SPLINE)

    nX = 6
    nU = 3
    ns = 5

    nV = nX * (ns + 1) + nU * ns
    V_bounds = Bounds([0] * nV, [0] * nV, interpolation_type=InterpolationType.CONSTANT)
    V_init = InitialConditions([0] * nV, interpolation_type=InterpolationType.CONSTANT)
    X_bounds.min.nb_shooting = ns
    X_bounds.max.nb_shooting = ns
    X_init.init.nb_shooting = ns

    offset = 0
    for k in range(ns + 1):
        V_bounds.min[offset : offset + nX, 0] = X_bounds.min.evaluate_at(
            shooting_point=k, spline_time=spline_time, t0=0.0, tf=6.0,
        )
        V_bounds.max[offset : offset + nX, 0] = X_bounds.max.evaluate_at(
            shooting_point=k, spline_time=spline_time, t0=0.0, tf=6.0,
        )
        V_init.init[offset : offset + nX, 0] = X_init.init.evaluate_at(
            shooting_point=k, spline_time=spline_time, t0=0.0, tf=6.0,
        )
        offset += nX

        V_bounds.check_and_adjust_dimensions(nV, 1)
        V_init.check_and_adjust_dimensions(nV, 1)

    np.testing.assert_almost_equal(
        V_bounds.min,
        np.array(
            [
                [-4.0],
                [-5.0],
                [-0.0],
                [-5.0],
                [-5.0],
                [-5.0],
                [-6.33333333],
                [-5.33333333],
                [-2.83333333],
                [-5.33333333],
                [-3.16666667],
                [-5.33333333],
                [-7.68421053],
                [-2.31578947],
                [-2.15789474],
                [-2.31578947],
                [-8.73684211],
                [-2.31578947],
                [-5.78947368],
                [-4.21052632],
                [-3.10526316],
                [-4.21052632],
                [-7.15789474],
                [-4.21052632],
                [-3.89473684],
                [-6.10526316],
                [-4.05263158],
                [-6.10526316],
                [-5.57894737],
                [-6.10526316],
                [-2.0],
                [-8.0],
                [-5.0],
                [-8.0],
                [-4.0],
                [-8.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ]
        ),
    )

    np.testing.assert_almost_equal(
        V_bounds.max,
        np.array(
            [
                [5.0],
                [4.0],
                [0.0],
                [5.0],
                [5.0],
                [5.0],
                [5.33333333],
                [6.33333333],
                [2.83333333],
                [3.16666667],
                [5.33333333],
                [5.33333333],
                [2.31578947],
                [7.68421053],
                [2.15789474],
                [8.73684211],
                [2.31578947],
                [2.31578947],
                [4.21052632],
                [5.78947368],
                [3.10526316],
                [7.15789474],
                [4.21052632],
                [4.21052632],
                [6.10526316],
                [3.89473684],
                [4.05263158],
                [5.57894737],
                [6.10526316],
                [6.10526316],
                [8.0],
                [2.0],
                [5.0],
                [4.0],
                [8.0],
                [8.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ]
        ),
    )

    np.testing.assert_almost_equal(
        V_init.init,
        np.array(
            [
                [0.5],
                [0.4],
                [0.0],
                [0.5],
                [0.5],
                [0.5],
                [0.53333333],
                [0.63333333],
                [0.28333333],
                [0.31666667],
                [0.53333333],
                [0.53333333],
                [0.23157895],
                [0.76842105],
                [0.21578947],
                [0.87368421],
                [0.23157895],
                [0.23157895],
                [0.42105263],
                [0.57894737],
                [0.31052632],
                [0.71578947],
                [0.42105263],
                [0.42105263],
                [0.61052632],
                [0.38947368],
                [0.40526316],
                [0.55789474],
                [0.61052632],
                [0.61052632],
                [0.8],
                [0.2],
                [0.5],
                [0.4],
                [0.8],
                [0.8],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ]
        ),
    )

    # CUSTOM
    def custom_bound_func(x, current_shooting_point, number_shooting_points):
        # Linear interpolation created with custom bound function
        return x[:, 0] + (x[:, 1] - x[:, 0]) * current_shooting_point / number_shooting_points

    x_init = np.array([[0.5, 0.8], [0.4, 0.2], [0.0, 0.5], [0.5, 0.4], [0.5, 0.8], [0.5, 0.8]])
    x_min = np.array([[-4.0, -2.0], [-5.0, -8.0], [-0.0, -5.0], [-5.0, -8.0], [-5.0, -4.0], [-5.0, -8.0]])
    x_max = np.array([[5.0, 8.0], [4.0, 2.0], [0.0, 5.0], [5.0, 4.0], [5.0, 8.0], [5.0, 8.0]])
    X_bounds = Bounds(x_min, x_max, interpolation_type=InterpolationType.CUSTOM)
    X_init = InitialConditions(x_init, interpolation_type=InterpolationType.CUSTOM)

    nX = 6
    nU = 3
    ns = 5

    nV = nX * (ns + 1) + nU * ns
    V_bounds = Bounds([0] * nV, [0] * nV, interpolation_type=InterpolationType.CONSTANT)
    V_init = InitialConditions([0] * nV, interpolation_type=InterpolationType.CONSTANT)
    X_bounds.min.nb_shooting = ns
    X_bounds.max.nb_shooting = ns
    X_init.init.nb_shooting = ns

    offset = 0
    for k in range(ns + 1):
        V_bounds.min[offset : offset + nX, 0] = X_bounds.min.evaluate_at(
            shooting_point=k, t0=0.0, tf=6.0, custom_bound_function=custom_bound_func,
        )
        V_bounds.max[offset : offset + nX, 0] = X_bounds.max.evaluate_at(
            shooting_point=k, t0=0.0, tf=6.0, custom_bound_function=custom_bound_func,
        )
        V_init.init[offset : offset + nX, 0] = X_init.init.evaluate_at(
            shooting_point=k, t0=0.0, tf=6.0, custom_bound_function=custom_bound_func,
        )
        offset += nX

    V_bounds.check_and_adjust_dimensions(nV, 1)
    V_init.check_and_adjust_dimensions(nV, 1)

    np.testing.assert_almost_equal(
        V_bounds.min,
        np.array(
            [
                [-4.0],
                [-5.0],
                [-0.0],
                [-5.0],
                [-5.0],
                [-5.0],
                [-3.6],
                [-5.6],
                [-1.0],
                [-5.6],
                [-4.8],
                [-5.6],
                [-3.2],
                [-6.2],
                [-2.0],
                [-6.2],
                [-4.6],
                [-6.2],
                [-2.8],
                [-6.8],
                [-3.0],
                [-6.8],
                [-4.4],
                [-6.8],
                [-2.4],
                [-7.4],
                [-4.0],
                [-7.4],
                [-4.2],
                [-7.4],
                [-2.0],
                [-8.0],
                [-5.0],
                [-8.0],
                [-4.0],
                [-8.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ]
        ),
    )

    np.testing.assert_almost_equal(
        V_bounds.max,
        np.array(
            [
                [5.0],
                [4.0],
                [0.0],
                [5.0],
                [5.0],
                [5.0],
                [5.6],
                [3.6],
                [1.0],
                [4.8],
                [5.6],
                [5.6],
                [6.2],
                [3.2],
                [2.0],
                [4.6],
                [6.2],
                [6.2],
                [6.8],
                [2.8],
                [3.0],
                [4.4],
                [6.8],
                [6.8],
                [7.4],
                [2.4],
                [4.0],
                [4.2],
                [7.4],
                [7.4],
                [8.0],
                [2.0],
                [5.0],
                [4.0],
                [8.0],
                [8.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ]
        ),
    )

    np.testing.assert_almost_equal(
        V_init.init,
        np.array(
            [
                [0.5],
                [0.4],
                [0.0],
                [0.5],
                [0.5],
                [0.5],
                [0.56],
                [0.36],
                [0.1],
                [0.48],
                [0.56],
                [0.56],
                [0.62],
                [0.32],
                [0.2],
                [0.46],
                [0.62],
                [0.62],
                [0.68],
                [0.28],
                [0.3],
                [0.44],
                [0.68],
                [0.68],
                [0.74],
                [0.24],
                [0.4],
                [0.42],
                [0.74],
                [0.74],
                [0.8],
                [0.2],
                [0.5],
                [0.4],
                [0.8],
                [0.8],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ]
        ),
    )


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


def test_state_transitions():
    # Load state_transitions
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "state_transitions", str(PROJECT_FOLDER) + "/examples/getting_started/state_transitions.py"
    )
    state_transitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state_transitions)

    ocp = state_transitions.prepare_ocp(
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
    np.testing.assert_almost_equal(f[0, 0], 853.5406085230834, decimal=6)

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


@pytest.mark.parametrize("problem_type_custom", [True, False])
def test_custom_problem_type_and_dynamics(problem_type_custom):
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "custom_problem_type_and_dynamics",
        str(PROJECT_FOLDER) + "/examples/getting_started/custom_problem_type_and_dynamics.py",
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
        problem_type_custom=problem_type_custom,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19767.5331257)

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
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516129, 9.81, 2.27903226)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.45161291, 9.81, -2.27903226)))
