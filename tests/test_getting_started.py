"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np
from casadi import MX
import biorbd

from biorbd_optim import (
    Data,
    OdeSolver,
    OptimalControlProgram,
    Bounds,
    InitialConditions,
    BidirectionalMapping,
    Mapping,
    InterpolationType,
)

# Load pendulum
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location("pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.py")
pendulum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pendulum)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_pendulum(ode_solver):
    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
        show_online_optim=False,
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
    np.testing.assert_almost_equal(tau[:, 0], np.array((9.55000247, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-21.18945819, 0)))


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_save_and_load(ode_solver):
    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
        show_online_optim=False,
    )
    sol = ocp.solve()

    # save and load
    ocp.save(sol, "pendulum_ocp_sol")
    ocp_load, sol_load = OptimalControlProgram.load(name="pendulum_ocp_sol.bo")
    for key in sol.keys():
        np.testing.assert_almost_equal(np.array(sol[key]), np.array(sol_load[key]))
    sol_from_load = ocp_load.solve()
    for key in sol.keys():
        np.testing.assert_almost_equal(np.array(sol[key]), np.array(sol_from_load[key]))

    def deep_assert(dict_loaded, dict_original):
        if isinstance(dict_loaded, dict):
            for key in dict_loaded:
                deep_assert(dict_loaded[key], dict_original[key])
        elif isinstance(dict_loaded, (list, tuple)):
            for i in range(len(dict_loaded)):
                deep_assert(dict_loaded[i], dict_original[i])
        elif isinstance(
            dict_loaded, (OptimalControlProgram, Bounds, InitialConditions, BidirectionalMapping, Mapping, OdeSolver)
        ):
            for key in dir(dict_loaded):
                deep_assert(getattr(dict_loaded, key), getattr(dict_original, key))
        else:
            if not callable(dict_loaded) and not isinstance(dict_loaded, (MX, biorbd.Model)):
                try:
                    elem_loaded = np.asarray(dict_loaded, dtype=float)
                    elem_original = np.array(dict_original, dtype=float)
                    np.testing.assert_almost_equal(elem_original, elem_loaded)
                except ValueError:
                    pass

    deep_assert(ocp_load, ocp)
    deep_assert(ocp, ocp_load)


# Load custom_constraint
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "custom_constraint", str(PROJECT_FOLDER) + "/examples/getting_started/custom_constraint.py"
)
custom_constraint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_constraint)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_custom_constraint_align_markers(ode_solver):
    ocp = custom_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 1317.835541713015)

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


# Load initial_guess
PROJECT_FOLDER2 = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "initial_guess", str(PROJECT_FOLDER2) + "/examples/getting_started/simple_ocp.py"
)
initial_guess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initial_guess)


@pytest.mark.parametrize("interpolation_type", InterpolationType)
def test_initial_guesses(interpolation_type):
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
    np.testing.assert_almost_equal(f[0, 0], 2790.947)

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
