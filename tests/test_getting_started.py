"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import Data, OdeSolver

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


# Load pendulum_min_time_Mayer
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "pendulum_min_time_Mayer", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum_min_time_Mayer.py",
)
pendulum_min_time_Mayer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pendulum_min_time_Mayer)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_pendulum_min_time_mayer(ode_solver):
    ocp = pendulum_min_time_Mayer.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
        show_online_optim=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.6209213032003106)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states, controls, param = Data.get_data(ocp, sol["x"], get_parameters=True)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]
    tf = param["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((59.95450138, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-99.99980141, 0)))

    # optimized time
    np.testing.assert_almost_equal(tf, 0.6209213032003106)


# Load pendulum_min_time_Lagrange
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "pendulum_min_time_Lagrange", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum_min_time_Lagrange.py",
)
pendulum_min_time_Lagrange = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pendulum_min_time_Lagrange)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_pendulum_min_time_lagrange(ode_solver):
    ocp = pendulum_min_time_Lagrange.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
        show_online_optim=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.003628839798820405)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=-6)

    # Check some of the results
    states, controls, param = Data.get_data(ocp, sol["x"], get_parameters=True)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]
    tf = param["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-20.36604829, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-99.98554494, 0)))

    # optimized time
    np.testing.assert_almost_equal(tf, 0.6023985224766413)
