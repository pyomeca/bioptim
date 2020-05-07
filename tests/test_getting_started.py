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
spec = importlib.util.spec_from_file_location(
    "pendulum", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.py"
)
pendulum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pendulum)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_pendulum(ode_solver):
    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2, number_shooting_points=50, show_online_optim=False
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 1.5970062033278107)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (200, 1))
    # TODO: remove or modify? np.testing.assert_almost_equal(g, np.zeros(200, 1))

    # Check some of the results
    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()


    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-5.9809981)))


# Load pendulum_min_time_Mayer
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "pendulum_min_time_Mayer", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum_min_time_Mayer.py"
)
pendulum_min_time_Mayer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pendulum_min_time_Mayer)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_pendulum_min_time_Mayer(ode_solver):
    ocp = pendulum_min_time_Mayer.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2, number_shooting_points=50, show_online_optim=False
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 2.18186597955319)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (200, 1))
    # TODO: remove or modify? np.testing.assert_almost_equal(g, np.zeros(200, 1))

    # Check some of the results
    states, controls, param = Data.get_data_from_V(ocp, sol["x"], get_parameters=True)
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()
    tf = param["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-0.5898199)))

    # optimized time
    np.testing.assert_almost_equal(tf, 1.5491434695819155)

# Load pendulum_min_time_Lagrange
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "pendulum_min_time_Lagrange", str(PROJECT_FOLDER) + "/examples/getting_started/pendulum_min_time_Lagrange.py"
)
pendulum_min_time_Lagrange = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pendulum_min_time_Lagrange)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_pendulum_min_time_Lagrange(ode_solver):
    ocp = pendulum_min_time_Lagrange.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2, number_shooting_points=50, show_online_optim=False
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.6082229673376528)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (200, 1))
    # TODO: remove or modify? np.testing.assert_almost_equal(g, np.zeros(200, 1))

    # Check some of the results
    states, controls, param = Data.get_data_from_V(ocp, sol["x"], get_parameters=True)
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()
    tf = param["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-3.7492435)))

    # optimized time
    np.testing.assert_almost_equal(tf, 1.6540746733051912)