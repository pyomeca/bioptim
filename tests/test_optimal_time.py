"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np
import biorbd

from biorbd_optim import Data, OdeSolver, Constraint, QAndQDotBounds, ProblemType, InitialConditions, Bounds, Instant, Objective, OptimalControlProgram
from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_pendulum_min_time_mayer(ode_solver):
    # Load pendulum_min_time_Mayer
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum_min_time_Mayer", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum_min_time_Mayer.py",
    )
    pendulum_min_time_Mayer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum_min_time_Mayer)

    ocp = pendulum_min_time_Mayer.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
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

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_pendulum_min_time_lagrange(ode_solver):
    # Load pendulum_min_time_Lagrange
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum_min_time_Lagrange", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum_min_time_Lagrange.py",
    )
    pendulum_min_time_Lagrange = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum_min_time_Lagrange)

    ocp = pendulum_min_time_Lagrange.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.062092703196434854)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

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
    np.testing.assert_almost_equal(tau[:, 0], np.array((59.9529745, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-99.9980341, 0)))

    # optimized time
    np.testing.assert_almost_equal(tf, 0.6209270319643485)

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_time_constraint(ode_solver):
    # Load time_constraint
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "time_constraint", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/time_constraint.py",
    )
    time_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(time_constraint)

    ocp = time_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum.bioMod",
        final_time=2,
        number_shooting_points=10,
        time_min=0.6,
        time_max=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 1451.2202233368012)

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
    np.testing.assert_almost_equal(tau[:, 0], np.array((22.49775, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-33.9047809, 0)))

    # optimized time
    np.testing.assert_almost_equal(tf, 1.0)

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_monophase_time_constraint(ode_solver):
    # Load time_constraint
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "monophase_time_constraint", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/multiphase_time_constraint.py",
    )
    monophase_time_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monophase_time_constraint)

    ocp = monophase_time_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=[1, 3, 0.1],
        time_max=[2, 4, 0.8],
        nb_phases=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 10826.61745874204)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (126, 1))
    np.testing.assert_almost_equal(g, np.zeros((126, 1)))

    # Check some of the results
    states, controls, param = Data.get_data(ocp, sol["x"], get_parameters=True)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]
    tf = param["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 0)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((5.71428583, 9.81, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-5.71428583, 9.81, 0)))

    # optimized time
    np.testing.assert_almost_equal(tf, 1.0)

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

# Load time_constraint
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "multiphase_time_constraint", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/multiphase_time_constraint.py",
)
multiphase_time_constraint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiphase_time_constraint)

@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_multiphase_time_constraint(ode_solver):

    ocp = multiphase_time_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=[1, 3, 0.1],
        time_max=[2, 4, 0.8],
        nb_phases=3,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 55582.04125059745)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (444, 1))
    np.testing.assert_almost_equal(g, np.zeros((444, 1)))

    # Check some of the results
    states, controls, param = Data.get_data(ocp, sol["x"], get_parameters=True)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]
    tf = param["time"][0, 0]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((5.71428583, 9.81, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-8.92857121, 9.81, -14.01785679)))

    # optimized time
    np.testing.assert_almost_equal(tf, 1.0)

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

def partial_ocp_parameters():
    biorbd_model_path = str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/cube.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)
    biorbd_model = biorbd_model, biorbd_model, biorbd_model
    number_shooting_points = (2, 2, 2)
    final_time = (2, 5, 4)
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    torque_min, torque_max, torque_init = -100, 100, 0
    problem_type = (ProblemType.torque_driven, ProblemType.torque_driven, ProblemType.torque_driven)
    X_bounds = [QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0]), QAndQDotBounds(biorbd_model[0])]
    for bounds in X_bounds:
        for i in [1, 3, 4, 5]:
            bounds.min[i, [0, -1]] = 0
            bounds.max[i, [0, -1]] = 0
    X_bounds[0].min[2, 0] = 0.0
    X_bounds[0].max[2, 0] = 0.0
    X_bounds[2].min[2, [0, -1]] = [0.0, 1.57]
    X_bounds[2].max[2, [0, -1]] = [0.0, 1.57]
    X_init = InitialConditions([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    X_init = (X_init, X_init, X_init)
    U_bounds = [
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
        Bounds(
            [torque_min] * biorbd_model[0].nbGeneralizedTorque(), [torque_max] * biorbd_model[0].nbGeneralizedTorque(),
        ),
    ]
    U_init = InitialConditions([torque_init] * biorbd_model[0].nbGeneralizedTorque())
    U_init = (U_init, U_init, U_init)

    return biorbd_model, number_shooting_points,final_time, time_min, time_max, torque_min, torque_max, torque_init, problem_type, X_bounds, X_init, U_bounds, U_init
