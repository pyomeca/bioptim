"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np
import biorbd

from biorbd_optim import (
    Data,
    ConstraintList,
    Constraint,
    QAndQDotBounds,
    DynamicsTypeList,
    DynamicsType,
    InitialConditionsList,
    BoundsList,
    Instant,
    ObjectiveList,
    Objective,
    OptimalControlProgram,
)
from .utils import TestUtils


def test_pendulum_min_time_mayer():
    # Load pendulum_min_time_Mayer
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum_min_time_Mayer", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum_min_time_Mayer.py"
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


def test_pendulum_min_time_lagrange():
    # Load pendulum_min_time_Lagrange
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum_min_time_Lagrange", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/pendulum_min_time_Lagrange.py"
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


def test_time_constraint():
    # Load time_constraint
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "time_constraint", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/time_constraint.py"
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


def test_monophase_time_constraint():
    # Load time_constraint
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "monophase_time_constraint", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/multiphase_time_constraint.py"
    )
    monophase_time_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monophase_time_constraint)

    ocp = monophase_time_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=[1, 3, 0.1],
        time_max=[2, 4, 0.8],
        number_shooting_points=(20,),
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


def test_multiphase_time_constraint():
    # Load time_constraint
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "multiphase_time_constraint", str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/multiphase_time_constraint.py"
    )
    multiphase_time_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multiphase_time_constraint)

    ocp = multiphase_time_constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/cube.bioMod",
        final_time=(2, 5, 4),
        time_min=[1, 3, 0.1],
        time_max=[2, 4, 0.8],
        number_shooting_points=(20, 30, 20),
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


def partial_ocp_parameters(nb_phases):
    if nb_phases != 1 and nb_phases != 3:
        raise RuntimeError("nb_phases should be 1 or 3")

    biorbd_model_path = str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/cube.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path)
    number_shooting_points = (2, 2, 2)
    final_time = (2, 5, 4)
    time_min = [1, 3, 0.1]
    time_max = [2, 4, 0.8]
    tau_min, tau_max, tau_init = -100, 100, 0
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)
    if nb_phases > 1:
        dynamics.add(DynamicsType.TORQUE_DRIVEN)
        dynamics.add(DynamicsType.TORQUE_DRIVEN)

    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model[0]))
    if nb_phases > 1:
        x_bounds.add(QAndQDotBounds(biorbd_model[0]))
        x_bounds.add(QAndQDotBounds(biorbd_model[0]))
    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds.min[i, [0, -1]] = 0
            bounds.max[i, [0, -1]] = 0
    x_bounds[0].min[2, 0] = 0.0
    x_bounds[0].max[2, 0] = 0.0
    if nb_phases > 1:
        x_bounds[2].min[2, [0, -1]] = [0.0, 1.57]
        x_bounds[2].max[2, [0, -1]] = [0.0, 1.57]

    x_init = InitialConditionsList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    if nb_phases > 1:
        x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
        x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    u_bounds = BoundsList()
    u_bounds.add([[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()])
    if nb_phases > 1:
        u_bounds.add(
            [[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()]
        )
        u_bounds.add(
            [[tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque()]
        )

    u_init = InitialConditionsList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    if nb_phases > 1:
        u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
        u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    return (
        biorbd_model[:nb_phases],
        number_shooting_points[:nb_phases],
        final_time[:nb_phases],
        time_min[:nb_phases],
        time_max[:nb_phases],
        tau_min,
        tau_max,
        tau_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    )


PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "test_mayer_neg_monophase_time_constraint",
    str(PROJECT_FOLDER) + "/examples/optimal_time_ocp/multiphase_time_constraint.py",
)
test_mayer_neg_monophase_time_constraint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_mayer_neg_monophase_time_constraint)


def test_mayer_neg_monophase_time_constraint():
    (
        biorbd_model,
        number_shooting_points,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(nb_phases=1)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_TIME)

    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0])

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            biorbd_model,
            dynamics,
            number_shooting_points,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


def test_mayer1_neg_multiphase_time_constraint():
    (
        biorbd_model,
        number_shooting_points,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(nb_phases=3)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=0)
    objective_functions.add(Objective.Mayer.MINIMIZE_TIME, phase=0)

    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0])

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            biorbd_model,
            dynamics,
            number_shooting_points,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


def test_mayer2_neg_multiphase_time_constraint():
    (
        biorbd_model,
        number_shooting_points,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(nb_phases=3)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=2)
    objective_functions.add(Objective.Mayer.MINIMIZE_TIME, phase=2)

    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1, phase=2)
    constraints.add(Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0], phase=2)

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            biorbd_model,
            dynamics,
            number_shooting_points,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


def test_mayer_multiphase_time_constraint():
    (
        biorbd_model,
        number_shooting_points,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(nb_phases=3)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=0)
    objective_functions.add(Objective.Mayer.MINIMIZE_TIME, phase=0)

    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1, phase=2)
    constraints.add(Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0], phase=2)

    OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
    )


def test_lagrange_neg_monophase_time_constraint():
    (
        biorbd_model,
        number_shooting_points,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(nb_phases=1)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TIME)

    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0])

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            biorbd_model,
            dynamics,
            number_shooting_points,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


def test_lagrange1_neg_multiphase_time_constraint():
    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        (
            biorbd_model,
            number_shooting_points,
            final_time,
            time_min,
            time_max,
            torque_min,
            torque_max,
            torque_init,
            dynamics,
            x_bounds,
            x_init,
            u_bounds,
            u_init,
        ) = partial_ocp_parameters(nb_phases=3)

        objective_functions = ObjectiveList()
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=0)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TIME, phase=0)

        constraints = ConstraintList()
        constraints.add(
            Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1, phase=0
        )
        constraints.add(
            Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0], phase=0
        )

        OptimalControlProgram(
            biorbd_model,
            dynamics,
            number_shooting_points,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


def test_lagrange2_neg_multiphase_time_constraint():
    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        (
            biorbd_model,
            number_shooting_points,
            final_time,
            time_min,
            time_max,
            torque_min,
            torque_max,
            torque_init,
            dynamics,
            x_bounds,
            x_init,
            u_bounds,
            u_init,
        ) = partial_ocp_parameters(nb_phases=3)

        objective_functions = ObjectiveList()
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=2)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TIME, phase=2)

        constraints = ConstraintList()
        constraints.add(
            Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1, phase=2
        )
        constraints.add(
            Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0], phase=2
        )

        OptimalControlProgram(
            biorbd_model,
            dynamics,
            number_shooting_points,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
            constraints,
        )


def test_lagrange_multiphase_time_constraint():
    (
        biorbd_model,
        number_shooting_points,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(nb_phases=3)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100, phase=0)
    objective_functions.add(Objective.Lagrange.MINIMIZE_TIME, phase=0)

    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=0, second_marker_idx=1, phase=2)
    constraints.add(Constraint.TIME_CONSTRAINT, instant=Instant.END, minimum=time_min[0], maximum=time_max[0], phase=2)

    OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
    )


def test_mayer_neg_two_objectives():
    (
        biorbd_model,
        number_shooting_points,
        final_time,
        time_min,
        time_max,
        torque_min,
        torque_max,
        torque_init,
        dynamics,
        x_bounds,
        x_init,
        u_bounds,
        u_init,
    ) = partial_ocp_parameters(nb_phases=1)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_TIME, phase=0)
    objective_functions.add(Objective.Mayer.MINIMIZE_TIME, phase=0)

    with pytest.raises(RuntimeError, match="Time constraint/objective cannot declare more than once"):
        OptimalControlProgram(
            biorbd_model,
            dynamics,
            number_shooting_points,
            final_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            objective_functions,
        )
