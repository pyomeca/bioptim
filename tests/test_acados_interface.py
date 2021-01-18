"""
Test for file IO.
It tests the results of an optimal control problem with acados regarding the proper functioning of :
- the handling of mayer and lagrange obj
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np
import os
import shutil

import biorbd
from bioptim import (
    Axe,
    Data,
    Solver,
    ObjectiveList,
    ObjectiveFcn,
    Bounds,
    QAndQDotBounds,
    OdeSolver,
    ConstraintList,
    ConstraintFcn,
    Node,
)


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_no_obj(cost_type):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        nbs=10,
        tf=2,
    )

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_mayer(cost_type):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        nbs=10,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0], target=np.array([[1.0]]).T)
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod")
    q = np.array(sol["qqdot"])[: model.nbQ()]
    np.testing.assert_almost_equal(q[0, -1], 1.0)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_several_mayer(cost_type):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        nbs=10,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0, 1], target=np.array([[1.0, 2.0]]).T)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[2], target=np.array([[3.0]]))
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod")
    q = np.array(sol["qqdot"])[: model.nbQ()]
    np.testing.assert_almost_equal(q[0, -1], 1.0)
    np.testing.assert_almost_equal(q[1, -1], 2.0)
    np.testing.assert_almost_equal(q[2, -1], 3.0)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_lagrange(cost_type):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    nbs = 10
    target = np.expand_dims(np.arange(0, nbs + 1), axis=0)
    target[0, -1] = nbs - 2
    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        nbs=nbs,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=10, index=[0], target=target)
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod")
    q = np.array(sol["qqdot"])[: model.nbQ()]
    np.testing.assert_almost_equal(q[0, :], target[0, :].squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_lagrange_and_one_mayer(cost_type):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    nbs = 10
    target = np.expand_dims(np.arange(0, nbs + 1), axis=0)
    target[0, -1] = nbs - 2
    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        nbs=nbs,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=10, index=[0], target=target)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0], target=target[:, -1:])
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod")
    q = np.array(sol["qqdot"])[: model.nbQ()]
    np.testing.assert_almost_equal(q[0, :], target[0, :].squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_control_lagrange_and_state_mayer(cost_type):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    nbs = 10
    target = np.array([[2]])
    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        nbs=nbs,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_ALL_CONTROLS,
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0], target=target)
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod")
    q = np.array(sol["qqdot"])[: model.nbQ()]
    np.testing.assert_almost_equal(q[0, -1], target.squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_mhe(cost_type):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    nbs = 5
    nbsample = 20
    target = np.expand_dims(np.cos(np.arange(0, nbsample + 1)), axis=0)

    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        nbs=nbs,
        tf=2,
    )

    model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod")
    for i in range(nbsample - nbs):
        objective_functions = ObjectiveList()
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_STATE, weight=10, index=[0], target=target[:, i : i + nbs + 1]
        )
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0], target=target[:, i + nbs : i + nbs + 1])
        ocp.update_objectives(objective_functions)
        sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

        # Check end state value
        q = np.array(sol["qqdot"])[: model.nbQ()]
        np.testing.assert_almost_equal(q[0, :], target[0, i : i + nbs + 1].squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_options(cost_type):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum",
        str(PROJECT_FOLDER) + "/examples/acados/pendulum.py",
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/pendulum.bioMod",
        final_time=3,
        number_shooting_points=12,
    )

    tol = [1e-1, 1e-0, 1e1]
    iter = []
    for i in range(3):
        solver_options = {"nlp_solver_tol_stat": tol[i], "cost_type": cost_type}
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=solver_options)
        iter += [sol["iter"]]

    # Check that tol impacted convergence
    np.testing.assert_array_less(iter[1], iter[0])
    np.testing.assert_array_less(iter[2], iter[1])

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_fail_external():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "pendulum",
        str(PROJECT_FOLDER) + "/examples/acados/pendulum.py",
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/pendulum.bioMod",
        final_time=1,
        number_shooting_points=2,
    )

    solver_options = {"cost_type": "EXTERNAL"}

    with pytest.raises(RuntimeError, match="EXTERNAL is not interfaced yet, please use NONLINEAR_LS"):
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=solver_options)


def test_acados_fail_lls():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "arm",
        str(PROJECT_FOLDER) + "/examples/acados/static_arm.py",
    )
    arm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arm)

    ocp = arm.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/arm26.bioMod",
        final_time=1,
        number_shooting_points=2,
        use_SX=True,
    )

    solver_options = {"cost_type": "LINEAR_LS"}

    with pytest.raises(RuntimeError, match="ALIGN_MARKERS is an incompatible objective term with LINEAR_LS cost type"):
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=solver_options)


@pytest.mark.parametrize("problem_type_custom", [True, False])
def test_acados_custom_dynamics(problem_type_custom):
    # Load pendulum
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "custom_problem_type_and_dynamics",
        str(PROJECT_FOLDER) + "/examples/getting_started/custom_dynamics.py",
    )
    pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pendulum)

    ocp = pendulum.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod",
        problem_type_custom=problem_type_custom,
        ode_solver=OdeSolver.RK4,
        use_SX=True,
    )
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)
    ocp.update_constraints(constraints)
    sol = ocp.solve(solver=Solver.ACADOS)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((2, 0, 0)), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.81, 2.27903226)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 9.81, -2.27903226)))


def test_acados_one_parameter():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "parameters",
        str(PROJECT_FOLDER) + "/examples/getting_started/custom_parameters.py",
    )
    parameters = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parameters)

    ocp = parameters.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        number_shooting_points=100,
        min_g=-10,
        max_g=-6,
        target_g=-8,
        use_SX=True,
    )
    model = ocp.nlp[0].model
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, index=[0, 1], target=np.array([[0, 3.14]]).T, weight=100000)
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, index=[2, 3], target=np.array([[0, 0]]).T, weight=100)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, index=1, weight=10)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, index=[2, 3], weight=0.000000010)
    ocp.update_objectives(objectives)

    # Path constraint
    x_bounds = QAndQDotBounds(model)
    x_bounds[[0, 1, 2, 3], 0] = 0
    u_bounds = Bounds([-300] * model.nbQ(), [300] * model.nbQ())
    ocp.update_bounds(x_bounds, u_bounds)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"print_level": 0})

    # Check some of the results
    states, controls, params = Data.get_data(ocp, sol["x"], concatenate=False, get_parameters=True)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]
    gravity = params["gravity_z"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)), decimal=6)

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((189.674313, 0)), decimal=3)
    np.testing.assert_almost_equal(tau[:, -1], np.array((-260.150570, 0)), decimal=3)

    # gravity parameter
    np.testing.assert_almost_equal(gravity, np.array([[-8]]), decimal=6)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_one_end_constraints():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "constraint",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constraint)

    ocp = constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        nbs=10,
        tf=2,
    )

    model = ocp.nlp[0].model
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, index=0, weight=100, target=np.array([[1]]))
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=100)
    ocp.update_objectives(objective_functions)

    # Path constraint
    x_bounds = QAndQDotBounds(model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[0, 0] = 0
    ocp.update_bounds(x_bounds=x_bounds)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"print_level": 0})

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 0)), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((2.72727272, 9.81, 0)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -1], np.array((-2.72727272, 9.81, 0)), decimal=6)


def test_acados_constraints_all():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "constraint",
        str(PROJECT_FOLDER) + "/examples/align/align_marker_on_segment.py",
    )
    constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constraint)

    ocp = constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod",
        number_shooting_points=30,
        final_time=2,
        initialize_near_solution=True,
        constr=False,
        use_SX=True,
    )

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.ALIGN_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL, marker_idx=1, segment_idx=2, axis=(Axe.X)
    )
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"print_level": 0})

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.8385190835, 0, 0, -0.212027938]), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((0.8385190835, 0, 1.57, -0.212027938)), decimal=6)

    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.81, 2.27903226, 0)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 9.81, -2.27903226, 0)), decimal=6)


def test_acados_constraints_end_all():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "constraint",
        str(PROJECT_FOLDER) + "/examples/align/align_marker_on_segment.py",
    )
    constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constraint)

    ocp = constraint.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod",
        number_shooting_points=30,
        final_time=2,
        initialize_near_solution=True,
        constr=False,
        use_SX=True,
    )

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.ALIGN_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=5)
    constraints.add(
        ConstraintFcn.ALIGN_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL, marker_idx=1, segment_idx=2, axis=(Axe.X)
    )
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"print_level": 0})

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, 0], np.array([2, 0, 0, -0.139146705]), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57, -0.139146705)), decimal=6)

    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.81, 2.27903226, 0)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 9.81, -2.27903226, 0)), decimal=6)
