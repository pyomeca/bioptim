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
from bioptim import Data, Solver, ObjectiveList, Objective

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
    objective_functions.add(Objective.Mayer.MINIMIZE_STATE, index=[0], target=np.array([[1.0]]).T)
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
    objective_functions.add(Objective.Mayer.MINIMIZE_STATE, index=[0, 1], target=np.array([[1.0, 2.0]]).T)
    objective_functions.add(Objective.Mayer.MINIMIZE_STATE, index=[2], target=np.array([[3.0]]))
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
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=10, index=[0], target=target)
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
    objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=10, index=[0], target=target)
    objective_functions.add(Objective.Mayer.MINIMIZE_STATE, index=[0], target=target[:, -1:])
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
        Objective.Lagrange.MINIMIZE_ALL_CONTROLS,
    )
    objective_functions.add(Objective.Mayer.MINIMIZE_STATE, index=[0], target=target)
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
        objective_functions.add(Objective.Lagrange.TRACK_STATE, weight=10, index=[0], target=target[:, i : i + nbs + 1])
        objective_functions.add(Objective.Mayer.MINIMIZE_STATE, index=[0], target=target[:, i + nbs : i + nbs + 1])
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

    with pytest.raises(RuntimeError, match="Incompatible objective term with LINEAR_LS cost type"):
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=solver_options)