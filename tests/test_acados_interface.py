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


def test_acados_no_obj():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
    )

    sol = ocp.solve(solver=Solver.ACADOS)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_one_mayer():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "cube",
        str(PROJECT_FOLDER) + "/examples/acados/cube.py",
    )
    cube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cube)

    ocp = cube.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
    )
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Mayer.MINIMIZE_STATE, weight=1000, states_idx=[0], target=np.array([[1.]]).T)
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS)

    # Check end state value
    model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod")
    q = np.array(sol["qqdot"])[:model.nbQ()]
    np.testing.assert_almost_equal(q[0, -1], 1.)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_several_mayer():
        PROJECT_FOLDER = Path(__file__).parent / ".."
        spec = importlib.util.spec_from_file_location(
            "cube",
            str(PROJECT_FOLDER) + "/examples/acados/cube.py",
        )
        cube = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cube)

        ocp = cube.prepare_ocp(
            biorbd_model_path=str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod",
        )
        objective_functions = ObjectiveList()
        objective_functions.add(Objective.Mayer.MINIMIZE_STATE, weight=1000, states_idx=[0, 1],
                                target=np.array([[1., 2.]]).T)
        objective_functions.add(Objective.Mayer.MINIMIZE_STATE, weight=10000, states_idx=[2], target=np.array([[3.]]))
        ocp.update_objectives(objective_functions)

        sol = ocp.solve(solver=Solver.ACADOS)

        # Check end state value
        model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/acados/cube.bioMod")
        q = np.array(sol["qqdot"])[:model.nbQ()]
        np.testing.assert_almost_equal(q[0, -1], 1.)
        np.testing.assert_almost_equal(q[1, -1], 2.)
        np.testing.assert_almost_equal(q[2, -1], 3.)

        # Clean test folder
        os.remove(f"./acados_ocp.json")
        shutil.rmtree(f"./c_generated_code/")