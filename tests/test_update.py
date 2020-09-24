from pathlib import Path
import pytest

from casadi import DM
from casadi import vertcat
import numpy as np
import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsOption,
    InitialConditionsOption,
    ObjectiveOption,
    Objective,
    Axe,
    Constraint,
    ConstraintOption,
)


def prepare_test_ocp():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    biorbd_model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, 10, 1.0)
    ocp.nlp[0].J = [list()]
    ocp.nlp[0].g = [list()]
    ocp.nlp[0].g_bounds = [list()]
    return ocp


def test_penalty_minimize_time():
    ocp = prepare_test_ocp()
    X_init = InitialConditionsOption(np.zeros((4 * 2, 1)))
    U_init = InitialConditionsOption(np.zeros((4, 1)))
    X_bounds = BoundsOption([np.zeros((8, 1)), np.zeros((4 * 2, 1))])
    U_bounds = BoundsOption([np.zeros((4, 1)), np.zeros((4, 1))])
    ocp.update_bounds(X_bounds, U_bounds)
    ocp.update_initial_guess(X_init, U_init)
    x = [DM.ones((12, 1)) * -10]
    penalty_type = Objective.Lagrange.MINIMIZE_TIME
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(1),
    )
