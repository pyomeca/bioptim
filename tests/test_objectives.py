from pathlib import Path

import numpy as np
import pytest

import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsOption,
    InitialConditionsOption,
    ObjectiveOption,
    Objective,
)


def prepare_test_ocp():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    biorbd_model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)
    x_bounds = BoundsOption([np.zeros((nq * 2, 1)), np.zeros((nq * 2, 1))])
    x_init = InitialConditionsOption(np.zeros((nq * 2, 1)))
    u_bounds = BoundsOption([np.zeros((nq, 1)), np.zeros((nq, 1))])
    u_init = InitialConditionsOption(np.zeros((nq, 1)))
    ocp = OptimalControlProgram(biorbd_model, dynamics, 10, 1.0, x_init, u_init, x_bounds, u_bounds)
    ocp.nlp[0].J = [list()]
    return ocp


def test_objective_minimize_time():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_TIME
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(1),
    )


def test_objective_minimize_state():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_STATE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
            ]
        ),
    )


def test_objective_track_state():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.TRACK_STATE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
            ]
        ),
    )


def test_objective_minimize_marker():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_MARKERS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [0.1, 0.99517075, 1.9901749, 1.0950042, 1, 2, 0.49750208],
                [0, 0, 0, 0, 0, 0, 0],
                [0.1, -0.9948376, -1.094671, 0.000166583, 0, 0, -0.0499167],
            ]
        ),
    )
