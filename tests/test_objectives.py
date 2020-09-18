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
    Axe,
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


def test_objective_minimize_markers():
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


def test_objective_track_markers():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.TRACK_MARKERS
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


def test_objective_minimize_markers_displacement():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_MARKERS_DISPLACEMENT
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_minimize_markers_velocity():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_MARKERS_VELOCITY
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [0.1],
                [0],
                [0.1],
            ]
        ),
    )


def test_objective_track_markers_velocity():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.TRACK_MARKERS_VELOCITY
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [0.1],
                [0],
                [0.1],
            ]
        ),
    )


def test_objective_align_markers():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.ALIGN_MARKERS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], first_marker_idx=0, second_marker_idx=1)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [-0.8951707],
                [0],
                [1.0948376],
            ]
        ),
    )


def test_objective_proportional_state():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.PROPORTIONAL_STATE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], which_var="states", first_dof=0, second_dof=1, coef=2)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array([-0.1]),
    )


def test_objective_proportional_control():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.PROPORTIONAL_CONTROL
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], which_var="controls", first_dof=0, second_dof=1, coef=2)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_minimize_torque():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_TORQUE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_track_torque():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.TRACK_TORQUE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_minimize_torque_derivative():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_minimize_muscles_control():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_track_muscles_control():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.TRACK_MUSCLES_CONTROL
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_minimize_all_controls():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_ALL_CONTROLS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_track_all_controls():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.TRACK_ALL_CONTROLS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_minimize_contact_forces():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.MINIMIZE_CONTACT_FORCES
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_minimize_contact_forces():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.TRACK_CONTACT_FORCES
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


def test_objective_align_segment_with_custom_rt():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.ALIGN_SEGMENT_WITH_CUSTOM_RT
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], segment_idx=1, rt_idx=0)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array([
            [0],
            [0.1],
            [0]
            ]),
    )


def test_objective_align_marker_with_segment_axis():
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) / 10]
    penalty_type = Objective.Lagrange.ALIGN_MARKER_WITH_SEGMENT_AXIS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], marker_idx=0, segment_idx=1, axis=Axe.X)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array([[0]]),
    )
