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


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_time(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_TIME
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(1),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_state(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_STATE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [value],
                [value],
                [value],
                [value],
                [value],
                [value],
                [value],
                [value],
            ]
        ),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_track_state(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.TRACK_STATE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [1], x, [], [], target=np.ones((8, 11)) * value)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [value],
                [value],
                [value],
                [value],
                [value],
                [value],
                [value],
                [value],
            ]
        ),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_markers(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_MARKERS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    res = np.array(
        [
            [0.1, 0.99517075, 1.9901749, 1.0950042, 1, 2, 0.49750208],
            [0, 0, 0, 0, 0, 0, 0],
            [0.1, -0.9948376, -1.094671, 0.000166583, 0, 0, -0.0499167],
        ]
    )
    if value == -10:
        res = np.array(
            [
                [-10, -11.3830926, -12.2221642, -10.8390715, 1.0, 2.0, -0.4195358],
                [0, 0, 0, 0, 0, 0, 0],
                [-10, -9.7049496, -10.2489707, -10.5440211, 0, 0, -0.2720106],
            ]
        )

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        res,
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_track_markers(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.TRACK_MARKERS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [2], x, [], [], target=np.ones((3, 7, 11)) * value)

    res = np.array(
        [
            [0.1, 0.99517075, 1.9901749, 1.0950042, 1, 2, 0.49750208],
            [0, 0, 0, 0, 0, 0, 0],
            [0.1, -0.9948376, -1.094671, 0.000166583, 0, 0, -0.0499167],
        ]
    )
    if value == -10:
        res = np.array(
            [
                [-10, -11.3830926, -12.2221642, -10.8390715, 1.0, 2.0, -0.4195358],
                [0, 0, 0, 0, 0, 0, 0],
                [-10, -9.7049496, -10.2489707, -10.5440211, 0, 0, -0.2720106],
            ]
        )

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        res,
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_markers_displacement(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_MARKERS_DISPLACEMENT
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_markers_velocity(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_MARKERS_VELOCITY
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [value],
                [0],
                [value],
            ]
        ),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_track_markers_velocity(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.TRACK_MARKERS_VELOCITY
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [3], x, [], [], target=np.ones((3, 7, 11)) * value)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(
            [
                [value],
                [0],
                [value],
            ]
        ),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_align_markers(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.ALIGN_MARKERS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], first_marker_idx=0, second_marker_idx=1)

    res = np.array(
        [
            [-0.8951707],
            [0],
            [1.0948376],
        ]
    )
    if value == -10:
        res = np.array(
            [
                [1.3830926],
                [0],
                [-0.2950504],
            ]
        )

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        res,
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_proportional_state(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.PROPORTIONAL_STATE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](
        penalty, ocp, ocp.nlp[0], [], x, [], [], which_var="states", first_dof=0, second_dof=1, coef=2
    )

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array([-value]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_proportional_control(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.PROPORTIONAL_CONTROL
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](
        penalty, ocp, ocp.nlp[0], [], x, [], [], which_var="controls", first_dof=0, second_dof=1, coef=2
    )

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_torque(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_TORQUE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_track_torque(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = Objective.Lagrange.TRACK_TORQUE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [4], x, [], [], target=np.ones((4, 10)) * value)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_torque_derivative(value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_muscles_control(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_MUSCLES_CONTROL
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_track_muscles_control(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.TRACK_MUSCLES_CONTROL
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [5], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_all_controls(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_ALL_CONTROLS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_track_all_controls(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.TRACK_ALL_CONTROLS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [6], x, [], [], target=np.ones((4, 10)) * value)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_contact_forces(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.MINIMIZE_CONTACT_FORCES
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_contact_forces(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.TRACK_CONTACT_FORCES
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [7], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_minimize_predicted_com_height(value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = Objective.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    res = np.array(0.0501274)
    if value == -10:
        res = np.array([[-3.72579]])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        res,
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_align_segment_with_custom_rt(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.ALIGN_SEGMENT_WITH_CUSTOM_RT
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], segment_idx=1, rt_idx=0)

    res = np.array([[0], [0.1], [0]])
    if value == -10:
        res = np.array([[3.1415927], [0.575222], [3.1415927]])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        res,
    )


@pytest.mark.parametrize("lagrange_or_mayer", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_objective_align_marker_with_segment_axis(lagrange_or_mayer, value):
    ocp = prepare_test_ocp()
    x = [np.ones((12, 1)) * value]
    penalty_type = lagrange_or_mayer.ALIGN_MARKER_WITH_SEGMENT_AXIS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], marker_idx=0, segment_idx=1, axis=Axe.X)

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array([[0]]),
    )
