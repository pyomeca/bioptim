from pathlib import Path
import pytest

from casadi import DM
import numpy as np
import biorbd

from bioptim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsOption,
    InitialGuessOption,
    ObjectiveOption,
    Objective,
    Axe,
    Constraint,
    ConstraintOption,
)


def prepare_test_ocp(with_muscles=False, with_contact=False):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    if with_muscles and with_contact:
        raise RuntimeError("With muscles and with contact together is not defined")
    elif with_muscles:
        biorbd_model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod")
        dynamics = DynamicsTypeList()
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
        nx = biorbd_model.nbQ() + biorbd_model.nbQdot()
        nu = biorbd_model.nbGeneralizedTorque() + biorbd_model.nbMuscles()
    elif with_contact:
        biorbd_model = biorbd.Model(
            str(PROJECT_FOLDER) + "/examples/muscle_driven_with_contact/2segments_4dof_2contacts_1muscle.bioMod"
        )
        dynamics = DynamicsTypeList()
        dynamics.add(DynamicsType.TORQUE_DRIVEN_WITH_CONTACT)
        nx = biorbd_model.nbQ() + biorbd_model.nbQdot()
        nu = biorbd_model.nbGeneralizedTorque()
    else:
        biorbd_model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod")
        dynamics = DynamicsTypeList()
        dynamics.add(DynamicsType.TORQUE_DRIVEN)
        nx = biorbd_model.nbQ() + biorbd_model.nbQdot()
        nu = biorbd_model.nbGeneralizedTorque()
    x_init = InitialGuessOption(np.zeros((nx, 1)))
    u_init = InitialGuessOption(np.zeros((nu, 1)))
    x_bounds = BoundsOption([np.zeros((nx, 1)), np.zeros((nx, 1))])
    u_bounds = BoundsOption([np.zeros((nu, 1)), np.zeros((nu, 1))])
    ocp = OptimalControlProgram(biorbd_model, dynamics, 10, 1.0, x_init, u_init, x_bounds, u_bounds)
    ocp.nlp[0].J = [list()]
    ocp.nlp[0].g = [list()]
    ocp.nlp[0].g_bounds = [list()]
    return ocp


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_time(penalty_origin, value):
    ocp = prepare_test_ocp()
    penalty_type = penalty_origin.MINIMIZE_TIME
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], [], [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(1),
    )


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_state(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_STATE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array([[value]] * 8),
    )


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_state(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_STATE
    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [1], x, [], [], target=np.ones((8, 1)) * value)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    expected = np.array([[value]] * 8)

    np.testing.assert_almost_equal(
        res,
        expected,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0]] * 8))
        np.testing.assert_almost_equal(
            ocp.nlp[0].g_bounds[0][0].max,
            np.array([[0]] * 8),
        )


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_MARKERS
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


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_MARKERS

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [2], x, [], [], target=np.ones((3, 7, 1)) * value)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    expected = np.array(
        [
            [0.1, 0.99517075, 1.9901749, 1.0950042, 1, 2, 0.49750208],
            [0, 0, 0, 0, 0, 0, 0],
            [0.1, -0.9948376, -1.094671, 0.000166583, 0, 0, -0.0499167],
        ]
    )
    if value == -10:
        expected = np.array(
            [
                [-10, -11.3830926, -12.2221642, -10.8390715, 1.0, 2.0, -0.4195358],
                [0, 0, 0, 0, 0, 0, 0],
                [-10, -9.7049496, -10.2489707, -10.5440211, 0, 0, -0.2720106],
            ]
        )

    np.testing.assert_almost_equal(
        res,
        expected,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(
            ocp.nlp[0].g_bounds[0][0].min,
            np.array([[0]] * 3),
        )
        np.testing.assert_almost_equal(
            ocp.nlp[0].g_bounds[0][0].max,
            np.array([[0]] * 3),
        )


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers_displacement(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_MARKERS_DISPLACEMENT
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers_velocity(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_MARKERS_VELOCITY
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


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers_velocity(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_MARKERS_VELOCITY

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [3], x, [], [], target=np.ones((3, 7, 1)) * value)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array(
            [
                [value],
                [0],
                [value],
            ]
        ),
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(
            ocp.nlp[0].g_bounds[0][0].min,
            np.array([[0]] * 3),
        )
        np.testing.assert_almost_equal(
            ocp.nlp[0].g_bounds[0][0].max,
            np.array([[0]] * 3),
        )


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_align_markers(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.ALIGN_MARKERS

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], first_marker_idx=0, second_marker_idx=1)

    expected = np.array(
        [
            [-0.8951707],
            [0],
            [1.0948376],
        ]
    )
    if value == -10:
        expected = np.array(
            [
                [1.3830926],
                [0],
                [-0.2950504],
            ]
        )

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        expected,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(
            ocp.nlp[0].g_bounds[0][0].min,
            np.array([[0]] * 3),
        )
        np.testing.assert_almost_equal(
            ocp.nlp[0].g_bounds[0][0].max,
            np.array([[0]] * 3),
        )


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_proportional_state(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.PROPORTIONAL_STATE

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](
        penalty, ocp, ocp.nlp[0], [], x, [], [], which_var="states", first_dof=0, second_dof=1, coef=2
    )

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[-value]]),
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0]]))


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_proportional_control(penalty_origin, value):
    ocp = prepare_test_ocp()
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.PROPORTIONAL_CONTROL

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    first = 0
    second = 1
    coef = 2
    penalty_type.value[0](
        penalty, ocp, ocp.nlp[0], [], [], u, [], which_var="controls", first_dof=first, second_dof=second, coef=coef
    )

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array(u[0][first] - coef * u[0][second]),
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0]]))


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_torque(penalty_origin, value):
    ocp = prepare_test_ocp()
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_TORQUE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], [], u, [])

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value]]).T,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0, 0, 0, 0]]))


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_torque(penalty_origin, value):
    ocp = prepare_test_ocp()
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_TORQUE

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [4], [], u, [], target=np.ones((4, 1)) * value)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value]]).T,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.zeros((4, 1)))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.zeros((4, 1)))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_torque_derivative(value):
    ocp = prepare_test_ocp()
    u = [DM.ones((12, 1)) * value, DM.ones((12, 1)) * value * 3]
    penalty_type = Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], [], u, [])

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[value * 2, value * 2, value * 2, value * 2]]).T,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0, 0, 0, 0]]))


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_muscles_control(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_MUSCLES_CONTROL
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], [], u, [])

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value, value, value]]).T,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0, 0, 0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0, 0, 0, 0, 0, 0]]))


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_muscles_control(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_MUSCLES_CONTROL

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [5], [], u, [], muscles_idx=0, target=np.ones((1, 1)) * value)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[value]]),
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0]]))


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_all_controls(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_ALL_CONTROLS
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], [], u, [])

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value, value, value, value, value]]).T,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]))


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_all_controls(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_ALL_CONTROLS

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [6], [], u, [], target=np.ones((8, 1)) * value)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value, value, value, value, value]]).T,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]).T)
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]).T)


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_contact_forces(penalty_origin, value):
    ocp = prepare_test_ocp(with_contact=True)
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_CONTACT_FORCES
    penalty = ObjectiveOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, u, [])

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    if value == 0.1:
        np.testing.assert_almost_equal(
            res,
            np.array([[-9.6680105, 127.2360329, 5.0905995]]).T,
        )
    else:
        np.testing.assert_almost_equal(
            res,
            np.array([[25.6627161, 462.7973306, -94.0182191]]).T,
        )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0]]).T)
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0]]).T)


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_contact_forces(penalty_origin, value):
    ocp = prepare_test_ocp(with_contact=True)
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.TRACK_CONTACT_FORCES

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [7], x, u, [], contacts_idx=0, target=np.ones((1, 1)) * value)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    if value == 0.1:
        np.testing.assert_almost_equal(
            res,
            np.array([[-9.6680105]]),
        )
    else:
        np.testing.assert_almost_equal(
            res,
            np.array([[25.6627161]]),
        )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0.0]]).T)
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0.0]]).T)


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_predicted_com_height(value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
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


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_align_segment_with_custom_rt(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.ALIGN_SEGMENT_WITH_CUSTOM_RT

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], segment_idx=1, rt_idx=0)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    expected = np.array([[0], [0.1], [0]])
    if value == -10:
        expected = np.array([[3.1415927], [0.575222], [3.1415927]])

    np.testing.assert_almost_equal(
        res,
        expected,
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0], [0], [0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0], [0], [0]]))


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_align_marker_with_segment_axis(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.ALIGN_MARKER_WITH_SEGMENT_AXIS

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], marker_idx=0, segment_idx=1, axis=Axe.X)

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(
        res,
        np.array([[0]]),
    )

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0]]))


@pytest.mark.parametrize("penalty_origin", [Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("direction", ["GREATER_THAN", "LESSER_THAN"])
def test_penalty_contact_force_inequality(penalty_origin, value, direction):
    ocp = prepare_test_ocp(with_contact=True)
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.CONTACT_FORCE_INEQUALITY
    penalty = ConstraintOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, u, [], direction=direction, contact_force_idx=0, boundary=1)
    res = ocp.nlp[0].g[0][0]

    if value == 0.1 and direction == "GREATER_THAN":
        expected = [-9.6680105, 1.0, np.inf]
    elif value == -10 and direction == "GREATER_THAN":
        expected = [25.6627161, 1.0, np.inf]
    elif value == 0.1 and direction == "LESSER_THAN":
        expected = [-9.6680105, -np.inf, 1.0]
    elif value == -10 and direction == "LESSER_THAN":
        expected = [25.6627161, -np.inf, 1.0]

    np.testing.assert_almost_equal(res, np.array([[expected[0]]]))

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[expected[1]]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[expected[2]]]))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_non_slipping(value):
    ocp = prepare_test_ocp(with_contact=True)
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = Constraint.NON_SLIPPING
    penalty = ConstraintOption(penalty_type)
    penalty_type.value[0](
        penalty,
        ocp,
        ocp.nlp[0],
        [],
        x,
        u,
        [],
        tangential_component_idx=0,
        normal_component_idx=1,
        static_friction_coefficient=2,
    )
    res = ocp.nlp[0].g[0]

    if value == 0.1:
        expected = [[264.1400764, 244.8040553], 0, np.inf]
    elif value == -10:
        expected = [[899.9319451, 951.2573773], 0, np.inf]

    np.testing.assert_almost_equal(res, np.array(expected[0]))

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[expected[1]]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[expected[2]]]))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_time_constraint(value):
    ocp = prepare_test_ocp()
    penalty_type = Constraint.TIME_CONSTRAINT
    penalty = ConstraintOption(penalty_type)
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], [], [], [])
    res = ocp.nlp[0].g[0]

    np.testing.assert_almost_equal(
        res,
        np.array([]),
    )
    np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0], np.array([]))


def custom(ocp, nlp, t, x, u, p, **extra_params):
    my_values = DM.zeros((12, 1)) + x[0]
    return my_values


@pytest.mark.parametrize("penalty_origin", [Objective.Lagrange, Objective.Mayer, Constraint])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.CUSTOM

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        penalty = ObjectiveOption(penalty_type)
    else:
        penalty = ConstraintOption(penalty_type)

    penalty.custom_function = custom
    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    if isinstance(penalty_type, (Objective.Lagrange, Objective.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]

    np.testing.assert_almost_equal(res, np.array([[value]] * 12))

    if isinstance(penalty_type, Constraint):
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].min, np.array([[0]] * 12))
        np.testing.assert_almost_equal(ocp.nlp[0].g_bounds[0][0].max, np.array([[0]] * 12))
