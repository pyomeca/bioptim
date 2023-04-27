import pytest
from casadi import DM, MX
import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    Bounds,
    InitialGuess,
    Objective,
    ObjectiveFcn,
    Axis,
    ConstraintFcn,
    Constraint,
    Node,
    RigidBodyDynamics,
    ControlType,
)
from bioptim.limits.penalty_node import PenaltyNodeList
from bioptim.limits.penalty import PenaltyOption
from bioptim.misc.mapping import BiMapping
from bioptim.optimization.non_linear_program import NonLinearProgram as NLP
from bioptim.optimization.optimization_variable import OptimizationVariableList
from .utils import TestUtils


def prepare_test_ocp(with_muscles=False, with_contact=False, with_actuator=False, implicit=False, use_sx=True):
    bioptim_folder = TestUtils.bioptim_folder()
    if with_muscles and with_contact or with_muscles and with_actuator or with_contact and with_actuator:
        raise RuntimeError("With muscles and with contact and with_actuator together is not defined")
    if with_muscles and implicit or implicit and with_actuator:
        raise RuntimeError("With muscles and implicit and with_actuator together is not defined")
    elif with_muscles:
        bio_model = BiorbdModel(bioptim_folder + "/examples/muscle_driven_ocp/models/arm26.bioMod")
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True)
        nx = bio_model.nb_q + bio_model.nb_qdot
        nu = bio_model.nb_tau + bio_model.nb_muscles
    elif with_contact:
        bio_model = BiorbdModel(
            bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
        )
        dynamics = DynamicsList()
        rigidbody_dynamics = RigidBodyDynamics.DAE_INVERSE_DYNAMICS if implicit else RigidBodyDynamics.ODE
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, expand=False, rigidbody_dynamics=rigidbody_dynamics)
        nx = bio_model.nb_q + bio_model.nb_qdot
        nu = bio_model.nb_tau
    elif with_actuator:
        bio_model = BiorbdModel(bioptim_folder + "/examples/torque_driven_ocp/models/cube.bioMod")
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
        nx = bio_model.nb_q + bio_model.nb_qdot
        nu = bio_model.nb_tau
    else:
        bio_model = BiorbdModel(bioptim_folder + "/examples/track/models/cube_and_line.bioMod")
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
        nx = bio_model.nb_q + bio_model.nb_qdot
        nu = bio_model.nb_tau
    x_init = InitialGuess(np.zeros((nx, 1)))

    if implicit:
        nu *= 2
        if with_contact:
            nu += 3

    u_init = InitialGuess(np.zeros((nu, 1)))
    x_bounds = Bounds(np.zeros((nx, 1)), np.zeros((nx, 1)))
    u_bounds = Bounds(np.zeros((nu, 1)), np.zeros((nu, 1)))
    ocp = OptimalControlProgram(bio_model, dynamics, 10, 1.0, x_init, u_init, x_bounds, u_bounds, use_sx=use_sx)
    ocp.nlp[0].J = [[]]
    ocp.nlp[0].g = [[]]
    return ocp


def get_penalty_value(ocp, penalty, t, x, u, p):
    val = penalty.type(penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, [], [], []), **penalty.params)
    if isinstance(val, float):
        return val

    states = ocp.nlp[0].states.cx if ocp.nlp[0].states.cx.shape != (0, 0) else ocp.cx(0, 0)
    controls = ocp.nlp[0].controls.cx if ocp.nlp[0].controls.cx.shape != (0, 0) else ocp.cx(0, 0)
    parameters = ocp.nlp[0].parameters.cx if ocp.nlp[0].parameters.cx.shape != (0, 0) else ocp.cx(0, 0)
    return ocp.nlp[0].to_casadi_func("penalty", val, states, controls, parameters)(x[0], u[0], p)


def test_penalty_targets_shapes():
    p = ObjectiveFcn.Parameter
    np.testing.assert_equal(Objective([], custom_type=p, target=1).target[0].shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=np.array(1)).target[0].shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[1]).target[0].shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[1, 2]).target[0].shape, (2, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[[1], [2]]).target[0].shape, (2, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[[1, 2]]).target[0].shape, (1, 2))
    np.testing.assert_equal(Objective([], custom_type=p, target=np.array([[1, 2]])).target[0].shape, (1, 2))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_time(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]

    penalty_type = penalty_origin.MINIMIZE_TIME
    penalty = Objective(penalty_type)
    penalty_type(penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], [], [], [], [], []))
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    np.testing.assert_almost_equal(res, np.array(1))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_state(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty = Objective(penalty_origin.MINIMIZE_STATE, key="qdot")
    res = get_penalty_value(ocp, penalty, t, x, u, [])
    np.testing.assert_almost_equal(res, np.array([[value]] * 4))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_qddot(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0, 1]
    x = [DM.ones((8, 1)) * value, DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    if penalty_origin == ConstraintFcn:
        with pytest.raises(AttributeError, match="MINIMIZE_QDDOT"):
            _ = penalty_origin.MINIMIZE_QDDOT
        return
    else:
        penalty_type = penalty_origin.MINIMIZE_QDDOT
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, []).T

    np.testing.assert_almost_equal(res, [[value, -9.81 + value, value, value]])


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_state(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_STATE
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="qdot", target=np.ones((4, 1)) * value)
    else:
        penalty = Constraint(penalty_type, key="qdot", target=np.ones((4, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, x, u, [])
    np.testing.assert_almost_equal(res, [[value]] * 4)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.MINIMIZE_MARKERS
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = np.array(
        [
            [0.1, 0.99517075, 1.9901749, 1.0950042, 0, 1, 2, 0.49750208],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0.1, -0.9948376, -1.094671, 0.000166583, 0, 0, 0, -0.0499167],
        ]
    )
    if value == -10:
        expected = np.array(
            [
                [-10, -11.3830926, -12.2221642, -10.8390715, 0, 1.0, 2.0, -0.4195358],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [-10, -9.7049496, -10.2489707, -10.5440211, 0, 0, 0, -0.2720106],
            ]
        )

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = np.array(
        [
            [0.1, 0.99517075, 1.9901749, 1.0950042, 0, 1, 2, 0.49750208],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0.1, -0.9948376, -1.094671, 0.000166583, 0, 0, 0, -0.0499167],
        ]
    )
    if value == -10:
        expected = np.array(
            [
                [-10, -11.3830926, -12.2221642, -10.8390715, 0, 1.0, 2.0, -0.4195358],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [-10, -9.7049496, -10.2489707, -10.5440211, 0, 0, 0, -0.2720106],
            ]
        )

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers_velocity(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.MINIMIZE_MARKERS_VELOCITY
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    if value == 0.1:
        np.testing.assert_almost_equal(
            res,
            np.array(
                [
                    [0.1, -0.00948376, -0.0194671, 0.0900167, 0, 00, 00, -0.00499167],
                    [0, 0, 0, 0, 0, 00, 00, 0],
                    [0.1, 0.0104829, -0.0890175, 0.000499583, 0, 0, 0, -0.0497502],
                ]
            ),
        )
    else:
        np.testing.assert_almost_equal(
            res,
            np.array(
                [
                    [-10, -12.9505, -7.51029, -4.55979, 0, 00, 00, 2.72011],
                    [0, 0, 0, 0, 0, 00, 00, 0],
                    [-10, -23.8309, -32.2216, -18.3907, 0, 0, 0, -4.19536],
                ]
            ),
            decimal=4,
        )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers_velocity(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_MARKERS_VELOCITY

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    if value == 0.1:
        np.testing.assert_almost_equal(
            res,
            np.array(
                [
                    [0.1, -0.00948376, -0.0194671, 0.0900167, 0, 00, 00, -0.00499167],
                    [0, 0, 0, 0, 0, 00, 00, 0],
                    [0.1, 0.0104829, -0.0890175, 0.000499583, 0, 0, 0, -0.0497502],
                ]
            ),
        )
    else:
        np.testing.assert_almost_equal(
            res,
            np.array(
                [
                    [-10, -12.9505, -7.51029, -4.55979, 0, 00, 00, 2.72011],
                    [0, 0, 0, 0, 0, 00, 00, 0],
                    [-10, -23.8309, -32.2216, -18.3907, 0, 0, 0, -4.19536],
                ]
            ),
            decimal=4,
        )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_super_impose_marker(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.SUPERIMPOSE_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, first_marker=0, second_marker=1)
    else:
        penalty = Constraint(penalty_type, first_marker=0, second_marker=1)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = [[0.8951707, 0, -1.0948376]] if value == 0.1 else [[-1.3830926, 0, 0.2950504]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("value_intercept", [0.0, 1.0])
def test_penalty_proportional_state(penalty_origin, value, value_intercept):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.PROPORTIONAL_STATE

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(
            penalty_type,
            key="qdot",
            first_dof=0,
            second_dof=1,
            coef=2,
            first_dof_intercept=value_intercept,
            second_dof_intercept=value_intercept,
        )
    else:
        penalty = Constraint(
            penalty_type,
            key="qdot",
            first_dof=0,
            second_dof=1,
            coef=2,
            first_dof_intercept=value_intercept,
            second_dof_intercept=value_intercept,
        )
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    if value_intercept == 0.0:
        np.testing.assert_almost_equal(res, -value)
    else:
        if value == 0.1:
            np.testing.assert_almost_equal(res, 0.9)
        else:
            np.testing.assert_almost_equal(res, 11)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_proportional_control(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.PROPORTIONAL_CONTROL

    first = 0
    second = 1
    coef = 2

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="tau", first_dof=first, second_dof=second, coef=coef)
    else:
        penalty = Constraint(penalty_type, key="tau", first_dof=first, second_dof=second, coef=coef)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    np.testing.assert_almost_equal(res, np.array(u[0][first] - coef * u[0][second]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_torque(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0, 1]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    penalty = Objective(penalty_origin.MINIMIZE_CONTROL, key="tau")
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    np.testing.assert_almost_equal(res, np.array([[value, value, value, value]]).T)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_torque(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0, 1]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.TRACK_CONTROL

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="tau", target=np.ones((4, 1)) * value)
    else:
        penalty = Constraint(penalty_type, key="tau", target=np.ones((4, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    np.testing.assert_almost_equal(res, np.array([[value, value, value, value]]).T)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_muscles_control(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    t = [0, 1]
    x = [0]
    u = [DM.ones((8, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_CONTROL
    penalty = Objective(penalty_type, key="muscles")
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    np.testing.assert_almost_equal(res, np.array([[value, value, value, value, value, value]]).T)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_contact_forces(penalty_origin, value):
    ocp = prepare_test_ocp(with_contact=True)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_CONTACT_FORCES
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    if value == 0.1:
        np.testing.assert_almost_equal(res, np.array([[-9.6680105, 127.2360329, 5.0905995]]).T)
    else:
        np.testing.assert_almost_equal(res, np.array([[25.6627161, 462.7973306, -94.0182191]]).T)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_contact_forces(penalty_origin, value):
    ocp = prepare_test_ocp(with_contact=True)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.TRACK_CONTACT_FORCES

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((1, 1)) * value, index=0)
    else:
        penalty = Constraint(penalty_type, target=np.ones((1, 1)) * value, index=0)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    if value == 0.1:
        np.testing.assert_almost_equal(res.T, [[-9.6680105, 127.2360329, 5.0905995]])
    else:
        np.testing.assert_almost_equal(res.T, [[25.6627161, 462.7973306, -94.0182191]])


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_predicted_com_height(value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = np.array(0.0501274 if value == 0.1 else -3.72579)
    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_com_position(value, penalty_origin):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    if "TRACK_COM_POSITION" in penalty_origin._member_names_:
        penalty_type = penalty_origin.TRACK_COM_POSITION
    else:
        penalty_type = penalty_origin.MINIMIZE_COM_POSITION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = np.array([[0.05], [0.05], [0.05]])
    if value == -10:
        expected = np.array([[-5], [0.05], [-5]])

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_angular_momentum(value, penalty_origin):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]

    penalty_type = penalty_origin.MINIMIZE_ANGULAR_MOMENTUM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = np.array([[-0.005], [0.2], [0.005]])
    if value == -10:
        expected = np.array([[0.5], [-20], [-0.5]])

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("use_sx", [True, False])
def test_penalty_minimize_linear_momentum(value, penalty_origin, use_sx):
    ocp = prepare_test_ocp(use_sx=use_sx)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]

    penalty_type = penalty_origin.MINIMIZE_LINEAR_MOMENTUM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = np.array([[0.1], [0], [0.1]])
    if value == -10:
        expected = np.array([[-10], [0], [-10]])

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("implicit", [True, False])
def test_penalty_minimize_comddot(value, penalty_origin, implicit):
    ocp = prepare_test_ocp(with_contact=True, implicit=implicit)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]

    penalty_type = penalty_origin.MINIMIZE_COM_ACCELERATION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    if not implicit:
        res = get_penalty_value(ocp, penalty, t, x, u, [])

        expected = np.array([[0.0], [-0.7168803], [-0.0740871]])
        if value == -10:
            expected = np.array([[0.0], [1.455063], [16.3741091]])

        np.testing.assert_almost_equal(res, expected)
    else:
        res = get_penalty_value(ocp, penalty, t, x, u, [])

        expected = np.array([[0], [-0.0008324], [0.002668]])
        if value == -10:
            expected = np.array([[0], [-17.5050533], [-18.2891901]])

        np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_segment_with_custom_rt(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_SEGMENT_WITH_CUSTOM_RT

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, segment="ground", rt=0)
    else:
        penalty = Constraint(penalty_type, segment="ground", rt=0)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = np.array([[0], [0.1], [0]])
    if value == -10:
        expected = np.array([[3.1415927], [0.575222], [3.1415927]])

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_marker_with_segment_axis(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_MARKER_WITH_SEGMENT_AXIS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, marker="m0", segment="ground", axis=Axis.X)
    else:
        penalty = Constraint(penalty_type, marker="m0", segment="ground", axis=Axis.X)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = [[value, 0, value]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_segment_rotation(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]

    if penalty_origin == ObjectiveFcn.Lagrange or penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = penalty_origin.MINIMIZE_SEGMENT_ROTATION
        penalty = Objective(penalty_type, segment=2)
    else:
        penalty_type = penalty_origin.TRACK_SEGMENT_ROTATION
        penalty = Constraint(penalty_type, segment=2)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = [[0, value, 0]] if value == 0.1 else [[3.1415927, 0.575222, 3.1415927]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_segment_velocity(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]

    if penalty_origin == ObjectiveFcn.Lagrange or penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = penalty_origin.MINIMIZE_SEGMENT_VELOCITY
        penalty = Objective(penalty_type, segment=2)
    else:
        penalty_type = penalty_origin.TRACK_SEGMENT_VELOCITY
        penalty = Constraint(penalty_type, segment=2)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = [[0, value, 0]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_vector_orientation(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM(np.array([0, 0, value, 0, 0, 0, 0, 0]))]
    u = [0]
    penalty_type = penalty_origin.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(
            penalty_type,
            vector_0_marker_0="m0",
            vector_0_marker_1="m3",
            vector_1_marker_0="origin",
            vector_1_marker_1="m6",
        )
    else:
        penalty = Constraint(
            penalty_type,
            vector_0_marker_0="m0",
            vector_0_marker_1="m3",
            vector_1_marker_0="origin",
            vector_1_marker_1="m6",
        )

    res = get_penalty_value(ocp, penalty, t, x, u, [])

    if value == 0.1:
        np.testing.assert_almost_equal(float(res), 1.0529423391195598)
    else:
        np.testing.assert_almost_equal(float(res), 1.2110674089002156)


@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_contact_force_inequality(penalty_origin, value):
    ocp = prepare_test_ocp(with_contact=True)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]

    penalty_type = penalty_origin.TRACK_CONTACT_FORCES
    penalty = Constraint(penalty_type, contact_index=0)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = [[-9.6680105, 127.2360329, 5.0905995]] if value == 0.1 else [[25.6627161, 462.7973306, -94.0182191]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_non_slipping(value):
    ocp = prepare_test_ocp(with_contact=True)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = ConstraintFcn.NON_SLIPPING
    penalty = Constraint(
        penalty_type, tangential_component_idx=0, normal_component_idx=1, static_friction_coefficient=2
    )
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    expected = [[64662.56185612, 64849.5027121]] if value == 0.1 else [[856066.90177734, 857384.05177395]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("value", [2])
@pytest.mark.parametrize("threshold", [None, 15, -15])
def test_tau_max_from_actuators(value, threshold):
    ocp = prepare_test_ocp(with_actuator=True)
    t = [0]
    x = [DM.zeros((6, 1)), DM.zeros((6, 1))]
    u = [DM.ones((3, 1)) * value, DM.ones((3, 1)) * value]
    penalty_type = ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT
    penalty = Constraint(penalty_type, min_torque=threshold)
    if threshold and threshold < 0:
        with pytest.raises(ValueError, match="min_torque cannot be negative in tau_max_from_actuators"):
            get_penalty_value(ocp, penalty, t, x, u, [])
        return
    else:
        res = get_penalty_value(ocp, penalty, t, x, u, [])

    if threshold:
        np.testing.assert_almost_equal(res, np.repeat([value + threshold, value - threshold], 3)[:, np.newaxis])
    else:
        np.testing.assert_almost_equal(res, np.repeat([value + 5, value - 10], 3)[:, np.newaxis])


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_time_constraint(value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [0]
    u = [0]
    penalty_type = ConstraintFcn.TIME_CONSTRAINT
    penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    np.testing.assert_almost_equal(res, np.array([]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom(penalty_origin, value):
    def custom(pn, mult):
        my_values = pn.nlp.states["q"].cx * mult
        return my_values

    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.CUSTOM

    mult = 2
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(custom, index=0, mult=mult, custom_type=penalty_origin)
    else:
        penalty = Constraint(custom, index=0, mult=mult)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    np.testing.assert_almost_equal(res, [[value * mult]] * 4)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_fail(penalty_origin, value):
    def custom_no_mult(ocp, nlp, t, x, u, p):
        my_values = DM.zeros((12, 1)) + x[0]
        return my_values

    def custom_with_mult(ocp, nlp, t, x, u, p, mult):
        my_values = DM.zeros((12, 1)) + x[0] * mult
        return my_values

    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.CUSTOM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    with pytest.raises(TypeError):
        penalty.custom_function = custom_no_mult
        penalty_type(penalty, ocp, ocp.nlp[0], [], x, [], [], mult=2)

    with pytest.raises(TypeError):
        penalty.custom_function = custom_with_mult
        penalty_type(penalty, ocp, ocp.nlp[0], [], x, [], [])

    with pytest.raises(TypeError):
        keywords = [
            "phase",
            "list_index",
            "name",
            "type",
            "params",
            "node",
            "quadratic",
            "index",
            "target",
            "min_bound",
            "max_bound",
            "custom_function",
            "weight",
        ]
        for keyword in keywords:
            exec(
                f"""def custom_with_keyword(ocp, nlp, t, x, u, p, {keyword}):
                            my_values = DM.zeros((12, 1)) + x[index]
                            return my_values"""
            )
            exec("""penalty.custom_function = custom_with_keyword""")
            exec(f"""penalty_type(penalty, ocp, ocp.nlp[0], [], x, [], [], {keyword}=0)""")


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds(value):
    def custom_with_bounds(pn):
        return -10, pn.nlp.states["q"].cx, 10

    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]

    penalty = Constraint(custom_with_bounds)
    res = get_penalty_value(ocp, penalty, t, x, u, [])

    np.testing.assert_almost_equal(res, [[value]] * 4)
    np.testing.assert_almost_equal(penalty.min_bound, -10)
    np.testing.assert_almost_equal(penalty.max_bound, 10)


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds_failing_min_bound(value):
    def custom_with_bounds(pn):
        return -10, pn.nlp.states["q"].cx, 10

    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.min_bound = 0
    penalty.custom_function = custom_with_bounds

    with pytest.raises(RuntimeError):
        penalty_type(penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, [], [], [], []))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds_failing_max_bound(value):
    def custom_with_bounds(pn):
        return -10, pn.nlp.states["q"].cx, 10

    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.max_bound = 0
    penalty.custom_function = custom_with_bounds

    with pytest.raises(
        RuntimeError,
        match="You cannot have non linear bounds for custom constraints and min_bound or max_bound defined",
    ):
        penalty_type(penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, [], [], [], []))


@pytest.mark.parametrize(
    "node",
    [
        Node.ALL,
        Node.ALL_SHOOTING,
        Node.INTERMEDIATES,
        Node.START,
        Node.MID,
        Node.PENULTIMATE,
        Node.END,
        Node.TRANSITION,
    ],
)
@pytest.mark.parametrize("ns", [1, 10, 11])
def test_PenaltyFunctionAbstract_get_node(node, ns):
    nlp = NLP()
    nlp.control_type = ControlType.CONSTANT
    nlp.ns = ns
    nlp.X = np.linspace(0, -10, ns + 1)
    nlp.U = np.linspace(10, 19, ns)
    nlp.X_scaled = nlp.X
    nlp.U_scaled = nlp.U
    tp = OptimizationVariableList()
    tp.cx_constructor = MX
    tp.append("param", [MX(), MX()], MX(), BiMapping([], []))
    nlp.parameters = tp["param"]

    pn = []
    penalty = PenaltyOption(pn)
    penalty.node = node

    if node == Node.MID and ns % 2 != 0:
        with pytest.raises(ValueError, match="Number of shooting points must be even to use MID"):
            _ = penalty._get_penalty_node_list([], nlp)
        return
    elif node == Node.TRANSITION:
        with pytest.raises(RuntimeError, match=" is not a valid node"):
            _ = penalty._get_penalty_node_list([], nlp)
        return
    elif ns == 1 and node == Node.PENULTIMATE:
        with pytest.raises(ValueError, match="Number of shooting points must be greater than 1"):
            _ = penalty._get_penalty_node_list([], nlp)
        return
    else:
        all_pn = penalty._get_penalty_node_list([], nlp)

    x_expected = nlp.X
    u_expected = nlp.U

    if node == Node.ALL:
        np.testing.assert_almost_equal(all_pn.t, [i for i in range(ns + 1)])
        np.testing.assert_almost_equal(np.array(all_pn.x), np.linspace(0, -10, ns + 1))
        np.testing.assert_almost_equal(np.array(all_pn.u), np.linspace(10, 19, ns))
        np.testing.assert_almost_equal(np.array(all_pn.x_scaled), np.linspace(0, -10, ns + 1))
        np.testing.assert_almost_equal(np.array(all_pn.u_scaled), np.linspace(10, 19, ns))
    elif node == Node.ALL_SHOOTING:
        np.testing.assert_almost_equal(all_pn.t, [i for i in range(ns)])
        np.testing.assert_almost_equal(np.array(all_pn.x), nlp.X[:-1])
        np.testing.assert_almost_equal(np.array(all_pn.u), nlp.U)
        np.testing.assert_almost_equal(np.array(all_pn.x_scaled), nlp.X[:-1])
        np.testing.assert_almost_equal(np.array(all_pn.u_scaled), nlp.U)
    elif node == Node.INTERMEDIATES:
        np.testing.assert_almost_equal(all_pn.t, [i for i in range(1, ns - 1)])
        np.testing.assert_almost_equal(np.array(all_pn.x), x_expected[1 : ns - 1])
        np.testing.assert_almost_equal(np.array(all_pn.u), u_expected[1 : ns - 1])
        np.testing.assert_almost_equal(np.array(all_pn.x_scaled), x_expected[1 : ns - 1])
        np.testing.assert_almost_equal(np.array(all_pn.u_scaled), u_expected[1 : ns - 1])
    elif node == Node.START:
        np.testing.assert_almost_equal(all_pn.t, [0])
        np.testing.assert_almost_equal(np.array(all_pn.x), x_expected[0])
        np.testing.assert_almost_equal(np.array(all_pn.u), u_expected[0])
        np.testing.assert_almost_equal(np.array(all_pn.x_scaled), x_expected[0])
        np.testing.assert_almost_equal(np.array(all_pn.u_scaled), u_expected[0])
    elif node == Node.MID:
        np.testing.assert_almost_equal(all_pn.t, [ns // 2])
        np.testing.assert_almost_equal(np.array(all_pn.x), x_expected[ns // 2])
        np.testing.assert_almost_equal(np.array(all_pn.u), u_expected[ns // 2])
        np.testing.assert_almost_equal(np.array(all_pn.x_scaled), x_expected[ns // 2])
        np.testing.assert_almost_equal(np.array(all_pn.u_scaled), u_expected[ns // 2])
    elif node == Node.PENULTIMATE:
        np.testing.assert_almost_equal(all_pn.t, [ns - 1])
        np.testing.assert_almost_equal(np.array(all_pn.x), x_expected[-2])
        np.testing.assert_almost_equal(np.array(all_pn.u), u_expected[-1])
        np.testing.assert_almost_equal(np.array(all_pn.x_scaled), x_expected[-2])
        np.testing.assert_almost_equal(np.array(all_pn.u_scaled), u_expected[-1])
    elif node == Node.END:
        np.testing.assert_almost_equal(all_pn.t, [ns])
        np.testing.assert_almost_equal(np.array(all_pn.x), x_expected[ns])
        np.testing.assert_almost_equal(all_pn.u, [])
        np.testing.assert_almost_equal(np.array(all_pn.x_scaled), x_expected[ns])
        np.testing.assert_almost_equal(all_pn.u_scaled, [])
    else:
        raise RuntimeError("Something went wrong")
