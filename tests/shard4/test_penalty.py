import pytest
from casadi import DM, MX, vertcat, horzcat
import numpy as np
import pytest
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    Axis,
    ConstraintFcn,
    Constraint,
    MultinodeConstraintFcn,
    MultinodeConstraintList,
    MultinodeConstraint,
    MultinodeObjective,
    Node,
    RigidBodyDynamics,
    ControlType,
    PhaseDynamics,
    ObjectiveList,
)
from bioptim.limits.penalty import PenaltyOption
from bioptim.limits.penalty_controller import PenaltyController
from bioptim.misc.mapping import BiMapping
from bioptim.optimization.non_linear_program import NonLinearProgram as NLP
from bioptim.optimization.optimization_variable import OptimizationVariableList
from casadi import DM, MX
from tests.utils import TestUtils


def prepare_test_ocp(
    phase_dynamics,
    with_muscles=False,
    with_contact=False,
    with_actuator=False,
    implicit=False,
    use_sx=True,
):
    bioptim_folder = TestUtils.bioptim_folder()
    if with_muscles and with_contact or with_muscles and with_actuator or with_contact and with_actuator:
        raise RuntimeError("With muscles and with contact and with_actuator together is not defined")
    if with_muscles and implicit or implicit and with_actuator:
        raise RuntimeError("With muscles and implicit and with_actuator together is not defined")
    elif with_muscles:
        bio_model = BiorbdModel(bioptim_folder + "/examples/muscle_driven_ocp/models/arm26.bioMod")
        dynamics = DynamicsList()
        dynamics.add(
            DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True, expand_dynamics=True, phase_dynamics=phase_dynamics
        )
    elif with_contact:
        bio_model = BiorbdModel(
            bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
        )
        dynamics = DynamicsList()
        rigidbody_dynamics = RigidBodyDynamics.DAE_INVERSE_DYNAMICS if implicit else RigidBodyDynamics.ODE
        dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN,
            with_contact=True,
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
            rigidbody_dynamics=rigidbody_dynamics,
        )
    elif with_actuator:
        bio_model = BiorbdModel(bioptim_folder + "/examples/torque_driven_ocp/models/cube.bioMod")
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)
    else:
        bio_model = BiorbdModel(bioptim_folder + "/examples/track/models/cube_and_line.bioMod")
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)

    objective_functions = Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME)

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        10,
        1.0,
        objective_functions=objective_functions,
        use_sx=use_sx,
    )

    ocp.nlp[0].J = [[]]
    ocp.nlp[0].g = [[]]
    return ocp


def prepare_multinode_test_ocp(
    phase_dynamics,
    with_muscles=False,
    with_contact=False,
    with_actuator=False,
    implicit=False,
    use_sx=True,
):
    bioptim_folder = TestUtils.bioptim_folder()
    if with_muscles and with_contact or with_muscles and with_actuator or with_contact and with_actuator:
        raise RuntimeError("With muscles and with contact and with_actuator together is not defined")
    if with_muscles and implicit or implicit and with_actuator:
        raise RuntimeError("With muscles and implicit and with_actuator together is not defined")
    elif with_muscles:
        bio_model = []
        bio_model.append(BiorbdModel(bioptim_folder + "/examples/muscle_driven_ocp/models/arm26.bioMod"))
        bio_model.append(BiorbdModel(bioptim_folder + "/examples/muscle_driven_ocp/models/arm26.bioMod"))
        dynamics = DynamicsList()
        dynamics.add(
            DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True, expand_dynamics=True, phase_dynamics=phase_dynamics
        )
        dynamics.add(
            DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True, expand_dynamics=True, phase_dynamics=phase_dynamics
        )
    elif with_contact:
        bio_model = []
        bio_model.append(
            BiorbdModel(
                bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
            )
        )
        bio_model.append(
            BiorbdModel(
                bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod"
            )
        )
        dynamics = DynamicsList()
        rigidbody_dynamics = RigidBodyDynamics.DAE_INVERSE_DYNAMICS if implicit else RigidBodyDynamics.ODE
        dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN,
            with_contact=True,
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
            rigidbody_dynamics=rigidbody_dynamics,
        )
        dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN,
            with_contact=True,
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
            rigidbody_dynamics=rigidbody_dynamics,
        )
    elif with_actuator:
        bio_model = []
        bio_model.append(BiorbdModel(bioptim_folder + "/examples/torque_driven_ocp/models/cube.bioMod"))
        bio_model.append(BiorbdModel(bioptim_folder + "/examples/torque_driven_ocp/models/cube.bioMod"))

        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)

    else:
        bio_model = []
        bio_model.append(BiorbdModel(bioptim_folder + "/examples/track/models/cube_and_line.bioMod"))
        bio_model.append(BiorbdModel(bioptim_folder + "/examples/track/models/cube_and_line.bioMod"))
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME)

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        [10, 10],
        [1.0, 1.0],
        objective_functions=objective_functions,
        use_sx=use_sx,
    )

    ocp.nlp[0].J = [[]]
    ocp.nlp[1].J = [[]]

    ocp.nlp[0].g = [[]]
    ocp.nlp[1].g = [[]]

    return ocp


def get_penalty_value(ocp, penalty, t, x, u, p, s):
    if isinstance(penalty, MultinodeConstraint) or isinstance(penalty, MultinodeObjective):
        controller = [
            PenaltyController(ocp, ocp.nlp[0], [], x[0], u[0], [], [], p, s, [], 0),
            PenaltyController(ocp, ocp.nlp[1], [], x[1], u[1], [], [], p, s, [], 0),
        ]
    else:
        controller = PenaltyController(ocp, ocp.nlp[0], t, x, u, [], [], p, s, [], 0)
    val = penalty.type(penalty, controller, **penalty.params)
    # changed only this one
    if isinstance(val, float):
        return val

    time = ocp.nlp[0].time_cx if ocp.nlp[0].time_cx.shape == (0, 0) else ocp.cx(0, 0)
    if isinstance(penalty, MultinodeConstraint) or isinstance(penalty, MultinodeObjective):
        states = (
            vertcat(ocp.nlp[0].states.cx_start, ocp.nlp[1].states.cx_start)
            if ocp.nlp[0].states.cx_start.shape != (0, 0)
            else ocp.cx(0, 0)
        )
        controls = (
            vertcat(ocp.nlp[0].controls.cx_start, ocp.nlp[1].controls.cx_start)
            if ocp.nlp[0].controls.cx_start.shape != (0, 0)
            else ocp.cx(0, 0)
        )
        # stochastic_variables = (vertcat(ocp.nlp[0].stochastic_variables.cx_start,
        #    ocp.nlp[1].stochastic_variables.cx.start)
        #    if ocp.nlp[0].stochastic_variables.cx_start.shape != (0, 0)
        #    else ocp.cx(0, 0)
        # )
    else:
        states = ocp.nlp[0].states.cx_start if ocp.nlp[0].states.cx_start.shape != (0, 0) else ocp.cx(0, 0)
        controls = ocp.nlp[0].controls.cx_start if ocp.nlp[0].controls.cx_start.shape != (0, 0) else ocp.cx(0, 0)
    stochastic_variables = (
        ocp.nlp[0].stochastic_variables.cx_start
        if ocp.nlp[0].stochastic_variables.cx_start.shape != (0, 0)
        else ocp.cx(0, 0)
    )

    parameters = ocp.nlp[0].parameters.cx if ocp.nlp[0].parameters.cx.shape != (0, 0) else ocp.cx(0, 0)
    return ocp.nlp[0].to_casadi_func("penalty", val, time, states, controls, parameters, stochastic_variables)(
        t, x[0], u[0], p, s
    )


def test_penalty_targets_shapes():
    p = ObjectiveFcn.Parameter
    np.testing.assert_equal(Objective([], custom_type=p, target=1).target[0].shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=np.array(1)).target[0].shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[1]).target[0].shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[1, 2]).target[0].shape, (2, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[[1], [2]]).target[0].shape, (2, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[[1, 2]]).target[0].shape, (1, 2))
    np.testing.assert_equal(Objective([], custom_type=p, target=np.array([[1, 2]])).target[0].shape, (1, 2))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_time(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = [1]
    s = []

    penalty_type = penalty_origin.MINIMIZE_TIME
    penalty = Objective(penalty_type)
    penalty_type(penalty, PenaltyController(ocp, ocp.nlp[0], [], [], [], [], [], p, s, [], 0))
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(res, np.array(1))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_state(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty = Objective(penalty_origin.MINIMIZE_STATE, key="qdot")
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)
    np.testing.assert_almost_equal(res, np.array([[value]] * 4))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_joint_power(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [1]
    p = []
    s = []
    penalty = Objective(penalty_origin.MINIMIZE_POWER, key_control="tau")
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)
    np.testing.assert_almost_equal(res, np.array([[value]] * 4))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_muscle_power(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_muscles=True, phase_dynamics=phase_dynamics)
    t = [0]
    x = [1]
    u = [DM.ones((8, 1)) * value]
    p = []
    s = []

    penalty = Objective(penalty_origin.MINIMIZE_POWER, key_control="muscles")
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)
    if value == 0.1:
        np.testing.assert_almost_equal(
            res, np.array([[0.00475812, -0.00505504, -0.000717714, 0.00215864, 0.00215864, -0.00159915]]).T
        )
    else:
        np.testing.assert_almost_equal(
            res, np.array([[-0.475812, 0.505504, 0.0717714, -0.215864, -0.215864, 0.159915]]).T, decimal=5
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_qddot(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [1]
    x = [DM.ones((8, 1)) * value, DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    if penalty_origin == ConstraintFcn:
        with pytest.raises(AttributeError, match="MINIMIZE_QDDOT"):
            _ = penalty_origin.MINIMIZE_QDDOT
        return
    else:
        penalty_type = penalty_origin.MINIMIZE_QDDOT
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s).T

    np.testing.assert_almost_equal(res, [[value, -9.81 + value, value, value]])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_state(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_STATE
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="qdot", target=np.ones((4, 1)) * value)
    else:
        penalty = Constraint(penalty_type, key="qdot", target=np.ones((4, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)
    np.testing.assert_almost_equal(res, [[value]] * 4)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_joint_power(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [1]
    p = []
    s = []
    penalty_type = penalty_origin.TRACK_POWER
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key_control="tau")
    else:
        penalty = Constraint(penalty_type, key_control="tau")
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)
    np.testing.assert_almost_equal(res, [[value]] * 4)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.MINIMIZE_MARKERS
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.MINIMIZE_MARKERS_VELOCITY
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("implicit", [True, False])
def test_penalty_minimize_markers_acceleration(penalty_origin, implicit, value, phase_dynamics):
    ocp = prepare_test_ocp(implicit=implicit, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = [0]
    s = []
    penalty_type = penalty_origin.MINIMIZE_MARKERS_ACCELERATION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    if not implicit:
        res = get_penalty_value(ocp, penalty, t, x, u, p, s)

        expected = np.array(
            [
                [2.15106e-16, -0.00895171, -0.0189017, -0.00995004, 00, 00, 00, -0.00497502],
                [0, 0, 0, 0, 00, 00, 00, 0],
                [-9.81, -9.79905, -9.79805, -9.809, 00, 00, 00, 0.000499167],
            ]
        )
        if value == -10:
            expected = np.array(
                [
                    [0.0, 138.309264, 222.2164169, 83.90715291, 0.0, 0.0, 0.0, 41.95357645],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-9.81, -39.31504182, 15.08706927, 44.59211109, 0.0, 0.0, 0.0, 27.20105554],
                ]
            )

        np.testing.assert_almost_equal(res, expected, decimal=5)
    else:
        res = get_penalty_value(ocp, penalty, t, x, u, p, s)

        expected = np.array(
            [
                [2.15105711e-16, -8.95170749e-03, -1.89017491e-02, -9.95004165e-03, 00, 00, 00, -4.97502083e-03],
                [0, 0, 0, 0, 00, 00, 00, 0],
                [-9.81, -9.79905162e00, -9.79805329e00, -9.80900167e00, 00, 00, 00, 4.99167083e-04],
            ]
        )
        if value == -10:
            expected = np.array(
                [
                    [0.0, 138.309264, 222.2164169, 83.90715291, 0.0, 0.0, 0.0, 41.95357645],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-9.81, -39.31504182, 15.08706927, 44.59211109, 0.0, 0.0, 0.0, 27.20105554],
                ]
            )
        np.testing.assert_almost_equal(
            res,
            expected,
            decimal=5,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_MARKERS_VELOCITY

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("implicit", [True, False])
def test_penalty_track_markers_acceleration(penalty_origin, value, implicit, phase_dynamics):
    ocp = prepare_test_ocp(implicit=implicit, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_MARKERS_ACCELERATION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)

    if not implicit:
        res = get_penalty_value(ocp, penalty, t, x, u, [], [])

        expected = np.array(
            [
                [2.15106e-16, -0.00895171, -0.0189017, -0.00995004, 00, 00, 00, -0.00497502],
                [0, 0, 0, 0, 00, 00, 00, 0],
                [-9.81, -9.79905, -9.79805, -9.809, 00, 00, 00, 0.000499167],
            ]
        )
        if value == -10:
            expected = np.array(
                [
                    [0.0, 138.309264, 222.2164169, 83.90715291, 0.0, 0.0, 0.0, 41.95357645],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-9.81, -39.31504182, 15.08706927, 44.59211109, 0.0, 0.0, 0.0, 27.20105554],
                ]
            )

        np.testing.assert_almost_equal(res, expected, decimal=5)
    else:
        res = get_penalty_value(ocp, penalty, t, x, u, [], [])

        expected = np.array(
            [
                [2.15105711e-16, -8.95170749e-03, -1.89017491e-02, -9.95004165e-03, 00, 00, 00, -4.97502083e-03],
                [0, 0, 0, 0, 00, 00, 00, 0],
                [-9.81, -9.79905162e00, -9.79805329e00, -9.80900167e00, 00, 00, 00, 4.99167083e-04],
            ]
        )
        if value == -10:
            expected = np.array(
                [
                    [0.0, 138.309264, 222.2164169, 83.90715291, 0.0, 0.0, 0.0, 41.95357645],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-9.81, -39.31504182, 15.08706927, 44.59211109, 0.0, 0.0, 0.0, 27.20105554],
                ]
            )
        np.testing.assert_almost_equal(
            res,
            expected,
            decimal=5,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_super_impose_marker(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.SUPERIMPOSE_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, first_marker=0, second_marker=1)
    else:
        penalty = Constraint(penalty_type, first_marker=0, second_marker=1)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = [[0.8951707, 0, -1.0948376]] if value == 0.1 else [[-1.3830926, 0, 0.2950504]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_super_impose_marker_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.SUPERIMPOSE_MARKERS_VELOCITY

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, first_marker=0, second_marker=1)
    else:
        penalty = Constraint(penalty_type, first_marker=0, second_marker=1)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = [[-0.1094838, 0.0, -0.0895171]] if value == 0.1 else [[-2.9505042, 0.0, -13.8309264]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("value_intercept", [0.0, 1.0])
def test_penalty_proportional_state(penalty_origin, value, value_intercept, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

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
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    if value_intercept == 0.0:
        np.testing.assert_almost_equal(res, -value)
    else:
        if value == 0.1:
            np.testing.assert_almost_equal(res, 0.9)
        else:
            np.testing.assert_almost_equal(res, 11)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_proportional_control(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    penalty_type = penalty_origin.PROPORTIONAL_CONTROL

    first = 0
    second = 1
    coef = 2

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="tau", first_dof=first, second_dof=second, coef=coef)
    else:
        penalty = Constraint(penalty_type, key="tau", first_dof=first, second_dof=second, coef=coef)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(res, np.array(u[0][first] - coef * u[0][second]))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_torque(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    penalty = Objective(penalty_origin.MINIMIZE_CONTROL, key="tau")
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(res, np.array([[value, value, value, value]]).T)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_torque(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_CONTROL

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="tau", target=np.ones((4, 1)) * value)
    else:
        penalty = Constraint(penalty_type, key="tau", target=np.ones((4, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(res, np.array([[value, value, value, value]]).T)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_muscles_control(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_muscles=True, phase_dynamics=phase_dynamics)
    t = [0]
    x = [0]
    u = [DM.ones((8, 1)) * value]
    p = []
    s = []

    penalty_type = penalty_origin.MINIMIZE_CONTROL
    penalty = Objective(penalty_type, key="muscles")
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(res, np.array([[value, value, value, value, value, value]]).T)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_contact_forces(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    penalty_type = penalty_origin.MINIMIZE_CONTACT_FORCES
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    if value == 0.1:
        np.testing.assert_almost_equal(res, np.array([[-9.6680105, 127.2360329, 5.0905995]]).T)
    else:
        np.testing.assert_almost_equal(res, np.array([[25.6627161, 462.7973306, -94.0182191]]).T)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_contact_forces(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_CONTACT_FORCES

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((1, 1)) * value, index=0)
    else:
        penalty = Constraint(penalty_type, target=np.ones((1, 1)) * value, index=0)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    if value == 0.1:
        np.testing.assert_almost_equal(res.T, [[-9.6680105, 127.2360329, 5.0905995]])
    else:
        np.testing.assert_almost_equal(res.T, [[25.6627161, 462.7973306, -94.0182191]])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_predicted_com_height(value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = np.array(0.0501274 if value == 0.1 else -3.72579)
    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_com_position(value, penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    if "TRACK_COM_POSITION" in penalty_origin._member_names_:
        penalty_type = penalty_origin.TRACK_COM_POSITION
    else:
        penalty_type = penalty_origin.MINIMIZE_COM_POSITION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = np.array([[0.05], [0.05], [0.05]])
    if value == -10:
        expected = np.array([[-5], [0.05], [-5]])

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_angular_momentum(value, penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.MINIMIZE_ANGULAR_MOMENTUM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = np.array([[-0.005], [0.2], [0.005]])
    if value == -10:
        expected = np.array([[0.5], [-20], [-0.5]])

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("use_sx", [True, False])
def test_penalty_minimize_linear_momentum(value, penalty_origin, use_sx, phase_dynamics):
    ocp = prepare_test_ocp(use_sx=use_sx, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.MINIMIZE_LINEAR_MOMENTUM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = np.array([[0.1], [0], [0.1]])
    if value == -10:
        expected = np.array([[-10], [0], [-10]])

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("implicit", [True, False])
def test_penalty_minimize_comddot(value, penalty_origin, implicit, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, implicit=implicit, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.MINIMIZE_COM_ACCELERATION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    if not implicit:
        res = get_penalty_value(ocp, penalty, t, x, u, p, s)

        expected = np.array([[0.0], [-0.7168803], [-0.0740871]])
        if value == -10:
            expected = np.array([[0.0], [1.455063], [16.3741091]])

        np.testing.assert_almost_equal(res, expected)
    else:
        res = get_penalty_value(ocp, penalty, t, x, u, [], [])

        expected = np.array([[0], [-0.0008324], [0.002668]])
        if value == -10:
            expected = np.array([[0], [-17.5050533], [-18.2891901]])

        np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_segment_with_custom_rt(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_SEGMENT_WITH_CUSTOM_RT

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, segment="ground", rt=0)
    else:
        penalty = Constraint(penalty_type, segment="ground", rt=0)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = np.array([[0], [0.1], [0]])
    if value == -10:
        expected = np.array([[3.1415927], [0.575222], [3.1415927]])

    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_marker_with_segment_axis(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_MARKER_WITH_SEGMENT_AXIS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, marker="m0", segment="ground", axis=Axis.X)
    else:
        penalty = Constraint(penalty_type, marker="m0", segment="ground", axis=Axis.X)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = [[value, 0, value]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_segment_rotation(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    if penalty_origin == ObjectiveFcn.Lagrange or penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = penalty_origin.MINIMIZE_SEGMENT_ROTATION
        penalty = Objective(penalty_type, segment=2)
    else:
        penalty_type = penalty_origin.TRACK_SEGMENT_ROTATION
        penalty = Constraint(penalty_type, segment=2)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = [[0, value, 0]] if value == 0.1 else [[3.1415927, 0.575222, 3.1415927]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_segment_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    if penalty_origin == ObjectiveFcn.Lagrange or penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = penalty_origin.MINIMIZE_SEGMENT_VELOCITY
        penalty = Objective(penalty_type, segment=2)
    else:
        penalty_type = penalty_origin.TRACK_SEGMENT_VELOCITY
        penalty = Constraint(penalty_type, segment=2)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = [[0, value, 0]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_state_continuity(penalty_origin, value, phase_dynamics):
    ocp = prepare_multinode_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x1 = DM.zeros((16, 1))
    x1[8:, :] = 1.0 * value
    x2 = DM.ones((16, 1)) * value
    x2[8:, :] = 0
    u = [0, 0]
    p = []
    s = []

    penalty_type = MultinodeConstraintFcn.STATES_EQUALITY
    penalty = MultinodeConstraintList()
    penalty.add(
        penalty_type,
        min_bound=0.01,
        max_bound=20,
        nodes_phase=(0, 1),
        nodes=(Node.START, Node.START),
    )

    res = get_penalty_value(ocp, penalty[0], t, [x1, x2], u, p, s)

    if value == 0.1:
        np.testing.assert_almost_equal(np.array(res.T)[0], np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]))
    else:
        np.testing.assert_almost_equal(np.array(res.T)[0], np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_controls_equality(value, phase_dynamics):
    ocp = prepare_multinode_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [0, 0]
    u1 = DM.zeros((8, 1))
    u1[4:, :] = 1.0 * value
    u2 = DM.ones((8, 1)) * value
    u2[4:, :] = 0
    p = []
    s = []

    penalty_type = MultinodeConstraintFcn.CONTROLS_EQUALITY
    penalty = MultinodeConstraintList()
    penalty.add(
        penalty_type,
        min_bound=0.01,
        max_bound=20,
        nodes_phase=(0, 1),
        nodes=(Node.START, Node.START),
    )

    res = get_penalty_value(ocp, penalty[0], t, x, [u1, u2], p, s)
    if value == 0.1:
        np.testing.assert_almost_equal(np.array(res.T)[0], [-0.1, -0.1, -0.1, -0.1])
    else:
        np.testing.assert_almost_equal(np.array(res.T)[0], [10.0, 10.0, 10.0, 10.0])


# TODO stochastic
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.ONE_PER_NODE])  # PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_stochastic_equality(value, phase_dynamics):
#     ocp = prepare_multinode_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = [0, 0]
#     u = [0, 0]
#     p = []
#     s = [2]
#
#     penalty_type = MultinodeConstraintFcn.STOCHASTIC_EQUALITY
#     penalty = MultinodeConstraintList()
#     penalty.add(
#         penalty_type,
#         min_bound=0.01,
#         max_bound=20,
#         nodes_phase=(0, 1),
#         nodes=(Node.START, Node.START),
#     )
#
#     res = get_penalty_value(ocp, penalty[0], t, x, u, p, s)
#
#     np.testing.assert_almost_equal(res.T, np.array(1))

# TODO uses sx but doesnt work with sx (according to Benjamin it uses cx)
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_com_equality(value, phase_dynamics):
#     ocp = prepare_multinode_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = [0, 0]
#     u1 = DM.zeros((8, 1))
#     u1[4:, :] = 1.0 * value
#     u2 = DM.ones((8, 1)) * value
#     u2[4:, :] = 0
#     p = []
#     s = []
#
#     penalty_type = MultinodeConstraintFcn.COM_EQUALITY
#     penalty = MultinodeConstraintList()
#     penalty.add(
#         penalty_type,
#         min_bound=0.01,
#         max_bound=20,
#         nodes_phase=(0, 1),
#         nodes=(Node.START, Node.START),
#     )
#
#     res = get_penalty_value(ocp, penalty[0], t, x, [u1, u2], p, s)
#
#     np.testing.assert_almost_equal(res.T, np.array(1))

# TODO uses sx but doesnt work with sx (according to Benjamin it uses cx)
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_com_velocity_equality(value, phase_dynamics):
#     ocp = prepare_multinode_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = [0, 0]
#     u1 = DM.zeros((8, 1))
#     u1[4:, :] = 1.0 * value
#     u2 = DM.ones((8, 1)) * value
#     u2[4:, :] = 0
#     p = []
#     s = []
#
#     penalty_type = MultinodeConstraintFcn.COM_VELOCITY_EQUALITY
#     penalty = MultinodeConstraintList()
#     penalty.add(
#         penalty_type,
#         min_bound=0.01,
#         max_bound=20,
#         nodes_phase=(0, 1),
#         nodes=(Node.START, Node.START),
#     )
#
#     res = get_penalty_value(ocp, penalty[0], t, x, [u1, u2], p, s)
#     if value == 0.1:
#         np.testing.assert_almost_equal(np.array(res.T)[0], [-0.1, -0.1, -0.1, -0.1])
#     else:
#         np.testing.assert_almost_equal(np.array(res.T)[0], [10., 10., 10., 10.])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_vector_orientation(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM(np.array([0, 0, value, 0, 0, 0, 0, 0]))]
    u = [0]
    p = []
    s = []

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

    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    if value == 0.1:
        np.testing.assert_almost_equal(float(res), 0.09999999999999999)
    else:
        np.testing.assert_almost_equal(float(res), 2.566370614359173)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_vector_orientation(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_ANGULAR_MOMENTUM

    penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)
    if value == 0.1:
        np.testing.assert_almost_equal(np.array(res.T)[0], np.array([-0.005, 0.2, 0.005]))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("implicit", [True, False])
def test_penalty_track_com_velocity(value, penalty_origin, implicit, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, implicit=implicit, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = ConstraintFcn.TRACK_COM_VELOCITY

    penalty = Constraint(penalty_type)

    if implicit:
        res = get_penalty_value(ocp, penalty, t, x, u, p, s)
        expected = np.array([0, 0.12724064, 0.10555671])
        if value == -10:
            expected = np.array([0.0, -7.9133082, -11.63528216])
        np.testing.assert_almost_equal(np.array(res.T)[0], expected)
    else:
        res = get_penalty_value(ocp, penalty, t, x, u, [], [])

        expected = np.array([0.0, 0.1272406, 0.1055567])
        if value == -10:
            expected = np.array([0.0, -7.91330821, -11.63528216])

        np.testing.assert_almost_equal(np.array(res.T)[0], expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_linear_momentum(value, penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = ConstraintFcn.TRACK_LINEAR_MOMENTUM

    penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = np.array([0.1, 0, 0.1])
    if value == -10:
        expected = np.array([-10, 0, -10])

    np.testing.assert_almost_equal(np.array(res.T)[0], expected)


# TODO Allow free problem
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_first_collocation_helper_equals_state(value, penalty_origin, phase_dynamics):
#     ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = [DM.ones((8, 1)) * value]
#     x[0][4:, :] = 0
#     u = [0]
#     p = [1]
#     s = []
#
#     penalty_type = ConstraintFcn.FIRST_COLLOCATION_HELPER_EQUALS_STATE
#
#     penalty = Constraint(penalty_type)
#     res = get_penalty_value(ocp, penalty, t, x, u, p, s)
#
#     np.testing.assert_almost_equal(np.array(res.T)[0], np.array(1))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_parameter(value, penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    x[0][4:, :] = 0
    u = [0]
    p = [1]
    s = []

    penalty_type = ConstraintFcn.TRACK_PARAMETER

    penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(np.array(res.T)[0], np.array(1))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_qddot(value, penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    x[0][4:, :] = 0
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    penalty_type = ConstraintFcn.TRACK_QDDOT

    penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)
    expected = np.array([value, -9.71, value, value])

    if value == -10:
        expected = np.array([value, -19.81, value, value])
    np.testing.assert_almost_equal(np.array(res.T)[0], expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_com_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.MINIMIZE_COM_VELOCITY

    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = np.array([0.05, 0, 0.05]) if value == 0.1 else np.array([-5.0, 0.0, -5.0])
    np.testing.assert_almost_equal(np.array(res.T)[0], expected)


# TODO add width fatigue to prepare_ocp
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_minimize_fatigue(penalty_origin, value, phase_dynamics):
#     ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = [DM.ones((8, 1)) * value]
#     u = [0]
#     p = []
#     s = []
#
#     penalty_type = penalty_origin.MINIMIZE_FATIGUE
#
#     penalty = Objective(penalty_type, key='tau')
#     res = get_penalty_value(ocp, penalty, t, x, u, p, s)
#
#     expected = np.array([0.05, 0, 0.05]) if value == 0.1 else np.array([-5., 0., -5.])
#     np.testing.assert_almost_equal(np.array(res.T)[0], expected)

# TODO add width soft contact to prepare_ocp
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_minimize_soft_contact_force(penalty_origin, value, phase_dynamics):
#     ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = DM.ones((8, 1)) * value
#     x[4:, :] = 0
#     u = DM.ones((4, 1)) * value
#     p = []
#     s = []
#
#     penalty_type = ObjectiveFcn.Lagrange.MINIMIZE_SOFT_CONTACT_FORCES
#
#     penalty = Objective(penalty_type)
#     res = get_penalty_value(ocp, penalty, t, x, u, p, s)
#
#     expected = np.array([0.05, 0, 0.05]) if value == 0.1 else np.array([-5., 0., -5.])
#     np.testing.assert_almost_equal(np.array(res.T)[0], 0)

# TODO add width soft contact to prepare_ocp
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_track_soft_contact_forces(penalty_origin, value, phase_dynamics):
#     ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = DM.ones((8, 1)) * value
#     x[4:, :] = 0
#     u = DM.ones((4, 1)) * value
#     p = []
#     s = []
#
#     penalty_type = ObjectiveFcn.Lagrange.TRACK_SOFT_CONTACT_FORCES
#
#     penalty = Objective(penalty_type)
#     res = get_penalty_value(ocp, penalty, t, x, u, p, s)
#
#     expected = np.array([0.05, 0, 0.05]) if value == 0.1 else np.array([-5., 0., -5.])
#     np.testing.assert_almost_equal(np.array(res.T)[0], 0)
# TODO stochastic
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_stochastic_minimize_variable(penalty_origin, value, phase_dynamics):
#     ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = DM.ones((8, 1)) * value
#     x[4:, :] = 0
#     u = DM.ones((4, 1)) * value
#     p = []
#     s = []
#
#     penalty_type = penalty_origin.STOCHASTIC_MINIMIZE_VARIABLE
#
#
#     penalty = Objective(penalty_type, key="tau")
#     res = get_penalty_value(ocp, penalty, t, x, u, p, s)
#
#     expected = np.array([0.05, 0, 0.05]) if value == 0.1 else np.array([-5., 0., -5.])
#     np.testing.assert_almost_equal(np.array(res.T)[0], 0)
# TODO stochastic


# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_stochastic_minimize_expected_feedback_efforts(penalty_origin, value, phase_dynamics):
#     ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x = DM.ones((8, 1)) * value
#     x[4:, :] = 0
#     u = DM.ones((4, 1)) * value
#     p = []
#     s = []
#
#     penalty_type = penalty_origin.STOCHASTIC_MINIMIZE_EXPECTED_FEEDBACK_EFFORTS
#
#
#
#     penalty = Objective(penalty_type)
#     res = get_penalty_value(ocp, penalty, t, x, u, p, s)
#
#     expected = np.array([0.05, 0, 0.05]) if value == 0.1 else np.array([-5., 0., -5.])
#     np.testing.assert_almost_equal(np.array(res.T)[0], 0)
@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_contact_force_inequality(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    penalty_type = penalty_origin.TRACK_CONTACT_FORCES
    penalty = Constraint(penalty_type, contact_index=0)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = [[-9.6680105, 127.2360329, 5.0905995]] if value == 0.1 else [[25.6627161, 462.7973306, -94.0182191]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_non_slipping(value, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    s = []

    penalty_type = ConstraintFcn.NON_SLIPPING
    penalty = Constraint(
        penalty_type, tangential_component_idx=0, normal_component_idx=1, static_friction_coefficient=2
    )
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    expected = [[64662.56185612, 64849.5027121]] if value == 0.1 else [[856066.90177734, 857384.05177395]]
    np.testing.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [2])
@pytest.mark.parametrize("threshold", [None, 15, -15])
def test_tau_max_from_actuators(value, threshold, phase_dynamics):
    ocp = prepare_test_ocp(with_actuator=True, phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.zeros((6, 1)), DM.zeros((6, 1))]
    u = [DM.ones((3, 1)) * value, DM.ones((3, 1)) * value]
    p = []
    s = []

    penalty_type = ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT
    penalty = Constraint(penalty_type, min_torque=threshold)
    if threshold and threshold < 0:
        with pytest.raises(ValueError, match="min_torque cannot be negative in tau_max_from_actuators"):
            get_penalty_value(ocp, penalty, t, x, u, p, s)
        return
    else:
        res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    if threshold:
        np.testing.assert_almost_equal(res, np.repeat([value + threshold, value - threshold], 3)[:, np.newaxis])
    else:
        np.testing.assert_almost_equal(res, np.repeat([value + 5, value - 10], 3)[:, np.newaxis])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_time_constraint(value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [0]
    u = [0]
    p = [0]
    s = []

    penalty_type = ConstraintFcn.TIME_CONSTRAINT
    penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(res, np.array(0))


# TODO doesn't work because it doesnt have a time parameter
# @pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
# @pytest.mark.parametrize("value", [0.1, -10])
# def test_penalty_constraint_total_time(value, phase_dynamics):
#     ocp = prepare_multinode_test_ocp(phase_dynamics=phase_dynamics)
#     t = [0]
#     x1 = [DM.ones((8, 1)) * value]
#     x2 = [DM.zeros((8, 1)) * value]
#     u = [0, 0]
#     p = [0.1]
#     s = []
#
#     penalty_type = MultinodeConstraintFcn.TRACK_TOTAL_TIME
#     penalty = MultinodeConstraintList()
#     penalty.add(
#         penalty_type,
#         min_bound=0.01,
#         max_bound=20,
#         nodes_phase=(0, 1),
#         nodes=(Node.END, Node.END),
#     )
#     res = get_penalty_value(ocp, penalty[0], t, [x1, x2], u, p, s)
#
#     np.testing.assert_almost_equal(res, np.array(0.2))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom(penalty_origin, value, phase_dynamics):
    def custom(controller: PenaltyController, mult):
        my_values = controller.states["q"].cx_start * mult
        return my_values

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = penalty_origin.CUSTOM

    mult = 2
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(custom, index=0, mult=mult, custom_type=penalty_origin)
    else:
        penalty = Constraint(custom, index=0, mult=mult)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(res, [[value * mult]] * 4)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_fail(penalty_origin, value, phase_dynamics):
    def custom_no_mult(ocp, nlp, t, x, u, p):
        my_values = DM.zeros((12, 1)) + x[0]
        return my_values

    def custom_with_mult(ocp, nlp, t, x, u, p, mult):
        my_values = DM.zeros((12, 1)) + x[0] * mult
        return my_values

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds(value, phase_dynamics):
    def custom_with_bounds(controller: PenaltyController):
        return -10, controller.states["q"].cx_start, 10

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty = Constraint(custom_with_bounds)
    res = get_penalty_value(ocp, penalty, t, x, u, p, s)

    np.testing.assert_almost_equal(res, [[value]] * 4)
    np.testing.assert_almost_equal(penalty.min_bound, -10)
    np.testing.assert_almost_equal(penalty.max_bound, 10)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds_failing_min_bound(value, phase_dynamics):
    def custom_with_bounds(controller: PenaltyController):
        return -10, controller.states["q"].cx_start, 10

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.min_bound = 0
    penalty.custom_function = custom_with_bounds

    with pytest.raises(RuntimeError):
        penalty_type(penalty, PenaltyController(ocp, ocp.nlp[0], t, x, [], [], [], p, s, [], 0))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds_failing_max_bound(value, phase_dynamics):
    def custom_with_bounds(controller: PenaltyController):
        return -10, controller.states["q"].cx_start, 10

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    p = []
    s = []

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.max_bound = 0
    penalty.custom_function = custom_with_bounds

    with pytest.raises(
        RuntimeError,
        match="You cannot have non linear bounds for custom constraints and min_bound or max_bound defined",
    ):
        penalty_type(penalty, PenaltyController(ocp, ocp.nlp[0], t, x, [], [], [], p, s, [], 0))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("node", [*Node, 2])
@pytest.mark.parametrize("ns", [3, 10, 11])
def test_PenaltyFunctionAbstract_get_node(node, ns, phase_dynamics):
    nlp = NLP(phase_dynamics=phase_dynamics)
    nlp.control_type = ControlType.CONSTANT
    nlp.ns = ns
    nlp.X = np.linspace(0, -10, ns + 1)
    nlp.U = np.linspace(10, 19, ns)
    nlp.X_scaled = nlp.X
    nlp.U_scaled = nlp.U
    nlp.S = np.linspace(0, 0, ns + 1)
    nlp.S_scaled = nlp.S
    tp = OptimizationVariableList(MX, phase_dynamics=phase_dynamics)
    tp.append(name="param", cx=[MX(), MX(), MX()], mx=MX(), bimapping=BiMapping([], []))
    nlp.parameters = tp["param"]

    pn = []
    penalty = PenaltyOption(pn)
    penalty.node = node

    if node == Node.MID and ns % 2 != 0:
        with pytest.raises(ValueError, match="Number of shooting points must be even to use MID"):
            _ = penalty._get_penalty_controller([], nlp)
        return
    elif node == Node.TRANSITION:
        with pytest.raises(RuntimeError, match="Node.TRANSITION is not a valid node"):
            _ = penalty._get_penalty_controller([], nlp)
        return
    elif node == Node.MULTINODES:
        with pytest.raises(RuntimeError, match="Node.MULTINODES is not a valid node"):
            _ = penalty._get_penalty_controller([], nlp)
        return
    elif node == Node.DEFAULT:
        with pytest.raises(RuntimeError, match="Node.DEFAULT is not a valid node"):
            _ = penalty._get_penalty_controller([], nlp)
        return
    elif ns == 1 and node == Node.PENULTIMATE:
        with pytest.raises(ValueError, match="Number of shooting points must be greater than 1"):
            _ = penalty._get_penalty_controller([], nlp)
        return
    else:
        controller = penalty._get_penalty_controller([], nlp)

    x_expected = nlp.X
    u_expected = nlp.U

    if node == Node.ALL:
        np.testing.assert_almost_equal(controller.t, [i for i in range(ns + 1)])
        np.testing.assert_almost_equal(np.array(controller.x), np.linspace(0, -10, ns + 1))
        np.testing.assert_almost_equal(np.array(controller.u), np.linspace(10, 19, ns))
        np.testing.assert_almost_equal(np.array(controller.x_scaled), np.linspace(0, -10, ns + 1))
        np.testing.assert_almost_equal(np.array(controller.u_scaled), np.linspace(10, 19, ns))
    elif node == Node.ALL_SHOOTING:
        np.testing.assert_almost_equal(controller.t, [i for i in range(ns)])
        np.testing.assert_almost_equal(np.array(controller.x), nlp.X[:-1])
        np.testing.assert_almost_equal(np.array(controller.u), nlp.U)
        np.testing.assert_almost_equal(np.array(controller.x_scaled), nlp.X[:-1])
        np.testing.assert_almost_equal(np.array(controller.u_scaled), nlp.U)
    elif node == Node.INTERMEDIATES:
        np.testing.assert_almost_equal(controller.t, [i for i in range(1, ns - 1)])
        np.testing.assert_almost_equal(np.array(controller.x), x_expected[1 : ns - 1])
        np.testing.assert_almost_equal(np.array(controller.u), u_expected[1 : ns - 1])
        np.testing.assert_almost_equal(np.array(controller.x_scaled), x_expected[1 : ns - 1])
        np.testing.assert_almost_equal(np.array(controller.u_scaled), u_expected[1 : ns - 1])
    elif node == Node.START:
        np.testing.assert_almost_equal(controller.t, [0])
        np.testing.assert_almost_equal(np.array(controller.x), x_expected[0])
        np.testing.assert_almost_equal(np.array(controller.u), u_expected[0])
        np.testing.assert_almost_equal(np.array(controller.x_scaled), x_expected[0])
        np.testing.assert_almost_equal(np.array(controller.u_scaled), u_expected[0])
    elif node == Node.MID:
        np.testing.assert_almost_equal(controller.t, [ns // 2])
        np.testing.assert_almost_equal(np.array(controller.x), x_expected[ns // 2])
        np.testing.assert_almost_equal(np.array(controller.u), u_expected[ns // 2])
        np.testing.assert_almost_equal(np.array(controller.x_scaled), x_expected[ns // 2])
        np.testing.assert_almost_equal(np.array(controller.u_scaled), u_expected[ns // 2])
    elif node == Node.PENULTIMATE:
        np.testing.assert_almost_equal(controller.t, [ns - 1])
        np.testing.assert_almost_equal(np.array(controller.x), x_expected[-2])
        np.testing.assert_almost_equal(np.array(controller.u), u_expected[-1])
        np.testing.assert_almost_equal(np.array(controller.x_scaled), x_expected[-2])
        np.testing.assert_almost_equal(np.array(controller.u_scaled), u_expected[-1])
    elif node == Node.END:
        np.testing.assert_almost_equal(controller.t, [ns])
        np.testing.assert_almost_equal(np.array(controller.x), x_expected[ns])
        np.testing.assert_almost_equal(controller.u, [])
        np.testing.assert_almost_equal(np.array(controller.x_scaled), x_expected[ns])
        np.testing.assert_almost_equal(controller.u_scaled, [])
    elif node == 2:
        np.testing.assert_almost_equal(controller.t, [2])
        np.testing.assert_almost_equal(np.array(controller.x), x_expected[2])
        np.testing.assert_almost_equal(controller.u, u_expected[2])
        np.testing.assert_almost_equal(np.array(controller.x_scaled), x_expected[2])
        np.testing.assert_almost_equal(controller.u_scaled, u_expected[2])
    else:
        raise RuntimeError("Something went wrong")


# TODO what is left:
# Constraint:
# STOCHASTIC_DF_DX_IMPLICIT
# STOCHASTIC_HELPER_MATRIX_COLLOCATION
# STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION
# STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE

# Lagrange:
# MINIMIZE_FATIGUE (already a written but not working)
# TRACK_SOFT_CONTACT_FORCES (already a written but not working)
# STOCHASTIC_MINIMIZE_EXPECTED_FEEDBACK_EFFORTS (already a written but not working)

# Mayer and Lagrange:
# STOCHASTIC_MINIMIZE_VARIABLE (already a written but not working)

# Multinode:
# COM_EQUALITY (already a written but not working)
# COM_VELOCITY_EQUALITY (already a written but not working)
# STOCHASTIC_EQUALITY (already a written but not working)
# STOCHASTIC_HELPER_MATRIX_EXPLICIT
# STOCHASTIC_HELPER_MATRIX_IMPLICIT
# STOCHASTIC_DF_DW_IMPLICIT

# Multinode constraint:
# STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_IMPLICIT
