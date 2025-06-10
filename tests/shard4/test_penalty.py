import numpy as np
import numpy.testing as npt
import pytest
from casadi import DM, MX, vertcat, horzcat, Function

from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
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
    ControlType,
    PhaseDynamics,
    ConstraintList,
    ExternalForceSetTimeSeries,
    ContactType,
    MusclesBiorbdModel,
    TorqueActivationBiorbdModel,
)
from bioptim.limits.penalty import PenaltyOption
from bioptim.limits.penalty_controller import PenaltyController
from bioptim.misc.mapping import BiMapping
from bioptim.optimization.non_linear_program import NonLinearProgram as NLP
from bioptim.optimization.optimization_variable import OptimizationVariableList
from tests.utils import TestUtils


N_SHOOTING = 10
EXTERNAL_FORCE_ARRAY = np.zeros((9, N_SHOOTING))
EXTERNAL_FORCE_ARRAY[:, 0] = [
    0.374540118847362,
    0.950714306409916,
    0.731993941811405,
    0.598658484197037,
    0.156018640442437,
    0.155994520336203,
    0,
    0,
    0,
]
EXTERNAL_FORCE_ARRAY[:, 1] = [
    0.058083612168199,
    0.866176145774935,
    0.601115011743209,
    0.708072577796045,
    0.020584494295802,
    0.969909852161994,
    0,
    0,
    0,
]
EXTERNAL_FORCE_ARRAY[:, 2] = [
    0.832442640800422,
    0.212339110678276,
    0.181824967207101,
    0.183404509853434,
    0.304242242959538,
    0.524756431632238,
    0,
    0,
    0,
]
EXTERNAL_FORCE_ARRAY[:, 3] = [
    0.431945018642116,
    0.291229140198042,
    0.611852894722379,
    0.139493860652042,
    0.292144648535218,
    0.366361843293692,
    0,
    0,
    0,
]
EXTERNAL_FORCE_ARRAY[:, 4] = [
    0.456069984217036,
    0.785175961393014,
    0.19967378215836,
    0.514234438413612,
    0.592414568862042,
    0.046450412719998,
    0,
    0,
    0,
]


def prepare_test_ocp(
    phase_dynamics,
    with_muscles=False,
    with_contact=False,
    with_actuator=False,
    use_sx=True,
    with_external_forces=False,
):
    bioptim_folder = TestUtils.bioptim_folder()

    if with_external_forces:
        if not with_contact:
            raise NotImplementedError("with_external_forces=True is only tested for with_contact=True")
        external_forces = ExternalForceSetTimeSeries(nb_frames=N_SHOOTING)
        external_forces.add(
            "force0", "Seg0", EXTERNAL_FORCE_ARRAY[:6, :], point_of_application=EXTERNAL_FORCE_ARRAY[6:, :]
        )
        numerical_time_series = {"external_forces": external_forces.to_numerical_time_series()}

    if with_muscles and with_contact or with_muscles and with_actuator or with_contact and with_actuator:
        raise RuntimeError("With muscles and with contact and with_actuator together is not defined")
    elif with_muscles:
        bio_model = MusclesBiorbdModel(
            bioptim_folder + "/examples/muscle_driven_ocp/models/arm26.bioMod", with_residual_torque=True
        )
        dynamics = DynamicsList()
        dynamics.add(expand_dynamics=True, phase_dynamics=phase_dynamics)
    elif with_contact:
        dynamics = DynamicsList()
        if with_external_forces:
            bio_model = MusclesBiorbdModel(
                bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod",
                with_residual_torque=True,
                contact_types=[ContactType.RIGID_EXPLICIT],
                external_force_set=external_forces,
            )
            dynamics.add(
                Dynamics(
                    expand_dynamics=True,
                    phase_dynamics=phase_dynamics,
                    numerical_data_timeseries=numerical_time_series,
                )
            )
        else:
            bio_model = MusclesBiorbdModel(
                bioptim_folder + "/examples/muscle_driven_with_contact/models/2segments_4dof_2contacts_1muscle.bioMod",
                with_residual_torque=True,
                contact_types=[ContactType.RIGID_EXPLICIT],
            )
            dynamics.add(
                Dynamics(
                    expand_dynamics=True,
                    phase_dynamics=phase_dynamics,
                )
            )
    elif with_actuator:
        bio_model = TorqueActivationBiorbdModel(bioptim_folder + "/examples/torque_driven_ocp/models/cube.bioMod")
        dynamics = Dynamics(expand_dynamics=True, phase_dynamics=phase_dynamics)
    else:
        bio_model = TorqueBiorbdModel(bioptim_folder + "/examples/track/models/cube_and_line.bioMod")
        dynamics = DynamicsList(expand_dynamics=True, phase_dynamics=phase_dynamics)

    objective_functions = Objective(ObjectiveFcn.Mayer.MINIMIZE_TIME)

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        N_SHOOTING,
        1.0,
        objective_functions=objective_functions,
        use_sx=use_sx,
    )

    ocp.nlp[0].J = [[]]
    ocp.nlp[0].g = [[]]
    return ocp


def get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d):
    if isinstance(penalty, MultinodeConstraint) or isinstance(penalty, MultinodeObjective):
        controller = [
            PenaltyController(ocp, ocp.nlp[0], t, x, u, [], [], p, a, [], d, 0) for i in range(len(penalty.nodes_phase))
        ]
    else:
        controller = PenaltyController(ocp, ocp.nlp[0], t, x, u, [], [], p, a, [], d, 0)
    val = penalty.type(penalty, controller, **penalty.extra_parameters)
    # changed only this one
    if isinstance(val, float):
        return val

    time = ocp.nlp[0].time_cx
    phases_dt_cx = vertcat(*[nlp.dt for nlp in ocp.nlp])
    states = ocp.nlp[0].states.cx_start if ocp.nlp[0].states.cx_start.shape != (0, 0) else ocp.cx(0, 0)
    controls = ocp.nlp[0].controls.cx_start if ocp.nlp[0].controls.cx_start.shape != (0, 0) else ocp.cx(0, 0)
    parameters = ocp.nlp[0].parameters.cx if ocp.nlp[0].parameters.cx.shape != (0, 0) else ocp.cx(0, 0)
    algebraic_states = (
        ocp.nlp[0].algebraic_states.cx_start if ocp.nlp[0].algebraic_states.cx_start.shape != (0, 0) else ocp.cx(0, 0)
    )
    numerical_timeseries = (
        ocp.nlp[0].numerical_timeseries.cx if ocp.nlp[0].numerical_timeseries.cx.shape != (0, 0) else ocp.cx(0, 0)
    )

    return Function(
        "penalty", [time, phases_dt_cx, states, controls, parameters, algebraic_states, numerical_timeseries], [val]
    )(t, phases_dt, x[0], u[0], p, a, d)


def test_penalty_targets_shapes():
    p = ObjectiveFcn.Parameter
    npt.assert_equal(Objective([], custom_type=p, target=1).target.shape, (1, 1))
    npt.assert_equal(Objective([], custom_type=p, target=np.array(1)).target.shape, (1, 1))
    npt.assert_equal(Objective([], custom_type=p, target=[1]).target.shape, (1, 1))
    npt.assert_equal(Objective([], custom_type=p, target=[1, 2]).target.shape, (2, 1))
    npt.assert_equal(Objective([], custom_type=p, target=[[1], [2]]).target.shape, (2, 1))
    npt.assert_equal(Objective([], custom_type=p, target=[[1, 2]]).target.shape, (1, 2))
    npt.assert_equal(Objective([], custom_type=p, target=np.array([[1, 2]])).target.shape, (1, 2))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_time(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0.05 * ocp.nlp[0].ns]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = [1]
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_TIME
    penalty = Objective(penalty_type)
    penalty_type(penalty, PenaltyController(ocp, ocp.nlp[0], [], [], [], [], [], p, a, [], d, 0))
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    if penalty_origin == ObjectiveFcn.Lagrange:
        npt.assert_almost_equal(res, np.array(1))
    else:
        npt.assert_almost_equal(res, np.array(0.05) * ocp.nlp[0].ns)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_state(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty = Objective(penalty_origin.MINIMIZE_STATE, key="qdot")
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)
    npt.assert_almost_equal(res, np.array([[value]] * 4))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_joint_power(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [1]
    p = []
    a = []
    d = []

    penalty = Objective(penalty_origin.MINIMIZE_POWER, key_control="tau")
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)
    npt.assert_almost_equal(res, np.array([[value]] * 4))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_muscle_power(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_muscles=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [1]
    u = [DM.ones((8, 1)) * value]
    p = []
    a = []
    d = []

    penalty = Objective(penalty_origin.MINIMIZE_POWER, key_control="muscles")
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)
    if value == 0.1:
        npt.assert_almost_equal(
            res, np.array([[0.00475812, -0.00505504, -0.000717714, 0.00215864, 0.00215864, -0.00159915]]).T
        )
    else:
        npt.assert_almost_equal(
            res, np.array([[-0.475812, 0.505504, 0.0717714, -0.215864, -0.215864, 0.159915]]).T, decimal=5
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_qddot(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [1]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value, DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    a = []
    d = []

    if penalty_origin == ConstraintFcn:
        with pytest.raises(AttributeError, match="MINIMIZE_QDDOT"):
            _ = penalty_origin.MINIMIZE_QDDOT
        return
    else:
        penalty_type = penalty_origin.MINIMIZE_QDDOT
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d).T

    npt.assert_almost_equal(res, [[value, -9.81 + value, value, value]])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_state(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_STATE
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="qdot", target=np.ones((4, 1)) * value)
    else:
        penalty = Constraint(penalty_type, key="qdot", target=np.ones((4, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)
    npt.assert_almost_equal(res, [[value]] * 4)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_joint_power(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [1]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_POWER
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key_control="tau")
    else:
        penalty = Constraint(penalty_type, key_control="tau")
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)
    npt.assert_almost_equal(res, [[value]] * 4)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_MARKERS
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

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

    npt.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

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

    npt.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_MARKERS_VELOCITY
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    if value == 0.1:
        npt.assert_almost_equal(
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
        npt.assert_almost_equal(
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
def test_penalty_minimize_markers_acceleration(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = [0]
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_MARKERS_ACCELERATION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

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

    npt.assert_almost_equal(res, expected, decimal=5)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_MARKERS_VELOCITY

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    if value == 0.1:
        npt.assert_almost_equal(
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
        npt.assert_almost_equal(
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
def test_penalty_track_markers_acceleration(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_MARKERS_ACCELERATION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)

    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, [], [], [])

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

    npt.assert_almost_equal(res, expected, decimal=5)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_super_impose_marker(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.SUPERIMPOSE_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, first_marker=0, second_marker=1)
    else:
        penalty = Constraint(penalty_type, first_marker=0, second_marker=1)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = [[0.8951707, 0, -1.0948376]] if value == 0.1 else [[-1.3830926, 0, 0.2950504]]
    npt.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_super_impose_marker_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.SUPERIMPOSE_MARKERS_VELOCITY

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, first_marker=0, second_marker=1)
    else:
        penalty = Constraint(penalty_type, first_marker=0, second_marker=1)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = [[-0.1094838, 0.0, -0.0895171]] if value == 0.1 else [[-2.9505042, 0.0, -13.8309264]]
    npt.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("value_intercept", [0.0, 1.0])
def test_penalty_proportional_state(penalty_origin, value, value_intercept, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

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
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    if value_intercept == 0.0:
        npt.assert_almost_equal(res, -value)
    else:
        if value == 0.1:
            npt.assert_almost_equal(res, 0.9)
        else:
            npt.assert_almost_equal(res, 11)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_proportional_control(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.PROPORTIONAL_CONTROL

    first = 0
    second = 1
    coef = 2

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="tau", first_dof=first, second_dof=second, coef=coef)
    else:
        penalty = Constraint(penalty_type, key="tau", first_dof=first, second_dof=second, coef=coef)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res, np.array(u[0][first] - coef * u[0][second]))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_torque(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    p = []
    a = []
    d = []

    penalty = Objective(penalty_origin.MINIMIZE_CONTROL, key="tau")
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res, np.array([[value, value, value, value]]).T)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_torque(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [0]
    u = [DM.ones((4, 1)) * value]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_CONTROL

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, key="tau", target=np.ones((4, 1)) * value)
    else:
        penalty = Constraint(penalty_type, key="tau", target=np.ones((4, 1)) * value)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res, np.array([[value, value, value, value]]).T)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_muscles_control(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_muscles=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [0]
    u = [DM.ones((8, 1)) * value]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_CONTROL
    penalty = Objective(penalty_type, key="muscles")
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res, np.array([[value, value, value, value, value, value]]).T)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_contact_forces(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_EXPLICIT_RIGID_CONTACT_FORCES
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    if value == 0.1:
        npt.assert_almost_equal(res, np.array([[-9.6680105, 127.2360329, 5.0905995]]).T)
    else:
        npt.assert_almost_equal(res, np.array([[25.6627161, 462.7973306, -94.0182191]]).T)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_contact_forces(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_EXPLICIT_RIGID_CONTACT_FORCES

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((1, 1)) * value, index=0)
    else:
        penalty = Constraint(penalty_type, target=np.ones((1, 1)) * value, index=0)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    if value == 0.1:
        npt.assert_almost_equal(res.T, [[-9.6680105, 127.2360329, 5.0905995]])
    else:
        npt.assert_almost_equal(res.T, [[25.6627161, 462.7973306, -94.0182191]])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_predicted_com_height(value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT
    penalty = Objective(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = np.array(0.0501274 if value == 0.1 else -3.72579)
    npt.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_com_position(value, penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    if "TRACK_COM_POSITION" in penalty_origin._member_names_:
        penalty_type = penalty_origin.TRACK_COM_POSITION
    else:
        penalty_type = penalty_origin.MINIMIZE_COM_POSITION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = np.array([[0.05], [0.05], [0.05]])
    if value == -10:
        expected = np.array([[-5], [0.05], [-5]])

    npt.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_angular_momentum(value, penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_ANGULAR_MOMENTUM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = np.array([[-0.005], [0.2], [0.005]])
    if value == -10:
        expected = np.array([[0.5], [-20], [-0.5]])

    npt.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("use_sx", [True, False])
def test_penalty_minimize_linear_momentum(value, penalty_origin, use_sx, phase_dynamics):
    ocp = prepare_test_ocp(use_sx=use_sx, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_LINEAR_MOMENTUM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = np.array([[0.1], [0], [0.1]])
    if value == -10:
        expected = np.array([[-10], [0], [-10]])

    npt.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_comddot(value, penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.MINIMIZE_COM_ACCELERATION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = np.array([[0.0], [-0.7168803], [-0.0740871]])
    if value == -10:
        expected = np.array([[0.0], [1.455063], [16.3741091]])

    npt.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_segment_with_custom_rt(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_SEGMENT_WITH_CUSTOM_RT

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, segment="ground", rt_index=0)
    else:
        penalty = Constraint(penalty_type, segment="ground", rt_index=0)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = np.array([[0], [0.1], [0]])
    if value == -10:
        expected = np.array([[3.1415927], [0.575222], [3.1415927]])

    npt.assert_almost_equal(res, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_marker_with_segment_axis(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_MARKER_WITH_SEGMENT_AXIS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, marker="m0", segment="ground", axis=Axis.X)
    else:
        penalty = Constraint(penalty_type, marker="m0", segment="ground", axis=Axis.X)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = [[value, 0, value]]
    npt.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_segment_rotation(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    if penalty_origin == ObjectiveFcn.Lagrange or penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = penalty_origin.MINIMIZE_SEGMENT_ROTATION
        penalty = Objective(penalty_type, segment=2)
    else:
        penalty_type = penalty_origin.TRACK_SEGMENT_ROTATION
        penalty = Constraint(penalty_type, segment=2)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = [[0, value, 0]] if value == 0.1 else [[3.1415927, 0.575222, 3.1415927]]
    npt.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_segment_velocity(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    if penalty_origin == ObjectiveFcn.Lagrange or penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = penalty_origin.MINIMIZE_SEGMENT_VELOCITY
        penalty = Objective(penalty_type, segment=2)
    else:
        penalty_type = penalty_origin.TRACK_SEGMENT_VELOCITY
        penalty = Constraint(penalty_type, segment=2)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = [[0, value, 0]]
    npt.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_contact_force_inequality(penalty_origin, value, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.TRACK_EXPLICIT_RIGID_CONTACT_FORCES
    penalty = Constraint(penalty_type, contact_index=0)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = [[-9.6680105, 127.2360329, 5.0905995]] if value == 0.1 else [[25.6627161, 462.7973306, -94.0182191]]
    npt.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_non_slipping(value, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    p = []
    a = []
    d = []

    penalty_type = ConstraintFcn.NON_SLIPPING
    penalty = Constraint(
        penalty_type, tangential_component_idx=0, normal_component_idx=1, static_friction_coefficient=2
    )
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    expected = [[64662.56185612, 64849.5027121]] if value == 0.1 else [[856066.90177734, 857384.05177395]]
    npt.assert_almost_equal(res.T, expected)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Mayer, ConstraintFcn])
def test_penalty_minimize_contact_forces_end_of_interval(penalty_origin, phase_dynamics):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * 0.1]
    u = [DM.ones((4, 1)) * 0.1]
    p = []
    a = []
    d = []

    if penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = ObjectiveFcn.Mayer.MINIMIZE_EXPLICIT_RIGID_CONTACT_FORCES_END_OF_INTERVAL
        penalty_object = Objective

    else:
        penalty_type = ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES_END_OF_INTERVAL

        penalty_object = Constraint

    penalty = penalty_object(
        penalty_type,
        contact_index=0,
        node=Node.PENULTIMATE,
    )
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res.T, [[-10.5199265, 126.8712299, 5.0900292]])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Mayer, ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize(
    "with_external_forces",
    [False, True],
)
def test_penalty_minimize_sum_reaction_forces(penalty_origin, phase_dynamics, with_external_forces):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics, with_external_forces=with_external_forces)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * 0.1]
    u = [DM.ones((4, 1)) * 0.1]
    p = []
    a = []
    d = []

    if penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = ObjectiveFcn.Mayer.TRACK_SUM_REACTION_FORCES
        penalty_object = Objective
    elif penalty_origin == ObjectiveFcn.Lagrange:
        penalty_type = ObjectiveFcn.Lagrange.TRACK_SUM_REACTION_FORCES
        penalty_object = Objective
    else:
        penalty_type = ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES_END_OF_INTERVAL
        penalty_object = Constraint

    penalty = penalty_object(
        penalty_type,
        contact_index=[0, 1],
    )
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    if penalty_object == Constraint:
        npt.assert_almost_equal(res.T, [[-10.5199, 126.871, 5.09003]], decimal=2)
    else:
        npt.assert_almost_equal(res.T, [[0, -9.66801, 132.327]], decimal=2)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Mayer, ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize(
    "with_external_forces",
    [False, True],
)
def test_penalty_minimize_center_of_pressure(penalty_origin, phase_dynamics, with_external_forces):
    ocp = prepare_test_ocp(with_contact=True, phase_dynamics=phase_dynamics, with_external_forces=with_external_forces)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * 0.1]
    u = [DM.ones((4, 1)) * 0.1]
    p = []
    a = []
    d = []

    if penalty_origin == ObjectiveFcn.Mayer:
        penalty_type = ObjectiveFcn.Mayer.TRACK_CENTER_OF_PRESSURE
        penalty_object = Objective
    elif penalty_origin == ObjectiveFcn.Lagrange:
        penalty_type = ObjectiveFcn.Lagrange.TRACK_CENTER_OF_PRESSURE
        penalty_object = Objective
    else:
        penalty_type = ConstraintFcn.TRACK_CENTER_OF_PRESSURE
        penalty_object = Constraint

    penalty = penalty_object(
        penalty_type,
        contact_index=[0, 1],
    )
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res[0], 0)
    npt.assert_almost_equal(res[2], -0.394827)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [2])
@pytest.mark.parametrize("threshold", [None, 15, -15])
def test_tau_max_from_actuators(value, threshold, phase_dynamics):
    ocp = prepare_test_ocp(with_actuator=True, phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.zeros((6, 1)), DM.zeros((6, 1))]
    u = [DM.ones((3, 1)) * value, DM.ones((3, 1)) * value]
    p = []
    a = []
    d = []

    penalty_type = ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT
    penalty = Constraint(penalty_type, min_torque=threshold)
    if threshold and threshold < 0:
        with pytest.raises(ValueError, match="min_torque cannot be negative in tau_max_from_actuators"):
            get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)
        return
    else:
        res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    if threshold:
        npt.assert_almost_equal(res, np.repeat([value + threshold, value - threshold], 3)[:, np.newaxis])
    else:
        npt.assert_almost_equal(res, np.repeat([value + 5, value - 10], 3)[:, np.newaxis])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_time_constraint(value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [0]
    u = [0]
    p = [0]
    a = []
    d = []

    penalty_type = ConstraintFcn.TIME_CONSTRAINT
    penalty = Constraint(penalty_type)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res, np.array(0.05) * ocp.nlp[0].ns)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_constraint_total_time(value, phase_dynamics):
    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = [0.1]
    a = []
    d = []

    penalty_type = MultinodeConstraintFcn.TRACK_TOTAL_TIME
    penalty = MultinodeConstraintList()
    penalty.add(
        penalty_type,
        min_bound=0.01,
        max_bound=20,
        nodes_phase=(0, 1),
        nodes=(Node.END, Node.END),
    )
    penalty[0].multinode_idx = (ocp.nlp[0].ns, ocp.nlp[0].ns)

    penalty_type(
        penalty[0],
        [
            PenaltyController(ocp, ocp.nlp[0], [], [], [], [], [], p, a, [], d, 0),
            PenaltyController(ocp, ocp.nlp[0], [], [], [], [], [], p, a, [], d, 0),
        ],
    )
    res = get_penalty_value(ocp, penalty[0], t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res, np.array(0.05) * ocp.nlp[0].ns * 2)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom(penalty_origin, value, phase_dynamics):
    def custom(controller: PenaltyController, mult):
        my_values = controller.q * mult
        return my_values

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty_type = penalty_origin.CUSTOM

    mult = 2
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(custom, index=0, mult=mult, custom_type=penalty_origin)
    else:
        penalty = Constraint(custom, index=0, mult=mult)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res, [[value * mult]] * 4)


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
        return -10, controller.q, 10

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    phases_dt = [0.05]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    p = []
    a = []
    d = []

    penalty = Constraint(custom_with_bounds)
    res = get_penalty_value(ocp, penalty, t, phases_dt, x, u, p, a, d)

    npt.assert_almost_equal(res, [[value]] * 4)
    npt.assert_almost_equal(penalty.min_bound, -10)
    npt.assert_almost_equal(penalty.max_bound, 10)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds_failing_min_bound(value, phase_dynamics):
    def custom_with_bounds(controller: PenaltyController):
        return -10, controller.q, 10

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    p = []
    a = []

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.min_bound = 0
    penalty.custom_function = custom_with_bounds

    with pytest.raises(RuntimeError):
        penalty_type(penalty, PenaltyController(ocp, ocp.nlp[0], t, x, [], [], [], p, a, [], [], 0))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds_failing_max_bound(value, phase_dynamics):
    def custom_with_bounds(controller: PenaltyController):
        return -10, controller.q, 10

    ocp = prepare_test_ocp(phase_dynamics=phase_dynamics)
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    p = []
    a = []

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.max_bound = 0
    penalty.custom_function = custom_with_bounds

    with pytest.raises(
        RuntimeError,
        match="You cannot have non linear bounds for custom constraints and min_bound or max_bound defined",
    ):
        penalty_type(penalty, PenaltyController(ocp, ocp.nlp[0], t, x, [], [], [], p, a, [], [], 0))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("node", [*Node, 2])
@pytest.mark.parametrize("ns", [3, 10, 11])
def test_PenaltyFunctionAbstract_get_node(node, ns, phase_dynamics):
    nlp = NLP(phase_dynamics=phase_dynamics, use_sx=False)
    nlp.control_type = ControlType.CONSTANT
    nlp.ns = ns
    nlp.X = np.linspace(0, -10, ns + 1)
    nlp.U = np.linspace(10, 19, ns)
    nlp.X_scaled = nlp.X
    nlp.U_scaled = nlp.U
    nlp.A = np.linspace(0, 0, ns + 1)
    nlp.A_scaled = nlp.A
    tp = OptimizationVariableList(MX, phase_dynamics=phase_dynamics)
    tp.append(name="param", cx=[MX(), MX(), MX()], bimapping=BiMapping([], []))
    nlp.parameters = tp["param"]

    pn = []
    penalty = PenaltyOption(pn)
    penalty.node = node

    if node == Node.MID and ns % 2 != 0:
        with pytest.raises(ValueError, match="Number of shooting points must be even to use MID"):
            _ = penalty.get_penalty_controller([], nlp)
        return
    elif node == Node.TRANSITION:
        with pytest.raises(RuntimeError, match="Node.TRANSITION is not a valid node"):
            _ = penalty.get_penalty_controller([], nlp)
        return
    elif node == Node.MULTINODES:
        with pytest.raises(RuntimeError, match="Node.MULTINODES is not a valid node"):
            _ = penalty.get_penalty_controller([], nlp)
        return
    elif node == Node.DEFAULT:
        with pytest.raises(RuntimeError, match="Node.DEFAULT is not a valid node"):
            _ = penalty.get_penalty_controller([], nlp)
        return
    elif ns == 1 and node == Node.PENULTIMATE:
        with pytest.raises(ValueError, match="Number of shooting points must be greater than 1"):
            _ = penalty.get_penalty_controller([], nlp)
        return
    else:
        controller = penalty.get_penalty_controller([], nlp)

    x_expected = nlp.X
    u_expected = nlp.U

    if node == Node.ALL:
        npt.assert_almost_equal(controller.t, [i for i in range(ns + 1)])
        npt.assert_almost_equal(np.array(controller.x), np.linspace(0, -10, ns + 1))
        npt.assert_almost_equal(np.array(controller.u), np.linspace(10, 19, ns))
        npt.assert_almost_equal(np.array(controller.x_scaled), np.linspace(0, -10, ns + 1))
        npt.assert_almost_equal(np.array(controller.u_scaled), np.linspace(10, 19, ns))
    elif node == Node.ALL_SHOOTING:
        npt.assert_almost_equal(controller.t, [i for i in range(ns)])
        npt.assert_almost_equal(np.array(controller.x), nlp.X[:-1])
        npt.assert_almost_equal(np.array(controller.u), nlp.U)
        npt.assert_almost_equal(np.array(controller.x_scaled), nlp.X[:-1])
        npt.assert_almost_equal(np.array(controller.u_scaled), nlp.U)
    elif node == Node.INTERMEDIATES:
        npt.assert_almost_equal(controller.t, [i for i in range(1, ns - 1)])
        npt.assert_almost_equal(np.array(controller.x), x_expected[1 : ns - 1])
        npt.assert_almost_equal(np.array(controller.u), u_expected[1 : ns - 1])
        npt.assert_almost_equal(np.array(controller.x_scaled), x_expected[1 : ns - 1])
        npt.assert_almost_equal(np.array(controller.u_scaled), u_expected[1 : ns - 1])
    elif node == Node.START:
        npt.assert_almost_equal(controller.t, [0])
        npt.assert_almost_equal(np.array(controller.x), x_expected[0])
        npt.assert_almost_equal(np.array(controller.u), u_expected[0])
        npt.assert_almost_equal(np.array(controller.x_scaled), x_expected[0])
        npt.assert_almost_equal(np.array(controller.u_scaled), u_expected[0])
    elif node == Node.MID:
        npt.assert_almost_equal(controller.t, [ns // 2])
        npt.assert_almost_equal(np.array(controller.x), x_expected[ns // 2])
        npt.assert_almost_equal(np.array(controller.u), u_expected[ns // 2])
        npt.assert_almost_equal(np.array(controller.x_scaled), x_expected[ns // 2])
        npt.assert_almost_equal(np.array(controller.u_scaled), u_expected[ns // 2])
    elif node == Node.PENULTIMATE:
        npt.assert_almost_equal(controller.t, [ns - 1])
        npt.assert_almost_equal(np.array(controller.x), x_expected[-2])
        npt.assert_almost_equal(np.array(controller.u), u_expected[-1])
        npt.assert_almost_equal(np.array(controller.x_scaled), x_expected[-2])
        npt.assert_almost_equal(np.array(controller.u_scaled), u_expected[-1])
    elif node == Node.END:
        npt.assert_almost_equal(controller.t, [ns])
        npt.assert_almost_equal(np.array(controller.x), x_expected[ns])
        npt.assert_almost_equal(controller.u, [])
        npt.assert_almost_equal(np.array(controller.x_scaled), x_expected[ns])
        npt.assert_almost_equal(controller.u_scaled, [])
    elif node == 2:
        npt.assert_almost_equal(controller.t, [2])
        npt.assert_almost_equal(np.array(controller.x), x_expected[2])
        npt.assert_almost_equal(controller.u, u_expected[2])
        npt.assert_almost_equal(np.array(controller.x_scaled), x_expected[2])
        npt.assert_almost_equal(controller.u_scaled, u_expected[2])
    else:
        raise RuntimeError("Something went wrong")


def test_bad_shape_output_penalty():
    def bad_custom_function(controller: PenaltyController):
        """
        This custom function returns a matrix, thus some terms will be ignored by bioptim!
        """
        return horzcat(controller.states["q"].cx, controller.states["qdot"].cx, controller.controls["tau"].cx)

    def prepare_test_ocp_error():
        bioptim_folder = TestUtils.bioptim_folder()

        bio_model = TorqueBiorbdModel(bioptim_folder + "/examples/track/models/cube_and_line.bioMod")
        dynamics = Dynamics()

        constraints = ConstraintList()
        constraints.add(bad_custom_function, node=Node.START)

        ocp = OptimalControlProgram(
            bio_model,
            dynamics,
            10,
            1.0,
            constraints=constraints,
        )
        return ocp

    with pytest.raises(RuntimeError, match="The constraint must return a vector not a matrix."):
        ocp = prepare_test_ocp_error()
