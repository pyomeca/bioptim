import re

import numpy as np
import numpy.testing as npt
import pytest
from casadi import MX, SX, vertcat

from bioptim import (
    VariableScalingList,
    ConfigureProblem,
    DynamicsFunctions,
    BiorbdModel,
    ControlType,
    NonLinearProgram,
    DynamicsFcn,
    Dynamics,
    DynamicsEvaluation,
    ConstraintList,
    ParameterContainer,
    ParameterList,
    PhaseDynamics,
    ExternalForceSetTimeSeries,
)

from ..utils import TestUtils


class OptimalControlProgram:
    def __init__(self, nlp, use_sx):
        self.cx = nlp.cx
        self.phase_dynamics = PhaseDynamics.SHARED_DURING_THE_PHASE
        self.n_phases = 1
        self.nlp = [nlp]
        parameters_list = ParameterList(use_sx=use_sx)
        self.parameters = ParameterContainer(use_sx=use_sx)
        self.parameters.initialize(parameters_list)
        self.implicit_constraints = ConstraintList()
        self.n_threads = 1


N_SHOOTING = 5
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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize(
    "with_external_force",
    [False, True],
)
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_driven(with_contact, with_external_force, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.ns = N_SHOOTING

    external_forces = None
    numerical_time_series = None
    if with_external_force:

        external_forces = ExternalForceSetTimeSeries(nb_frames=nlp.ns)
        external_forces.add(
            "force0", "Seg0", EXTERNAL_FORCE_ARRAY[:6, :], point_of_application=EXTERNAL_FORCE_ARRAY[6:, :]
        )
        numerical_time_series = {"external_forces": external_forces.to_numerical_time_series()}

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
        external_force_set=external_forces,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        with_contact=with_contact,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_time_series,
    )

    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    np.random.seed(42)
    if with_external_force:
        np.random.rand(nlp.ns, 6)  # just not to change the values of the next random values

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
    time = np.random.rand(2)
    x_out = np.array(
        nlp.dynamics_func(
            time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], numerical_timeseries
        )
    )
    if with_contact:
        contact_out = np.array(
            nlp.contact_forces_func(time, states, controls, params, algebraic_states, numerical_timeseries)
        )
        if with_external_force:
            npt.assert_almost_equal(
                x_out[:, 0],
                [0.9695846, 0.9218742, 0.3886773, 0.5426961, -2.2030836, -0.3463042, 4.4577117, -3.5917074],
            )
            npt.assert_almost_equal(contact_out[:, 0], [-14.3821076, 126.2899884, 4.1631847])

        else:
            npt.assert_almost_equal(
                x_out[:, 0],
                [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3214905, -0.1912131, 0.6507164, -0.2359716],
            )
            npt.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        if with_external_force:
            npt.assert_almost_equal(
                x_out[:, 0],
                [0.9695846, 0.9218742, 0.3886773, 0.5426961, -1.090359, -10.1284375, 4.8896337, 13.5217526],
            )
        else:
            npt.assert_almost_equal(
                x_out[:, 0],
                [
                    0.61185289,
                    0.78517596,
                    0.60754485,
                    0.80839735,
                    -0.30241366,
                    -10.38503791,
                    1.60445173,
                    35.80238642,
                ],
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize("implicit_contact", [False, True])
def test_torque_driven_soft_contacts_dynamics(with_contact, cx, implicit_contact, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        with_contact=with_contact,
        soft_contacts_dynamics=implicit_contact,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
    )

    nlp.ns = N_SHOOTING
    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)

    nlp.x_bounds = np.zeros((nlp.model.nb_q * (2 + 3), 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q * 2, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT

    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )

    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    # Prepare the dynamics
    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    np.random.seed(42)
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = []
    time = np.random.rand(2)
    x_out = np.array(nlp.dynamics_func(time, states, controls, params, algebraic_states, numerical_timeseries))

    if with_contact:
        contact_out = np.array(
            nlp.contact_forces_func(time, states, controls, params, algebraic_states, numerical_timeseries)
        )
        npt.assert_almost_equal(
            x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3214905, -0.1912131, 0.6507164, -0.2359716]
        )

        npt.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        npt.assert_almost_equal(
            x_out[:, 0],
            [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3024137, -10.3850379, 1.6044517, 35.8023864],
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [True])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_derivative_driven(with_contact, with_external_force, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.ns = N_SHOOTING

    external_forces = None
    numerical_timeseries = None
    if with_external_force:
        external_forces = ExternalForceSetTimeSeries(nb_frames=nlp.ns)
        external_forces.add(
            "force0", "Seg0", EXTERNAL_FORCE_ARRAY[:6, :], point_of_application=EXTERNAL_FORCE_ARRAY[6:, :]
        )
        numerical_timeseries = {"external_forces": external_forces.to_numerical_time_series()}

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
        external_force_set=external_forces,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
        with_contact=with_contact,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_timeseries,
    )

    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT

    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )

    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    np.random.seed(42)
    if with_external_force:
        np.random.rand(nlp.ns, 6)

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
    time = np.random.rand(2)
    x_out = np.array(
        nlp.dynamics_func(
            time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], numerical_timeseries
        )
    )

    if with_contact:
        contact_out = np.array(
            nlp.contact_forces_func(time, states, controls, params, algebraic_states, numerical_timeseries)
        )
        if with_external_force:
            npt.assert_almost_equal(
                x_out[:, 0],
                [
                    0.9695846,
                    0.9218742,
                    0.3886773,
                    0.5426961,
                    -2.2030836,
                    -0.3463042,
                    4.4577117,
                    -3.5917074,
                    0.1195942,
                    0.4937956,
                    0.0314292,
                    0.2492922,
                ],
            )
            npt.assert_almost_equal(contact_out[:, 0], [-14.3821076, 126.2899884, 4.1631847])
        else:
            npt.assert_almost_equal(
                x_out[:, 0],
                [
                    0.61185289,
                    0.78517596,
                    0.60754485,
                    0.80839735,
                    -0.32149054,
                    -0.19121314,
                    0.65071636,
                    -0.23597164,
                    0.38867729,
                    0.54269608,
                    0.77224477,
                    0.72900717,
                ],
            )
            npt.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        if with_external_force:
            npt.assert_almost_equal(
                x_out[:, 0],
                [
                    0.9695846,
                    0.9218742,
                    0.3886773,
                    0.5426961,
                    -1.090359,
                    -10.1284375,
                    4.8896337,
                    13.5217526,
                    0.1195942,
                    0.4937956,
                    0.0314292,
                    0.2492922,
                ],
            )
        else:
            npt.assert_almost_equal(
                x_out[:, 0],
                [
                    0.61185289,
                    0.78517596,
                    0.60754485,
                    0.80839735,
                    -0.30241366,
                    -10.38503791,
                    1.60445173,
                    35.80238642,
                    0.38867729,
                    0.54269608,
                    0.77224477,
                    0.72900717,
                ],
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize("implicit_contact", [False, True])
def test_torque_derivative_driven_soft_contacts_dynamics(with_contact, cx, implicit_contact, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
        with_contact=with_contact,
        soft_contacts_dynamics=implicit_contact,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
    )

    nlp.ns = N_SHOOTING
    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)

    nlp.x_bounds = np.zeros((nlp.model.nb_q * (2 + 3), 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q * 4, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )

    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    # Prepare the dynamics
    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    np.random.seed(42)
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = []
    time = np.random.rand(2)
    x_out = np.array(nlp.dynamics_func(time, states, controls, params, algebraic_states, numerical_timeseries))

    if with_contact:
        contact_out = np.array(
            nlp.contact_forces_func(time, states, controls, params, algebraic_states, numerical_timeseries)
        )
        npt.assert_almost_equal(
            x_out[:, 0],
            [
                0.6118529,
                0.785176,
                0.6075449,
                0.8083973,
                -0.3214905,
                -0.1912131,
                0.6507164,
                -0.2359716,
                0.3886773,
                0.5426961,
                0.7722448,
                0.7290072,
            ],
        )
        npt.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        npt.assert_almost_equal(
            x_out[:, 0],
            [
                0.6118529,
                0.785176,
                0.6075449,
                0.8083973,
                -0.3024137,
                -10.3850379,
                1.6044517,
                35.8023864,
                0.3886773,
                0.5426961,
                0.7722448,
                0.7290072,
            ],
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize(
    "dynamics",
    [DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, DynamicsFcn.MUSCLE_DRIVEN],
)
def test_soft_contacts_dynamics_errors(dynamics, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=False)
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.dynamics_type = Dynamics(
        dynamics, soft_contacts_dynamics=True, expand_dynamics=True, phase_dynamics=phase_dynamics
    )

    nlp.ns = N_SHOOTING
    nlp.cx = MX

    nlp.u_bounds = np.zeros((nlp.model.nb_q * 4, 1))
    nlp.u_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=True)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    # Prepare the dynamics
    with pytest.raises(
        TypeError,
        match=re.escape(f"{dynamics.name.lower()}() got an unexpected keyword argument " "'soft_contacts_dynamics'"),
    ):
        ConfigureProblem.initialize(ocp, nlp)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_activation_driven(with_contact, with_external_force, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.ns = N_SHOOTING

    external_forces = None
    numerical_timeseries = None
    if with_external_force:
        external_forces = ExternalForceSetTimeSeries(nb_frames=nlp.ns)
        external_forces.add(
            "force0", "Seg0", EXTERNAL_FORCE_ARRAY[:6, :], point_of_application=EXTERNAL_FORCE_ARRAY[6:, :]
        )
        numerical_timeseries = {"external_forces": external_forces.to_numerical_time_series()}

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
        external_force_set=external_forces,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN,
        with_contact=with_contact,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_timeseries,
    )

    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    np.random.seed(42)
    if with_external_force:
        np.random.rand(nlp.ns, 6)

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
    time = np.random.rand(2)
    x_out = np.array(
        nlp.dynamics_func(
            time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], numerical_timeseries
        )
    )

    if with_contact:
        contact_out = np.array(
            nlp.contact_forces_func(time, states, controls, params, algebraic_states, numerical_timeseries)
        )
        if with_external_force:
            npt.assert_almost_equal(
                x_out[:, 0],
                [0.96958, 0.92187, 0.38868, 0.5427, -8.22427, -1.08479, 16.59032, -15.72432],
                decimal=5,
            )
            npt.assert_almost_equal(contact_out[:, 0], [-126.9614581, 179.6585112, -125.8079563])
        else:
            npt.assert_almost_equal(
                x_out[:, 0],
                [0.61185289, 0.78517596, 0.60754485, 0.80839735, 0.78455384, -0.16844256, -1.56184114, 1.97658587],
                decimal=5,
            )
            npt.assert_almost_equal(contact_out[:, 0], [-7.88958997, 329.70828173, -263.55516549])

    else:
        if with_external_force:
            npt.assert_almost_equal(
                x_out[:, 0],
                [
                    9.69584628e-01,
                    9.21874235e-01,
                    3.88677290e-01,
                    5.42696083e-01,
                    -6.35312971e01,
                    -3.16877667e01,
                    3.09696095e02,
                    1.36002265e03,
                ],
                decimal=5,
            )
        else:
            npt.assert_almost_equal(
                x_out[:, 0],
                [
                    6.11852895e-01,
                    7.85175961e-01,
                    6.07544852e-01,
                    8.08397348e-01,
                    -2.38262975e01,
                    -5.82033454e01,
                    1.27439020e02,
                    3.66531163e03,
                ],
                decimal=5,
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_residual_torque", [False, True])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_passive_torque", [False, True])
def test_torque_activation_driven_with_residual_torque(
    with_residual_torque, with_external_force, with_passive_torque, cx, phase_dynamics
):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.ns = N_SHOOTING

    external_forces = None
    numerical_timeseries = None
    if with_external_force:
        external_forces = ExternalForceSetTimeSeries(nb_frames=nlp.ns)
        external_forces.add(
            "force0", "Seg0", EXTERNAL_FORCE_ARRAY[:6, :], point_of_application=EXTERNAL_FORCE_ARRAY[6:, :]
        )
        numerical_timeseries = {"external_forces": external_forces.to_numerical_time_series()}

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/2segments_2dof_2contacts.bioMod",
        external_force_set=external_forces,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN,
        with_residual_torque=with_residual_torque,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_timeseries,
    )

    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    np.random.seed(42)
    if with_external_force:
        np.random.rand(nlp.ns, 6)

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
    time = np.random.rand(2)
    x_out = np.array(
        nlp.dynamics_func(
            time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], numerical_timeseries
        )
    )

    if with_residual_torque:
        if with_external_force:
            if with_passive_torque:
                npt.assert_almost_equal(
                    x_out[:, 0],
                    [1.22038235e-01, 6.62522284e-01, 1.52446740e02, 1.79223051e03],
                    decimal=5,
                )
            else:
                npt.assert_almost_equal(
                    x_out[:, 0],
                    [1.22038235e-01, 6.62522284e-01, 1.52446740e02, 1.79223051e03],
                    decimal=5,
                )
        else:
            if with_passive_torque:
                npt.assert_almost_equal(
                    x_out[:, 0],
                    [0.020584, 0.183405, 55.393940, 54.222523],
                    decimal=5,
                )
            else:
                npt.assert_almost_equal(
                    x_out[:, 0],
                    [0.020584, 0.183405, 55.393940, 54.222523],
                    decimal=5,
                )

    else:
        if with_external_force:
            if with_passive_torque:
                npt.assert_almost_equal(
                    x_out[:, 0],
                    [1.22038235e-01, 6.62522284e-01, 1.51341897e02, 1.77042854e03],
                    decimal=5,
                )
            else:
                npt.assert_almost_equal(
                    x_out[:, 0],
                    [1.22038235e-01, 6.62522284e-01, 1.51341897e02, 1.77042854e03],
                    decimal=5,
                )
        else:
            if with_passive_torque:
                npt.assert_almost_equal(
                    x_out[:, 0],
                    [0.020584, 0.183405, 55.204243, 24.411235],
                    decimal=5,
                )
            else:
                npt.assert_almost_equal(
                    x_out[:, 0],
                    [0.020584, 0.183405, 55.204243, 24.411235],
                    decimal=5,
                )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
def test_torque_driven_free_floating_base(cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN_FREE_FLOATING_BASE, expand_dynamics=True, phase_dynamics=phase_dynamics
    )

    nlp.ns = N_SHOOTING
    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_tau - nlp.model.nb_root, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    np.random.seed(42)

    # Prepare the dynamics
    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = []
    time = np.random.rand(2, 1)
    x_out = np.array(nlp.dynamics_func(time, states, controls, params, algebraic_states, numerical_timeseries))

    npt.assert_almost_equal(
        x_out[:, 0],
        [0.61185289, 0.78517596, 0.60754485, 0.80839735, 0.04791036, -9.96778948, -0.01986505, 4.39786051],
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize("with_residual_torque", [False, True])
@pytest.mark.parametrize("with_excitations", [False, True])
def test_muscle_driven(with_excitations, with_contact, with_residual_torque, with_external_force, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.ns = N_SHOOTING

    external_forces = None
    numerical_timeseries = None
    if with_external_force:
        external_forces = ExternalForceSetTimeSeries(nb_frames=nlp.ns)
        external_forces.add(
            "force0",
            "r_ulna_radius_hand_rotation1",
            EXTERNAL_FORCE_ARRAY[:6, :],
            point_of_application=EXTERNAL_FORCE_ARRAY[6:, :],
        )
        numerical_timeseries = {"external_forces": external_forces.to_numerical_time_series()}

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26_with_contact.bioMod",
        external_force_set=external_forces,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.MUSCLE_DRIVEN,
        with_residual_torque=with_residual_torque,
        with_excitations=with_excitations,
        with_contact=with_contact,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_timeseries,
    )

    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2 + nlp.model.nb_muscles, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_muscles, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()
    nlp.phase_idx = 0

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    np.random.seed(42)
    if with_external_force:
        np.random.rand(nlp.ns, 6)  # just to make sure the next random is the same as before

    # Prepare the dynamics
    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
    time = np.random.rand(2)
    x_out = np.array(
        nlp.dynamics_func(
            time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], numerical_timeseries
        )
    )

    if with_contact:  # Warning this test is a bit bogus, there since the model does not have contacts
        if with_residual_torque:
            if with_excitations:
                if with_external_force:
                    npt.assert_almost_equal(
                        x_out[:, 0],
                        [
                            0.6625223,
                            0.9695846,
                            0.9218742,
                            0.2123157,
                            -29.9955403,
                            -37.8135747,
                            -3.773906,
                            -8.3095101,
                            5.9827416,
                            4.9220243,
                            -19.5615453,
                            9.336912,
                        ],
                    )
                else:
                    npt.assert_almost_equal(
                        x_out[:, 0],
                        [
                            1.83404510e-01,
                            6.11852895e-01,
                            7.85175961e-01,
                            -3.94658983e00,
                            1.23227027e02,
                            -4.38936797e02,
                            8.60630831e00,
                            3.19433638e00,
                            2.97405608e01,
                            -2.02754226e01,
                            -2.32467778e01,
                            -4.19135012e01,
                        ],
                        decimal=6,
                    )
            else:
                if with_external_force:
                    npt.assert_almost_equal(
                        x_out[:, 0],
                        [0.662522, 0.969585, 0.921874, 1.151072, -56.094393, 49.109365],
                        decimal=6,
                    )
                else:
                    npt.assert_almost_equal(
                        x_out[:, 0],
                        [0.18340451, 0.61185289, 0.78517596, -0.8671376, 22.51194682, -153.29477496],
                        decimal=6,
                    )

        else:
            if with_excitations:
                if with_external_force:
                    npt.assert_almost_equal(
                        x_out[:, 0],
                        [
                            0.6625223,
                            0.9695846,
                            0.9218742,
                            0.2684853,
                            -33.7252751,
                            -30.3079326,
                            -7.2855306,
                            -1.6064349,
                            -30.7136058,
                            -19.1107728,
                            -25.7242266,
                            55.3038169,
                        ],
                    )
                else:
                    npt.assert_almost_equal(
                        x_out[:, 0],
                        [
                            1.83404510e-01,
                            6.11852895e-01,
                            7.85175961e-01,
                            -4.37708456e00,
                            1.33221135e02,
                            -4.71307550e02,
                            -7.72228930e00,
                            -1.13759732e01,
                            9.51906209e01,
                            4.45077128e00,
                            -5.20261014e00,
                            -2.80864106e01,
                        ],
                        decimal=6,
                    )
            else:
                if with_external_force:
                    npt.assert_almost_equal(
                        x_out[:, 0],
                        [0.6625223, 0.9695846, 0.9218742, 0.2684853, -33.7252751, -30.3079326],
                    )
                else:
                    npt.assert_almost_equal(
                        x_out[:, 0],
                        [
                            1.83404510e-01,
                            6.11852895e-01,
                            7.85175961e-01,
                            -4.37708456e00,
                            1.33221135e02,
                            -4.71307550e02,
                        ],
                        decimal=6,
                    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
def test_joints_acceleration_driven(cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = BiorbdModel(TestUtils.bioptim_folder() + "/examples/getting_started/models/double_pendulum.bioMod")
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.JOINTS_ACCELERATION_DRIVEN,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
    )

    nlp.ns = N_SHOOTING
    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(nlp.cx)

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))
    nlp.control_type = ControlType.CONSTANT

    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )
    np.random.seed(42)
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    # Prepare the dynamics
    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = []
    time = np.random.rand(2)
    x_out = np.array(nlp.dynamics_func(time, states, controls, params, algebraic_states, numerical_timeseries))

    # obtained using Ipuch reference implementation. [https://github.com/Ipuch/OnDynamicsForSomersaults]
    npt.assert_almost_equal(x_out[:, 0], [0.02058449, 0.18340451, -2.95556261, 0.61185289])


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("with_contact", [False, True])
def test_custom_dynamics(with_contact, phase_dynamics):
    def custom_dynamic(
        time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp, with_contact=False
    ) -> DynamicsEvaluation:
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)

        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)

    def configure(ocp, nlp, with_contact=None, numerical_data_timeseries=None):
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, with_contact=with_contact)

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=False)
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.dynamics_type = Dynamics(
        configure,
        dynamic_function=custom_dynamic,
        with_contact=with_contact,
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
    )

    nlp.ns = N_SHOOTING
    nlp.cx = MX
    nlp.time_cx = nlp.cx.sym("time", 1, 1)
    nlp.dt = nlp.cx.sym("dt", 1, 1)
    nlp.initialize(nlp.cx)
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=False)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        nlp.dynamics_type,
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)
    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)

    np.random.seed(42)

    # Prepare the dynamics
    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = []
    time = np.random.rand(2)
    x_out = np.array(nlp.dynamics_func(time, states, controls, params, algebraic_states, numerical_timeseries))

    if with_contact:
        contact_out = np.array(
            nlp.contact_forces_func(time, states, controls, params, algebraic_states, numerical_timeseries)
        )
        npt.assert_almost_equal(
            x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3214905, -0.1912131, 0.6507164, -0.2359716]
        )
        npt.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        npt.assert_almost_equal(
            x_out[:, 0],
            [0.61185289, 0.78517596, 0.60754485, 0.80839735, -0.30241366, -10.38503791, 1.60445173, 35.80238642],
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize(
    "dynamics_fcn",
    [
        DynamicsFcn.TORQUE_DRIVEN,
        DynamicsFcn.MUSCLE_DRIVEN,
        DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
        DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN,
    ],
)
def test_with_contact_error(dynamics_fcn, phase_dynamics):
    from bioptim.examples.getting_started import pendulum as ocp_module
    from bioptim import BoundsList, ObjectiveList, OdeSolver, OptimalControlProgram

    bioptim_folder = TestUtils.module_folder(ocp_module)

    bio_model = BiorbdModel(bioptim_folder + "/models/pendulum.bioMod")

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = Dynamics(
        dynamics_fcn, with_contact=True, ode_solver=OdeSolver.RK4(), expand_dynamics=True, phase_dynamics=phase_dynamics
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

    # Define control path constraint
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [100] * n_tau, [100] * n_tau
    u_bounds["tau"][1, :] = 0  # Prevent the model from actively rotate

    with pytest.raises(ValueError, match="No contact defined in the .bioMod of phase 0, set with_contact to False"):
        OptimalControlProgram(
            bio_model=bio_model,
            dynamics=dynamics,
            n_shooting=5,
            phase_time=1,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
        )
