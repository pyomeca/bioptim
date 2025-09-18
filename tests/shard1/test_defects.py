import numpy as np
import numpy.testing as npt
import pytest
from casadi import MX, SX

from bioptim import (
    VariableScalingList,
    ConfigureProblem,
    TorqueBiorbdModel,
    TorqueDerivativeBiorbdModel,
    TorqueActivationBiorbdModel,
    TorqueFreeFloatingBaseBiorbdModel,
    MusclesBiorbdModel,
    JointAccelerationBiorbdModel,
    ControlType,
    NonLinearProgram,
    DynamicsOptions,
    ParameterContainer,
    ParameterList,
    PhaseDynamics,
    ExternalForceSetTimeSeries,
    ContactType,
    OdeSolver,
    DefectType,
    MusclesWithExcitationsBiorbdModel,
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
@pytest.mark.parametrize("contact_types", [[ContactType.RIGID_EXPLICIT], [ContactType.RIGID_IMPLICIT], ()])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_torque_driven(contact_types, with_external_force, cx, phase_dynamics, defects_type):
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

    if ContactType.RIGID_IMPLICIT in contact_types and with_external_force:
        with pytest.raises(NotImplementedError):
            # "Your contact_types [<ContactType.RIGID_IMPLICIT: 'rigid_implicit'>] is not supported yet with external_force_set of type ExternalForceSetTimeSeries."
            nlp.model = TorqueBiorbdModel(
                TestUtils.bioptim_folder() + "/examples/models/2segments_4dof_2contacts.bioMod",
                contact_types=contact_types,
                external_force_set=external_forces,
            )
    else:
        nlp.model = TorqueBiorbdModel(
            TestUtils.bioptim_folder() + "/examples/models/2segments_4dof_2contacts.bioMod",
            contact_types=contact_types,
            external_force_set=external_forces,
        )
        nlp.dynamics_type = DynamicsOptions(
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
            numerical_data_timeseries=numerical_time_series,
            ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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
        if ContactType.RIGID_EXPLICIT in contact_types and defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
            with pytest.raises(
                NotImplementedError, match="Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet"
            ):
                ConfigureProblem.initialize(ocp, nlp)
        else:
            ConfigureProblem.initialize(ocp, nlp)

            # Test the results
            states = np.random.rand(nlp.states.shape, nlp.ns)
            states_dot = np.random.rand(nlp.states.shape, nlp.ns)
            controls = np.random.rand(nlp.controls.shape, nlp.ns)
            params = np.random.rand(nlp.parameters.shape, nlp.ns)
            algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
            numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
            time = np.random.rand(2)
            x_defects = np.array(
                nlp.dynamics_defects_func(
                    time,
                    states[:, 0],
                    controls[:, 0],
                    params[:, 0],
                    algebraic_states[:, 0],
                    numerical_timeseries,
                    states_dot[:, 0],
                )
            )
            if ContactType.RIGID_EXPLICIT in contact_types:
                if with_external_force:
                    if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                        npt.assert_almost_equal(
                            x_defects[:, 0],
                            np.array(
                                [
                                    -0.19733986,
                                    -0.19286707,
                                    0.47442614,
                                    -0.21751276,
                                    2.36303867,
                                    0.84504995,
                                    -4.50760903,
                                    3.92232611,
                                ]
                            ),
                        )
                else:
                    if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                        npt.assert_almost_equal(
                            x_defects[:, 0],
                            np.array(
                                [
                                    -0.48981466,
                                    -0.12265368,
                                    0.36203978,
                                    0.11347689,
                                    0.27198618,
                                    0.7248882,
                                    0.99807742,
                                    0.08842979,
                                ]
                            ),
                        )
            elif ContactType.RIGID_IMPLICIT in contact_types:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -0.48981466,
                                -0.12265368,
                                0.36203978,
                                0.11347689,
                                -0.20592043,
                                10.66048726,
                                2.36458005,
                                -19.80096095,
                                0.77091916,
                                0.73515993,
                                1.10755289,
                            ]
                        ),
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -4.89814660e-01,
                                -1.22653677e-01,
                                3.62039776e-01,
                                1.13476887e-01,
                                -6.86116772e00,
                                -1.41889201e02,
                                -4.93022460e00,
                                -1.49950526e00,
                                7.70919162e-01,
                                7.35159931e-01,
                                1.10755289e00,
                            ]
                        ),
                        decimal=6,
                    )
            else:
                if with_external_force:
                    if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                        npt.assert_almost_equal(
                            x_defects[:, 0],
                            np.array(
                                [
                                    -0.19733986,
                                    -0.19286707,
                                    0.47442614,
                                    -0.21751276,
                                    1.29981099,
                                    10.66675032,
                                    -5.20925874,
                                    -17.23035513,
                                ]
                            ),
                        )
                    else:
                        npt.assert_almost_equal(
                            x_defects[:, 0],
                            np.array(
                                [
                                    -0.19733986,
                                    -0.19286707,
                                    0.47442614,
                                    -0.21751276,
                                    -1.0734216,
                                    -139.60897337,
                                    -5.3638614,
                                    -0.9738995,
                                ]
                            ),
                            decimal=6,
                        )
                else:
                    if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                        npt.assert_almost_equal(
                            x_defects[:, 0],
                            np.array(
                                [
                                    -0.48981466,
                                    -0.12265368,
                                    0.36203978,
                                    0.11347689,
                                    -0.21226781,
                                    10.68042395,
                                    2.39425844,
                                    -19.95918011,
                                ]
                            ),
                        )
                    else:
                        npt.assert_almost_equal(
                            x_defects[:, 0],
                            np.array(
                                [
                                    -4.89814660e-01,
                                    -1.22653677e-01,
                                    3.62039776e-01,
                                    1.13476887e-01,
                                    -6.89259691e00,
                                    -1.42138493e02,
                                    -4.94850146e00,
                                    -1.49950526e00,
                                ]
                            ),
                            decimal=6,
                        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("contact_types", [[ContactType.SOFT_IMPLICIT], [ContactType.SOFT_EXPLICIT]])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_torque_driven_soft_contacts_dynamics(contact_types, cx, phase_dynamics, defects_type):

    np.random.seed(42)

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))

    nlp.model = TorqueBiorbdModel(
        TestUtils.bioptim_folder() + "/examples/models/2segments_4dof_2soft_contacts_1muscle.bioMod",
        contact_types=contact_types,
    )
    nlp.dynamics_type = DynamicsOptions(
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    states_dot = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    time = np.random.rand(2)
    x_defects = np.array(
        nlp.dynamics_defects_func(
            time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], [], states_dot[:, 0]
        )
    )
    if ContactType.SOFT_IMPLICIT in contact_types:
        if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -4.89814660e-01,
                        -1.22653677e-01,
                        3.62039776e-01,
                        1.13476887e-01,
                        4.13788060e-02,
                        1.11261314e01,
                        1.08042394e00,
                        -5.18792532e01,
                        -3.14291857e-02,
                        -2.49292229e-01,
                        -2.89751453e-01,
                        -8.71460590e-01,
                        -8.07440155e-01,
                        -4.27107789e-01,
                        -4.17411003e-01,
                        -3.23202932e-01,
                        -9.62447295e-01,
                        -3.68869474e-02,
                        -9.08265886e-01,
                        -2.42055272e-01,
                    ]
                ),
            )
        else:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -4.89814660e-01,
                        -1.22653677e-01,
                        3.62039776e-01,
                        1.13476887e-01,
                        -6.64330468e00,
                        -1.41848741e02,
                        -4.01919389e00,
                        -6.97799376e-01,
                        -3.14291857e-02,
                        -2.49292229e-01,
                        -2.89751453e-01,
                        -8.71460590e-01,
                        -8.07440155e-01,
                        -4.27107789e-01,
                        -4.17411003e-01,
                        -3.23202932e-01,
                        -9.62447295e-01,
                        -3.68869474e-02,
                        -9.08265886e-01,
                        -2.42055272e-01,
                    ]
                ),
                decimal=6,
            )
    if ContactType.SOFT_EXPLICIT in contact_types:
        if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -0.48981466,
                        -0.12265368,
                        0.36203978,
                        0.11347689,
                        -0.21226781,
                        10.68042395,
                        2.39425844,
                        -19.95918011,
                    ]
                ),
            )
        else:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -4.89814660e-01,
                        -1.22653677e-01,
                        3.62039776e-01,
                        1.13476887e-01,
                        -6.89259691e00,
                        -1.42138493e02,
                        -4.94850146e00,
                        -1.49950526e00,
                    ]
                ),
                decimal=6,
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [True])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_torque_derivative_driven(with_contact, with_external_force, cx, phase_dynamics, defects_type):

    np.random.seed(42)

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

    nlp.model = TorqueDerivativeBiorbdModel(
        TestUtils.bioptim_folder() + "/examples/models/2segments_4dof_2contacts.bioMod",
        contact_types=[ContactType.RIGID_EXPLICIT] if with_contact else (),
        external_force_set=external_forces,
    )
    nlp.dynamics_type = DynamicsOptions(
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_timeseries,
        ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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

    if with_external_force:
        np.random.rand(nlp.ns, 6)  # just not to change the values of the next random values

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    if with_contact and defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
        with pytest.raises(
            NotImplementedError, match="Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet"
        ):
            ConfigureProblem.initialize(ocp, nlp)

    else:
        ConfigureProblem.initialize(ocp, nlp)

        # Test the results
        states = np.random.rand(nlp.states.shape, nlp.ns)
        states_dot = np.random.rand(nlp.states.shape, nlp.ns)
        controls = np.random.rand(nlp.controls.shape, nlp.ns)
        params = np.random.rand(nlp.parameters.shape, nlp.ns)
        algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
        numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
        time = np.random.rand(2)
        x_defects = np.array(
            nlp.dynamics_defects_func(
                time,
                states[:, 0],
                controls[:, 0],
                params[:, 0],
                algebraic_states[:, 0],
                numerical_timeseries,
                states_dot[:, 0],
            )
        )
        if with_contact:
            if with_external_force:
                npt.assert_almost_equal(
                    x_defects[:, 0],
                    np.array(
                        [
                            -0.84999038,
                            -0.42807864,
                            -0.3572481,
                            -0.29340385,
                            2.49283506,
                            1.21776478,
                            -3.65027157,
                            4.01881518,
                            -0.49085488,
                            0.08114766,
                            0.59466416,
                            -0.79841555,
                        ]
                    ),
                )
            else:
                npt.assert_almost_equal(
                    x_defects[:, 0],
                    np.array(
                        [
                            -0.31172315,
                            -0.07805808,
                            0.23040588,
                            0.07221787,
                            0.17309483,
                            0.4613264,
                            0.63518686,
                            0.05627764,
                        ]
                    ),
                )

        else:
            if with_external_force:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -0.84999038,
                                -0.42807864,
                                -0.3572481,
                                -0.29340385,
                                1.38011043,
                                10.99989806,
                                -4.08219355,
                                -13.09464479,
                                -0.49085488,
                                0.08114766,
                                0.59466416,
                                -0.79841555,
                            ]
                        ),
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -8.49990382e-01,
                                -4.28078639e-01,
                                -3.57248104e-01,
                                -2.93403854e-01,
                                -5.70258892e00,
                                -1.45376254e02,
                                -7.08538520e00,
                                -1.08402622e00,
                                -4.90854883e-01,
                                8.11476605e-02,
                                5.94664162e-01,
                                -7.98415548e-01,
                            ]
                        ),
                        decimal=6,
                    )
            else:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -0.31172315,
                                -0.07805808,
                                0.23040588,
                                0.07221787,
                                -0.13508945,
                                6.797133,
                                1.523731,
                                -12.70223002,
                            ]
                        ),
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -3.11723149e-01,
                                -7.80580770e-02,
                                2.30405883e-01,
                                7.22178723e-02,
                                -6.89259691e00,
                                -1.42138493e02,
                                -4.94850146e00,
                                -1.49950526e00,
                            ]
                        ),
                        decimal=6,
                    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("contact_types", [[ContactType.SOFT_IMPLICIT], [ContactType.SOFT_EXPLICIT]])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_torque_derivative_driven_soft_contacts_dynamics(contact_types, cx, phase_dynamics, defects_type):

    np.random.seed(42)

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))

    nlp.model = TorqueDerivativeBiorbdModel(
        TestUtils.bioptim_folder() + "/examples/models/2segments_4dof_2soft_contacts_1muscle.bioMod",
        contact_types=contact_types,
    )
    nlp.dynamics_type = DynamicsOptions(
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    states_dot = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    time = np.random.rand(2)
    x_defects = np.array(
        nlp.dynamics_defects_func(
            time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], [], states_dot[:, 0]
        )
    )
    if ContactType.SOFT_IMPLICIT in contact_types:
        if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -2.23175605e-01,
                        -2.42479878e-01,
                        1.64699917e-01,
                        -7.93901801e-02,
                        1.22119681e00,
                        1.07432483e01,
                        -1.76659607e00,
                        -4.21293061e01,
                        -7.76010969e-01,
                        -1.77815559e-01,
                        -1.27659550e-01,
                        5.48257658e-01,
                        -9.62447295e-01,
                        -3.68869474e-02,
                        -9.08265886e-01,
                        -2.42055272e-01,
                        -3.67783133e-01,
                        -8.35302496e-01,
                        -6.77564362e-01,
                        -1.74366429e-01,
                        -3.41066351e-01,
                        -6.59984046e-01,
                        -9.31027678e-02,
                        -3.49209575e-01,
                    ]
                ),
            )
        else:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -2.23175605e-01,
                        -2.42479878e-01,
                        1.64699917e-01,
                        -7.93901801e-02,
                        -1.16039573e01,
                        -1.37716732e02,
                        -4.76824404e00,
                        -8.16422811e-01,
                        -7.76010969e-01,
                        -1.77815559e-01,
                        -1.27659550e-01,
                        5.48257658e-01,
                        -9.62447295e-01,
                        -3.68869474e-02,
                        -9.08265886e-01,
                        -2.42055272e-01,
                        -3.67783133e-01,
                        -8.35302496e-01,
                        -6.77564362e-01,
                        -1.74366429e-01,
                        -3.41066351e-01,
                        -6.59984046e-01,
                        -9.31027678e-02,
                        -3.49209575e-01,
                    ]
                ),
                decimal=6,
            )
    if ContactType.SOFT_EXPLICIT in contact_types:
        if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -0.22317561,
                        -0.24247988,
                        0.16469992,
                        -0.07939018,
                        1.16551709,
                        10.71022124,
                        -1.48485748,
                        -35.30859082,
                        -0.77601097,
                        -0.17781556,
                        -0.12765955,
                        0.54825766,
                    ]
                ),
            )
        else:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -2.23175605e-01,
                        -2.42479878e-01,
                        1.64699917e-01,
                        -7.93901801e-02,
                        -1.16408442e01,
                        -1.38624998e02,
                        -4.97910457e00,
                        -9.99496342e-01,
                        -7.76010969e-01,
                        -1.77815559e-01,
                        -1.27659550e-01,
                        5.48257658e-01,
                    ]
                ),
                decimal=6,
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_torque_activation_driven(with_contact, with_external_force, cx, phase_dynamics, defects_type):

    np.random.seed(42)

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

    nlp.model = TorqueActivationBiorbdModel(
        TestUtils.bioptim_folder() + "/examples/models/2segments_4dof_2contacts.bioMod",
        contact_types=[ContactType.RIGID_EXPLICIT] if with_contact else (),
        external_force_set=external_forces,
    )
    nlp.dynamics_type = DynamicsOptions(
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_timeseries,
        ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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

    if with_external_force:
        np.random.rand(nlp.ns, 6)  # just not to change the values of the next random values

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)

    if with_contact and defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
        with pytest.raises(
            NotImplementedError, match="Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet"
        ):
            ConfigureProblem.initialize(ocp, nlp)
    else:
        ConfigureProblem.initialize(ocp, nlp)

        # Test the results
        states = np.random.rand(nlp.states.shape, nlp.ns)
        states_dot = np.random.rand(nlp.states.shape, nlp.ns)
        controls = np.random.rand(nlp.controls.shape, nlp.ns)
        params = np.random.rand(nlp.parameters.shape, nlp.ns)
        algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
        numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
        time = np.random.rand(2)
        x_defects = np.array(
            nlp.dynamics_defects_func(
                time,
                states[:, 0],
                controls[:, 0],
                params[:, 0],
                algebraic_states[:, 0],
                numerical_timeseries,
                states_dot[:, 0],
            )
        )
        if with_contact:
            if with_external_force:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -0.19733986,
                                -0.19286707,
                                0.47442614,
                                -0.21751276,
                                12.37994595,
                                2.07360077,
                                -24.69154011,
                                24.10625719,
                            ]
                        ),
                        decimal=5,
                    )
                # else: NotImplemented
            else:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -0.48981466,
                                -0.12265368,
                                0.36203978,
                                0.11347689,
                                -44.21404209,
                                -0.19096356,
                                89.98898695,
                                -88.90247974,
                            ]
                        ),
                        decimal=5,
                    )
                # else: NotImplemented
        else:
            if with_external_force:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -1.97339858e-01,
                                -1.92867067e-01,
                                4.74426136e-01,
                                -2.17512761e-01,
                                7.26366684e01,
                                3.66332876e01,
                                -3.44770087e02,
                                -1.75556283e03,
                            ]
                        ),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -0.19733986,
                                -0.19286707,
                                0.47442614,
                                -0.21751276,
                                27.61197224,
                                -53.33437494,
                                74.57271396,
                                41.30977157,
                            ]
                        ),
                        decimal=5,
                    )
            else:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -4.89814660e-01,
                                -1.22653677e-01,
                                3.62039776e-01,
                                1.13476887e-01,
                                -6.61209021e01,
                                3.40150373e01,
                                1.95979764e02,
                                -2.15316271e03,
                            ]
                        ),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -0.48981466,
                                -0.12265368,
                                0.36203978,
                                0.11347689,
                                78.55464226,
                                -109.94534391,
                                6.89132889,
                                47.38625878,
                            ]
                        ),
                        decimal=5,
                    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_residual_torque", [False, True])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_passive_torque", [False, True])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_torque_activation_driven_with_residual_torque(
    with_residual_torque, with_external_force, with_passive_torque, cx, phase_dynamics, defects_type
):

    np.random.seed(42)

    if with_passive_torque:
        model_filename = (
            TestUtils.bioptim_folder() + "/examples/models/2segments_2dof_2contacts_with_passive_torque.bioMod"
        )
    else:
        model_filename = TestUtils.bioptim_folder() + "/examples/models/2segments_2dof_2contacts.bioMod"

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

    nlp.model = TorqueActivationBiorbdModel(
        model_filename,
        with_residual_torque=with_residual_torque,
        external_force_set=external_forces,
    )
    nlp.dynamics_type = DynamicsOptions(
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_timeseries,
        ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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

    if with_external_force:
        np.random.rand(nlp.ns, 6)  # just not to change the values of the next random values

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)

    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    states_dot = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
    time = np.random.rand(2)
    x_defects = np.array(
        nlp.dynamics_defects_func(
            time,
            states[:, 0],
            controls[:, 0],
            params[:, 0],
            algebraic_states[:, 0],
            numerical_timeseries,
            states_dot[:, 0],
        )
    )
    if with_residual_torque:
        if with_passive_torque:
            if with_external_force:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([8.47546393e-01, 2.59351951e-01, -1.22084636e02, -1.45062357e03]),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([0.84754639, 0.25935195, 83.57439118, 34.05025566]),
                        decimal=5,
                    )
            else:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.5912684, 0.60177145, -3.91786284, -100.33736802]), decimal=5
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.5912684, 0.60177145, 6.29890182, 2.76062726]), decimal=5
                    )
        else:
            if with_external_force:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([8.47546393e-01, 2.59351951e-01, -1.17121402e02, -1.40586142e03]),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([0.84754639, 0.25935195, 79.57439118, 33.05025566]),
                        decimal=5,
                    )
            else:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.5912684, 0.60177145, -1.16847683, -64.58799018]), decimal=5
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.5912684, 0.60177145, 2.29890182, 1.76062726]), decimal=5
                    )
    else:
        if with_passive_torque:
            if with_external_force:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([8.47546393e-01, 2.59351951e-01, -1.20860807e02, -1.43667920e03]),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([0.84754639, 0.25935195, 82.71128776, 33.72507234]),
                        decimal=5,
                    )
            else:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.5912684, 0.60177145, -3.48729129, -66.43746144]), decimal=5
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.5912684, 0.60177145, 5.32931719, 1.83875302]), decimal=5
                    )
        else:
            if with_external_force:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([8.47546393e-01, 2.59351951e-01, -1.15897572e02, -1.39191705e03]),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([0.84754639, 0.25935195, 78.71128776, 32.72507234]),
                        decimal=5,
                    )
            else:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.5912684, 0.60177145, -0.73790529, -30.6880836]), decimal=5
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.5912684, 0.60177145, 1.32931719, 0.83875302]), decimal=5
                    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_torque_driven_free_floating_base(cx, phase_dynamics, defects_type):

    np.random.seed(42)

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = TorqueFreeFloatingBaseBiorbdModel(
        TestUtils.bioptim_folder() + "/examples/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.dynamics_type = DynamicsOptions(
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)

    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    states_dot = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)

    time = np.random.rand(2)
    x_defects = np.array(
        nlp.dynamics_defects_func(
            time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], [], states_dot[:, 0]
        )
    )

    if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
        npt.assert_almost_equal(
            x_defects[:, 0],
            np.array(
                [-0.48981466, -0.12265368, 0.36203978, 0.11347689, -0.3436476, 10.90983566, 2.93724966, -35.59258422]
            ),
            decimal=5,
        )
    else:
        npt.assert_almost_equal(
            x_defects[:, 0],
            np.array(
                [-0.48981466, -0.12265368, 0.36203978, 0.11347689, -1.99330085, -1.99330085, -1.99330085, -1.13019743]
            ),
            decimal=6,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize(
    "with_external_force",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "contact_types",
    [
        [ContactType.RIGID_EXPLICIT],
        [ContactType.RIGID_IMPLICIT],
        (),
    ],
)
@pytest.mark.parametrize(
    "with_residual_torque",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "with_excitation",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_muscle_driven(
    with_excitation, contact_types, with_residual_torque, with_external_force, cx, phase_dynamics, defects_type
):

    np.random.seed(42)

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

    if ContactType.RIGID_IMPLICIT in contact_types and with_external_force:
        with pytest.raises(NotImplementedError):
            nlp.model = MusclesBiorbdModel(
                TestUtils.bioptim_folder()
                + "/examples/toy_examples/muscle_driven_ocp/models/arm26_with_contact.bioMod",
                contact_types=contact_types,
                external_force_set=external_forces,
            )
    elif ContactType.RIGID_IMPLICIT in contact_types:
        with pytest.raises(RuntimeError, match="The segment for the rigid contact index 0 was not found."):
            # TODO: This is a bug... The index of the parent of the contact is not correctly identified when it is the root
            nlp.model = MusclesBiorbdModel(
                TestUtils.bioptim_folder()
                + "/examples/toy_examples/muscle_driven_ocp/models/arm26_with_contact.bioMod",
                contact_types=contact_types,
                external_force_set=external_forces,
            )
    else:
        muscle_class = MusclesWithExcitationsBiorbdModel if with_excitation else MusclesBiorbdModel
        nlp.model = muscle_class(
            TestUtils.bioptim_folder() + "/examples/toy_examples/muscle_driven_ocp/models/arm26_with_contact.bioMod",
            contact_types=contact_types,
            external_force_set=external_forces,
            with_residual_torque=with_residual_torque,
        )
        nlp.dynamics_type = DynamicsOptions(
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
            numerical_data_timeseries=numerical_timeseries,
            ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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

        if with_external_force:
            np.random.rand(nlp.ns, 6)  # just not to change the values of the next random values

        nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)

        if ContactType.RIGID_EXPLICIT in contact_types and defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
            with pytest.raises(
                NotImplementedError, match="Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet"
            ):
                ConfigureProblem.initialize(ocp, nlp)
        else:
            ConfigureProblem.initialize(ocp, nlp)

            # Test the results
            states = np.random.rand(nlp.states.shape, nlp.ns)
            states_dot = np.random.rand(nlp.states.shape, nlp.ns)
            controls = np.random.rand(nlp.controls.shape, nlp.ns)
            params = np.random.rand(nlp.parameters.shape, nlp.ns)
            algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
            numerical_timeseries = EXTERNAL_FORCE_ARRAY[:, 0] if with_external_force else []
            time = np.random.rand(2)
            x_defects = np.array(
                nlp.dynamics_defects_func(
                    time,
                    states[:, 0],
                    controls[:, 0],
                    params[:, 0],
                    algebraic_states[:, 0],
                    numerical_timeseries,
                    states_dot[:, 0],
                )
            )

            if ContactType.RIGID_EXPLICIT in contact_types:
                if with_residual_torque:
                    if with_excitation:
                        if with_external_force:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        -0.54292804,
                                        -0.47578903,
                                        -0.89044505,
                                        -0.44641779,
                                        38.99903987,
                                        15.22840121,
                                        -40.43158428,
                                        -9.8364957,
                                        25.20495604,
                                        15.77789951,
                                        10.07567827,
                                        5.76797109,
                                    ]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        2.05272780e-01,
                                        -6.91568116e-02,
                                        -1.29311921e-02,
                                        4.72584539e00,
                                        -1.28005642e02,
                                        4.56345900e02,
                                        1.01520109e01,
                                        -8.50131737e00,
                                        1.48550951e00,
                                        -1.62017941e01,
                                        3.58364164e01,
                                        2.69526860e01,
                                    ]
                                ),
                                decimal=5,
                            )
                    else:
                        if with_external_force:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [-0.27384499, -0.42688854, -0.14962947, -0.36567318, 61.25857721, -44.54620577]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array([0.42414034, 0.19654445, -0.66313773, 1.13868998, -4.0691168, 120.74475469]),
                                decimal=5,
                            )
                else:
                    if with_excitation:
                        if with_external_force:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        -5.42928038e-01,
                                        -4.75789031e-01,
                                        -8.90445049e-01,
                                        -1.91930588e-02,
                                        3.40150265e01,
                                        3.11793932e01,
                                        -4.71686403e01,
                                        1.03034901e01,
                                        1.71860583e01,
                                        -6.34730917e00,
                                        9.28691316e00,
                                        3.76121580e00,
                                    ]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        2.05272780e-01,
                                        -6.91568116e-02,
                                        -1.29311921e-02,
                                        5.10609172e00,
                                        -1.32358031e02,
                                        4.71632733e02,
                                        -1.40441453e01,
                                        1.68186702e01,
                                        -4.32113186e01,
                                        1.29210328e01,
                                        6.38477420e-01,
                                        4.25280634e01,
                                    ]
                                ),
                                decimal=5,
                            )
                    else:
                        if with_external_force:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array([-0.27384499, -0.42688854, -0.14962947, 0.55562059, 15.460567, 10.819210193]),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        4.24140342e-01,
                                        1.96544453e-01,
                                        -6.63137727e-01,
                                        2.41386864e00,
                                        -4.88604389e01,
                                        2.05011756e02,
                                    ]
                                ),
                                decimal=5,
                            )
            else:
                if with_residual_torque:
                    if with_excitation:
                        if with_external_force:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            -0.54292804,
                                            -0.47578903,
                                            -0.89044505,
                                            -0.44641779,
                                            38.99903987,
                                            15.22840121,
                                            -40.43158428,
                                            -9.8364957,
                                            25.20495604,
                                            15.77789951,
                                            10.07567827,
                                            5.76797109,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            -0.54292804,
                                            -0.47578903,
                                            -0.89044505,
                                            0.31041669,
                                            -19.22248632,
                                            -6.93204239,
                                            -40.43158428,
                                            -9.8364957,
                                            25.20495604,
                                            15.77789951,
                                            10.07567827,
                                            5.76797109,
                                        ]
                                    ),
                                    decimal=5,
                                )
                        else:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            2.05272780e-01,
                                            -6.91568116e-02,
                                            -1.29311921e-02,
                                            4.72584539e00,
                                            -1.28005642e02,
                                            4.56345900e02,
                                            1.01520109e01,
                                            -8.50131737e00,
                                            1.48550951e00,
                                            -1.62017941e01,
                                            3.58364164e01,
                                            2.69526860e01,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            2.05272780e-01,
                                            -6.91568116e-02,
                                            -1.29311921e-02,
                                            -1.70266428e00,
                                            -1.31826147e01,
                                            -1.28447704e01,
                                            1.01520109e01,
                                            -8.50131737e00,
                                            1.48550951e00,
                                            -1.62017941e01,
                                            3.58364164e01,
                                            2.69526860e01,
                                        ]
                                    ),
                                    decimal=5,
                                )
                    else:
                        if with_external_force:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [-0.27384499, -0.42688854, -0.14962947, -0.36567318, 61.25857721, -44.54620577]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [-0.27384499, -0.42688854, -0.14962947, -2.14326015, -19.93936948, -6.05447738]
                                    ),
                                    decimal=5,
                                )
                        else:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [0.42414034, 0.19654445, -0.66313773, 1.13868998, -4.0691168, 120.74475469]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [0.42414034, 0.19654445, -0.66313773, -1.89342602, -16.46421687, -7.88393296]
                                    ),
                                    decimal=5,
                                )
                else:
                    if with_excitation:
                        if with_external_force:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            -5.42928038e-01,
                                            -4.75789031e-01,
                                            -8.90445049e-01,
                                            -1.91930588e-02,
                                            3.40150265e01,
                                            3.11793932e01,
                                            -4.71686403e01,
                                            1.03034901e01,
                                            1.71860583e01,
                                            -6.34730917e00,
                                            9.28691316e00,
                                            3.76121580e00,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            -0.54292804,
                                            -0.47578903,
                                            -0.89044505,
                                            -0.59784919,
                                            -19.46454159,
                                            -7.29982552,
                                            -47.16864032,
                                            10.30349009,
                                            17.18605833,
                                            -6.34730917,
                                            9.28691316,
                                            3.7612158,
                                        ]
                                    ),
                                    decimal=5,
                                )
                        else:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            2.05272780e-01,
                                            -6.91568116e-02,
                                            -1.29311921e-02,
                                            5.10609172e00,
                                            -1.32358031e02,
                                            4.71632733e02,
                                            -1.40441453e01,
                                            1.68186702e01,
                                            -4.32113186e01,
                                            1.29210328e01,
                                            6.38477420e-01,
                                            4.25280634e01,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            2.05272780e-01,
                                            -6.91568116e-02,
                                            -1.29311921e-02,
                                            -2.51010444e00,
                                            -1.36097225e01,
                                            -1.32621814e01,
                                            -1.40441453e01,
                                            1.68186702e01,
                                            -4.32113186e01,
                                            1.29210328e01,
                                            6.38477420e-01,
                                            4.25280634e01,
                                        ]
                                    ),
                                    decimal=5,
                                )
                    else:
                        if with_external_force:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [-0.27384499, -0.42688854, -0.14962947, 0.55562059, 15.460567, 10.81921019]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [-0.27384499, -0.42688854, -0.14962947, -2.2628544, -8.36653992, -3.06929409]
                                    ),
                                    decimal=5,
                                )
                        else:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            4.24140342e-01,
                                            1.96544453e-01,
                                            -6.63137727e-01,
                                            2.41386864e00,
                                            -4.88604389e01,
                                            2.05011756e02,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [0.42414034, 0.19654445, -0.66313773, -2.28210331, -9.70670533, -7.06713657]
                                    ),
                                    decimal=5,
                                )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_joints_acceleration_driven(cx, phase_dynamics, defects_type):

    np.random.seed(42)

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = JointAccelerationBiorbdModel(TestUtils.bioptim_folder() + "/examples/models/double_pendulum.bioMod")
    nlp.dynamics_type = DynamicsOptions(
        expand_dynamics=True,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.COLLOCATION(defects_type=defects_type),
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

    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)

    nlp.numerical_timeseries = TestUtils.initialize_numerical_timeseries(nlp, dynamics=nlp.dynamics_type)
    if defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
        with pytest.raises(
            NotImplementedError,
            match="The defect type DefectType.TAU_EQUALS_INVERSE_DYNAMICS is not implemented yet for joints acceleration driven dynamics.",
        ):
            ConfigureProblem.initialize(ocp, nlp)

    else:
        ConfigureProblem.initialize(ocp, nlp)

        # Test the results
        states = np.random.rand(nlp.states.shape, nlp.ns)
        states_dot = np.random.rand(nlp.states.shape, nlp.ns)
        controls = np.random.rand(nlp.controls.shape, nlp.ns)
        params = np.random.rand(nlp.parameters.shape, nlp.ns)
        algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
        time = np.random.rand(2)
        x_defects = np.array(
            nlp.dynamics_defects_func(
                time, states[:, 0], controls[:, 0], params[:, 0], algebraic_states[:, 0], [], states_dot[:, 0]
            )
        )
        npt.assert_almost_equal(x_defects[:, 0], np.array([0.5912684, 0.60177145, 3.36951637, 0.68635911]), decimal=5)
