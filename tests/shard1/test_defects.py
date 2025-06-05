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
    ParameterContainer,
    ParameterList,
    PhaseDynamics,
    ExternalForceSetTimeSeries,
    ContactType,
    OdeSolver,
    DefectType,
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
            nlp.model = BiorbdModel(
                TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
                contact_types=contact_types,
                external_force_set=external_forces,
            )
    else:
        nlp.model = BiorbdModel(
            TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
            contact_types=contact_types,
            external_force_set=external_forces,
        )
        nlp.dynamics_type = Dynamics(
            DynamicsFcn.TORQUE_DRIVEN,
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
                                    -0.04383072,
                                    -0.04283728,
                                    0.10537375,
                                    -0.04831128,
                                    0.52484935,
                                    0.18769219,
                                    -1.00117517,
                                    0.87117926,
                                ]
                            ),
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
                                    0.17309483,
                                    0.4613264,
                                    0.63518686,
                                    0.05627764,
                                ]
                            ),
                        )
            elif ContactType.RIGID_IMPLICIT in contact_types:
                if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array(
                            [
                                -0.39365036,
                                -0.09857334,
                                0.29096126,
                                0.09119821,
                                -0.1654925,
                                8.56753594,
                                1.90034696,
                                -15.91347941,
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
                                -3.93650365e-01,
                                -9.85733354e-02,
                                2.90961259e-01,
                                9.11982054e-02,
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
                                    -0.04383072,
                                    -0.04283728,
                                    0.10537375,
                                    -0.04831128,
                                    0.28869817,
                                    2.36916856,
                                    -1.15701705,
                                    -3.82699645,
                                ]
                            ),
                        )
                    else:
                        npt.assert_almost_equal(
                            x_defects[:, 0],
                            np.array(
                                [
                                    -4.38307239e-02,
                                    -4.28372820e-02,
                                    1.05373750e-01,
                                    -4.83112831e-02,
                                    -1.07342160e00,
                                    -1.39608973e02,
                                    -5.36386140e00,
                                    -9.73899501e-01,
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
def test_torque_driven_soft_contacts_dynamics(contact_types, cx, phase_dynamics, defects_type):

    np.random.seed(42)

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder()
        + "/examples/muscle_driven_with_contact/models/2segments_4dof_2soft_contacts_1muscle.bioMod",
        contact_types=contact_types,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
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
                        -3.09712665e-01,
                        -7.75546351e-02,
                        2.28919861e-01,
                        7.17520972e-02,
                        2.61640603e-02,
                        7.03511775e00,
                        6.83158357e-01,
                        -3.28035543e01,
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
                        -3.09712665e-01,
                        -7.75546351e-02,
                        2.28919861e-01,
                        7.17520972e-02,
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
                    [-0.31172315, -0.07805808, 0.23040588, 0.07221787, -0.13508945, 6.797133, 1.523731, -12.70223002]
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

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
        contact_types=[ContactType.RIGID_EXPLICIT] if with_contact else (),
        external_force_set=external_forces,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
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
                    np.array([-0.0140995 , -0.0071009 , -0.00592597, -0.00486693,  0.04135072,
                        0.02020007, -0.06055008,  0.06666342, -0.00814222,  0.00134606,
                        0.00986419, -0.01324398]),
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
                        np.array([-0.0140995 , -0.0071009 , -0.00592597, -0.00486693,  0.02289304,
                            0.18246443, -0.06771473, -0.21721173, -0.00814222,  0.00134606,
                            0.00986419, -0.01324398]),
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([-1.40994950e-02, -7.10089523e-03, -5.92597043e-03, -4.86693294e-03,
                           -5.70258892e+00, -1.45376254e+02, -7.08538520e+00, -1.08402622e+00,
                           -8.14221682e-03,  1.34606351e-03,  9.86418739e-03, -1.32439805e-02]),
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

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder()
        + "/examples/muscle_driven_with_contact/models/2segments_4dof_2soft_contacts_1muscle.bioMod",
        contact_types=contact_types,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
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
                        -0.01877799,
                        -0.02040225,
                        0.01385785,
                        -0.00667989,
                        0.10275146,
                        0.90393653,
                        -0.14864133,
                        -3.54475834,
                        -0.06529354,
                        -0.01496139,
                        -0.01074127,
                        0.04613038,
                        -0.96244729,
                        -0.03688695,
                        -0.90826589,
                        -0.24205527,
                        -0.36778313,
                        -0.8353025,
                        -0.67756436,
                        -0.17436643,
                        -0.34106635,
                        -0.65998405,
                        -0.09310277,
                        -0.34920957,
                    ]
                ),
            )
        else:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -1.87779876e-02,
                        -2.04022485e-02,
                        1.38578453e-02,
                        -6.67988697e-03,
                        -1.16039573e01,
                        -1.37716732e02,
                        -4.76824404e00,
                        -8.16422811e-01,
                        -6.52935358e-02,
                        -1.49613950e-02,
                        -1.07412701e-02,
                        4.61303802e-02,
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
                        -0.05619167,
                        -0.06105214,
                        0.04146852,
                        -0.01998904,
                        0.29345657,
                        2.69664409,
                        -0.37386083,
                        -8.89007806,
                        -0.19538582,
                        -0.04477081,
                        -0.03214241,
                        0.13804157,
                    ]
                ),
            )
        else:
            npt.assert_almost_equal(
                x_defects[:, 0],
                np.array(
                    [
                        -5.61916662e-02,
                        -6.10521404e-02,
                        4.14685233e-02,
                        -1.99890418e-02,
                        -1.16408442e01,
                        -1.38624998e02,
                        -4.97910457e00,
                        -9.99496342e-01,
                        -1.95385823e-01,
                        -4.47708098e-02,
                        -3.21424146e-02,
                        1.38041572e-01,
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

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
        contact_types=[ContactType.RIGID_EXPLICIT] if with_contact else (),
        external_force_set=external_forces,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN,
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
    if defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
        with pytest.raises(
            NotImplementedError,
            match="The defect type DefectType.TAU_EQUALS_INVERSE_DYNAMICS is not implemented yet for torque activations driven dynamics.",
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
                                -0.04383072,
                                -0.04283728,
                                0.10537375,
                                -0.04831128,
                                2.74968269,
                                0.46056293,
                                -5.48418391,
                                5.354188,
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
                                -0.31172315,
                                -0.07805808,
                                0.23040588,
                                0.07221787,
                                -28.13827671,
                                -0.1215312,
                                57.2699282,
                                -56.5784637,
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
                            [
                                -4.38307239e-02,
                                -4.28372820e-02,
                                1.05373750e-01,
                                -4.83112831e-02,
                                1.61331714e01,
                                8.13653930e00,
                                -7.65761291e01,
                                -3.89924217e02,
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
                                -3.11723149e-01,
                                -7.80580770e-02,
                                2.30405883e-01,
                                7.22178723e-02,
                                -4.20800305e01,
                                2.16475239e01,
                                1.24723562e02,
                                -1.37029516e03,
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
            TestUtils.bioptim_folder()
            + "/examples/torque_driven_ocp/models/2segments_2dof_2contacts_with_passive_torque.bioMod"
        )
    else:
        model_filename = (
            TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/2segments_2dof_2contacts.bioMod"
        )

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
        model_filename,
        external_force_set=external_forces,
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN,
        with_residual_torque=with_residual_torque,
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
    if defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
        with pytest.raises(
            NotImplementedError,
            match="The defect type DefectType.TAU_EQUALS_INVERSE_DYNAMICS is not implemented yet for torque activations driven dynamics.",
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
        if with_residual_torque:
            if with_passive_torque:
                if with_external_force:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([6.04508047e-01, 1.84981427e-01, -8.70762306e01, -1.03464970e03]),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.16044011, 0.1632901, -1.06310829, -27.22644766]), decimal=5
                    )
            else:
                if with_external_force:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([6.04508047e-01, 1.84981427e-01, -8.35362294e01, -1.00272333e03]),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.16044011, 0.1632901, -0.31706506, -17.5258886]), decimal=5
                    )
        else:
            if with_passive_torque:
                if with_external_force:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([5.28274079e-01, 1.61653585e-01, -7.53323145e01, -8.95479455e02]),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.45831154, 0.4664528, -2.70311395, -51.49785706]), decimal=5
                    )
            else:
                if with_external_force:
                    npt.assert_almost_equal(
                        x_defects[:, 0],
                        np.array([5.28274079e-01, 1.61653585e-01, -7.22387397e01, -8.67579288e02]),
                        decimal=5,
                    )
                else:
                    npt.assert_almost_equal(
                        x_defects[:, 0], np.array([0.45831154, 0.4664528, -0.57197461, -23.78734089]), decimal=5
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
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN_FREE_FLOATING_BASE,
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
                [-0.3573718, -0.08948888, 0.26414646, 0.08279344, -0.25072741, 7.9598835, 2.1430355, -25.96856935]
            ),
            decimal=5,
        )
    else:
        npt.assert_almost_equal(
            x_defects[:, 0],
            np.array(
                [-0.3573718, -0.08948888, 0.26414646, 0.08279344, -1.99330085, -1.99330085, -1.99330085, -1.13019743]
            ),
            decimal=6,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("contact_types", [[ContactType.RIGID_EXPLICIT], [ContactType.RIGID_IMPLICIT], ()])
@pytest.mark.parametrize("with_residual_torque", [False, True])
@pytest.mark.parametrize("with_excitations", [False, True])
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
def test_muscle_driven(
    with_excitations, contact_types, with_residual_torque, with_external_force, cx, phase_dynamics, defects_type
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
            nlp.model = BiorbdModel(
                TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26_with_contact.bioMod",
                contact_types=contact_types,
                external_force_set=external_forces,
            )
    elif ContactType.RIGID_IMPLICIT in contact_types:
        with pytest.raises(RuntimeError, match="The segment for the rigid contact index 0 was not found."):
            # TODO: This is a bug... The index of the parent of the contact is not correctly identified when it is the root
            nlp.model = BiorbdModel(
                TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26_with_contact.bioMod",
                contact_types=contact_types,
                external_force_set=external_forces,
            )
    else:
        nlp.model = BiorbdModel(
            TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26_with_contact.bioMod",
            contact_types=contact_types,
            external_force_set=external_forces,
        )
        nlp.dynamics_type = Dynamics(
            DynamicsFcn.MUSCLE_DRIVEN,
            with_residual_torque=with_residual_torque,
            with_excitations=with_excitations,
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
                    if with_excitations:
                        if with_external_force:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        -0.39414169,
                                        -0.34540175,
                                        -0.64642364,
                                        -0.32407953,
                                        28.31157446,
                                        11.05514434,
                                        -29.35153822,
                                        -7.14085992,
                                        18.29768098,
                                        11.45405575,
                                        7.31449586,
                                        4.18729136,
                                    ]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        6.58474157e-02,
                                        -2.21841265e-02,
                                        -4.14806864e-03,
                                        1.51595699e00,
                                        -4.10616583e01,
                                        1.46386667e02,
                                        3.25656270e00,
                                        -2.72705314e00,
                                        4.76521837e-01,
                                        -5.19721258e00,
                                        1.14956080e01,
                                        8.64588438e00,
                                    ]
                                ),
                                decimal=5,
                            )
                    else:
                        if with_external_force:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [-0.14206822, -0.22146577, -0.07762636, -0.18970781, 31.78037536, -23.11015379]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array([0.17405995, 0.08065849, -0.2721404, 0.46729892, -1.66989605, 49.55158537]),
                                decimal=5,
                            )
                else:
                    if with_excitations:
                        if with_external_force:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        -6.16079563e-02,
                                        -5.39894568e-02,
                                        -1.01041935e-01,
                                        -2.17790397e-03,
                                        3.85980483e00,
                                        3.53803553e00,
                                        -5.35239171e00,
                                        1.16917330e00,
                                        1.95016256e00,
                                        -7.20251522e-01,
                                        1.05381874e00,
                                        4.26798401e-01,
                                    ]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        4.91755352e-02,
                                        -1.65673365e-02,
                                        -3.09782083e-03,
                                        1.22322499e00,
                                        -3.17079402e01,
                                        1.12985229e02,
                                        -3.36444200e00,
                                        4.02911242e00,
                                        -1.03517852e01,
                                        3.09538706e00,
                                        1.52954858e-01,
                                        1.01881033e01,
                                    ]
                                ),
                                decimal=5,
                            )
                    else:
                        if with_external_force:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array([-0.24539012, -0.38253111, -0.13408166, 0.49788677, 13.85407958, 9.69500013]),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        3.02515888e-01,
                                        1.40184307e-01,
                                        -4.72979527e-01,
                                        1.72167922e00,
                                        -3.48494533e01,
                                        1.46223566e02,
                                    ]
                                ),
                                decimal=5,
                            )
            elif ContactType.RIGID_IMPLICIT in contact_types:
                if with_residual_torque:
                    if with_excitations:
                        if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        6.58474157e-02,
                                        -2.21841265e-02,
                                        -4.14806864e-03,
                                        1.51595699e00,
                                        -4.10616583e01,
                                        1.46386667e02,
                                        3.25656270e00,
                                        -2.72705314e00,
                                        4.76521837e-01,
                                        -5.19721258e00,
                                        1.14956080e01,
                                        8.64588438e00,
                                    ]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        6.58474157e-02,
                                        -2.21841265e-02,
                                        -4.14806864e-03,
                                        -1.70266428e00,
                                        -1.31826147e01,
                                        -1.28447704e01,
                                        3.25656270e00,
                                        -2.72705314e00,
                                        4.76521837e-01,
                                        -5.19721258e00,
                                        1.14956080e01,
                                        8.64588438e00,
                                    ]
                                ),
                                decimal=5,
                            )
                    else:
                        if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array([0.17405995, 0.08065849, -0.2721404, 0.46729892, -1.66989605, 49.55158537]),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array([0.17405995, 0.08065849, -0.2721404, -1.89342602, -16.46421687, -7.88393296]),
                                decimal=5,
                            )
                else:
                    if with_excitations:
                        if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        4.91755352e-02,
                                        -1.65673365e-02,
                                        -3.09782083e-03,
                                        1.22322499e00,
                                        -3.17079402e01,
                                        1.12985229e02,
                                        -3.36444200e00,
                                        4.02911242e00,
                                        -1.03517852e01,
                                        3.09538706e00,
                                        1.52954858e-01,
                                        1.01881033e01,
                                    ]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array(
                                    [
                                        4.91755352e-02,
                                        -1.65673365e-02,
                                        -3.09782083e-03,
                                        -2.51010444e00,
                                        -1.36097225e01,
                                        -1.32621814e01,
                                        -3.36444200e00,
                                        4.02911242e00,
                                        -1.03517852e01,
                                        3.09538706e00,
                                        1.52954858e-01,
                                        1.01881033e01,
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
                                        3.02515888e-01,
                                        1.40184307e-01,
                                        -4.72979527e-01,
                                        1.72167922e00,
                                        -3.48494533e01,
                                        1.46223566e02,
                                    ]
                                ),
                                decimal=5,
                            )
                        else:
                            npt.assert_almost_equal(
                                x_defects[:, 0],
                                np.array([0.30251589, 0.14018431, -0.47297953, -2.28210331, -9.70670533, -7.06713657]),
                                decimal=5,
                            )
            else:
                if with_residual_torque:
                    if with_excitations:
                        if with_external_force:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            -0.39414169,
                                            -0.34540175,
                                            -0.64642364,
                                            -0.32407953,
                                            28.31157446,
                                            11.05514434,
                                            -29.35153822,
                                            -7.14085992,
                                            18.29768098,
                                            11.45405575,
                                            7.31449586,
                                            4.18729136,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            -0.39414169,
                                            -0.34540175,
                                            -0.64642364,
                                            0.31041669,
                                            -19.22248632,
                                            -6.93204239,
                                            -29.35153822,
                                            -7.14085992,
                                            18.29768098,
                                            11.45405575,
                                            7.31449586,
                                            4.18729136,
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
                                            6.58474157e-02,
                                            -2.21841265e-02,
                                            -4.14806864e-03,
                                            1.51595699e00,
                                            -4.10616583e01,
                                            1.46386667e02,
                                            3.25656270e00,
                                            -2.72705314e00,
                                            4.76521837e-01,
                                            -5.19721258e00,
                                            1.14956080e01,
                                            8.64588438e00,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            6.58474157e-02,
                                            -2.21841265e-02,
                                            -4.14806864e-03,
                                            -1.70266428e00,
                                            -1.31826147e01,
                                            -1.28447704e01,
                                            3.25656270e00,
                                            -2.72705314e00,
                                            4.76521837e-01,
                                            -5.19721258e00,
                                            1.14956080e01,
                                            8.64588438e00,
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
                                        [-0.14206822, -0.22146577, -0.07762636, -0.18970781, 31.78037536, -23.11015379]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [-0.14206822, -0.22146577, -0.07762636, -2.14326015, -19.93936948, -6.05447738]
                                    ),
                                    decimal=5,
                                )
                        else:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [0.17405995, 0.08065849, -0.2721404, 0.46729892, -1.66989605, 49.55158537]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [0.17405995, 0.08065849, -0.2721404, -1.89342602, -16.46421687, -7.88393296]
                                    ),
                                    decimal=5,
                                )
                else:
                    if with_excitations:
                        if with_external_force:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            -6.16079563e-02,
                                            -5.39894568e-02,
                                            -1.01041935e-01,
                                            -2.17790397e-03,
                                            3.85980483e00,
                                            3.53803553e00,
                                            -5.35239171e00,
                                            1.16917330e00,
                                            1.95016256e00,
                                            -7.20251522e-01,
                                            1.05381874e00,
                                            4.26798401e-01,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            -0.06160796,
                                            -0.05398946,
                                            -0.10104194,
                                            -0.59784919,
                                            -19.46454159,
                                            -7.29982552,
                                            -5.35239171,
                                            1.1691733,
                                            1.95016256,
                                            -0.72025152,
                                            1.05381874,
                                            0.4267984,
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
                                            4.91755352e-02,
                                            -1.65673365e-02,
                                            -3.09782083e-03,
                                            1.22322499e00,
                                            -3.17079402e01,
                                            1.12985229e02,
                                            -3.36444200e00,
                                            4.02911242e00,
                                            -1.03517852e01,
                                            3.09538706e00,
                                            1.52954858e-01,
                                            1.01881033e01,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            4.91755352e-02,
                                            -1.65673365e-02,
                                            -3.09782083e-03,
                                            -2.51010444e00,
                                            -1.36097225e01,
                                            -1.32621814e01,
                                            -3.36444200e00,
                                            4.02911242e00,
                                            -1.03517852e01,
                                            3.09538706e00,
                                            1.52954858e-01,
                                            1.01881033e01,
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
                                        [-0.24539012, -0.38253111, -0.13408166, 0.49788677, 13.85407958, 9.69500013]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [-0.24539012, -0.38253111, -0.13408166, -2.2628544, -8.36653992, -3.06929409]
                                    ),
                                    decimal=5,
                                )
                        else:
                            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [
                                            3.02515888e-01,
                                            1.40184307e-01,
                                            -4.72979527e-01,
                                            1.72167922e00,
                                            -3.48494533e01,
                                            1.46223566e02,
                                        ]
                                    ),
                                    decimal=5,
                                )
                            else:
                                npt.assert_almost_equal(
                                    x_defects[:, 0],
                                    np.array(
                                        [0.30251589, 0.14018431, -0.47297953, -2.28210331, -9.70670533, -7.06713657]
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
    nlp.model = BiorbdModel(TestUtils.bioptim_folder() + "/examples/getting_started/models/double_pendulum.bioMod")
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.JOINTS_ACCELERATION_DRIVEN,
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
        npt.assert_almost_equal(x_defects[:, 0], np.array([0.18430491, 0.18757883, 1.05031557, 0.21394574]), decimal=5)
