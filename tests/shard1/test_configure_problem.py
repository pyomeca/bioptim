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
def test_configure_q(with_contact, with_external_force, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.ns = N_SHOOTING
    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))

    external_forces = None
    numerical_time_series = None
    if with_external_force:

        external_forces = ExternalForceSetTimeSeries(nb_frames=nlp.ns)
        external_forces.add("Seg0", EXTERNAL_FORCE_ARRAY[:6, :], point_of_application=EXTERNAL_FORCE_ARRAY[6:, :])
        numerical_time_series = {"external_forces": external_forces.to_numerical_time_series()}

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
        external_force_set=external_forces,
    )

    # Nothing in there
    npt.assert_equal(nlp.states.shape, 0)
    npt.assert_equal(nlp.controls.shape, 0)
    npt.assert_equal(nlp.parameters.shape, 0)
    npt.assert_equal(nlp.algebraic_states.shape, 0)

    # q
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    npt.assert_equal(nlp.states.shape, 4)
    npt.assert_equal(nlp.states.keys(), ['q'])



####### TODO: CHARBIE --------------------------------------

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
        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False)
        ConfigureProblem.configure_tau(ocp, nlp, False, True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, with_contact=with_contact)

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=False)
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = N_SHOOTING
    nlp.cx = MX
    nlp.time_cx = nlp.cx.sym("time", 1, 1)
    nlp.dt = nlp.cx.sym("dt", 1, 1)
    nlp.initialize(nlp.cx)
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    nlp.x_scaling = VariableScalingList()
    nlp.xdot_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    ocp = OptimalControlProgram(nlp, use_sx=False)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            configure,
            dynamic_function=custom_dynamic,
            with_contact=with_contact,
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
        ),
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
