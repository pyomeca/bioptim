import pytest

import numpy as np
from casadi import MX, SX
from bioptim import (
    ConfigureProblem,
    ControlType,
    RigidBodyDynamics,
    BiorbdModel,
    NonLinearProgram,
    DynamicsFcn,
    Dynamics,
    ConstraintList,
    Solver,
    VariableScalingList,
    ParameterList,
    PhaseDynamics,
    SolutionMerge,
)
from tests.utils import TestUtils
import os


class OptimalControlProgram:
    def __init__(self, nlp):
        self.cx = nlp.cx
        self.n_phases = 1
        self.nlp = [nlp]
        self.parameters = ParameterList()
        self.implicit_constraints = ConstraintList()
        self.n_threads = 1


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_passive_torque", [False, True])
@pytest.mark.parametrize("rigidbody_dynamics", [RigidBodyDynamics.ODE])
def test_torque_driven_with_passive_torque(with_passive_torque, cx, rigidbody_dynamics, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics)
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.time_mx = MX.sym("time", 1, 1)
    nlp.dt_mx = MX.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_scaling = VariableScalingList()
    nlp.xdot_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.TORQUE_DRIVEN,
            rigidbody_dynamics=rigidbody_dynamics,
            with_passive_torque=with_passive_torque,
            phase_dynamics=phase_dynamics,
        ),
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)
    use_states_from_phase_idx = [i for i in range(ocp.n_phases)]
    use_states_dot_from_phase_idx = [i for i in range(ocp.n_phases)]
    use_controls_from_phase_idx = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "use_states_from_phase_idx", use_states_from_phase_idx, False)
    NonLinearProgram.add(ocp, "use_states_dot_from_phase_idx", use_states_dot_from_phase_idx, False)
    NonLinearProgram.add(ocp, "use_controls_from_phase_idx", use_controls_from_phase_idx, False)

    np.random.seed(42)

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    time = np.random.rand(2)
    x_out = np.array(nlp.dynamics_func[0](time, states, controls, params, algebraic_states))
    if rigidbody_dynamics == RigidBodyDynamics.ODE:
        if with_passive_torque:
            np.testing.assert_almost_equal(
                x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, -5.0261535, -10.5570666, 18.569191, 24.2237134]
            )
        else:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [0.61185289, 0.78517596, 0.60754485, 0.80839735, -0.30241366, -10.38503791, 1.60445173, 35.80238642],
            )
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
        if with_passive_torque:
            np.testing.assert_almost_equal(
                x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072]
            )
        else:
            np.testing.assert_almost_equal(
                x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072]
            )
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        if with_passive_torque:
            np.testing.assert_almost_equal(
                x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072]
            )
        else:
            np.testing.assert_almost_equal(
                x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072]
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_passive_torque", [False, True])
def test_torque_derivative_driven_with_passive_torque(with_passive_torque, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics)
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.time_mx = MX.sym("time", 1, 1)
    nlp.dt_mx = MX.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_scaling = VariableScalingList()
    nlp.xdot_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT

    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
            with_passive_torque=with_passive_torque,
            phase_dynamics=phase_dynamics,
        ),
        False,
    )

    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)
    use_states_from_phase_idx = [i for i in range(ocp.n_phases)]
    use_states_dot_from_phase_idx = [i for i in range(ocp.n_phases)]
    use_controls_from_phase_idx = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "use_states_from_phase_idx", use_states_from_phase_idx, False)
    NonLinearProgram.add(ocp, "use_states_dot_from_phase_idx", use_states_dot_from_phase_idx, False)
    NonLinearProgram.add(ocp, "use_controls_from_phase_idx", use_controls_from_phase_idx, False)

    np.random.seed(42)

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    time = np.random.rand(2)
    x_out = np.array(nlp.dynamics_func[0](time, states, controls, params, algebraic_states))
    if with_passive_torque:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [
                0.6118529,
                0.785176,
                0.6075449,
                0.8083973,
                -5.0261535,
                -10.5570666,
                18.569191,
                24.2237134,
                0.3886773,
                0.5426961,
                0.7722448,
                0.7290072,
            ],
        )
    else:
        np.testing.assert_almost_equal(
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
@pytest.mark.parametrize("with_passive_torque", [False, True])
@pytest.mark.parametrize("with_residual_torque", [False, True])
def test_torque_activation_driven_with_passive_torque(with_passive_torque, with_residual_torque, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics)
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.time_mx = MX.sym("time", 1, 1)
    nlp.dt_mx = MX.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_scaling = VariableScalingList()
    nlp.xdot_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN,
            with_passive_torque=with_passive_torque,
            with_residual_torque=with_residual_torque,
            phase_dynamics=phase_dynamics,
        ),
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)
    use_states_from_phase_idx = [i for i in range(ocp.n_phases)]
    use_states_dot_from_phase_idx = [i for i in range(ocp.n_phases)]
    use_controls_from_phase_idx = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "use_states_from_phase_idx", use_states_from_phase_idx, False)
    NonLinearProgram.add(ocp, "use_states_dot_from_phase_idx", use_states_dot_from_phase_idx, False)
    NonLinearProgram.add(ocp, "use_controls_from_phase_idx", use_controls_from_phase_idx, False)

    np.random.seed(42)

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    time = np.random.rand(2)
    x_out = np.array(nlp.dynamics_func[0](time, states, controls, params, algebraic_states))
    if with_residual_torque:
        if with_passive_torque:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [
                    0.6118528947,
                    0.7851759614,
                    0.6075448519,
                    0.8083973481,
                    -28.6265970388,
                    -58.7530113476,
                    145.0864163235,
                    3682.9683657415,
                ],
                decimal=5,
            )
        else:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [
                    0.6118528947,
                    0.7851759614,
                    0.6075448519,
                    0.8083973481,
                    -23.9028572107,
                    -58.5809826745,
                    128.1216770837,
                    3694.5470387809,
                ],
                decimal=5,
            )

    else:
        if with_passive_torque:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [
                    6.1185289472e-01,
                    7.8517596139e-01,
                    6.0754485190e-01,
                    8.0839734812e-01,
                    -2.8550037341e01,
                    -5.8375374025e01,
                    1.4440375924e02,
                    3.6537329536e03,
                ],
                decimal=6,
            )
        else:
            np.testing.assert_almost_equal(
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
@pytest.mark.parametrize("with_passive_torque", [False, True])
@pytest.mark.parametrize("rigidbody_dynamics", [RigidBodyDynamics.ODE])
def test_muscle_driven_with_passive_torque(with_passive_torque, rigidbody_dynamics, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics)
    nlp.model = BiorbdModel(TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26_with_contact.bioMod")
    nlp.ns = 5
    nlp.cx = cx
    nlp.time_mx = MX.sym("time", 1, 1)
    nlp.dt_mx = MX.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_scaling = VariableScalingList()
    nlp.xdot_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2 + nlp.model.nb_muscles, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_muscles, 1))

    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.MUSCLE_DRIVEN,
            rigidbody_dynamics=rigidbody_dynamics,
            with_passive_torque=with_passive_torque,
            phase_dynamics=phase_dynamics,
        ),
        False,
    )
    phase_index = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "phase_idx", phase_index, False)
    use_states_from_phase_idx = [i for i in range(ocp.n_phases)]
    use_states_dot_from_phase_idx = [i for i in range(ocp.n_phases)]
    use_controls_from_phase_idx = [i for i in range(ocp.n_phases)]
    NonLinearProgram.add(ocp, "use_states_from_phase_idx", use_states_from_phase_idx, False)
    NonLinearProgram.add(ocp, "use_states_dot_from_phase_idx", use_states_dot_from_phase_idx, False)
    NonLinearProgram.add(ocp, "use_controls_from_phase_idx", use_controls_from_phase_idx, False)

    np.random.seed(42)

    # Prepare the dynamics
    if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        pass
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    algebraic_states = np.random.rand(nlp.algebraic_states.shape, nlp.ns)
    time = np.random.rand(2)
    x_out = np.array(nlp.dynamics_func[0](time, states, controls, params, algebraic_states))

    if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        if with_passive_torque:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [0.183405, 0.611853, 0.785176, 0.388677, 0.542696, 0.772245],
                decimal=6,
            )
        else:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [0.183405, 0.611853, 0.785176, 0.388677, 0.542696, 0.772245],
                decimal=6,
            )
    else:
        if with_passive_torque:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [
                    1.8340450985e-01,
                    6.1185289472e-01,
                    7.8517596139e-01,
                    -5.3408086130e00,
                    1.6890917494e02,
                    -5.4766884856e02,
                ],
                decimal=6,
            )
        else:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [1.83404510e-01, 6.11852895e-01, 7.85175961e-01, -4.37708456e00, 1.33221135e02, -4.71307550e02],
                decimal=6,
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize(
    "rigidbody_dynamics", [RigidBodyDynamics.DAE_FORWARD_DYNAMICS, RigidBodyDynamics.DAE_INVERSE_DYNAMICS]
)
@pytest.mark.parametrize("with_passive_torque", [False, True])
def test_pendulum_passive_torque(rigidbody_dynamics, with_passive_torque, phase_dynamics):
    from bioptim.examples.torque_driven_ocp import pendulum_with_passive_torque as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/pendulum_with_passive_torque.bioMod"
    final_time = 1
    n_shooting = 30

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path,
        final_time,
        n_shooting,
        rigidbody_dynamics=rigidbody_dynamics,
        with_passive_torque=with_passive_torque,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    solver = Solver.IPOPT()

    # solver.set_maximum_iterations(10)
    sol = ocp.solve(solver)

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        if with_passive_torque:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.0, 3.14]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0]))
            # initial and final controls
            np.testing.assert_almost_equal(
                tau[:, 0],
                np.array([-1.071535, 0.0]),
                decimal=6,
            )
            np.testing.assert_almost_equal(tau[:, -1], np.array([-19.422394, 0.0]), decimal=6)

        else:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.0, 3.14]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0]))
            # initial and final controls
            np.testing.assert_almost_equal(
                tau[:, 0],
                np.array([2.531529, 0.0]),
                decimal=6,
            )
            np.testing.assert_almost_equal(
                tau[:, -1],
                np.array([-18.254416, 0.0]),
                decimal=6,
            )

    else:
        if with_passive_torque:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.0, 3.14]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0]))
            # initial and final controls
            np.testing.assert_almost_equal(
                tau[:, 0],
                np.array([1.587319, 0.0]),
                decimal=6,
            )
            np.testing.assert_almost_equal(
                tau[:, -1],
                np.array([-39.19793, 0.0]),
                decimal=6,
            )

        else:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.0, 3.14]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0]))
            # initial and final controls
            np.testing.assert_almost_equal(
                tau[:, 0],
                np.array([2.606971, 0.0]),
                decimal=6,
            )
            np.testing.assert_almost_equal(
                tau[:, -1],
                np.array([-24.611219, 0.0]),
                decimal=6,
            )
