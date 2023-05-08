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
)
from bioptim.optimization.optimization_vector import OptimizationVector
from .utils import TestUtils
import os


class OptimalControlProgram:
    def __init__(self, nlp):
        self.cx = nlp.cx
        self.assume_phase_dynamics = True
        self.n_phases = 1
        self.nlp = [nlp]
        self.v = OptimizationVector(self)
        self.implicit_constraints = ConstraintList()


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize(
    "with_ligament",
    [
        False,
        True,
    ],
)
def test_torque_driven_with_ligament(with_ligament, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/mass_point_with_ligament.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.initialize(cx)
    nlp.x_scaling = {}
    nlp.xdot_scaling = {}
    nlp.u_scaling = {}

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(DynamicsFcn.TORQUE_DRIVEN, rigidbody_dynamics=RigidBodyDynamics.ODE, with_ligament=with_ligament),
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
    x_out = np.array(nlp.dynamics_func(states, controls, params))
    if with_ligament:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.1559945, -47.2537196],
        )
    else:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.1559945, -9.7997078],
        )


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_ligament", [False, True])
def test_torque_derivative_driven_with_ligament(with_ligament, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/mass_point_with_ligament.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.initialize(cx)
    nlp.x_scaling = {}
    nlp.xdot_scaling = {}
    nlp.u_scaling = {}

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT

    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, with_ligament=with_ligament),
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
    x_out = np.array(nlp.dynamics_func(states, controls, params))
    if with_ligament:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.1559945, -47.2537196, 0.1834045],
        )
    else:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.1559945, -9.7997078, 0.1834045],
        )


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_ligament", [False, True])
def test_torque_activation_driven_with_ligament(with_ligament, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/mass_point_with_ligament.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.initialize(cx)
    nlp.x_scaling = {}
    nlp.xdot_scaling = {}
    nlp.u_scaling = {}
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_ligament=with_ligament),
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
    x_out = np.array(nlp.dynamics_func(states, controls, params))
    if with_ligament:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.155995, -46.234787],
            decimal=6,
        )
    else:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.15599, -8.78078],
            decimal=5,
        )


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_ligament", [False, True])
def test_muscle_driven_with_ligament(with_ligament, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26_with_ligament.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.initialize(cx)
    nlp.x_scaling = {}
    nlp.xdot_scaling = {}
    nlp.u_scaling = {}
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2 + nlp.model.nb_muscles, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_muscles, 1))

    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.MUSCLE_DRIVEN,
            rigidbody_dynamics=RigidBodyDynamics.ODE,
            with_ligament=with_ligament,
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
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_ligament:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [2.0584494e-02, 1.8340451e-01, -6.0300944e00, -9.4582028e01],
            decimal=6,
        )
    else:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [2.0584494e-02, 1.8340451e-01, -7.3880194e00, -9.0642142e01],
            decimal=6,
        )


@pytest.mark.parametrize(
    "rigidbody_dynamics",
    [
        RigidBodyDynamics.DAE_FORWARD_DYNAMICS,
        RigidBodyDynamics.DAE_INVERSE_DYNAMICS,
    ],
)
@pytest.mark.parametrize(
    "with_ligament",
    [
        False,
        True,
    ],
)
def test_ocp_mass_ligament(rigidbody_dynamics, with_ligament):
    from bioptim.examples.torque_driven_ocp import ocp_mass_with_ligament as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/mass_point_with_ligament.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path,
        rigidbody_dynamics=rigidbody_dynamics,
        with_ligament=with_ligament,
    )
    solver = Solver.IPOPT()

    # solver.set_maximum_iterations(10)
    sol = ocp.solve(solver)

    # Check some results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        if with_ligament:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.0]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.0194773]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.3061592]))
            # initial and final controls
            np.testing.assert_almost_equal(
                tau[:, 0],
                np.array([2.158472e-16]),
                decimal=6,
            )
            np.testing.assert_almost_equal(tau[:, -2], np.array([1.423733e-17]), decimal=6)

        else:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.0]))
            np.testing.assert_almost_equal(q[:, -1], np.array([-3.1415927]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([-7.2608855]))
            # initial and final controls
            np.testing.assert_almost_equal(
                tau[:, 0],
                np.array([24.594638]),
                decimal=6,
            )
            np.testing.assert_almost_equal(
                tau[:, -2],
                np.array([0.123591]),
                decimal=6,
            )

    else:
        if with_ligament:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.0]))
            np.testing.assert_almost_equal(q[:, -1], np.array([0.0194773]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.3061592]))
            # initial and final controls
            np.testing.assert_almost_equal(
                tau[:, 0],
                np.array([2.158472e-16]),
                decimal=6,
            )
            np.testing.assert_almost_equal(
                tau[:, -2],
                np.array([1.423733e-17]),
                decimal=6,
            )

        else:
            # initial and final position
            np.testing.assert_almost_equal(q[:, 0], np.array([0.0]))
            np.testing.assert_almost_equal(q[:, -1], np.array([-3.1415927]))
            # initial and final velocities
            np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0]))
            np.testing.assert_almost_equal(qdot[:, -1], np.array([-7.2608855]))
            # initial and final controls
            np.testing.assert_almost_equal(
                tau[:, 0],
                np.array([24.594638]),
                decimal=6,
            )
            np.testing.assert_almost_equal(
                tau[:, -2],
                np.array([0.123591]),
                decimal=6,
            )
