import pytest
import re

import numpy as np
from casadi import MX, SX, vertcat
import biorbd_casadi as biorbd
from bioptim.dynamics.configure_problem import ConfigureProblem
from bioptim.dynamics.dynamics_functions import DynamicsFunctions
from bioptim.interfaces.biorbd_interface import BiorbdInterface
from bioptim.misc.enums import ControlType, RigidBodyDynamics, SoftContactDynamics
from bioptim.optimization.non_linear_program import NonLinearProgram
from bioptim.optimization.optimization_vector import OptimizationVector
from bioptim.dynamics.configure_problem import DynamicsFcn, Dynamics
from bioptim.dynamics.dynamics_evaluation import DynamicsEvaluation
from bioptim.limits.constraints import ConstraintList
from .utils import TestUtils


class OptimalControlProgram:
    def __init__(self, nlp):
        self.n_phases = 1
        self.nlp = [nlp]
        self.v = OptimizationVector(self)
        self.implicit_constraints = ConstraintList()


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize(
    "rigidbody_dynamics",
    [RigidBodyDynamics.ODE, RigidBodyDynamics.DAE_FORWARD_DYNAMICS, RigidBodyDynamics.DAE_INVERSE_DYNAMICS],
)
def test_torque_driven(with_contact, with_external_force, cx, rigidbody_dynamics):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.phase_idx = 0

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=with_contact, rigidbody_dynamics=rigidbody_dynamics),
        False,
    )

    np.random.seed(42)
    if with_external_force:
        external_forces = [np.random.rand(6, nlp.model.nbSegment(), nlp.ns)]
        nlp.external_forces = BiorbdInterface.convert_array_to_external_forces(external_forces)[0]

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))
    if rigidbody_dynamics == RigidBodyDynamics.ODE:
        if with_contact:
            contact_out = np.array(nlp.contact_forces_func(states, controls, params))
            if with_external_force:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.8631034, 0.3251833, 0.1195942, 0.4937956, -7.7700092, -7.5782306, 21.7073786, -16.3059315],
                )
                np.testing.assert_almost_equal(contact_out[:, 0], [-47.8131136, 111.1726516, -24.4449121])
            else:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3214905, -0.1912131, 0.6507164, -0.2359716],
                )
                np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

        else:
            if with_external_force:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.86310343, 0.32518332, 0.11959425, 0.4937956, 0.30731739, -9.97912778, 1.15263778, 36.02430956],
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
                    ],
                )
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
        if with_contact:
            contact_out = np.array(nlp.contact_forces_func(states, controls, params))
            if with_external_force:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.8631034, 0.3251833, 0.1195942, 0.4937956, 0.8074402, 0.4271078, 0.417411, 0.3232029],
                )
                np.testing.assert_almost_equal(contact_out[:, 0], [-47.8131136, 111.1726516, -24.4449121])
            else:
                np.testing.assert_almost_equal(
                    x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072]
                )
                np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

        else:
            if with_external_force:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.8631034, 0.3251833, 0.1195942, 0.4937956, 0.8074402, 0.4271078, 0.417411, 0.3232029],
                )
            else:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072],
                )
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        if with_contact:
            contact_out = np.array(nlp.contact_forces_func(states, controls, params))
            if with_external_force:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.8631034, 0.3251833, 0.1195942, 0.4937956, 0.8074402, 0.4271078, 0.417411, 0.3232029],
                )
                np.testing.assert_almost_equal(contact_out[:, 0], [-47.8131136, 111.1726516, -24.4449121])
            else:
                np.testing.assert_almost_equal(
                    x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072]
                )
                np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

        else:
            if with_external_force:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.8631034, 0.3251833, 0.1195942, 0.4937956, 0.8074402, 0.4271078, 0.417411, 0.3232029],
                )
            else:
                np.testing.assert_almost_equal(
                    x_out[:, 0],
                    [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072],
                )


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_driven_implicit(with_contact, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ() * 2, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    nlp.phase_idx = 0

    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.TORQUE_DRIVEN,
            with_contact=with_contact,
            rigidbody_dynamics=RigidBodyDynamics.DAE_INVERSE_DYNAMICS,
        ),
        False,
    )

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    np.random.seed(42)
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_contact:
        contact_out = np.array(nlp.contact_forces_func(states, controls, params))
        np.testing.assert_almost_equal(
            x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072]
        )

        np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.6118529, 0.785176, 0.6075449, 0.8083973, 0.3886773, 0.5426961, 0.7722448, 0.7290072],
        )


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize("implicit_contact", [False, True])
def test_torque_driven_soft_contacts_dynamics(with_contact, cx, implicit_contact):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * (2 + 3), 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ() * 2, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    nlp.phase_idx = 0

    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=with_contact, soft_contacts_dynamics=implicit_contact),
        False,
    )

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    np.random.seed(42)
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_contact:
        contact_out = np.array(nlp.contact_forces_func(states, controls, params))
        np.testing.assert_almost_equal(
            x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3214905, -0.1912131, 0.6507164, -0.2359716]
        )

        np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3024137, -10.3850379, 1.6044517, 35.8023864],
        )


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_derivative_driven(with_contact, with_external_force, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    nlp.phase_idx = 0

    NonLinearProgram.add(
        ocp, "dynamics_type", Dynamics(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, with_contact=with_contact), False
    )

    np.random.seed(42)
    if with_external_force:
        external_forces = [np.random.rand(6, nlp.model.nbSegment(), nlp.ns)]
        nlp.external_forces = BiorbdInterface.convert_array_to_external_forces(external_forces)[0]

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_contact:
        contact_out = np.array(nlp.contact_forces_func(states, controls, params))
        if with_external_force:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [
                    0.8631034,
                    0.3251833,
                    0.1195942,
                    0.4937956,
                    -7.7700092,
                    -7.5782306,
                    21.7073786,
                    -16.3059315,
                    0.8074402,
                    0.4271078,
                    0.417411,
                    0.3232029,
                ],
            )
            np.testing.assert_almost_equal(contact_out[:, 0], [-47.8131136, 111.1726516, -24.4449121])
        else:
            np.testing.assert_almost_equal(
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
            np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        if with_external_force:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [
                    0.86310343,
                    0.32518332,
                    0.11959425,
                    0.4937956,
                    0.30731739,
                    -9.97912778,
                    1.15263778,
                    36.02430956,
                    0.80744016,
                    0.42710779,
                    0.417411,
                    0.32320293,
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


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_derivative_driven_implicit(with_contact, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.phase_idx = 0
    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 4, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 2))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
            with_contact=with_contact,
            rigidbody_dynamics=RigidBodyDynamics.DAE_INVERSE_DYNAMICS,
        ),
        False,
    )

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    np.random.seed(42)
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_contact:
        contact_out = np.array(nlp.contact_forces_func(states, controls, params))
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [
                0.6118529,
                0.785176,
                0.6075449,
                0.8083973,
                0.3886773,
                0.5426961,
                0.7722448,
                0.7290072,
                0.8631034,
                0.3251833,
                0.1195942,
                0.4937956,
                0.0314292,
                0.2492922,
                0.2897515,
                0.8714606,
            ],
        )
        np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])
    else:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [
                0.6118529,
                0.785176,
                0.6075449,
                0.8083973,
                0.3886773,
                0.5426961,
                0.7722448,
                0.7290072,
                0.8631034,
                0.3251833,
                0.1195942,
                0.4937956,
                0.0314292,
                0.2492922,
                0.2897515,
                0.8714606,
            ],
        )


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize("implicit_contact", [False, True])
def test_torque_derivative_driven_soft_contacts_dynamics(with_contact, cx, implicit_contact):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * (2 + 3), 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ() * 4, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, with_contact=with_contact, soft_contacts_dynamics=implicit_contact
        ),
        False,
    )

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    np.random.seed(42)
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_contact:
        contact_out = np.array(nlp.contact_forces_func(states, controls, params))
        np.testing.assert_almost_equal(
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

        np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        np.testing.assert_almost_equal(
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


@pytest.mark.parametrize(
    "dynamics",
    [DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, DynamicsFcn.MUSCLE_DRIVEN],
)
def test_soft_contacts_dynamics_errors(dynamics):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = MX

    nlp.u_bounds = np.zeros((nlp.model.nbQ() * 4, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(dynamics, soft_contacts_dynamics=True),
        False,
    )

    # Prepare the dynamics
    with pytest.raises(
        TypeError,
        match=re.escape(f"{dynamics.name.lower()}() got an unexpected keyword argument " "'soft_contacts_dynamics'"),
    ):
        ConfigureProblem.initialize(ocp, nlp)


@pytest.mark.parametrize(
    "dynamics",
    [DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN],
)
def test_implicit_dynamics_errors(dynamics):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = MX

    nlp.u_bounds = np.zeros((nlp.model.nbQ() * 4, 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(dynamics, rigidbody_dynamics=RigidBodyDynamics.DAE_INVERSE_DYNAMICS),
        False,
    )

    # Prepare the dynamics
    with pytest.raises(
        TypeError,
        match=re.escape(f"{dynamics.name.lower()}() got an unexpected keyword argument " "'rigidbody_dynamics'"),
    ):
        ConfigureProblem.initialize(ocp, nlp)


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_activation_driven(with_contact, with_external_force, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx
    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 2, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp, "dynamics_type", Dynamics(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_contact=with_contact), False
    )

    np.random.seed(42)
    if with_external_force:
        external_forces = [np.random.rand(6, nlp.model.nbSegment(), nlp.ns)]
        nlp.external_forces = BiorbdInterface.convert_array_to_external_forces(external_forces)[0]

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_contact:
        contact_out = np.array(nlp.contact_forces_func(states, controls, params))
        if with_external_force:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [0.8631, 0.32518, 0.11959, 0.4938, 19.01887, 18.51503, -53.08574, 58.48719],
                decimal=5,
            )
            np.testing.assert_almost_equal(contact_out[:, 0], [109.8086936, 3790.3932439, -3571.7858574])
        else:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [0.61185289, 0.78517596, 0.60754485, 0.80839735, 0.78455384, -0.16844256, -1.56184114, 1.97658587],
                decimal=5,
            )
            np.testing.assert_almost_equal(contact_out[:, 0], [-7.88958997, 329.70828173, -263.55516549])

    else:
        if with_external_force:
            np.testing.assert_almost_equal(
                x_out[:, 0],
                [
                    8.63103426e-01,
                    3.25183322e-01,
                    1.19594246e-01,
                    4.93795596e-01,
                    1.73558072e01,
                    -4.69891264e01,
                    1.81396922e02,
                    3.61170139e03,
                ],
                decimal=5,
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


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
@pytest.mark.parametrize("with_torque", [False, True])
@pytest.mark.parametrize("with_excitations", [False, True])
@pytest.mark.parametrize("rigidbody_dynamics", [RigidBodyDynamics.ODE, RigidBodyDynamics.DAE_INVERSE_DYNAMICS])
def test_muscle_driven(with_excitations, with_contact, with_torque, with_external_force, rigidbody_dynamics, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26_with_contact.bioMod"
    )
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 2 + nlp.model.nbMuscles(), 1))
    nlp.u_bounds = np.zeros((nlp.model.nbMuscles(), 1))
    nlp.phase_idx = 0

    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.MUSCLE_DRIVEN,
            with_torque=with_torque,
            with_excitations=with_excitations,
            with_contact=with_contact,
            rigidbody_dynamics=rigidbody_dynamics,
        ),
        False,
    )

    np.random.seed(42)
    if with_external_force:
        external_forces = [np.random.rand(6, nlp.model.nbSegment(), nlp.ns)]
        nlp.external_forces = BiorbdInterface.convert_array_to_external_forces(external_forces)[0]

    # Prepare the dynamics
    if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        pass
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))
    print("x_out", x_out[:, 0], "\n")

    if with_contact:  # Warning this test is a bit bogus, there since the model does not have contacts
        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            if with_torque:
                if with_excitations:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.6158501,
                                0.5031363,
                                0.6424193,
                                0.7009691,
                                0.0848377,
                                0.9472486,
                                46.8792802,
                                -1.8018903,
                                53.3914525,
                                48.3005692,
                                63.6937337,
                                -28.1570099,
                            ],
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.183405,
                                0.611853,
                                0.785176,
                                0.249292,
                                0.289751,
                                0.871461,
                                8.606308,
                                3.194336,
                                29.740561,
                                -20.275423,
                                -23.246778,
                                -41.913501,
                            ],
                            decimal=6,
                        )
                else:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.61585, 0.503136, 0.642419, 0.895523, 0.319314, 0.448446],
                            decimal=6,
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.183405, 0.611853, 0.785176, 0.729007, 0.863103, 0.325183],
                            decimal=6,
                        )

            else:
                if with_excitations:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.6158501,
                                0.5031363,
                                0.6424193,
                                0.791579,
                                0.5495289,
                                0.1429917,
                                55.6555782,
                                50.4705269,
                                0.3602559,
                                58.9237749,
                                29.7009419,
                                -15.1353494,
                            ],
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                1.83404510e-01,
                                6.11852895e-01,
                                7.85175961e-01,
                                1.19594246e-01,
                                4.93795596e-01,
                                3.14291857e-02,
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
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.6158501, 0.5031363, 0.6424193, 0.9905051, 0.9307573, 0.1031239],
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.183405, 0.611853, 0.785176, 0.388677, 0.542696, 0.772245],
                            decimal=6,
                        )
        else:
            if with_torque:
                if with_excitations:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.6158501,
                                0.5031363,
                                0.6424193,
                                1.7685411,
                                -54.335106,
                                102.170402,
                                46.8792802,
                                -1.8018903,
                                53.3914525,
                                48.3005692,
                                63.6937337,
                                -28.1570099,
                            ],
                        )
                    else:
                        np.testing.assert_almost_equal(
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
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                6.15850098e-01,
                                5.03136259e-01,
                                6.42419278e-01,
                                -5.27793227e00,
                                1.57668597e02,
                                -5.16345913e02,
                            ],
                            decimal=6,
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.183405, 0.611853, 0.785176, -0.867138, 22.511947, -153.294775],
                            decimal=6,
                        )

            else:
                if with_excitations:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.6158501,
                                0.5031363,
                                0.6424193,
                                1.6162719,
                                -59.6216891,
                                111.2706156,
                                55.6555782,
                                50.4705269,
                                0.3602559,
                                58.9237749,
                                29.7009419,
                                -15.1353494,
                            ],
                        )
                    else:
                        np.testing.assert_almost_equal(
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
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.6158501, 0.50313626, 0.64241928, 1.61627195, -59.62168912, 111.27061562],
                        )
                    else:
                        np.testing.assert_almost_equal(
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
    else:
        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            if with_torque:
                if with_excitations:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.6158501,
                                0.5031363,
                                0.6424193,
                                0.7009691,
                                0.0848377,
                                0.9472486,
                                46.8792802,
                                -1.8018903,
                                53.3914525,
                                48.3005692,
                                63.6937337,
                                -28.1570099,
                            ],
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.183405,
                                0.611853,
                                0.785176,
                                0.249292,
                                0.289751,
                                0.871461,
                                8.606308,
                                3.194336,
                                29.740561,
                                -20.275423,
                                -23.246778,
                                -41.913501,
                            ],
                            decimal=6,
                        )
                else:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.61585, 0.503136, 0.642419, 0.895523, 0.319314, 0.448446],
                            decimal=6,
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.183405, 0.611853, 0.785176, 0.729007, 0.863103, 0.325183],
                            decimal=6,
                        )

            else:
                if with_excitations:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.6158501,
                                0.50313626,
                                0.64241928,
                                0.79157904,
                                0.54952888,
                                0.14299168,
                                55.65557816,
                                50.47052688,
                                0.36025589,
                                58.92377491,
                                29.70094194,
                                -15.13534937,
                            ],
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                1.83404510e-01,
                                6.11852895e-01,
                                7.85175961e-01,
                                1.19594246e-01,
                                4.93795596e-01,
                                3.14291857e-02,
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
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.6158501, 0.5031363, 0.6424193, 0.9905051, 0.9307573, 0.1031239],
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.183405, 0.611853, 0.785176, 0.388677, 0.542696, 0.772245],
                            decimal=6,
                        )
        else:
            if with_torque:
                if with_excitations:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.6158501,
                                0.5031363,
                                0.6424193,
                                1.7685411,
                                -54.335106,
                                102.170402,
                                46.8792802,
                                -1.8018903,
                                53.3914525,
                                48.3005692,
                                63.6937337,
                                -28.1570099,
                            ],
                        )
                    else:
                        np.testing.assert_almost_equal(
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
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                6.15850098e-01,
                                5.03136259e-01,
                                6.42419278e-01,
                                -5.27793227e00,
                                1.57668597e02,
                                -5.16345913e02,
                            ],
                            decimal=6,
                        )
                    else:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.18340451, 0.61185289, 0.78517596, -0.8671376, 22.51194682, -153.29477496],
                            decimal=6,
                        )

            else:
                if with_excitations:
                    if with_external_force:
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [
                                0.6158501,
                                0.5031363,
                                0.6424193,
                                1.6162719,
                                -59.6216891,
                                111.2706156,
                                55.6555782,
                                50.4705269,
                                0.3602559,
                                58.9237749,
                                29.7009419,
                                -15.1353494,
                            ],
                        )
                    else:
                        np.testing.assert_almost_equal(
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
                        np.testing.assert_almost_equal(
                            x_out[:, 0],
                            [0.6158501, 0.50313626, 0.64241928, 1.61627195, -59.62168912, 111.27061562],
                        )
                    else:
                        np.testing.assert_almost_equal(
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


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("rigid_body_dynamics", RigidBodyDynamics)
def test_joints_acceleration_driven(cx, rigid_body_dynamics):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/getting_started/models/double_pendulum.bioMod")
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT

    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN, rigidbody_dynamics=rigid_body_dynamics),
        False,
    )
    np.random.seed(42)

    # Prepare the dynamics
    if rigid_body_dynamics != RigidBodyDynamics.ODE:
        with pytest.raises(NotImplementedError, match=re.escape("Implicit dynamics not implemented yet.")):
            ConfigureProblem.initialize(ocp, nlp)
    else:
        ConfigureProblem.initialize(ocp, nlp)

        # Test the results
        states = np.random.rand(nlp.states.shape, nlp.ns)
        controls = np.random.rand(nlp.controls.shape, nlp.ns)
        params = np.random.rand(nlp.parameters.shape, nlp.ns)
        x_out = np.array(nlp.dynamics_func(states, controls, params))

        # obtained using Ipuch reference implementation. [https://github.com/Ipuch/OnDynamicsForSomersaults]
        np.testing.assert_almost_equal(x_out[:, 0], [0.02058449, 0.18340451, -2.95556261, 0.61185289])


@pytest.mark.parametrize("with_contact", [False, True])
def test_custom_dynamics(with_contact):
    def custom_dynamic(states, controls, parameters, nlp, with_contact=False) -> DynamicsEvaluation:
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)

        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)

    def configure(ocp, nlp, with_contact=None):
        ConfigureProblem.configure_q(nlp, True, False)
        ConfigureProblem.configure_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, with_contact=with_contact)

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod"
    )
    nlp.ns = 5
    nlp.cx = MX

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp, "dynamics_type", Dynamics(configure, dynamic_function=custom_dynamic, with_contact=with_contact), False
    )

    np.random.seed(42)

    # Prepare the dynamics
    ConfigureProblem.initialize(ocp, nlp)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_contact:
        contact_out = np.array(nlp.contact_forces_func(states, controls, params))
        np.testing.assert_almost_equal(
            x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3214905, -0.1912131, 0.6507164, -0.2359716]
        )
        np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071, 128.8816865, 2.7245124])

    else:
        np.testing.assert_almost_equal(
            x_out[:, 0],
            [0.61185289, 0.78517596, 0.60754485, 0.80839735, -0.30241366, -10.38503791, 1.60445173, 35.80238642],
        )
