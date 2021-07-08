import pytest

import numpy as np
from casadi import MX, SX
import biorbd
from bioptim.dynamics.configure_problem import ConfigureProblem
from bioptim.dynamics.dynamics_functions import DynamicsFunctions
from bioptim.interfaces.biorbd_interface import BiorbdInterface
from bioptim.misc.enums import ControlType
from bioptim.optimization.non_linear_program import NonLinearProgram
from bioptim.optimization.optimization_vector import OptimizationVector
from bioptim.dynamics.configure_problem import DynamicsFcn, Dynamics

from .utils import TestUtils


class OptimalControlProgram:
    def __init__(self, nlp):
        self.n_phases = 1
        self.nlp = [nlp]
        self.v = OptimizationVector(self)


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_driven(with_contact, with_external_force, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/getting_started/2segments_4dof_2contacts.bioMod")
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(ocp, "dynamics_type", Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=with_contact), False)

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
                [0.8631034, 0.3251833, 0.1195942, 0.4937956, -7.8965868, -7.7015214, 22.0607764, -16.6593293],
            )
            np.testing.assert_almost_equal(contact_out[:, 0], [-48.4505407, 111.4024916, -24.4449121])
        else:
            np.testing.assert_almost_equal(
                x_out[:, 0], [0.6118529, 0.785176, 0.6075449, 0.8083973, -0.3214905, -0.1912131, 0.6507164, -0.2359716]
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
                [0.61185289, 0.78517596, 0.60754485, 0.80839735, -0.30241366, -10.38503791, 1.60445173, 35.80238642],
            )


@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_derivative_driven(with_contact, with_external_force, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/getting_started/2segments_4dof_2contacts.bioMod")
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
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
                    0.86310343,
                    0.32518332,
                    0.11959425,
                    0.4937956,
                    -7.8965868,
                    -7.70152137,
                    22.06077635,
                    -16.65932927,
                    0.80744016,
                    0.42710779,
                    0.417411,
                    0.32320293,
                ],
            )
            np.testing.assert_almost_equal(contact_out[:, 0], [-48.4505407, 111.4024916, -24.4449121])
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
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_activation_driven(with_contact, with_external_force, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/getting_started/2segments_4dof_2contacts.bioMod")
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
                [0.86310343, 0.32518332, 0.11959425, 0.4937956, 18.89229596, 18.39174157, -52.73234125, 58.13378833],
                decimal=5,
            )
            np.testing.assert_almost_equal(contact_out[:, 0], [109.17126642, 3790.62308393, -3571.78585744])
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
@pytest.mark.parametrize("with_residual_torque", [False, True])
@pytest.mark.parametrize("with_excitations", [False, True])
def test_muscle_driven(with_excitations, with_contact, with_residual_torque, with_external_force, cx):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/arm26_with_contact.bioMod")
    nlp.ns = 5
    nlp.cx = cx

    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 2 + nlp.model.nbMuscles(), 1))
    nlp.u_bounds = np.zeros((nlp.model.nbMuscles(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(
        ocp,
        "dynamics_type",
        Dynamics(
            DynamicsFcn.MUSCLE_DRIVEN,
            with_residual_torque=with_residual_torque,
            with_excitations=with_excitations,
            with_contact=with_contact,
        ),
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

    if with_contact:  # Warning this test is a bit bogus, there since the model does not have contacts
        if with_residual_torque:
            if with_excitations:
                if with_external_force:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [
                            0.6158501,
                            0.50313626,
                            0.64241928,
                            0.3264777,
                            -1.57134516,
                            0.87073117,
                            46.87928022,
                            -1.80189035,
                            53.3914525,
                            48.30056919,
                            63.69373374,
                            -28.15700995,
                        ],
                    )
                else:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [
                            1.83404510e-01,
                            6.11852895e-01,
                            7.85175961e-01,
                            3.92710810e-02,
                            2.24914101e00,
                            -9.32712397e00,
                            8.60630831e00,
                            3.19433638e00,
                            2.97405608e01,
                            -2.02754226e01,
                            -2.32467778e01,
                            -4.19135012e01,
                        ],
                    )
            else:
                if with_external_force:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [0.6158501, 0.50313626, 0.64241928, 0.02002169, 2.81525506, -9.39083155],
                    )
                else:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [0.18340451, 0.61185289, 0.78517596, 0.16825028, -0.08046333, -3.94434684],
                    )

        else:
            if with_excitations:
                if with_external_force:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [
                            6.15850098e-01,
                            5.03136259e-01,
                            6.42419278e-01,
                            3.91853634e-02,
                            -1.76074913e00,
                            1.02811024e00,
                            5.56555782e01,
                            5.04705269e01,
                            3.60255887e-01,
                            5.89237749e01,
                            2.97009419e01,
                            -1.51353494e01,
                        ],
                    )
                else:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [
                            1.83404510e-01,
                            6.11852895e-01,
                            7.85175961e-01,
                            -7.74768714e-02,
                            2.30892158e00,
                            -9.64013318e00,
                            -7.72228930e00,
                            -1.13759732e01,
                            9.51906209e01,
                            4.45077128e00,
                            -5.20261014e00,
                            -2.80864106e01,
                        ],
                    )
            else:
                if with_external_force:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [0.6158501, 0.50313626, 0.64241928, 0.03918536, -1.76074913, 1.02811024],
                    )
                else:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [0.18340451, 0.61185289, 0.78517596, -0.07747687, 2.30892158, -9.64013318],
                    )

    else:
        if with_residual_torque:
            if with_excitations:
                if with_external_force:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [
                            0.6158501,
                            0.50313626,
                            0.64241928,
                            0.3264777,
                            -1.57134516,
                            0.87073117,
                            46.87928022,
                            -1.80189035,
                            53.3914525,
                            48.30056919,
                            63.69373374,
                            -28.15700995,
                        ],
                    )
                else:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [
                            1.83404510e-01,
                            6.11852895e-01,
                            7.85175961e-01,
                            3.92710810e-02,
                            2.24914101e00,
                            -9.32712397e00,
                            8.60630831e00,
                            3.19433638e00,
                            2.97405608e01,
                            -2.02754226e01,
                            -2.32467778e01,
                            -4.19135012e01,
                        ],
                    )
            else:
                if with_external_force:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [0.6158501, 0.50313626, 0.64241928, 0.02002169, 2.81525506, -9.39083155],
                    )
                else:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [0.18340451, 0.61185289, 0.78517596, 0.16825028, -0.08046333, -3.94434684],
                    )

        else:
            if with_excitations:
                if with_external_force:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [
                            6.15850098e-01,
                            5.03136259e-01,
                            6.42419278e-01,
                            3.91853634e-02,
                            -1.76074913e00,
                            1.02811024e00,
                            5.56555782e01,
                            5.04705269e01,
                            3.60255887e-01,
                            5.89237749e01,
                            2.97009419e01,
                            -1.51353494e01,
                        ],
                    )
                else:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [
                            1.83404510e-01,
                            6.11852895e-01,
                            7.85175961e-01,
                            -7.74768714e-02,
                            2.30892158e00,
                            -9.64013318e00,
                            -7.72228930e00,
                            -1.13759732e01,
                            9.51906209e01,
                            4.45077128e00,
                            -5.20261014e00,
                            -2.80864106e01,
                        ],
                    )
            else:
                if with_external_force:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [0.6158501, 0.50313626, 0.64241928, 0.03918536, -1.76074913, 1.02811024],
                    )
                else:
                    np.testing.assert_almost_equal(
                        x_out[:, 0],
                        [0.18340451, 0.61185289, 0.78517596, -0.07747687, 2.30892158, -9.64013318],
                    )


@pytest.mark.parametrize("with_contact", [False, True])
def test_custom_dynamics(with_contact):
    def custom_dynamic(states, controls, parameters, nlp, with_contact=False) -> tuple:
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)

        return dq, ddq

    def configure(ocp, nlp, with_contact=None):
        ConfigureProblem.configure_q(nlp, True, False)
        ConfigureProblem.configure_qdot(nlp, True, False)
        ConfigureProblem.configure_tau(nlp, False, True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, with_contact=with_contact)

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/getting_started/2segments_4dof_2contacts.bioMod")
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
