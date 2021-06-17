import pytest

import numpy as np
from casadi import MX, SX
import biorbd
from bioptim.dynamics.configure_problem import ConfigureProblem
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


@pytest.mark.parametrize("CX", [MX, SX])
@pytest.mark.parametrize("with_external_force", [False, True])
@pytest.mark.parametrize("with_contact", [False, True])
def test_torque_driven(with_contact, with_external_force, CX):
    # Prepare the program
    nlp = NonLinearProgram()
    nlp.model = biorbd.Model(TestUtils.bioptim_folder() + "/examples/getting_started/2segments_4dof_2contacts.bioMod")
    nlp.ns = 5
    nlp.cx = CX
    nlp.x_bounds = np.zeros((nlp.model.nbQ() * 2, 1))
    nlp.u_bounds = np.zeros((nlp.model.nbQ(), 1))
    ocp = OptimalControlProgram(nlp)
    nlp.control_type = ControlType.CONSTANT
    NonLinearProgram.add(ocp, "dynamics_type", Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=with_contact), False)

    np.random.seed(42)
    if with_external_force:
        external_forces = [np.random.rand(6, nlp.model.nbSegment(), nlp.ns)]
        nlp.external_forces = BiorbdInterface.convert_array_to_external_forces(external_forces)[0]

    # Prepare the dynamics
    ConfigureProblem.torque_driven(ocp, nlp, with_contact)

    # Test the results
    states = np.random.rand(nlp.states.shape, nlp.ns)
    controls = np.random.rand(nlp.controls.shape, nlp.ns)
    params = np.random.rand(nlp.parameters.shape, nlp.ns)
    x_out = np.array(nlp.dynamics_func(states, controls, params))

    if with_contact:
        contact_out = np.array(nlp.contact_forces_func(states, controls, params))
        if with_external_force:
            np.testing.assert_almost_equal(x_out[:, 0], [0.8631034,   0.3251833,   0.1195942,   0.4937956,  -7.8965868,
                       -7.7015214,  22.0607764, -16.6593293])
            np.testing.assert_almost_equal(contact_out[:, 0], [-48.4505407, 111.4024916, -24.4449121])
        else:
            np.testing.assert_almost_equal(x_out[:, 0], [0.6118529,  0.785176 ,  0.6075449,  0.8083973, -0.3214905,
                      -0.1912131,  0.6507164, -0.2359716])
            np.testing.assert_almost_equal(contact_out[:, 0], [-2.444071 , 128.8816865,   2.7245124])

    else:
        if with_external_force:
            np.testing.assert_almost_equal(x_out[:, 0], [0.86310343, 0.32518332, 0.11959425, 0.4937956, 0.30731739,
                                                         -9.97912778, 1.15263778, 36.02430956])
        else:
            np.testing.assert_almost_equal(x_out[:, 0], [0.61185289,   0.78517596,   0.60754485,   0.80839735,
        -0.30241366, -10.38503791,   1.60445173,  35.80238642])

