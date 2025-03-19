import numpy.testing as npt
import pytest
from casadi import MX, SX

from bioptim import (
    ConfigureProblem,
    BiorbdModel,
    NonLinearProgram,
    ConstraintList,
    ParameterContainer,
    ParameterList,
    PhaseDynamics,
    VariableScalingList,
    FatigueList,
    XiaFatigue,
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


@pytest.mark.parametrize("cx", [MX, SX])
def test_configures(cx):

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE, use_sx=(cx == SX))
    nlp.ns = 30
    nlp.cx = MX
    nlp.time_cx = nlp.cx.sym("time", 1, 1)
    nlp.dt = nlp.cx.sym("dt", 1, 1)
    nlp.initialize(nlp.cx)
    nlp.x_scaling = VariableScalingList()
    nlp.xdot_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()
    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/getting_started/models/2segments_4dof_2contacts.bioMod",
    )

    # We start with empty OptimizationVariableContainers (states, controls, algebraic states) and then fill them with the appropriate variables

    # Test states
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    npt.assert_equal(nlp.states.shape, 4)
    npt.assert_equal(nlp.states.keys(), ["q"])

    # Test multiple states + states dot
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False, as_states_dot=True)
    npt.assert_equal(nlp.states.shape, 4 + 4)
    npt.assert_equal(nlp.states.keys(), ["q", "qdot"])
    npt.assert_equal(nlp.states_dot.shape, 4)
    npt.assert_equal(nlp.states_dot.keys(), ["qdot"])

    # Test controls
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    npt.assert_equal(nlp.controls.shape, 4)
    npt.assert_equal(nlp.controls.keys(), ["tau"])

    # Test all other configures
    ConfigureProblem.configure_qddot(ocp, nlp, as_states=True, as_controls=False, as_states_dot=True)
    npt.assert_equal(nlp.states.shape, 4 + 4 + 4)
    npt.assert_equal(nlp.states.keys(), ["q", "qdot", "qddot"])
    npt.assert_equal(nlp.states_dot.shape, 4 + 4)
    npt.assert_equal(nlp.states_dot.keys(), ["qdot", "qddot"])

    ConfigureProblem.configure_qdddot(ocp, nlp, as_states=True, as_controls=False)
    npt.assert_equal(nlp.states.shape, 4 + 4 + 4 + 4)
    npt.assert_equal(nlp.states.keys(), ["q", "qdot", "qddot", "qdddot"])

    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=4, n_references=8)
    npt.assert_equal(nlp.controls.shape, 36)
    npt.assert_equal(nlp.controls.keys(), ["tau", "k"])

    ConfigureProblem.configure_residual_tau(ocp, nlp, as_states=False, as_controls=True)
    npt.assert_equal(nlp.controls.shape, 4 + 36)
    npt.assert_equal(nlp.controls.keys(), ["tau", "k", "residual_tau"])

    ConfigureProblem.configure_taudot(ocp, nlp, as_states=False, as_controls=True)
    npt.assert_equal(nlp.controls.shape, 4 + 36 + 4)
    npt.assert_equal(nlp.controls.keys(), ["tau", "k", "residual_tau", "taudot"])

    ConfigureProblem.configure_translational_forces(ocp, nlp, as_states=False, as_controls=True)
    npt.assert_equal(nlp.controls.shape, 4 + 36 + 4 + 3 + 3)
    npt.assert_equal(nlp.controls.keys(), ["tau", "k", "residual_tau", "taudot", "contact_forces", "contact_positions"])

    ConfigureProblem.configure_rigid_contact_forces(
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=False,
        as_states_dot=False,
    )
    npt.assert_equal(nlp.states.shape, 4 + 4 + 4 + 4 + 3)
    npt.assert_equal(nlp.states.keys(), ["q", "qdot", "qddot", "qdddot", "rigid_contact_forces"])

    ConfigureProblem.configure_rigid_contact_forces(
        ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False, as_states_dot=False
    )
    npt.assert_equal(nlp.controls.shape, 4 + 36 + 4 + 3 + 3 + 3)
    npt.assert_equal(
        nlp.controls.keys(),
        ["tau", "k", "residual_tau", "taudot", "contact_forces", "contact_positions", "rigid_contact_forces"],
    )

    ConfigureProblem.configure_rigid_contact_forces(
        ocp, nlp, as_states=False, as_controls=False, as_algebraic_states=True, as_states_dot=False
    )
    npt.assert_equal(nlp.algebraic_states.shape, 3)
    npt.assert_equal(nlp.algebraic_states.keys(), ["rigid_contact_forces"])


@pytest.mark.parametrize("cx", [MX, SX])
def test_configure_soft_contacts(cx):

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE, use_sx=(cx == SX))
    nlp.ns = 30
    nlp.cx = MX
    nlp.time_cx = nlp.cx.sym("time", 1, 1)
    nlp.dt = nlp.cx.sym("dt", 1, 1)
    nlp.initialize(nlp.cx)
    nlp.x_scaling = VariableScalingList()
    nlp.xdot_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()
    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/soft_contact_sphere.bioMod",
    )

    ConfigureProblem.configure_soft_contact_forces(
        ocp, nlp, as_states=True, as_controls=False, as_algebraic_states=False
    )
    npt.assert_equal(nlp.states.shape, 3)
    npt.assert_equal(nlp.states.keys(), ["soft_contact_forces"])


@pytest.mark.parametrize("cx", [MX, SX])
def test_configure_muscles(cx):

    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE, use_sx=(cx == SX))
    nlp.ns = 30
    nlp.cx = MX
    nlp.time_cx = nlp.cx.sym("time", 1, 1)
    nlp.dt = nlp.cx.sym("dt", 1, 1)
    nlp.initialize(nlp.cx)
    nlp.x_scaling = VariableScalingList()
    nlp.xdot_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()
    ocp = OptimalControlProgram(nlp, use_sx=(cx == SX))

    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26.bioMod",
    )

    fatigue = FatigueList()
    fatigue.add(XiaFatigue(LD=10, LR=10, F=0.01, R=0.002), state_only=False)

    ConfigureProblem.configure_muscles(ocp, nlp, as_states=True, as_controls=True, fatigue=fatigue)
    npt.assert_equal(nlp.states.shape, 24)
    npt.assert_equal(nlp.states.keys(), ["muscles", "muscles_ma", "muscles_mr", "muscles_mf"])
    npt.assert_equal(nlp.controls.shape, 6)
    npt.assert_equal(nlp.controls.keys(), ["muscles"])
    npt.assert_equal(nlp.controls.keys(), ["muscles"])
