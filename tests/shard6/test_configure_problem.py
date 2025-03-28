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
    n_states = 4
    keys_states = ["q"]
    npt.assert_equal(nlp.states.shape, n_states)
    npt.assert_equal(nlp.states.keys(), keys_states)
    npt.assert_equal(nlp.states_dot.shape, n_states)
    npt.assert_equal(nlp.states_dot.keys(), keys_states)

    # Test multiple states + states dot
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    n_states += 4
    keys_states += ["qdot"]
    npt.assert_equal(nlp.states.shape, n_states)
    npt.assert_equal(nlp.states.keys(), keys_states)
    npt.assert_equal(nlp.states_dot.shape, n_states)
    npt.assert_equal(nlp.states_dot.keys(), keys_states)

    # Test controls
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    n_controls = 4
    keys_controls = ["tau"]
    npt.assert_equal(nlp.controls.shape, n_controls)
    npt.assert_equal(nlp.controls.keys(), keys_controls)

    # Test all other configures
    ConfigureProblem.configure_qddot(ocp, nlp, as_states=True, as_controls=False)
    n_states += 4
    keys_states += ["qddot"]
    npt.assert_equal(nlp.states.shape, n_states)
    npt.assert_equal(nlp.states.keys(), keys_states)
    npt.assert_equal(nlp.states_dot.shape, n_states)
    npt.assert_equal(nlp.states_dot.keys(), keys_states)

    ConfigureProblem.configure_qdddot(ocp, nlp, as_states=True, as_controls=False)
    n_states += 4
    keys_states += ["qdddot"]
    npt.assert_equal(nlp.states.shape, n_states)
    npt.assert_equal(nlp.states.keys(), keys_states)
    npt.assert_equal(nlp.states_dot.shape, n_states)
    npt.assert_equal(nlp.states_dot.keys(), keys_states)

    ConfigureProblem.configure_stochastic_k(ocp, nlp, n_noised_controls=4, n_references=8)
    n_controls += 32
    keys_controls += ["k"]
    npt.assert_equal(nlp.controls.shape, n_controls)
    npt.assert_equal(nlp.controls.keys(), keys_controls)

    ConfigureProblem.configure_residual_tau(ocp, nlp, as_states=False, as_controls=True)
    n_controls += 4
    keys_controls += ["residual_tau"]
    npt.assert_equal(nlp.controls.shape, n_controls)
    npt.assert_equal(nlp.controls.keys(), keys_controls)

    ConfigureProblem.configure_taudot(ocp, nlp, as_states=False, as_controls=True)
    n_controls += 4
    keys_controls += ["taudot"]
    npt.assert_equal(nlp.controls.shape, n_controls)
    npt.assert_equal(nlp.controls.keys(), keys_controls)

    ConfigureProblem.configure_translational_forces(ocp, nlp, as_states=False, as_controls=True)
    n_controls += 6
    keys_controls += ["contact_forces", "contact_positions"]
    npt.assert_equal(nlp.controls.shape, n_controls)
    npt.assert_equal(nlp.controls.keys(), keys_controls)

    ConfigureProblem.configure_rigid_contact_forces(
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=False,
    )
    n_states += 3
    keys_states += ["rigid_contact_forces"]
    npt.assert_equal(nlp.states.shape, n_states)
    npt.assert_equal(nlp.states.keys(), keys_states)
    npt.assert_equal(nlp.states_dot.shape, n_states)
    npt.assert_equal(nlp.states_dot.keys(), keys_states)

    ConfigureProblem.configure_rigid_contact_forces(
        ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False
    )
    n_controls += 3
    keys_controls += ["rigid_contact_forces"]
    npt.assert_equal(nlp.controls.shape, n_controls)
    npt.assert_equal(nlp.controls.keys(), keys_controls)

    ConfigureProblem.configure_rigid_contact_forces(
        ocp, nlp, as_states=False, as_controls=False, as_algebraic_states=True
    )
    n_algebraic_states = 3
    keys_algebraic_states = ["rigid_contact_forces"]
    npt.assert_equal(nlp.algebraic_states.shape, n_algebraic_states)
    npt.assert_equal(nlp.algebraic_states.keys(), keys_algebraic_states)


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
    npt.assert_equal(nlp.states.shape, 6)
    npt.assert_equal(nlp.states.keys(), ["soft_contact_forces"])
    npt.assert_equal(nlp.states_dot.shape, 6)
    npt.assert_equal(nlp.states_dot.keys(), ["soft_contact_forces"])


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
    npt.assert_equal(nlp.states_dot.shape, 24)
    npt.assert_equal(nlp.states_dot.keys(), ["muscles", "muscles_ma", "muscles_mr", "muscles_mf"])
    npt.assert_equal(nlp.controls.shape, 6)
    npt.assert_equal(nlp.controls.keys(), ["muscles"])
