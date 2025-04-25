import os

import numpy as np
import numpy.testing as npt
import pytest
from casadi import MX, SX

from bioptim import (
    ConfigureProblem,
    ControlType,
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
    ParameterContainer,
)
from tests.utils import TestUtils


class OptimalControlProgram:
    def __init__(self, nlp, use_sx):
        self.cx = nlp.cx
        self.n_phases = 1
        self.nlp = [nlp]
        parameters_list = ParameterList(use_sx=use_sx)
        self.parameters = ParameterContainer(use_sx=use_sx)
        self.parameters.initialize(parameters_list)
        self.n_threads = 1


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_ligament", [False, True])
def test_torque_driven_with_ligament(with_ligament, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/mass_point_with_ligament.bioMod"
    )
    nlp.dynamics_type = Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_ligament=with_ligament)

    nlp.ns = 5
    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
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
    if with_ligament:
        npt.assert_almost_equal(
            x_out[:, 0],
            [0.1559945, -47.2537196],
        )
    else:
        npt.assert_almost_equal(
            x_out[:, 0],
            [0.1559945, -9.7997078],
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_ligament", [False, True])
def test_torque_derivative_driven_with_ligament(with_ligament, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/mass_point_with_ligament.bioMod"
    )
    nlp.dynamics_type = Dynamics(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, with_ligament=with_ligament)

    nlp.ns = 5
    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()

    nlp.x_bounds = np.zeros((nlp.model.nb_q * 3, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
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
    if with_ligament:
        npt.assert_almost_equal(
            x_out[:, 0],
            [0.1559945, -47.2537196, 0.1834045],
        )
    else:
        npt.assert_almost_equal(
            x_out[:, 0],
            [0.1559945, -9.7997078, 0.1834045],
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_ligament", [False, True])
def test_torque_activation_driven_with_ligament(with_ligament, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/torque_driven_ocp/models/mass_point_with_ligament.bioMod"
    )
    nlp.dynamics_type = Dynamics(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_ligament=with_ligament)

    nlp.ns = 5
    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_q, 1))
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
    if with_ligament:
        npt.assert_almost_equal(
            x_out[:, 0],
            [0.155995, -46.234787],
            decimal=6,
        )
    else:
        npt.assert_almost_equal(
            x_out[:, 0],
            [0.15599, -8.78078],
            decimal=5,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("cx", [MX, SX])
@pytest.mark.parametrize("with_ligament", [False, True])
def test_muscle_driven_with_ligament(with_ligament, cx, phase_dynamics):
    # Prepare the program
    nlp = NonLinearProgram(phase_dynamics=phase_dynamics, use_sx=(cx == SX))
    nlp.model = BiorbdModel(
        TestUtils.bioptim_folder() + "/examples/muscle_driven_ocp/models/arm26_with_ligament.bioMod"
    )
    nlp.dynamics_type = Dynamics(
        DynamicsFcn.MUSCLE_DRIVEN,
        with_ligament=with_ligament,
    )

    nlp.ns = 5
    nlp.cx = cx
    nlp.time_cx = cx.sym("time", 1, 1)
    nlp.dt = cx.sym("dt", 1, 1)
    nlp.initialize(cx)
    nlp.x_scaling = VariableScalingList()
    nlp.u_scaling = VariableScalingList()
    nlp.a_scaling = VariableScalingList()
    nlp.x_bounds = np.zeros((nlp.model.nb_q * 2 + nlp.model.nb_muscles, 1))
    nlp.u_bounds = np.zeros((nlp.model.nb_muscles, 1))

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

    if with_ligament:
        npt.assert_almost_equal(
            x_out[:, 0],
            [2.0584494e-02, 1.8340451e-01, -6.0300944e00, -9.4582028e01],
            decimal=6,
        )
    else:
        npt.assert_almost_equal(
            x_out[:, 0],
            [2.0584494e-02, 1.8340451e-01, -7.3880194e00, -9.0642142e01],
            decimal=6,
        )
