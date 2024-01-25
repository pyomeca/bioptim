"""
This test is designed to test the different integrators behavior to time dependent dynamics.
This example uses the data from the balanced pendulum example.
"""

import os
import pytest

import numpy as np
from casadi import MX, SX, vertcat, sin

from bioptim import (
    BiorbdModel,
    ConfigureProblem,
    ControlType,
    DynamicsEvaluation,
    DynamicsFunctions,
    DynamicsList,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    NonLinearProgram,
    PhaseDynamics,
)

from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

bioptim_folder = os.path.dirname(ocp_module.__file__)  # Get the path to the pendulum example


def time_dynamic(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    stochastic_variables: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, s)

    Parameters
    ----------
    time: MX | SX
        The time of the system
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    stochastic_variables: MX | SX
        The stochastic variables of the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls) * (sin(time) * time.ones(nlp.model.nb_tau) * 10)

    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = nlp.model.forward_dynamics(q, qdot, tau)

    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, time_dynamic)


def prepare_ocp(
    biorbd_model_path: str,
    ode_solver: OdeSolverBase,
    control_type: ControlType,
    use_sx: bool,
    phase_dynamics: PhaseDynamics = PhaseDynamics.ONE_PER_NODE,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    ode_solver: OdeSolverBase
        The ode solver to use
    control_type: ControlType
        The type of the controls
    use_sx: bool
        If the ocp should be built with SX. Please note that ACADOS requires SX
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = [BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path)]
    final_time = [1] * 2
    n_shooting = [50 if isinstance(ode_solver, OdeSolver.IRK) else 30] * 2

    # Dynamics
    dynamics = DynamicsList()
    expand = not isinstance(ode_solver, OdeSolver.IRK)
    for i in range(len(bio_model)):
        dynamics.add(
            custom_configure,
            dynamic_function=time_dynamic,
            phase=i,
            expand_dynamics=expand,
            phase_dynamics=phase_dynamics,
        )

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        ode_solver=ode_solver,
        control_type=control_type,
        use_sx=use_sx,
    )


@pytest.mark.parametrize("control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS])
@pytest.mark.parametrize("use_sx", [False, True])
def test_rk4_integrator(control_type, use_sx):
    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        ode_solver=OdeSolver.RK4(),
        control_type=control_type,
        use_sx=use_sx,
    )

    states = (0, 0, 0, 0)
    controls = 0
    parameters = 0
    stochastics = 0

    phase0_node0_dynamics = ocp.nlp[0].dynamics[0]
    phase0_node15_dynamics = ocp.nlp[0].dynamics[15]
    phase0_node29_dynamics = ocp.nlp[0].dynamics[29]
    phase1_node0_dynamics = ocp.nlp[1].dynamics[0]
    phase1_node15_dynamics = ocp.nlp[1].dynamics[15]
    phase1_node29_dynamics = ocp.nlp[1].dynamics[29]

    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0.00806965, -0.00845451, 0.478401, -0.501076]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 0.000326354, 0.00130369, 0.00292674, 0.00518647, 0.00806965],
                    [0, -0.000342014, -0.0013662, -0.00306689, -0.00543438, -0.00845451],
                    [0, 0.0978631, 0.195206, 0.291485, 0.386105, 0.478401],
                    [0, -0.102558, -0.204556, -0.30541, -0.404485, -0.501076],
                ]
            )
        ),
        decimal=5,
    )

    controls = 1

    np.testing.assert_almost_equal(
        phase0_node15_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0.00809725, -0.00563696, 0.481376, -0.331795]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node15_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 0.000326701, 0.00130542, 0.00293197, 0.00519927, 0.00809725],
                    [0, -0.000230274, -0.000917809, -0.00205544, -0.00363296, -0.00563696],
                    [0, 0.0979758, 0.195535, 0.292253, 0.387687, 0.481376],
                    [0, -0.0689774, -0.137104, -0.20393, -0.26899, -0.331795],
                ]
            )
        ),
        decimal=5,
    )

    states = (1, 1, 0, 0)

    np.testing.assert_almost_equal(
        phase0_node29_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[1.00589, 0.997204, 0.354366, -0.167234]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node29_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [1, 1.00023, 1.00094, 1.00212, 1.00377, 1.00589],
                    [1, 0.999887, 0.999551, 0.998991, 0.998208, 0.997204],
                    [0, 0.0704302, 0.141056, 0.211903, 0.282998, 0.354366],
                    [0, -0.0337252, -0.0672889, -0.100713, -0.134021, -0.167234],
                ]
            )
        ),
        decimal=5,
    )

    states = (0, 0, 0, 0)
    controls = 0

    np.testing.assert_almost_equal(
        np.array(phase1_node0_dynamics(states, controls, 1, 0)[0]),
        np.array(phase1_node15_dynamics(states, controls, 0, 1)[0]),
    )
    np.testing.assert_almost_equal(
        np.array(phase1_node0_dynamics(states, controls, 1, 0)[1]),
        np.array(phase1_node15_dynamics(states, controls, 0, 1)[1]),
    )

    states = (0, 0, 1, 1)
    stochastics = 1

    np.testing.assert_almost_equal(
        phase1_node29_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0.0428703, 0.0233604, 1.58314, 0.389636]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase1_node29_dynamics(states, controls, parameters, stochastics)[1],
        (
            np.array(
                [
                    [0, 0.00703216, 0.0148151, 0.0233721, 0.03272, 0.0428703],
                    [0, 0.00628509, 0.0117858, 0.0164766, 0.0203391, 0.0233604],
                    [1, 1.11045, 1.225, 1.34253, 1.46213, 1.58314],
                    [1, 0.884671, 0.764899, 0.641881, 0.516555, 0.389636],
                ]
            )
        ),
        decimal=5,
    )


def test_irk_integrator():
    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        ode_solver=OdeSolver.IRK(),
        control_type=ControlType.CONSTANT,
        use_sx=False,
    )

    states = (0, 0, 0, 0)
    controls = 0
    parameters = 0
    stochastics = 0

    phase0_node0_dynamics = ocp.nlp[0].dynamics[0]
    phase0_node25_dynamics = ocp.nlp[0].dynamics[25]
    phase0_node49_dynamics = ocp.nlp[0].dynamics[49]
    phase1_node0_dynamics = ocp.nlp[1].dynamics[0]
    phase1_node25_dynamics = ocp.nlp[1].dynamics[25]
    phase1_node49_dynamics = ocp.nlp[1].dynamics[49]

    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0.00292675, -0.00306689, 0.291485, -0.30541]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0.00292675], [0, -0.00306689], [0, 0.291485], [0, -0.30541]])),
        decimal=5,
    )

    controls = 1

    np.testing.assert_almost_equal(
        phase0_node25_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0.00293267, -0.00194324, 0.29236, -0.19336]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node25_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0.00293267], [0, -0.00194324], [0, 0.29236], [0, -0.19336]])),
        decimal=5,
    )

    states = (1, 1, 0, 0)

    np.testing.assert_almost_equal(
        phase0_node49_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[1.00217, 0.999049, 0.217027, -0.095098]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node49_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[1, 1.00217], [1, 0.999049], [0, 0.217027], [0, -0.095098]])),
        decimal=5,
    )

    states = (0, 0, 0, 0)
    controls = 0

    np.testing.assert_almost_equal(
        np.array(phase1_node0_dynamics(states, controls, 1, 0)[0]),
        np.array(phase1_node25_dynamics(states, controls, 0, 1)[0]),
    )
    np.testing.assert_almost_equal(
        np.array(phase1_node0_dynamics(states, controls, 1, 0)[1]),
        np.array(phase1_node25_dynamics(states, controls, 0, 1)[1]),
    )

    states = (0, 0, 1, 1)
    stochastics = 1

    np.testing.assert_almost_equal(
        phase1_node49_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0.0233721, 0.0164765, 1.34253, 0.641881]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase1_node49_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0.0233721], [0, 0.0164765], [1, 1.34253], [1, 0.641881]])),
        decimal=5,
    )


@pytest.mark.parametrize("use_sx", [False, True])
def test_collocation_integrator(use_sx):
    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        ode_solver=OdeSolver.COLLOCATION(),
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
    )

    states = (0, 0, 0, 0)
    controls = 0
    parameters = 0
    stochastics = 0

    phase0_node0_dynamics = ocp.nlp[0].dynamics[0]
    phase0_node15_dynamics = ocp.nlp[0].dynamics[15]
    phase0_node29_dynamics = ocp.nlp[0].dynamics[29]
    phase1_node0_dynamics = ocp.nlp[1].dynamics[0]
    phase1_node15_dynamics = ocp.nlp[1].dynamics[15]
    phase1_node29_dynamics = ocp.nlp[1].dynamics[29]

    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0, 0, 0, 0]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0], [0, 0], [0, 0], [0, 0]])),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[2].T,
        np.array(
            [
                [
                    0,
                    0,
                    -0.489745,
                    0.513252,
                    0,
                    0,
                    -0.489745,
                    0.513252,
                    0,
                    0,
                    -0.489745,
                    0.513252,
                    0,
                    0,
                    -0.489745,
                    0.513252,
                ]
            ]
        ),
        decimal=5,
    )

    controls = 1

    np.testing.assert_almost_equal(
        phase0_node15_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0, 0, 0, 0]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node15_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0], [0, 0], [0, 0], [0, 0]])),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node15_dynamics(states, controls, parameters, stochastics)[2].T,
        np.array(
            [
                [
                    0,
                    0,
                    -0.490282,
                    0.325471,
                    0,
                    0,
                    -0.490282,
                    0.325471,
                    0,
                    0,
                    -0.490282,
                    0.325471,
                    0,
                    0,
                    -0.490282,
                    0.325471,
                ]
            ]
        ),
        decimal=5,
    )

    states = (1, 1, 0, 0)

    np.testing.assert_almost_equal(
        phase0_node29_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[1, 1, 0, 0]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node29_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[1, 1], [1, 1], [0, 0], [0, 0]])),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node29_dynamics(states, controls, parameters, stochastics)[2].T,
        np.array(
            [
                [
                    1.66533e-15,
                    1.66533e-15,
                    -0.360026,
                    0.159984,
                    -7.21645e-16,
                    -7.21645e-16,
                    -0.360026,
                    0.159984,
                    -4.44089e-16,
                    -4.44089e-16,
                    -0.360026,
                    0.159984,
                    -8.88178e-16,
                    -8.88178e-16,
                    -0.360026,
                    0.159984,
                ]
            ]
        ),
        decimal=5,
    )

    states = (0, 0, 0, 0)
    controls = 0

    np.testing.assert_almost_equal(
        np.array(phase1_node0_dynamics(states, controls, 1, 0)[0]),
        np.array(phase1_node15_dynamics(states, controls, 0, 1)[0]),
    )
    np.testing.assert_almost_equal(
        np.array(phase1_node0_dynamics(states, controls, 1, 0)[1]),
        np.array(phase1_node15_dynamics(states, controls, 0, 1)[1]),
    )

    states = (0, 0, 1, 1)
    stochastics = 1

    np.testing.assert_almost_equal(
        phase1_node29_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0, 0, 1, 1]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase1_node29_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0], [0, 0], [1, 1], [1, 1]])),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase1_node29_dynamics(states, controls, parameters, stochastics)[2].T,
        np.array(
            [
                [
                    -0.0333333,
                    -0.0333333,
                    -0.539675,
                    0.563175,
                    -0.0333333,
                    -0.0333333,
                    -0.539675,
                    0.563175,
                    -0.0333333,
                    -0.0333333,
                    -0.539675,
                    0.563175,
                    -0.0333333,
                    -0.0333333,
                    -0.539675,
                    0.563175,
                ]
            ]
        ),
        decimal=5,
    )


@pytest.mark.parametrize("use_sx", [False, True])
def test_trapezoidal_integrator(use_sx):
    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        ode_solver=OdeSolver.TRAPEZOIDAL(),
        control_type=ControlType.LINEAR_CONTINUOUS,
        use_sx=use_sx,
    )

    states = (0, 0, 0, 0)
    controls = 0
    parameters = 0
    stochastics = 0

    phase0_node0_dynamics = ocp.nlp[0].dynamics[0]
    phase0_node15_dynamics = ocp.nlp[0].dynamics[15]
    phase0_node29_dynamics = ocp.nlp[0].dynamics[29]
    phase1_node0_dynamics = ocp.nlp[1].dynamics[0]
    phase1_node15_dynamics = ocp.nlp[1].dynamics[15]
    phase1_node29_dynamics = ocp.nlp[1].dynamics[29]

    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0, 0, 0.489745, -0.513252]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0], [0, 0], [0, 0.489745], [0, -0.513252]])),
        decimal=5,
    )

    controls = 1

    np.testing.assert_almost_equal(
        phase0_node15_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0, 0, 0.490223, -0.346274]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node15_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0], [0, 0], [0, 0.490223], [0, -0.346274]])),
        decimal=5,
    )

    states = (1, 1, 0, 0)

    np.testing.assert_almost_equal(
        phase0_node29_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[1, 1, 0.351708, -0.169067]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node29_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[1, 1], [1, 1], [0, 0.351708], [0, -0.169067]])),
        decimal=5,
    )

    states = (0, 0, 0, 0)
    controls = 0

    np.testing.assert_almost_equal(
        np.array(phase1_node0_dynamics(states, controls, 1, 0)[0]),
        np.array(phase1_node15_dynamics(states, controls, 0, 1)[0]),
    )
    np.testing.assert_almost_equal(
        np.array(phase1_node0_dynamics(states, controls, 1, 0)[1]),
        np.array(phase1_node15_dynamics(states, controls, 0, 1)[1]),
    )

    states = (0, 0, 1, 1)
    stochastics = 1

    np.testing.assert_almost_equal(
        phase1_node29_dynamics(states, controls, parameters, stochastics)[0].T,
        np.array([[0.0333333, 0.0333333, 1.53967, 0.436825]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase1_node29_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0.0333333], [0, 0.0333333], [1, 1.53967], [1, 0.436825]])),
        decimal=5,
    )
