"""
This test is designed to test the collocation integrators behavior for a stochastic problem.
This example uses the data from the balanced obstacle_avoidance_direct_collocation example.

"""

import pytest

import numpy as np

from bioptim import (
    ConfigureProblem,
    ControlType,
    DynamicsList,
    OptimalControlProgram,
    NonLinearProgram,
    PhaseDynamics,
    SocpType,
    StochasticOptimalControlProgram,
)

from bioptim.examples.stochastic_optimal_control.mass_point_model import MassPointModel


def configure_stochastic_optimal_control_problem(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    ConfigureProblem.configure_q(ocp, nlp, True, False, False)
    ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
    ConfigureProblem.configure_new_variable("u", nlp.model.name_u, ocp, nlp, as_states=False, as_controls=True)

    # Stochastic variables
    ConfigureProblem.configure_stochastic_m(
        ocp, nlp, n_noised_states=4, n_collocation_points=nlp.model.polynomial_degree + 1
    )
    ConfigureProblem.configure_stochastic_cov_implicit(ocp, nlp, n_noised_states=4)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, stochastic_variables, nlp, with_noise=False
        ),
    )
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        dyn_func=lambda time, states, controls, parameters, stochastic_variables, nlp: nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, stochastic_variables, nlp, with_noise=True
        ),
        allow_free_variables=True,
    )


def prepare_socp(
    use_sx: bool = False,
) -> StochasticOptimalControlProgram | OptimalControlProgram:
    bio_model = [
        MassPointModel(
            socp_type=SocpType.COLLOCATION(polynomial_degree=5, method="legendre"),
            motor_noise_magnitude=np.array([1, 1]),
            polynomial_degree=5,
        ),
        MassPointModel(
            socp_type=SocpType.COLLOCATION(polynomial_degree=5, method="legendre"),
            motor_noise_magnitude=np.array([1, 1]),
            polynomial_degree=5,
        ),
    ]

    final_time = [1] * 2
    n_shooting = [30] * 2

    # Dynamics
    dynamics = DynamicsList()
    for i in range(len(bio_model)):
        dynamics.add(
            configure_stochastic_optimal_control_problem,
            dynamic_function=lambda time, states, controls, parameters, stochastic_variables, nlp, with_noise: bio_model[
                i
            ].dynamics(
                states,
                controls,
                parameters,
                stochastic_variables,
                nlp,
                with_noise=with_noise,
            ),
            phase=i,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            expand_dynamics=True,
        )

    return StochasticOptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        control_type=ControlType.CONSTANT,
        n_threads=6,
        problem_type=SocpType.COLLOCATION(polynomial_degree=5, method="legendre"),
        use_sx=use_sx,
    )


@pytest.mark.parametrize("use_sx", [False])  # TODO add True in the 3.2 version
def test_collocation_integrator_for_stochastic(use_sx):
    socp = prepare_socp(use_sx=use_sx)

    states = (0, 0, 0, 0)
    controls = 0
    parameters = 0
    stochastics = 0

    phase0_node0_dynamics = socp.nlp[0].dynamics[0]
    phase0_node15_dynamics = socp.nlp[0].dynamics[15]
    phase0_node29_dynamics = socp.nlp[0].dynamics[29]
    phase1_node0_dynamics = socp.nlp[1].dynamics[0]
    phase1_node15_dynamics = socp.nlp[1].dynamics[15]
    phase1_node29_dynamics = socp.nlp[1].dynamics[29]

    # phase0_node0_extra_dynamics = socp.nlp[0].extra_dynamics[0][0]  TODO add in the 3.2 version
    # phase0_node15_extra_dynamics = socp.nlp[0].extra_dynamics[0][15]
    # phase0_node29_extra_dynamics = socp.nlp[0].extra_dynamics[0][29]
    # phase1_node0_extra_dynamics = socp.nlp[1].extra_dynamics[0][0]
    # phase1_node15_extra_dynamics = socp.nlp[1].extra_dynamics[0][15]
    # phase1_node29_extra_dynamics = socp.nlp[1].extra_dynamics[0][29]

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
        (np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])),
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
        phase0_node0_dynamics(states, controls, parameters, stochastics)[2].T,
        (
            np.array(
                [
                    [
                        0,
                        0,
                        -0.333333,
                        -0.333333,
                        0,
                        0,
                        -0.333333,
                        -0.333333,
                        0,
                        0,
                        -0.333333,
                        -0.333333,
                        0,
                        0,
                        -0.333333,
                        -0.333333,
                        0,
                        0,
                        -0.333333,
                        -0.333333,
                    ]
                ]
            )
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
        phase0_node0_dynamics(states, controls, parameters, stochastics)[2].T,
        (
            np.array(
                [
                    [
                        -2.51882e-15,
                        -2.51882e-15,
                        0,
                        0,
                        -1.59595e-15,
                        -1.59595e-15,
                        0,
                        0,
                        1.83187e-15,
                        1.83187e-15,
                        0,
                        0,
                        4.21885e-15,
                        4.21885e-15,
                        0,
                        0,
                        1.77636e-15,
                        1.77636e-15,
                        0,
                        0,
                    ]
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
        np.array([[0, 0, 1, 1]]),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase1_node29_dynamics(states, controls, parameters, stochastics)[1],
        (np.array([[0, 0], [0, 0], [1, 1], [1, 1]])),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        phase0_node0_dynamics(states, controls, parameters, stochastics)[2].T,
        (
            np.array(
                [
                    [
                        -0.0333333,
                        -0.0333333,
                        0.057735,
                        0.057735,
                        -0.0333333,
                        -0.0333333,
                        0.057735,
                        0.057735,
                        -0.0333333,
                        -0.0333333,
                        0.057735,
                        0.057735,
                        -0.0333333,
                        -0.0333333,
                        0.057735,
                        0.057735,
                        -0.0333333,
                        -0.0333333,
                        0.057735,
                        0.057735,
                    ]
                ]
            )
        ),
        decimal=5,
    )
