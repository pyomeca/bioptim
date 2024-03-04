import numpy as np
import pytest

from bioptim import PhaseDynamics, SolutionMerge
from tests.utils import TestUtils


@pytest.mark.parametrize(
    "phase_dynamics",
    [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE],
)
def test_continuity_as_objective(phase_dynamics):
    from bioptim.examples.getting_started import (
        example_continuity_as_objective as ocp_module,
    )

    np.random.seed(42)
    model_path = TestUtils.bioptim_folder() + "/examples/getting_started/models/pendulum_maze.bioMod"

    # first pass
    ocp = ocp_module.prepare_ocp_first_pass(
        biorbd_model_path=model_path,
        n_threads=1,
        final_time=1,
        n_shooting=3,
        state_continuity_weight=1000,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
        minimize_time=False,  # we only want to test the continuity as objective here
    )
    sol = ocp.solve()

    expected_q = [[0.0, -0.1820716, 0.0502083, -0.1376], [0.0, 0.2059882, -0.3885045, 2.9976372]]

    expected_qdot = [[0.0, 0.13105439, -3.43794783, -23.6570729], [0.0, -0.66178869, 3.07970721, -19.12526049]]

    expected_controls = [[-1.49607534, -0.24541618, -19.12881238], [0.0, 0.0, 0.0]]
    #
    expected_iterations = range(300, 600)  # 436 on my laptop @ipuch
    np.testing.assert_almost_equal(sol.decision_states(to_merge=SolutionMerge.NODES)["q"], expected_q)
    np.testing.assert_almost_equal(sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"], expected_qdot)
    np.testing.assert_almost_equal(sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"], expected_controls)
    assert sol.iterations in expected_iterations

    # second pass
    ocp_second_pass = ocp_module.prepare_ocp_second_pass(
        biorbd_model_path=model_path,
        n_threads=1,
        n_shooting=3,
        solution=sol,
        minimize_time=False,
    )
    sol_second_pass = ocp_second_pass.solve()
    # check q in integrality, qdot,controls, vector, cost, iterations

    expected_q = [[0.0, -0.12359617, 0.21522375, -0.1376], [0.0, 0.06184961, -0.37118107, 2.9976372]]
    expected_qdot = [[0.0, 0.14235975, -1.10526128, -4.21797828], [0.0, -0.55992744, 1.17407988, -1.06473819]]
    expected_tau = [[-1.16548046, 1.10283517, -27.94121882], [0.0, 0.0, 0.0]]

    expected_vector = [
        0.333333,
        0,
        0,
        0,
        0,
        -0.123596,
        0.0618496,
        0.14236,
        -0.559927,
        0.215224,
        -0.371181,
        -1.10526,
        1.17408,
        -0.1376,
        2.99764,
        -4.21798,
        -1.06474,
        -1.16548,
        0,
        1.10284,
        0,
        -27.9412,
        0,
    ]
    expected_constraints = [
        1.90126e-15,
        1.99146e-15,
        -3.33067e-16,
        -1.11022e-16,
        6.10623e-16,
        5.60663e-15,
        1.55431e-15,
        -1.33227e-15,
        -1.47937e-14,
        -5.77316e-15,
        -1.1573e-12,
        -5.08926e-13,
        0,
        -1.30451e-15,
        -4.04121e-14,
        1.08612,
        1.05124,
        0.991253,
        0.954385,
        0.949236,
        0.9216,
        0.492353,
        0.554696,
        0.620095,
        1.34023,
        1.36917,
        1.38148,
        1.97149,
        2.0114,
        2.03747,
        2.89311,
        2.93228,
        2.95656,
    ]

    expected_cost = 261.095

    expected_detailed_cost = [
        {
            "name": "Lagrange.MINIMIZE_CONTROL",
            "cost_value_weighted": 261.0954331500721,
            "cost_value": -28.003864114406223,
        }
    ]

    expected_iterations = range(5, 35)  # 20 on my laptop @ipuch
    np.testing.assert_almost_equal(sol_second_pass.decision_states(to_merge=SolutionMerge.NODES)["q"], expected_q)
    np.testing.assert_almost_equal(sol_second_pass.decision_states(to_merge=SolutionMerge.NODES)["qdot"], expected_qdot)
    np.testing.assert_almost_equal(sol_second_pass.decision_controls(to_merge=SolutionMerge.NODES)["tau"], expected_tau)

    np.testing.assert_almost_equal(sol_second_pass.vector, np.array([expected_vector]).T, decimal=4)
    np.testing.assert_almost_equal(sol_second_pass.constraints, np.array([expected_constraints]).T, decimal=4)
    np.testing.assert_almost_equal(float(sol_second_pass.cost), expected_cost, decimal=2)

    assert sol_second_pass.detailed_cost[0]["name"] == expected_detailed_cost[0]["name"]
    np.testing.assert_almost_equal(
        sol_second_pass.detailed_cost[0]["cost_value_weighted"], expected_detailed_cost[0]["cost_value_weighted"]
    )
    np.testing.assert_almost_equal(
        sol_second_pass.detailed_cost[0]["cost_value"], expected_detailed_cost[0]["cost_value"]
    )

    assert sol_second_pass.iterations in expected_iterations
