import os

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

    if os.sys.platform == "windows":
        expected_q = [[0.0, -0.1820716, 0.0502083, -0.1376], [0.0, 0.2059882, -0.3885045, 2.9976372]]
        expected_qdot = [[0.0, 0.13105439, -3.43794783, -23.6570729], [0.0, -0.66178869, 3.07970721, -19.12526049]]
        expected_controls = [[-1.49607534, -0.24541618, -19.12881238], [0.0, 0.0, 0.0]]
        expected_iterations = range(300, 700)  # 436 on my laptop @ipuch, 639 on Windows Github CI

    if os.sys.platform == "linux" or os.sys.platform == "darwin":
        np.printoptions(precision=15, suppress=True)

        print(sol.decision_states(to_merge=SolutionMerge.NODES)["q"])
        print(sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"])
        print(sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"])
        print(sol.iterations)

        expected_q = [[0.0, -0.1820716, 0.0502083, -0.1376], [0.0, 0.2059882, -0.3885045, 2.9976372]]
        expected_qdot = [[0.0, 0.13105439, -3.43794783, -23.6570729], [0.0, -0.66178869, 3.07970721, -19.12526049]]
        expected_controls = [[-1.49607534, -0.24541618, -19.12881238], [0.0, 0.0, 0.0]]
        expected_iterations = range(300, 700)

    assert sol.iterations in expected_iterations
    np.testing.assert_almost_equal(sol.decision_states(to_merge=SolutionMerge.NODES)["q"], expected_q)
    np.testing.assert_almost_equal(sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"], expected_qdot)
    np.testing.assert_almost_equal(sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"], expected_controls)

    # second pass
    ocp_second_pass = ocp_module.prepare_ocp_second_pass(
        biorbd_model_path=model_path,
        n_threads=1,
        n_shooting=3,
        solution=sol,
        minimize_time=False,
    )
    sol_second_pass = ocp_second_pass.solve()

    expected_q = [[0.0, -0.12359617, 0.21522375, -0.1376], [0.0, 0.06184961, -0.37118107, 2.9976372]]
    expected_qdot = [[0.0, 0.14235975, -1.10526128, -4.21797828], [0.0, -0.55992744, 1.17407988, -1.06473819]]
    expected_tau = [[-1.16548046, 1.10283517, -27.94121882], [0.0, 0.0, 0.0]]

    expected_vector = [
        [0.333333333333333],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [-0.123596172283089],
        [0.06184960919155],
        [0.142359752273267],
        [-0.559927440453811],
        [0.215223745187429],
        [-0.371181072742915],
        [-1.105261281320264],
        [1.174079883672672],
        [-0.137599999999438],
        [2.99763720171466],
        [-4.217978284100386],
        [-1.064738187592947],
        [-1.16548046161843],
        [0.0],
        [1.102835170494119],
        [0.0],
        [-27.94121882328191],
        [0.0],
    ]
    expected_constraints = [
        [0.000000000000002],
        [0.000000000000002],
        [-0.0],
        [-0.0],
        [0.000000000000001],
        [0.000000000000006],
        [0.000000000000002],
        [-0.000000000000001],
        [-0.000000000000015],
        [-0.000000000000006],
        [-0.000000000001157],
        [-0.000000000000509],
        [0.0],
        [-0.000000000000001],
        [-0.00000000000004],
        [1.086117433798022],
        [1.051237343203642],
        [0.991252668915217],
        [0.954385184294056],
        [0.949235691291338],
        [0.921600419952798],
        [0.492352597230887],
        [0.554696150939241],
        [0.620095068306614],
        [1.340227995529119],
        [1.369169603766323],
        [1.381480510093951],
        [1.97149463098432],
        [2.011400546027982],
        [2.037470886662517],
        [2.893114425666568],
        [2.932275980994855],
        [2.956560343780495],
    ]

    expected_cost = 261.0954331500721
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

    np.testing.assert_almost_equal(sol_second_pass.vector, expected_vector)
    np.testing.assert_almost_equal(sol_second_pass.constraints, expected_constraints)
    np.testing.assert_almost_equal(float(sol_second_pass.cost), expected_cost)

    assert sol_second_pass.detailed_cost[0]["name"] == expected_detailed_cost[0]["name"]
    np.testing.assert_almost_equal(
        sol_second_pass.detailed_cost[0]["cost_value_weighted"], expected_detailed_cost[0]["cost_value_weighted"]
    )
    np.testing.assert_almost_equal(
        sol_second_pass.detailed_cost[0]["cost_value"], expected_detailed_cost[0]["cost_value"]
    )

    assert sol_second_pass.iterations in expected_iterations
