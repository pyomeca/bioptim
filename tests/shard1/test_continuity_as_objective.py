from bioptim import PhaseDynamics
import numpy as np
import pytest

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
    model_path = (
        TestUtils.bioptim_folder()
        + "/examples/getting_started/models/pendulum_maze.bioMod"
    )

    # first pass
    ocp = ocp_module.prepare_ocp_first_pass(
        biorbd_model_path=model_path,
        n_threads=1,
        final_time=1,
        n_shooting=3,
        state_continuity_weight=1000000,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    sol = ocp.solve()

    # # check q in integrality, qdot, controls, iterations
    #
    # expected_q = [
    #     [0.0, -1.1923517, -1.42486622, -0.1376],
    #     [0.0, 1.16898343, 1.88747861, 2.9976372],
    # ]
    #
    # expected_qdot = [
    #     [0.0, -5.43509473, 2.40522723, 24.88229668],
    #     [0.0, 5.15109901, 3.59555194, 21.98924882],
    # ]
    #
    # expected_controls = [
    #     [-20.54589655, 26.78617256, 15.76500633, np.nan],
    #     [0.0, 0.0, 0.0, np.nan],
    # ]
    #
    # expected_iterations = range(200, 250)  # while the number of iterations is reproductible, it differs between debug and raw runs
    # np.testing.assert_almost_equal(sol.states["q"], expected_q)
    # np.testing.assert_almost_equal(sol.states["qdot"], expected_qdot)
    # np.testing.assert_almost_equal(sol.controls["tau"], expected_controls)
    # assert sol.iterations in expected_iterations

    # second pass
    ocp_second_pass = ocp_module.prepare_ocp_second_pass(
        biorbd_model_path=model_path, n_threads=1, solution=sol
    )
    sol_second_pass = ocp_second_pass.solve()
    # check q in integrality, qdot,controls, vector, cost, iterations

    # expected_q = [
    #     [0.0, -1.19252768, -1.42504253, -0.1376],
    #     [0.0, 1.16885905, 1.88734408, 2.9976372],
    # ]
    # expected_qdot = [
    #     [0.0, -5.43603845, 2.40642094, 24.89415277],
    #     [0.0, 5.15237646, 3.59628495, 21.99866436],
    # ]
    # expected_vector = [
    #     [
    #         0,
    #         0,
    #         0,
    #         0,
    #         -1.19253,
    #         1.16886,
    #         -5.43604,
    #         5.15238,
    #         -1.42504,
    #         1.88734,
    #         2.40642,
    #         3.59628,
    #         -0.1376,
    #         2.99764,
    #         24.8942,
    #         21.9987,
    #         -20.5528,
    #         0,
    #         26.7991,
    #         0,
    #         15.7774,
    #         0,
    #         0.553202,
    #     ]
    # ]
    #
    # expected_constraints = [
    #     [
    #         -9.10383e-15,
    #         5.10703e-15,
    #         6.21725e-15,
    #         8.88178e-16,
    #         5.10703e-15,
    #         -5.9952e-15,
    #         4.26326e-14,
    #         2.08722e-14,
    #         3.91354e-15,
    #         1.15463e-14,
    #         8.52651e-14,
    #         9.9476e-14,
    #         0,
    #         -3.20577e-15,
    #         -1.4988e-14,
    #         1.08612,
    #         0.35,
    #         0.37334,
    #         0.954385,
    #         0.458212,
    #         0.692135,
    #         0.492353,
    #         0.996479,
    #         1.63013,
    #         1.34023,
    #         1.15602,
    #         1.29985,
    #         1.97149,
    #         1.87178,
    #         1.94454,
    #         2.89311,
    #         2.74089,
    #         2.67422,
    #     ]
    # ]

    expected_cost = 311.551

    expected_detailed_cost = [
        {
            "cost_value": 22.02370080865957,
            "cost_value_weighted": 256.23075256985715,
            "name": "Lagrange.MINIMIZE_CONTROL",
        },
        {
            "cost_value": 0.5532019998370968,
            "cost_value_weighted": 55.320199983709685,
            "name": "Mayer.MINIMIZE_TIME",
        },
    ]

    # expected_iterations = range(8, 12)
    # np.testing.assert_almost_equal(sol_second_pass.states["q"], expected_q)
    # np.testing.assert_almost_equal(sol_second_pass.states["qdot"], expected_qdot)
    # np.testing.assert_almost_equal(sol_second_pass.vector.T, expected_vector, decimal=1)
    # np.testing.assert_almost_equal(
    #     sol_second_pass.constraints.T, expected_constraints, decimal=1
    # )
    # np.testing.assert_almost_equal(
    #     float(sol_second_pass.cost), expected_cost, decimal=2
    # )
    # assert sol_second_pass.iterations in expected_iterations
