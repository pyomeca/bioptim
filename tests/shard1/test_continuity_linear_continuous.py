import os

from bioptim import PhaseDynamics, ControlType, QuadratureRule, Solver
import numpy.testing as npt

from ..utils import TestUtils


def test_continuity_linear_continuous_global():
    """
    This test combines linear continuous controls and trapezoidal integration method.
    This is to make sure we take into account the combined effect of these two features.
    """
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=False,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
        control_type=ControlType.LINEAR_CONTINUOUS,
        quadrature_rule=QuadratureRule.TRAPEZOIDAL,
    )

    solution = ocp.solve(Solver.IPOPT(_max_iter=5))

    actual_costs = solution.detailed_cost

    expected_costs = [
        {
            "name": "Lagrange.MINIMIZE_CONTROL",
            "cost_value_weighted": 19401.119148753893,
            "cost_value": 196.2000000000025,
        },
        {
            "name": "MultinodeObjectiveFcn.CUSTOM",
            "cost_value_weighted": 0.08631534265793345,
            "cost_value": -0.03625119905959934,
        },
        {
            "name": "Lagrange.MINIMIZE_CONTROL",
            "cost_value_weighted": 48130.30361835311,
            "cost_value": 294.30000000000007,
        },
        {
            "name": "Lagrange.MINIMIZE_CONTROL",
            "cost_value_weighted": 38561.67005961334,
            "cost_value": 196.1999999999997,
        },
    ]

    # Assert that the expected and actual cost values are the same
    for expected, actual in zip(expected_costs, actual_costs):
        assert expected["name"] == actual["name"]

        npt.assert_almost_equal(expected["cost_value_weighted"], actual["cost_value_weighted"])
        npt.assert_almost_equal(expected["cost_value"], actual["cost_value"])
