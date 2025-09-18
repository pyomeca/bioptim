"""
Tests of the examples of the variational integrator.
"""

import os

import numpy.testing as npt
from bioptim import Solver, SolutionMerge
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("use_sx", [False, True])
def test_variational_pendulum(use_sx):
    """Test the variational integrator pendulum example"""
    from bioptim.examples.toy_examples.discrete_mechanics_and_optimal_control import example_variational_integrator_pendulum

    bioptim_folder = TestUtils.module_folder(example_variational_integrator_pendulum)

    # --- Prepare the ocp --- #
    ocp = example_variational_integrator_pendulum.prepare_ocp(
        bio_model_path=bioptim_folder + "examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=20,
        use_sx=use_sx,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(states["q"][:, 0], [0.0, 0.0], decimal=6)
    npt.assert_almost_equal(states["q"][:, 10], [-0.325653795765479, 0.514317755981177], decimal=6)
    npt.assert_almost_equal(states["q"][:, -1], [0.0, 3.14], decimal=6)

    npt.assert_almost_equal(controls["tau"][:, 0], [9.952650040830257, 0.0], decimal=6)
    npt.assert_almost_equal(controls["tau"][:, 20], [1.326124391015805, 0.0], decimal=6)
    npt.assert_almost_equal(controls["tau"][:, -4], [-24.871395482788490, 0.0], decimal=6)

    npt.assert_almost_equal(sol.parameters["qdot_start"], [0.0, 0.0], decimal=6)
    npt.assert_almost_equal(sol.parameters["qdot_end"], [0.0, 0.0], decimal=6)


@pytest.mark.parametrize("use_sx", [False, True])
def test_variational_pendulum_with_holonomic_constraints(use_sx):
    """Test the variational integrator pendulum with holonomic constraints example"""
    from bioptim.examples.toy_examples.discrete_mechanics_and_optimal_control import (
        example_variational_integrator_with_holonomic_constraints_pendulum,
    )

    bioptim_folder = TestUtils.module_folder(example_variational_integrator_with_holonomic_constraints_pendulum)

    # --- Prepare the ocp --- #
    ocp = example_variational_integrator_with_holonomic_constraints_pendulum.prepare_ocp(
        bio_model_path=bioptim_folder + "/../../models/pendulum_holonomic.bioMod",
        final_time=1,
        n_shooting=20,
        use_sx=use_sx,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(states["lambdas"][:, 0], [-16.478903], decimal=6)
    npt.assert_almost_equal(states["lambdas"][:, -1], [7.242878], decimal=6)

    npt.assert_almost_equal(states["q"][:, 0], [0.0, 0.0, 0.0], decimal=6)
    npt.assert_almost_equal(states["q"][:, 10], [-5.307718e-01, -2.969952e-14, 7.052470e-01], decimal=6)
    npt.assert_almost_equal(states["q"][:, -1], [0.0, 0.0, 3.14], decimal=6)

    npt.assert_almost_equal(controls["tau"][:, 0], [10.502854, 0.0, 0.0], decimal=6)
    npt.assert_almost_equal(controls["tau"][:, 20], [12.717297, 0.0, 0.0], decimal=6)
    npt.assert_almost_equal(controls["tau"][:, -4], [-19.131171, 0.0, 0.0], decimal=6)

    npt.assert_almost_equal(sol.parameters["qdot_start"], [1.000001e-02, 1.507920e-16, 1.000001e-02], decimal=6)
    npt.assert_almost_equal(sol.parameters["qdot_end"], [-1.000001e-02, 7.028717e-16, 1.000001e-02], decimal=6)
