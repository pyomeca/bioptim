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
    from bioptim.examples.toy_examples.discrete_mechanics_and_optimal_control import (
        example_variational_integrator_pendulum,
    )

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp = example_variational_integrator_pendulum.prepare_ocp(
        bio_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=20,
        use_sx=use_sx,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(states["q"][:, 0], [0.0, 0.0], decimal=6)
    npt.assert_almost_equal(states["q"][:, 10], [-0.227191682344467, 0.417947896973504], decimal=6)
    npt.assert_almost_equal(states["q"][:, -1], [0.0, 3.14], decimal=6)

    npt.assert_almost_equal(controls["tau"][:, 0], [11.973897222965608, 0.0], decimal=6)
    npt.assert_almost_equal(controls["tau"][:, 20], [-19.644558618117195, 0.0], decimal=6)
    npt.assert_almost_equal(controls["tau"][:, -4], [9.998764637575732, 0.0], decimal=6)

    npt.assert_almost_equal(sol.parameters["qdot_start"], [0.0, 0.0], decimal=6)
    npt.assert_almost_equal(sol.parameters["qdot_end"], [0.0, 0.0], decimal=6)


@pytest.mark.parametrize("use_sx", [False, True])
def test_variational_pendulum_with_holonomic_constraints(use_sx):
    """Test the variational integrator pendulum with holonomic constraints example"""
    from bioptim.examples.toy_examples.discrete_mechanics_and_optimal_control import (
        example_variational_integrator_with_holonomic_constraints_pendulum,
    )

    bioptim_folder = TestUtils.bioptim_folder()

    # --- Prepare the ocp --- #
    ocp = example_variational_integrator_with_holonomic_constraints_pendulum.prepare_ocp(
        bio_model_path=bioptim_folder + "/examples/models/pendulum_holonomic.bioMod",
        final_time=1,
        n_shooting=20,
        use_sx=use_sx,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT())
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    npt.assert_almost_equal(states["lambdas"][:, 0], [-12.569972544979578], decimal=6)
    npt.assert_almost_equal(states["lambdas"][:, -1], [22.847930707142613], decimal=6)

    npt.assert_almost_equal(states["q"][:, 0], [0.0, 0.0, 0.0], decimal=6)
    npt.assert_almost_equal(
        states["q"][:, 10], [-2.329813885524443e-01, -3.196131756793987e-14, 4.302587257051049e-01], decimal=6
    )
    npt.assert_almost_equal(states["q"][:, -1], [0.0, 0.0, 3.14], decimal=6)

    npt.assert_almost_equal(controls["tau"][:, 0], [11.370708961872488, 0.0, 0.0], decimal=6)
    npt.assert_almost_equal(controls["tau"][:, 20], [-18.763179166039833, 0.0, 0.0], decimal=6)
    npt.assert_almost_equal(controls["tau"][:, -4], [9.347683129357181, 0.0, 0.0], decimal=6)

    npt.assert_almost_equal(
        sol.parameters["qdot_start"], [1.000000976270894e-02, -5.248038150251689e-16, 1.000000975829636e-02], decimal=6
    )
    npt.assert_almost_equal(
        sol.parameters["qdot_end"], [-1.000000959168232e-02, 5.121224865078237e-16, 1.000000959323689e-02], decimal=6
    )
