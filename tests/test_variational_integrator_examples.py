"""
Tests of the examples of the variational integrator.
"""

import numpy as np
import os
from bioptim import Solver


def test_variational_pendulum():
    """Test the variational integrator pendulum example"""
    from bioptim.examples.discrete_mechanics_for_optimal_control import example_variational_integrator_pendulum

    bioptim_folder = os.path.dirname(example_variational_integrator_pendulum.__file__)

    # --- Prepare the ocp --- #
    ocp = example_variational_integrator_pendulum.prepare_ocp(
        bio_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=100,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    np.testing.assert_almost_equal(
        sol.states["q"][:, 0].squeeze(),
        [0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.states["q"][:, 50].squeeze(),
        [-0.726413733965370, 0.957513371856119],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.states["q"][:, -1].squeeze(),
        [0.0, 3.14],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, 0].squeeze(),
        [10.244275356612663, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, 50].squeeze(),
        [1.571057590776628, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, -2].squeeze(),
        [-11.075690668538043, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.parameters["qdot0"].squeeze(),
        [0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.parameters["qdotN"].squeeze(),
        [0.0, 0.0],
        decimal=6,
    )


def test_variational_pendulum_with_holonomic_constraints():
    """Test the variational integrator pendulum with holonomic constraints example"""
    from bioptim.examples.discrete_mechanics_for_optimal_control import (
        example_variational_integrator_with_holonomic_constraints_pendulum,
    )

    bioptim_folder = os.path.dirname(example_variational_integrator_with_holonomic_constraints_pendulum.__file__)

    # --- Prepare the ocp --- #
    ocp = example_variational_integrator_with_holonomic_constraints_pendulum.prepare_ocp(
        bio_model_path=bioptim_folder + "/models/pendulum_holonomic.bioMod",
        final_time=1,
        n_shooting=100,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    np.testing.assert_almost_equal(
        sol.states["q"][:, 0].squeeze(),
        [0.0, 0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.states["q"][:, 50].squeeze(),
        [-0.726414103487621, 0.0, 0.957513756633348],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.states["q"][:, -1].squeeze(),
        [0.0, 0.0, 3.14],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, 0].squeeze(),
        [10.244271905334511, 0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, 50].squeeze(),
        [1.571068734115470, 0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.controls["tau"][:, -2].squeeze(),
        [-11.075685503793608, 0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.parameters["qdot0"].squeeze(),
        [0.0, 0.0, 0.0],
        decimal=6,
    )

    np.testing.assert_almost_equal(
        sol.parameters["qdotN"].squeeze(),
        [0.0, 0.0, 0.0],
        decimal=6,
    )
