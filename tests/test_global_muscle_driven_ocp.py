"""
Test for file IO
"""
import os
import pytest

import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.IRK, OdeSolver.COLLOCATION])
def test_muscle_driven_ocp(ode_solver):
    from bioptim.examples.muscle_driven_ocp import static_arm as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        bioptim_folder + "/models/arm26.bioMod",
        final_time=0.1,
        n_shooting=5,
        weight=1,
        ode_solver=ode_solver(),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (20 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((20 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (20, 1))
        np.testing.assert_almost_equal(g, np.zeros((20, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 0.1295996956517354)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.21049937, 2.66347178]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.60354736, 14.13846729]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00742946, 0.01867067]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00225482, 0.00278352]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([5.56663119e-06, 5.88956271e-01, 3.45151119e-01, 1.12613757e-05, 1.15791172e-05, 3.66139526e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([4.68407227e-05, 4.18459135e-02, 2.57479023e-02, 1.98847006e-03, 1.99863757e-03, 3.36166011e-02]),
        )

    elif ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 0.1295996956517354)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.21049957, 2.66347197]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.60354823, 14.13846625]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00742946, 0.01867066]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00225482, 0.00278352]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([5.56662312e-06, 5.88956121e-01, 3.45150958e-01, 1.12613865e-05, 1.15791281e-05, 3.66139401e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([4.68408159e-05, 4.18458099e-02, 2.57478452e-02, 1.98847543e-03, 1.99864288e-03, 3.36165225e-02]),
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 0.1295997069118093)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.21049957, 2.66346524]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.60356895, 14.13838088]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00742974, 0.01867047]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00225483, 0.00278353]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([5.59017711e-06, 5.88953738e-01, 3.45148044e-01, 1.12186997e-05, 1.15352694e-05, 3.66135660e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([4.69994690e-05, 4.18470695e-02, 2.57503597e-02, 2.00270750e-03, 2.01283945e-03, 3.36178728e-02]),
        )
    else:
        raise ValueError("Test not ready")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)
