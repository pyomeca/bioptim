"""
Test for file IO
"""
import os
import pytest

import numpy as np
from bioptim import OdeSolver, Solver

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
        np.testing.assert_almost_equal(g, np.zeros((20 * 5, 1)), decimal=5)
    else:
        np.testing.assert_equal(g.shape, (20, 1))
        np.testing.assert_almost_equal(g, np.zeros((20, 1)), decimal=5)

    # Check some of the results
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]
    if ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 0.12369835031375455)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.16806756, 2.65468153]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-1.47973292, 14.21736795]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00837209, 0.01991103]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00231058, 0.00268708]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([4.58410571e-06, 5.97816053e-01, 3.31990595e-01, 1.10272434e-05, 1.16080375e-05, 3.60499511e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([5.20539825e-05, 6.32041077e-03, 3.64566577e-03, 4.91166408e-04, 5.07451759e-04, 8.87382356e-03]),
        )

    elif ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 0.12369837010671972)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.16806758, 2.65468191]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-1.4797281, 14.2173632]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00837208, 0.01991103]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00231058, 0.00268707]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([4.44790964e-06, 5.97816063e-01, 3.31990620e-01, 1.10272405e-05, 1.16080346e-05, 3.60499532e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([4.67158105e-05, 6.36064841e-03, 3.70441050e-03, 5.16633150e-04, 5.33757415e-04, 8.89768052e-03]),
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 0.12369838396015334)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.16806772, 2.65467554]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-1.47975105, 14.21729773]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00837235, 0.01991081]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00231059, 0.00268708]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([4.44786958e-06, 5.97813268e-01, 3.31987682e-01, 1.10273614e-05, 1.16081619e-05, 3.60495697e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([4.67154321e-05, 6.38489487e-03, 3.73719311e-03, 5.30535414e-04, 5.48115098e-04, 8.91280115e-03]),
        )
    else:
        raise ValueError("Test not ready")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)
