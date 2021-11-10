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
        np.testing.assert_almost_equal(f[0, 0], 0.1287891837070965)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.20964342, 2.66681312]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.55576371, 14.3208137]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00744364, 0.01844564]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00225168, 0.00274223]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([5.59842913e-06, 5.82963672e-01, 3.41202719e-01, 1.14045116e-05, 1.17266479e-05, 3.61743065e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([4.70953118e-05, 4.12630948e-02, 2.53947594e-02, 2.02570670e-03, 2.03584013e-03, 3.30582209e-02]),
        )

    elif ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 0.12878624563147562)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.20964664, 2.66682631]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.55579821, 14.3227702]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00744366, 0.01844438]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00225163, 0.0027421]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([5.59863034e-06, 5.82930799e-01, 3.41181230e-01, 1.14053268e-05, 1.17274885e-05, 3.61718752e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([4.70965781e-05, 4.12619188e-02, 2.53940987e-02, 2.02571975e-03, 2.03585343e-03, 3.30569074e-02]),
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 0.12878624563147562)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.20964665, 2.66681958]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.5558196, 14.32268489]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00744394, 0.01844419]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([0.00225164, 0.00274211]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([5.62246227e-06, 5.82928445e-01, 3.41178336e-01, 1.13618834e-05, 1.16828530e-05, 3.61715036e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([4.72570788e-05, 4.12631870e-02, 2.53966706e-02, 2.04023366e-03, 2.05033105e-03, 3.30582893e-02]),
        )
    else:
        raise ValueError("Test not ready")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)
