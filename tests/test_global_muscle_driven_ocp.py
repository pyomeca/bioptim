"""
Test for file IO
"""
import pytest
import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_muscle_driven_ocp(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    static_arm = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/static_arm.py")
    ode_solver = ode_solver()

    ocp = static_arm.prepare_ocp(
        bioptim_folder + "/examples/muscle_driven_ocp/arm26.bioMod",
        final_time=2,
        n_shooting=10,
        weight=1,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14351611580879933)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.94511299, 3.07048865]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.41149114, -0.55863385]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00147561, 0.00520749]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00027953, 0.00069257]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.29029533e-06, 1.64976642e-01, 1.00004898e-01, 4.01974257e-06, 4.13014984e-06, 1.03945583e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.25940361e-03, 3.21754460e-05, 3.12984790e-05, 2.00725054e-03, 1.99993619e-03, 1.81725854e-03]),
        )

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14350914060136277)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.94510844, 3.07048231]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.41151235, -0.55866253]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00147777, 0.00520795]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00027953, 0.00069258]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.28863414e-06, 1.65011897e-01, 1.00017224e-01, 4.01934660e-06, 4.12974244e-06, 1.03954780e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.25990460e-03, 3.21893307e-05, 3.13077447e-05, 2.01209936e-03, 2.00481801e-03, 1.82353344e-03]),
        )

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14350464848810182)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.9451058, 3.0704789]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.4115254, -0.5586797]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.0014793, 0.0052082]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-0.0002795, 0.0006926]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.2869218e-06, 1.6503522e-01, 1.0002514e-01, 4.0190181e-06, 4.1294041e-06, 1.0396051e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.2599283e-03, 3.2188697e-05, 3.1307377e-05, 2.0121186e-03, 2.0048373e-03, 1.8235679e-03]),
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)
