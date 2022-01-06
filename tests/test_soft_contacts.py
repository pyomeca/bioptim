import os
import pytest
import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils


@pytest.mark.parametrize("n_threads", [8])
@pytest.mark.parametrize("use_sx", [False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK8])
def test_soft_contact(ode_solver, use_sx, n_threads):
    from bioptim.examples.torque_driven_ocp import example_soft_contact as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/soft_contact_sphere.bioMod",
        final_time=0.37,
        n_shooting=37,
        n_threads=n_threads,
        use_sx=use_sx,
        ode_solver=ode_solver,
    )

    ocp.print(to_console=True, to_graph=False)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if isinstance(ode_solver, OdeSolver.RK8):
        np.testing.assert_almost_equal(f[0, 0], 23.679065887950706)
    else:
        np.testing.assert_almost_equal(f[0, 0], 41.58259426)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (228, 1))
    np.testing.assert_almost_equal(g, np.zeros((228, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0, 0)), decimal=1)
    np.testing.assert_almost_equal(q[:, -1], np.array([0.05, 0.0933177, -0.62620287]))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)), decimal=4)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([2.03004523e-01, -1.74795966e-05, -2.53770131e00]))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-0.16347455, 0.02123226, -13.25955361]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([0.00862357, -0.00298151, -0.16425701]))
