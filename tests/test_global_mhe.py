"""
Test for file IO
"""
import numpy as np
from bioptim import OdeSolver, Solver

from .utils import TestUtils


def test_cyclic_nmpc():

    def update_functions(_nmpc, cycle_idx, _sol):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    bioptim_folder = TestUtils.bioptim_folder()
    nmpc_module = TestUtils.load_module(bioptim_folder + "/examples/moving_horizon_estimation/cyclic_nmpc.py")

    n_cycles = 3
    window_len = 20
    nmpc = nmpc_module.prepare_nmpc(
        model_path=bioptim_folder + "/examples/moving_horizon_estimation/arm2.bioMod",
        window_len=window_len,
        window_duration=1,
        max_torque=50,
    )
    sol = nmpc.solve(update_functions, solver=Solver.IPOPT)

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    np.testing.assert_equal(q.shape, (3, n_cycles * window_len))
    np.testing.assert_almost_equal(q[:, 0], np.array((-3.14159265,  0.12979378,  2.77623291)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2.82743339, 0.63193395, 2.68235056)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((6.28268908, -11.63289399,   0.37215021)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((6.28368154, -7.73180135,  3.56900657)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.01984925, 17.53758229, -1.92204945)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.01984925, -3.09892348,  0.23160067)))
