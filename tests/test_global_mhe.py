"""
Test for file IO
"""
import os

import numpy as np
from bioptim import Solver


def test_cyclic_nmpc():
    def update_functions(_nmpc, cycle_idx, _sol):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    from bioptim.examples.moving_horizon_estimation import cyclic_nmpc as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_cycles = 3
    cycle_len = 20
    nmpc = ocp_module.prepare_nmpc(
        model_path=bioptim_folder + "/models/arm2.bioMod",
        cycle_len=cycle_len,
        cycle_duration=1,
        max_torque=50,
    )
    sol = nmpc.solve(update_functions, solver=Solver.IPOPT())

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    np.testing.assert_equal(q.shape, (3, n_cycles * cycle_len))
    np.testing.assert_almost_equal(q[:, 0], np.array((-3.14159265, 0.12979378, 2.77623291)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2.82743339, 0.63193395, 2.68235056)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((6.28268908, -11.63289399, 0.37215021)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((6.28368154, -7.73180135, 3.56900657)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.01984925, 17.53758229, -1.92204945)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.01984925, -3.09892348, 0.23160067)), decimal=6)


def test_multi_cyclic_nmpc():
    def update_functions(_nmpc, cycle_idx, _sol):
        return cycle_idx < n_cycles_total  # True if there are still some cycle to perform

    from bioptim.examples.moving_horizon_estimation import multi_cyclic_nmpc as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_cycles_simultaneous = 2
    n_cycles_to_advance = 1
    n_cycles_total = 3
    cycle_len = 20
    nmpc = ocp_module.prepare_nmpc(
        model_path=bioptim_folder + "/models/arm2.bioMod",
        cycle_len=cycle_len,
        cycle_duration=1,
        n_cycles_simultaneous=n_cycles_simultaneous,
        n_cycles_to_advance=n_cycles_to_advance,
        max_torque=50,
    )
    sol = nmpc.solve(update_functions, solver=Solver.IPOPT(), n_cycles_simultaneous=n_cycles_simultaneous)

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    np.testing.assert_equal(q.shape, (3, n_cycles_total * cycle_len))
    np.testing.assert_almost_equal(q[:, 0], np.array((-12.56637061, 1.04359174, 1.03625065)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-6.59734457, 0.89827771, 1.0842402)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((6.28293718, 2.5617072, -0.00942694)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((6.28343343, 3.28099958, -1.27304428)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.00992505, 4.88488618, 2.4400698)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.00992505, 9.18387711, 5.22418771)), decimal=6)
