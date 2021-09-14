from .utils import TestUtils

import numpy as np
from bioptim import OdeSolver


def test_fatigable_muscles():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/static_arm_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/models/arm26_constant.bioMod"
    ocp = fatigue.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 14.539133019717394)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (552, 1))
    np.testing.assert_almost_equal(g, np.zeros((552, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, ma, mr, mf = states["q"], states["qdot"], states["muscles_ma"], states["muscles_mr"], states["muscles_mf"]
    tau, muscles = controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
    np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.24599073, 3.18297013)))

    # fatigue parameters
    np.testing.assert_almost_equal(
        ma[:, 0],
        np.array((1.32807277e-06, 8.40690602e-01, 9.99999768e-01, 9.99999563e-01, 9.99999563e-01, 3.07301564e-01)),
    )
    np.testing.assert_almost_equal(
        ma[:, -1], np.array((0.01701839, 0.00497141, 0.00074378, 0.03855518, 0.03855518, 0.00014726))
    )
    np.testing.assert_almost_equal(
        mr[:, 0],
        np.array((9.99876906e-01, 1.59661558e-01, 4.17731415e-04, 4.04748305e-04, 4.04748305e-04, 6.92828272e-01)),
    )
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.98284038, 0.99372091, 0.9982237, 0.96056167, 0.96056167, 0.99955019))
    )
    np.testing.assert_almost_equal(
        mf[:, 0],
        np.array((3.14130154e-05, 3.27364674e-04, 3.87164371e-04, 3.67080374e-04, 3.67080374e-04, 1.22439033e-04)),
    )
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((0.00014123, 0.00130769, 0.00103254, 0.00088316, 0.00088316, 0.00030256)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.02542356, 0.04088353)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((1.00050822, -1.20250826)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((8.57521247e-09, 9.60056240e-02, 1.03204359e-01, 3.31493107e-06, 3.31493107e-06, 4.88470915e-02)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((2.04017431e-02, 5.42722758e-09, 7.02346537e-09, 2.96574162e-02, 2.96574162e-02, 1.18595463e-08)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_torque():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/pendulum_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/models/pendulum.bioMod"
    ocp = fatigue.prepare_ocp(biorbd_model_path=model_path, final_time=1, n_shooting=30, use_sx=False)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 42.36278033654877)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (480, 1))
    np.testing.assert_almost_equal(g, np.zeros((480, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
    ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
    tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((2.7617393e-02, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((9.2887246e-02, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((0.05456142, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88806261, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((3.09189806e-01, 8.33651471e-08)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((1.90501432e-02, 2.08896043e-12)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((4.42636375e-01, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((2.9294882e-06, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((0.02011347, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.98979468, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((1.66068159e-01, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((1.02023908e-02, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-1.21286654e00, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-9.80300169e00, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((1.11431088e-06, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)
