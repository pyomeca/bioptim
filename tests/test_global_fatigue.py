from .utils import TestUtils

import numpy as np
from bioptim import OdeSolver


def test_fatigable_muscles():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/static_arm_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/arm26_constant.bioMod"
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
    np.testing.assert_almost_equal(f[0, 0], 0.0001608)

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
    np.testing.assert_almost_equal(qdot[:, -1], np.array((5.64396402, -5.76061912)))

    # fatigue parameters
    np.testing.assert_almost_equal(
        ma[:, 0], np.array((0.00240059, 0.15905932, 0.11430391, 0.00319514, 0.00319514, 0.12065072))
    )
    np.testing.assert_almost_equal(
        ma[:, -1], np.array((0.00132612, 0.00129193, 0.00129276, 0.00133958, 0.00133958, 0.00129812))
    )
    np.testing.assert_almost_equal(
        mr[:, 0], np.array((0.9975931, 0.8410049, 0.88574217, 0.99679934, 0.99679934, 0.87939794))
    )
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.99865423, 0.99860923, 0.99863404, 0.99864057, 0.99864057, 0.9986249))
    )
    np.testing.assert_almost_equal(
        mf[:, 0],
        np.array((5.27901660e-06, 5.84681643e-05, 4.21000787e-05, 5.32340053e-06, 5.32340053e-06, 4.44211728e-05)),
    )
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((1.96475704e-05, 9.88472276e-05, 7.31988770e-05, 1.98550575e-05, 1.98550575e-05, 7.69833434e-05)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((5.14547083, 9.99818276)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((9.90666779, -7.26466591)))

    np.testing.assert_almost_equal(
        muscles[:, 0], np.array((0.00092469, 0.00011269, 0.00015668, 0.00096547, 0.00096547, 0.00015789))
    )
    np.testing.assert_almost_equal(
        muscles[:, -2], np.array((0.001281, 0.00131175, 0.00129413, 0.00128674, 0.00128674, 0.00130017))
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_torque():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/pendulum_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/pendulum.bioMod"
    ocp = fatigue.prepare_ocp(biorbd_model_path=model_path, final_time=3, n_shooting=30, use_sx=False)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 78.93205116298246)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (480, 1))
    np.testing.assert_almost_equal(g, np.zeros((480, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_ma_minus"], states["tau_mr_minus"], states["tau_mf_minus"]
    ma_plus, mr_plus, mf_plus = states["tau_ma_plus"], states["tau_mr_plus"], states["tau_mf_plus"]
    tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((6.43184345e-02, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((0.09384976, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.9117385, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((3.87370845e-01, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((2.39430682e-02, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((1, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((0, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((0.01424636, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.99951522, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((1.26659198e-01, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((4.84779734e-04, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-6.77598216, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.66844956, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)
