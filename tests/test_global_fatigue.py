from .utils import TestUtils

import numpy as np
from bioptim import OdeSolver


def test_xia_fatigable_muscles():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/static_arm_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/models/arm26_constant.bioMod"
    ocp = fatigue.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        fatigue_type="xia",
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 14.541159477787328)

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
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.2459991, 3.18299163)))

    # fatigue parameters
    np.testing.assert_almost_equal(
        ma[:, 0],
        np.array((1.48745294e-06, 8.40530104e-01, 9.99999760e-01, 9.99999536e-01, 9.99999536e-01, 3.07494906e-01)),
    )
    np.testing.assert_almost_equal(
        ma[:, -1], np.array((0.01701756, 0.00497107, 0.00074379, 0.03855388, 0.03855388, 0.00014729))
    )
    np.testing.assert_almost_equal(
        mr[:, 0],
        np.array((5.02286654e-01, 1.88234907e-02, 3.80716857e-04, 2.68711752e-03, 2.68711752e-03, 2.52713019e-01)),
    )
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.485251, 0.85272351, 0.99818739, 0.96284608, 0.96284608, 0.55962829))
    )
    np.testing.assert_almost_equal(
        mf[:, 0],
        np.array((5.41684445e-05, 3.99979682e-08, 3.97219808e-08, 5.19496978e-08, 5.19496978e-08, 1.58563641e-07)),
    )
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((7.37541597e-05, 1.65905174e-03, 1.44934200e-03, 1.28674686e-03, 1.28674686e-03, 4.32504808e-04)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.02575629, 0.04092752)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((1.00050166, -1.20249023)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((8.04095214e-09, 9.60100489e-02, 1.03211076e-01, 2.68214642e-06, 2.68214642e-06, 4.88423858e-02)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((2.04007424e-02, 5.43752305e-09, 7.08127869e-09, 2.96570420e-02, 2.96570420e-02, 1.24844428e-08)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_michaud_fatigable_muscles():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/static_arm_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/models/arm26_constant.bioMod"
    ocp = fatigue.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        fatigue_type="michaud",
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 14.73842884)

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
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.26213599, 3.20810705)))

    # fatigue parameters
    np.testing.assert_almost_equal(
        ma[:, 0],
        np.array((9.44228091e-08, 8.43601660e-01, 9.91888885e-01, 9.90149214e-01, 9.90149214e-01, 3.01951606e-01)),
    )
    np.testing.assert_almost_equal(
        ma[:, -1], np.array((0.01828051, 0.00596713, 0.00179076, 0.03971385, 0.03971385, 0.00119471))
    )
    np.testing.assert_almost_equal(
        mr[:, 0],
        np.array((9.84563128e-01, 1.53000959e-01, 2.35315505e-07, 4.56365919e-07, 4.56365919e-07, 6.84336510e-01)),
    )
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.97544329, 0.99265159, 0.99491161, 0.95628143, 0.95628143, 0.99323042))
    )
    np.testing.assert_almost_equal(
        mf[:, 0],
        np.array((2.02152351e-07, 9.02637623e-08, 7.70418078e-08, 5.88505434e-07, 5.88505434e-07, 1.02856071e-07)),
    )
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((1.61940091e-07, 4.44512938e-08, 2.88499039e-08, 1.13063984e-07, 1.13063984e-07, 7.93653462e-08)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.02614443, 0.04111782)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((1.00672255, -1.21169387)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((8.63947680e-10, 9.57745872e-02, 1.03402383e-01, 3.05354087e-06, 3.05354087e-06, 4.86013016e-02)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((2.08030796e-02, 5.47693177e-09, 7.17538706e-09, 3.00141373e-02, 3.00141373e-02, 1.25372170e-08)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_xia_torque():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/pendulum_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/models/pendulum.bioMod"
    ocp = fatigue.prepare_ocp(
        biorbd_model_path=model_path, final_time=1, n_shooting=30, fatigue_type="xia", use_sx=False
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 40.95718478849265)

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

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.01985313, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((7.37767532e-02, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((0.06325867, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.55150842, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((5.5779170e-01, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((1.56183245e-02, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0.40909385, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((6.85653498e-07, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((0.02136454, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.71084984, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((2.87872267e-01, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((7.48012913e-03, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-2.05780708, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-7.78768695, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((6.52663587e-07, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((1.64880042e-07, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_michaud_torque():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/pendulum_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/models/pendulum.bioMod"
    ocp = fatigue.prepare_ocp(
        biorbd_model_path=model_path, final_time=1, n_shooting=30, fatigue_type="michaud", use_sx=False
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 39.148770270880526)

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

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((3.17363764e-08, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((8.57856948e-02, 3.49993000e-05)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((0.50501489, 0.49184462)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.43233508, 0.50680829)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0.50572158, 0.49632324)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((0.48187971, 0.49315618)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0.98767835, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((6.90717846e-03, 3.49993000e-05)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((3.16861572e-08, 4.91844622e-01)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.99309224, 0.50680829)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((3.10122002e-04, 4.96323238e-01)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((3.16294492e-08, 4.93156177e-01)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-3.2944767e-07, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-12.86030016, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((4.0186369, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((9.60415772e-08, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)
