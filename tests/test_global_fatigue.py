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
    np.testing.assert_almost_equal(f[0, 0], 19.770521758810368)

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
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.93853331,  3.00564551)))

    # fatigue parameters
    np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        ma[:, -1], np.array((0.00739128, 0.00563555, 0.00159309, 0.02418655, 0.02418655, 0.00041913))
    )
    np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.99260018, 0.99281414, 0.99707397, 0.97566527, 0.97566527, 0.99904065))
    )
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((8.54868154e-06, 1.55030599e-03, 1.33293886e-03, 1.48176210e-04, 1.48176210e-04, 5.40217808e-04)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.80920008, 1.66855572)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.81847388, -0.85234628)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((6.22395441e-08, 4.38966513e-01, 3.80781292e-01, 2.80532297e-07, 2.80532297e-07, 2.26601989e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((8.86069119e-03, 1.17337666e-08, 1.28715148e-08, 2.02340603e-02, 2.02340603e-02, 2.16517945e-088)),
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
    np.testing.assert_almost_equal(f[0, 0], 16.32400654587575)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (702, 1))
    np.testing.assert_almost_equal(g, np.zeros((702, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, ma, mr, mf = states["q"], states["qdot"], states["muscles_ma"], states["muscles_mr"], states["muscles_mf"]
    tau, muscles = controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
    np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.8913551, 3.68787122)))

    # fatigue parameters
    np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        ma[:, -1], np.array((0.03924828, 0.01089071, 0.00208428, 0.05019898, 0.05019898, 0.00058203))
    )
    np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.96071394, 0.98795266, 0.99699829, 0.9496845 , 0.9496845 , 0.99917771))
    )
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1], np.array((0,  3.59773278e-04,  3.59740895e-04, 0, 0, 0)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.96697626, 0.7686893)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.59833412, -0.73455049)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((1.25202085e-07, 3.21982969e-01, 2.28408549e-01, 3.74330449e-07, 3.74330448e-07, 1.69987512e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((0.0441982 , 0.00474236, 0.0009076 , 0.04843388, 0.04843388, 0.00025345)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_effort_fatigable_muscles():
    bioptim_folder = TestUtils.bioptim_folder()
    fatigue = TestUtils.load_module(f"{bioptim_folder}/examples/fatigue/static_arm_with_fatigue.py")

    model_path = f"{bioptim_folder}/examples/fatigue/models/arm26_constant.bioMod"
    ocp = fatigue.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        fatigue_type="effort",
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 15.670921245579846)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (252, 1))
    np.testing.assert_almost_equal(g, np.zeros((252, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot, mf = states["q"], states["qdot"], states["muscles_mf"]
    tau, muscles = controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.07, 1.4)))
    np.testing.assert_almost_equal(q[:, -1], np.array((1.64470726, 2.25033212)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.88775204,  3.6333437)))

    # fatigue parameters
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1], np.array((0,  3.59773278e-04,  3.59740895e-04, 0, 0, 0)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.00151725, 0.75680955)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.5258675 , -0.65113292)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((-3.28714919e-09,  3.22449026e-01,  2.29706846e-01,  2.48558352e-08,
        2.48558352e-08,  1.68035357e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((3.86483779e-02, 1.10050544e-09, 2.74222976e-09, 4.25097691e-02,
       4.25097691e-02, 6.56233979e-09)),
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
