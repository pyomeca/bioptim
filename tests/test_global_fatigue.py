import platform
import os

import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils


def test_xia_fatigable_muscles():
    from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
    ocp = ocp_module.prepare_ocp(
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
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.93853331, 3.00564551)))

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


def test_xia_stabilized_fatigable_muscles():
    from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=0.9,
        n_shooting=5,
        fatigue_type="xia_stabilized",
        ode_solver=OdeSolver.COLLOCATION(),
        torque_level=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 18.904146147978263)

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
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-1.78119672,  1.76347727)))

    # fatigue parameters
    np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        ma[:, -1],
        np.array((3.39538234e-06, 6.02070039e-03, 3.21988283e-03, 1.95918805e-02,
       1.95918805e-02, 7.88670707e-04)),
    )
    np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.9999966 , 0.99236943, 0.99537329, 0.98030776, 0.98030776,
       0.9985729))
    )
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((2.02250160e-09, 1.58159634e-03, 1.38994305e-03, 7.92778775e-05,
       7.92778775e-05, 6.33575125e-04)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.11808666, 0.92779318)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.96700045, -0.92269412)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((5.70272945e-08, 3.85244803e-01, 3.11772581e-01, 5.96397281e-07,
       5.96397280e-07, 1.74119901e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((4.06360849e-06, 1.23286848e-08, 1.10684255e-08, 1.84235398e-02,
       1.84235398e-02, 1.98894149e-08)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_michaud_fatigable_muscles():
    from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
    ocp = ocp_module.prepare_ocp(
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
    if platform.system() == "Linux":
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
    np.testing.assert_almost_equal(ma[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((0, 3.59773278e-04, 3.59740895e-04, 0, 0, 0)),
    )
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.8913551, 3.68787122)))
        np.testing.assert_almost_equal(
            ma[:, -1], np.array((0.03924828, 0.01089071, 0.00208428, 0.05019898, 0.05019898, 0.00058203))
        )
        np.testing.assert_almost_equal(
            mr[:, -1], np.array((0.96071394, 0.98795266, 0.99699829, 0.9496845, 0.9496845, 0.99917771))
        )
        np.testing.assert_almost_equal(tau[:, 0], np.array((0.96697626, 0.7686893)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((0.59833412, -0.73455049)))
        np.testing.assert_almost_equal(
            muscles[:, 0],
            np.array((1.25202085e-07, 3.21982969e-01, 2.28408549e-01, 3.74330449e-07, 3.74330448e-07, 1.69987512e-01)),
        )
        np.testing.assert_almost_equal(
            muscles[:, -2],
            np.array((0.0441982, 0.00474236, 0.0009076, 0.04843388, 0.04843388, 0.00025345)),
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_effort_fatigable_muscles():
    from bioptim.examples.fatigue import static_arm_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/arm26_constant.bioMod"
    ocp = ocp_module.prepare_ocp(
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
    np.testing.assert_almost_equal(f[0, 0], 15.6707872174798)

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
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.88775196, 3.63334305)))

    # fatigue parameters
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((3.42522894e-09, 1.50441857e-07, 1.02811466e-07, 1.17766898e-08, 1.17766898e-08, 4.00293490e-08)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.00151693, 0.75680923)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.52586736, -0.6511327)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((-3.28714673e-09, 3.22448998e-01, 2.29706897e-01, 2.48558502e-08, 2.48558502e-08, 1.68035523e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((3.86484687e-02, 1.10050931e-09, 2.74223400e-09, 4.25098184e-02, 4.25098184e-02, 6.56234551e-09)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_xia_torque_non_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="xia",
        split_controls=False,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(f[0, 0], 681.4936347682981)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
    ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
    tau = controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))

    if platform.system() == "Linux":
        np.testing.assert_almost_equal(ma_minus[:, -1], np.array((2.05715389e-01, 0)))
        np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.71681593, 1)))
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((7.74686771e-02, 0)))
        np.testing.assert_almost_equal(ma_plus[:, -1], np.array((4.54576950e-03, 0)))
        np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.91265673, 1)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((8.27975034e-02, 0)))
        np.testing.assert_almost_equal(tau[:, 0], np.array((4.65387493, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-21.7531631, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_xia_torque_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path, final_time=1, n_shooting=30, fatigue_type="xia", split_controls=True, use_sx=False
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 46.97293026598778)

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

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((9.74835527e-02, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88266826, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((1.98481921e-02, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((5.69110401e-06, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.9891588, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((1.08355110e-02, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-10.29111867, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((7.0546191, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_xia_stabilized_torque_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=30,
        fatigue_type="xia_stabilized",
        split_controls=True,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 58.3380274895759)

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

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0.0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((1.28587581e-01, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.84200964, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((6.53188577e-03, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((2.91160036e-05, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.98087163, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array(( 3.59780294e-02, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-3.01735252e-07, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-13.61176916, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((6.8734662, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_michaud_torque_non_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="michaud",
        split_controls=False,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(f[0, 0], 752.2660291516361)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (200, 1))
    np.testing.assert_almost_equal(g, np.zeros((200, 1)), decimal=6)

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    ma_minus, mr_minus, mf_minus = states["tau_minus_ma"], states["tau_minus_mr"], states["tau_minus_mf"]
    ma_plus, mr_plus, mf_plus = states["tau_plus_ma"], states["tau_plus_mr"], states["tau_plus_mf"]
    tau = controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))

    if platform.system() == "Linux":
        np.testing.assert_almost_equal(ma_minus[:, -1], np.array((2.27726849e-01, 0)))
        np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.77154438, 1)))
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((2.99934839e-04, 0)))
        np.testing.assert_almost_equal(ma_plus[:, -1], np.array((2.94965705e-03, 0)))
        np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.99650902, 1)))
        np.testing.assert_almost_equal(tau[:, 0], np.array((4.59966318, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-22.86838109, 0)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((9.99805014e-05, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


def test_fatigable_michaud_torque_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="michaud",
        split_controls=True,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 66.4869989782804)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (200, 1))
    np.testing.assert_almost_equal(g, np.zeros((200, 1)))

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

    np.testing.assert_almost_equal(ma_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((1.14840287e-01, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.88501154, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((6.06085673e-04, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.99924023, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-2.39672721e-07, 0)))
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-11.53208375, 0)))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.03417941, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


def test_fatigable_effort_torque_non_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="effort",
        split_controls=False,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if platform.system() == "Linux":
        np.testing.assert_almost_equal(f[0, 0], 681.4936347682992)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (80, 1))
    np.testing.assert_almost_equal(g, np.zeros((80, 1)))

    # Check some of the results
    states, controls = sol.states, sol.controls
    q, qdot = states["q"], states["qdot"]
    mf_minus, mf_plus = states["tau_minus_mf"], states["tau_plus_mf"]
    tau = controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))

    if platform.system() == "Linux":
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((1.87657915e-07, 0)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((8.11859845e-08, 0)))
        np.testing.assert_almost_equal(tau[:, 0], np.array((4.65387489, 0)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((-21.75316318, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


def test_fatigable_effort_torque_split():
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="effort",
        split_controls=True,
        use_sx=False,
    )
    sol = ocp.solve()

    # Check objective function value
    if platform != "Darwin":
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 946.1348332835429)

        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (80, 1))
        np.testing.assert_almost_equal(g, np.zeros((80, 1)))

        # Check some of the results
        states, controls = sol.states, sol.controls
        q, qdot = states["q"], states["qdot"]
        mf_minus, mf_plus = states["tau_minus_mf"], states["tau_plus_mf"]
        tau_minus, tau_plus = controls["tau_minus"], controls["tau_plus"]

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

        np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((9.97591033e-08, 0)))
        np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((9.76072978e-08, 0)))

        np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-8.39444342e-08, 0)))
        np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-25.58603374, 0)))
        np.testing.assert_almost_equal(tau_plus[:, 0], np.array((4.83833953, 0)))
        np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)
