import platform
import os

import pytest
import numpy as np
from bioptim import OdeSolver, Solver

from .utils import TestUtils


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_xia_fatigable_muscles(assume_phase_dynamics):
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


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_xia_stabilized_fatigable_muscles(assume_phase_dynamics):
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
        assume_phase_dynamics=assume_phase_dynamics,
        n_threads=8 if assume_phase_dynamics else 1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19.770521758810393)

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
        ma[:, -1],
        np.array((0.00739128, 0.00563555, 0.00159309, 0.02418655, 0.02418655, 0.00041913)),
    )
    np.testing.assert_almost_equal(mr[:, 0], np.array((1, 1, 1, 1, 1, 1)))
    np.testing.assert_almost_equal(
        mr[:, -1], np.array((0.99260018, 0.99281414, 0.99707397, 0.97566527, 0.97566527, 0.99904065))
    )
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((8.54868155e-06, 1.55030599e-03, 1.33293886e-03, 1.48176210e-04, 1.48176210e-04, 5.40217808e-04)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.80920008, 1.66855572)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.81847388, -0.85234628)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((6.22395441e-08, 4.38966513e-01, 3.80781292e-01, 2.80532298e-07, 2.80532298e-07, 2.26601989e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((8.86069119e-03, 1.17337666e-08, 1.28715148e-08, 2.02340603e-02, 2.02340603e-02, 2.16517945e-08)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_michaud_fatigable_muscles(assume_phase_dynamics):
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
        assume_phase_dynamics=assume_phase_dynamics,
        n_threads=8 if assume_phase_dynamics else 1,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (702, 1))

    # Check some of the results
    # TODO: add tests

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_effort_fatigable_muscles(assume_phase_dynamics):
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
        assume_phase_dynamics=assume_phase_dynamics,
        n_threads=8 if assume_phase_dynamics else 1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 15.670790035133818)

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
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-3.88775177, 3.63334333)))

    # fatigue parameters
    np.testing.assert_almost_equal(mf[:, 0], np.array((0, 0, 0, 0, 0, 0)))
    np.testing.assert_almost_equal(
        mf[:, -1],
        np.array((0, 6.37439422e-05, 4.49189030e-05, 0, 0, 0)),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.00151692, 0.75680941)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((0.52586761, -0.65113307)))

    np.testing.assert_almost_equal(
        muscles[:, 0],
        np.array((-3.28714697e-09, 3.22448892e-01, 2.29707231e-01, 2.48558443e-08, 2.48558443e-08, 1.68035326e-01)),
    )
    np.testing.assert_almost_equal(
        muscles[:, -2],
        np.array((3.86483818e-02, 1.10050313e-09, 2.74222702e-09, 4.25097771e-02, 4.25097771e-02, 6.56233597e-09)),
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_fatigable_xia_torque_non_split(assume_phase_dynamics):
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    # it doesn't pass on macos
    if platform.system() == "Darwin":
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="xia",
        split_controls=False,
        use_sx=False,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (160, 1))

    # Check some of the results
    # TODO: add tests

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_fatigable_xia_torque_split(assume_phase_dynamics):
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="xia",
        split_controls=True,
        use_sx=False,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 73.27929222817079)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)))

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
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((1.14097518e-01, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.85128364, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((3.46188391e-02, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((1.05233076e-03, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.97572892, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((2.32187531e-02, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-12.0660082, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.2893453, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_fatigable_xia_stabilized_torque_split(assume_phase_dynamics):
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    if platform.system() == "Windows" and not assume_phase_dynamics:
        # This is a long test and CI is already long for Windows
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="xia_stabilized",
        split_controls=True,
        use_sx=False,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 73.2792922281799)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)))

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
    np.testing.assert_almost_equal(ma_minus[:, -1], np.array((1.14097518e-01, 0)))
    np.testing.assert_almost_equal(mr_minus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_minus[:, -1], np.array((0.85128364, 1)))
    np.testing.assert_almost_equal(mf_minus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_minus[:, -1], np.array((3.46188391e-02, 0)))
    np.testing.assert_almost_equal(ma_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(ma_plus[:, -1], np.array((1.05233076e-03, 0)))
    np.testing.assert_almost_equal(mr_plus[:, 0], np.array((1, 1)))
    np.testing.assert_almost_equal(mr_plus[:, -1], np.array((0.97572892, 1)))
    np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(mf_plus[:, -1], np.array((2.32187531e-02, 0)))

    np.testing.assert_almost_equal(tau_minus[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-12.0660082, 0)))
    np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.2893453, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_fatigable_michaud_torque_non_split(assume_phase_dynamics):
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    if platform.system() == "Windows":
        # This is a long test and CI is already long for Windows
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="michaud",
        split_controls=False,
        use_sx=False,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver=solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (200, 1))

    # Check some of the results
    # TODO: add some tests

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_fatigable_michaud_torque_split(assume_phase_dynamics):
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    if platform.system() == "Windows" and not assume_phase_dynamics:
        # This tst fails on the CI
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="michaud",
        split_controls=True,
        use_sx=False,
        assume_phase_dynamics=assume_phase_dynamics,
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
        np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.03417919, 0)))
    np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_fatigable_effort_torque_non_split(assume_phase_dynamics):
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
        assume_phase_dynamics=assume_phase_dynamics,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(0)
    sol = ocp.solve(solver=solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (80, 1))

    # Check some of the results
    # TODO: add some tests

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_fatigable_effort_torque_split(assume_phase_dynamics):
    from bioptim.examples.fatigue import pendulum_with_fatigue as ocp_module

    if platform.system() == "Windows" and not assume_phase_dynamics:
        # This tst fails on the CI
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    model_path = f"{bioptim_folder}/models/pendulum.bioMod"
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=model_path,
        final_time=1,
        n_shooting=10,
        fatigue_type="effort",
        split_controls=True,
        use_sx=False,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    if platform.system() == "Linux":
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 124.09811263203727)

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
        np.testing.assert_almost_equal(mf_minus[:, -1], np.array((4.51209384e-05, 1.99600599e-06)))
        np.testing.assert_almost_equal(mf_plus[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(mf_plus[:, -1], np.array((4.31950457e-05, 0)))

        np.testing.assert_almost_equal(tau_minus[:, 0], np.array((-8.39444342e-08, 0)))
        np.testing.assert_almost_equal(tau_minus[:, -2], np.array((-12.03087219, 0)))
        np.testing.assert_almost_equal(tau_plus[:, 0], np.array((5.85068579, 0)))
        np.testing.assert_almost_equal(tau_plus[:, -2], np.array((0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)
