"""
Test for file IO
"""

import tracemalloc
import gc
import pickle
import platform
import re
import shutil
import time

from bioptim import (
    InterpolationType,
    OdeSolver,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    Node,
    ControlType,
    PhaseDynamics,
    SolutionMerge,
)
from casadi import sum1, sum2
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


# Store results for all tests
global test_memory
test_memory = {}


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("n_threads", [1, 2])
@pytest.mark.parametrize("use_sx", [False, True])
@pytest.mark.parametrize(
    "ode_solver",
    [
        OdeSolver.RK1,
        OdeSolver.RK2,
        OdeSolver.CVODES,
        OdeSolver.RK4,
        OdeSolver.RK8,
        OdeSolver.IRK,
        OdeSolver.COLLOCATION,
        OdeSolver.TRAPEZOIDAL,
    ],
)
def test_pendulum(ode_solver, use_sx, n_threads, phase_dynamics):
    from bioptim.examples.getting_started import pendulum as ocp_module

    if platform.system() == "Windows":
        # These tests fail on CI for Windows
        pytest.skip("Skipping tests on Windows")

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    # For reducing time phase_dynamics=PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if n_threads > 1 and phase_dynamics == PhaseDynamics.ONE_PER_NODE:
        return
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver not in (OdeSolver.RK4, OdeSolver.COLLOCATION):
        return
    if ode_solver == OdeSolver.RK8 and not use_sx:
        return

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_obj = ode_solver()

    if isinstance(ode_solver_obj, OdeSolver.CVODES):
        with pytest.raises(
            NotImplementedError,
            match=f"CVODES is not yet implemented",
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                final_time=2,
                n_shooting=10,
                n_threads=n_threads,
                use_sx=use_sx,
                ode_solver=ode_solver_obj,
                phase_dynamics=phase_dynamics,
                expand_dynamics=False,
            )
        return

    if isinstance(ode_solver_obj, (OdeSolver.IRK, OdeSolver.CVODES)) and use_sx:
        with pytest.raises(
            RuntimeError,
            match=f"use_sx=True and OdeSolver.{ode_solver_obj.integrator.__name__} are not yet compatible",
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                final_time=2,
                n_shooting=10,
                n_threads=n_threads,
                use_sx=use_sx,
                ode_solver=ode_solver_obj,
                phase_dynamics=phase_dynamics,
                expand_dynamics=False,
            )
        return
    elif isinstance(ode_solver_obj, OdeSolver.CVODES):
        with pytest.raises(
            RuntimeError,
            match=f"CVODES cannot be used with dynamics that depends on time",
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                final_time=2,
                n_shooting=10,
                n_threads=n_threads,
                use_sx=use_sx,
                ode_solver=ode_solver_obj,
                phase_dynamics=phase_dynamics,
                expand_dynamics=False,
            )
        return

    if isinstance(ode_solver_obj, (OdeSolver.TRAPEZOIDAL)):
        control_type = ControlType.CONSTANT_WITH_LAST_NODE
    else:
        control_type = ControlType.CONSTANT

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=30,
        n_threads=n_threads,
        use_sx=use_sx,
        ode_solver=ode_solver_obj,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver not in (OdeSolver.IRK, OdeSolver.CVODES),
        control_type=control_type,
    )
    tak = time.time()  # Time after building, but before solving
    ocp.print(to_console=True, to_graph=False)

    # the test is too long with CVODES
    if isinstance(ode_solver_obj, OdeSolver.CVODES):
        return

    sol = ocp.solve()
    tok = time.time()  # This after solving

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    if n_threads > 1:
        with pytest.raises(
            NotImplementedError, match="Computing detailed cost with n_thread > 1 is not implemented yet"
        ):
            detailed_cost = sol.detailed_cost[0]
        detailed_cost = None
    else:
        detailed_cost = sol.detailed_cost[0]

    if isinstance(ode_solver_obj, OdeSolver.RK8):
        npt.assert_almost_equal(f[0, 0], 41.57063948309302)
        # detailed cost values
        if detailed_cost is not None:
            npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 41.57063948309302)
        npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [-0.5010317, 0.6824593])

    elif isinstance(ode_solver_obj, OdeSolver.IRK):
        npt.assert_almost_equal(f[0, 0], 65.8236055171619)
        # detailed cost values
        if detailed_cost is not None:
            npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 65.8236055171619)
        npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.5536468, -0.4129719])

    elif isinstance(ode_solver_obj, OdeSolver.COLLOCATION):
        npt.assert_almost_equal(f[0, 0], 46.667345680854794)
        # detailed cost values
        if detailed_cost is not None:
            npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 46.667345680854794)
        npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [-0.1780507, 0.3254202])

    elif isinstance(ode_solver_obj, OdeSolver.RK1):
        npt.assert_almost_equal(f[0, 0], 47.360621044913245)
        # detailed cost values
        if detailed_cost is not None:
            npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 47.360621044913245)
        npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.1463538, 0.0215651])

    elif isinstance(ode_solver_obj, OdeSolver.RK2):
        npt.assert_almost_equal(f[0, 0], 76.24887695462857)
        # detailed cost values
        if detailed_cost is not None:
            npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 76.24887695462857)
        npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.652476, -0.496652])

    elif isinstance(ode_solver_obj, OdeSolver.TRAPEZOIDAL):
        npt.assert_almost_equal(f[0, 0], 31.423389566303985)
        # detailed cost values
        if detailed_cost is not None:
            npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 31.423389566303985)
        npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.69364974, -0.48330043])

    else:
        npt.assert_almost_equal(f[0, 0], 41.58259426)
        # detailed cost values
        if detailed_cost is not None:
            npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 41.58259426)
        npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [-0.4961208, 0.6764171])

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver_obj.is_direct_collocation:
        npt.assert_equal(g.shape, (600, 1))
        npt.assert_almost_equal(g, np.zeros((600, 1)))
    else:
        npt.assert_equal(g.shape, (120, 1))
        npt.assert_almost_equal(g, np.zeros((120, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    if isinstance(ode_solver_obj, OdeSolver.RK8):
        npt.assert_almost_equal(tau[:, 0], np.array((6.03763589, 0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-13.59527556, 0)))
    elif isinstance(ode_solver_obj, OdeSolver.IRK):
        npt.assert_almost_equal(tau[:, 0], np.array((5.40765381, 0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-25.26494109, 0)))
    elif isinstance(ode_solver_obj, OdeSolver.COLLOCATION):
        npt.assert_almost_equal(tau[:, 0], np.array((5.78386563, 0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-18.22245512, 0)))
    elif isinstance(ode_solver_obj, OdeSolver.RK1):
        npt.assert_almost_equal(tau[:, 0], np.array((5.498956, 0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-17.6888209, 0)))
    elif isinstance(ode_solver_obj, OdeSolver.RK2):
        npt.assert_almost_equal(tau[:, 0], np.array((5.6934385, 0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-27.6610711, 0)))
    elif isinstance(ode_solver_obj, OdeSolver.TRAPEZOIDAL):
        npt.assert_almost_equal(tau[:, 0], np.array((6.79720006, 0.0)))
        npt.assert_almost_equal(tau[:, -2], np.array((-15.23562005, 0.0)))
    else:
        npt.assert_almost_equal(tau[:, 0], np.array((6.01549798, 0.0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-13.68877181, 0.0)))

    # simulate
    TestUtils.simulate(sol)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"pendulum-{ode_solver}-{use_sx}-{n_threads}-{phase_dynamics}"] = [
        building_duration,
        solving_duration,
        mem_used,
    ]

    return


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_custom_constraint_track_markers(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import custom_constraint as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (186, 1))
    npt.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 19767.53312569523)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((1.4516129, 9.81, 2.27903226)))
        npt.assert_almost_equal(tau[:, -1], np.array((-1.45161291, 9.81, -2.27903226)))

        # detailed cost values
        npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 19767.533125695227)
    else:
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 19767.533125695223)

        npt.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
        npt.assert_almost_equal(tau[:, -1], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))

        # detailed cost values
        npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 19767.533125695227)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"custom_constraint_track_markers-{ode_solver}-{phase_dynamics}"] = [
        building_duration,
        solving_duration,
        mem_used,
    ]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("random_init", [True, False])
@pytest.mark.parametrize("interpolation", [*InterpolationType])
@pytest.mark.parametrize("ode_solver", [OdeSolver.COLLOCATION])
def test_initial_guesses(ode_solver, interpolation, random_init, phase_dynamics):
    from bioptim.examples.getting_started import custom_initial_guess as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver = ode_solver()

    np.random.seed(42)

    if interpolation == InterpolationType.ALL_POINTS and ode_solver.is_direct_shooting:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            _ = ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
                final_time=1,
                n_shooting=5,
                random_init=random_init,
                initial_guess=interpolation,
                ode_solver=ode_solver,
                phase_dynamics=phase_dynamics,
                expand_dynamics=True,
            )
        return

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        final_time=1,
        n_shooting=5,
        random_init=random_init,
        initial_guess=interpolation,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 13954.735)

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver.is_direct_collocation:
        npt.assert_equal(g.shape, (156, 1))
        npt.assert_almost_equal(g, np.zeros((156, 1)))
    else:
        npt.assert_equal(g.shape, (36, 1))
        npt.assert_almost_equal(g, np.zeros((36, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([1, 0, 0]))
    npt.assert_almost_equal(q[:, -1], np.array([2, 0, 1.57]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([5.0, 9.81, 7.85]))
    npt.assert_almost_equal(tau[:, -1], np.array([-5.0, 9.81, -7.85]))

    # simulate
    TestUtils.simulate(sol)

    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 13954.735000000004)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"initial_guesses-{ode_solver}-{interpolation}-{random_init}-{phase_dynamics}"] = [
        building_duration,
        solving_duration,
        mem_used,
    ]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_cyclic_objective(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import example_cyclic_movement as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    np.random.seed(42)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        final_time=1,
        n_shooting=10,
        loop_from_constraint=False,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 56851.88181545)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (67, 1))
    npt.assert_almost_equal(g, np.zeros((67, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([1.60205103, -0.01069317, 0.62477988]))
    npt.assert_almost_equal(q[:, -1], np.array([1, 0, 1.57]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0.12902365, 0.09340155, -0.20256713)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([9.89210954, 9.39362112, -15.53061197]))
    npt.assert_almost_equal(tau[:, -1], np.array([17.16370432, 9.78643138, -26.94701577]))

    # simulate
    TestUtils.simulate(sol)

    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 13224.252515047212)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"cyclic_objective-{ode_solver}-{phase_dynamics}"] = [building_duration, solving_duration, mem_used]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_cyclic_constraint(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import example_cyclic_movement as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    np.random.seed(42)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        final_time=1,
        n_shooting=10,
        loop_from_constraint=True,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 78921.61000000016)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (73, 1))
    npt.assert_almost_equal(g, np.zeros((73, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([1, 0, 1.57]))
    npt.assert_almost_equal(q[:, -1], np.array([1, 0, 1.57]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([20.0, 9.81, -31.4]))
    npt.assert_almost_equal(tau[:, -1], np.array([20.0, 9.81, -31.4]))

    # simulate
    TestUtils.simulate(sol)

    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 78921.61000000013)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"cyclic_constraint-{ode_solver}-{phase_dynamics}"] = [building_duration, solving_duration, mem_used]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_phase_transitions(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import custom_phase_transitions as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    # For reducing time phase_dynamics=PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        return

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 109443.6239236211)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (516, 1))
    npt.assert_almost_equal(g, np.zeros((516, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # initial and final position
    npt.assert_almost_equal(states[0]["q"][:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(states[-1]["q"][:, -1], np.array((1, 0, 0)))
    # initial and final velocities
    npt.assert_almost_equal(states[0]["qdot"][:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(states[-1]["qdot"][:, -1], np.array((0, 0, 0)))

    # cyclic continuity (between phase 3 and phase 0)
    npt.assert_almost_equal(states[-1]["q"][:, -1], states[0]["q"][:, 0])

    # Continuity between phase 0 and phase 1
    npt.assert_almost_equal(states[0]["q"][:, -1], states[1]["q"][:, 0])

    # initial and final controls
    npt.assert_almost_equal(controls[0]["tau"][:, 0], np.array((0.73170732, 12.71705188, -0.0928732)))
    npt.assert_almost_equal(controls[-1]["tau"][:, -1], np.array((0.11614402, 8.70686126, 1.05599166)))

    # simulate
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Phase transition must have the same number of states (3) when integrating with Shooting.SINGLE. "
            "If it is not possible, please integrate with Shooting.SINGLE_DISCONTINUOUS_PHASE."
        ),
    ):
        TestUtils.simulate(sol)

    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 14769.760808687663)
    npt.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 38218.35341602849)
    npt.assert_almost_equal(sol.detailed_cost[2]["cost_value_weighted"], 34514.48724963841)
    npt.assert_almost_equal(sol.detailed_cost[3]["cost_value_weighted"], 21941.02244926652)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"phase_transition-{ode_solver}-{phase_dynamics}"] = [building_duration, solving_duration, mem_used]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.COLLOCATION])  # OdeSolver.IRK
def test_parameter_optimization(ode_solver, phase_dynamics):
    return  # TODO: Fix parameter scaling :(
    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver in (OdeSolver.RK8, OdeSolver.COLLOCATION):
        return

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=20,
        optim_gravity=True,
        optim_mass=False,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=np.array([0, 0, -9.81]),
        target_m=20,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    gravity = sol.parameters["gravity_xyz"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # Check objective and constraints function value
    f = np.array(sol.cost)
    g = np.array(sol.constraints)
    npt.assert_equal(f.shape, (1, 1))

    if isinstance(ode_solver, OdeSolver.RK4):
        npt.assert_equal(g.shape, (80, 1))
        npt.assert_almost_equal(g, np.zeros((80, 1)), decimal=6)

        npt.assert_almost_equal(f[0, 0], 55.29552160879171, decimal=6)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((7.08951794, 0.0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-15.21533398, 0.0)))

        # gravity parameter
        npt.assert_almost_equal(gravity, np.array([[0, 4.95762449e-03, -9.93171691e00]]).T)

        # detailed cost values
        cost_values_all = np.sum(cost["cost_value_weighted"] for cost in sol.detailed_cost)
        npt.assert_almost_equal(cost_values_all, f[0, 0])

    elif isinstance(ode_solver, OdeSolver.RK8):
        npt.assert_equal(g.shape, (80, 1))
        npt.assert_almost_equal(g, np.zeros((80, 1)), decimal=6)

        npt.assert_almost_equal(f[0, 0], 49.828261340026486, decimal=6)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((5.82740495, 0.0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-13.06649769, 0.0)))

        # gravity parameter
        npt.assert_almost_equal(gravity, np.array([[0, 5.19787253e-03, -9.84722491e00]]).T)

        # detailed cost values
        cost_values_all = np.sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
        npt.assert_almost_equal(cost_values_all, f[0, 0])

    else:
        npt.assert_equal(g.shape, (400, 1))
        npt.assert_almost_equal(g, np.zeros((400, 1)), decimal=6)

        npt.assert_almost_equal(f[0, 0], 100.59286910162214, decimal=6)

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((-0.23081842, 0.0)))
        npt.assert_almost_equal(tau[:, -1], np.array((-26.01316438, 0.0)))

        # gravity parameter
        npt.assert_almost_equal(gravity, np.array([[0, 6.82939855e-03, -1.00000000e01]]).T)

        # detailed cost values
        cost_values_all = np.sum(sol.detailed_cost[i]["cost_value_weighted"] for i in range(len(sol.detailed_cost)))
        npt.assert_almost_equal(cost_values_all, f[0, 0])

    # simulate
    TestUtils.simulate(sol, decimal_value=6)

    # Test warm starting
    TestUtils.assert_warm_start(ocp, sol)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"parameter_optimization-{ode_solver}-{phase_dynamics}"] = [
        building_duration,
        solving_duration,
        mem_used,
    ]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("problem_type_custom", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_custom_problem_type_and_dynamics(problem_type_custom, ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import custom_dynamics as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        return

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        problem_type_custom=problem_type_custom,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 19767.5331257)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (186, 1))
    npt.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((1.4516129, 9.81, 2.27903226)))
    npt.assert_almost_equal(tau[:, -1], np.array((-1.45161291, 9.81, -2.27903226)))

    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 19767.533125695227)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"custom_dynamics-{ode_solver}-{problem_type_custom}-{phase_dynamics}"] = [
        building_duration,
        solving_duration,
        mem_used,
    ]


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("n_threads", [1, 2])
@pytest.mark.parametrize("use_sx", [True, False])
@pytest.mark.parametrize("use_point_of_applications", [True, False])
def test_example_external_forces(ode_solver, phase_dynamics, n_threads, use_sx, use_point_of_applications):
    from bioptim.examples.getting_started import example_external_forces as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    if use_sx and ode_solver == OdeSolver.IRK:
        return
    if n_threads == 2 and phase_dynamics == PhaseDynamics.ONE_PER_NODE:
        return

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_with_forces.bioMod",
        ode_solver=ode_solver,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
        phase_dynamics=phase_dynamics,
        use_point_of_applications=use_point_of_applications,
        n_threads=n_threads,
        use_sx=use_sx,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    if not use_point_of_applications:
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 7067.851604540213)

        # Check constraints
        g = np.array(sol.constraints)
        npt.assert_equal(g.shape, (246, 1))
        npt.assert_almost_equal(g, np.zeros((246, 1)))

        # Check some of the results
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        q, qdot, tau = states["q"], states["qdot"], controls["tau"]

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([2.0377671e-09, 6.9841937e00, 4.3690494e-19, 0]))
        npt.assert_almost_equal(tau[:, 10], np.array([-8.2313903e-10, 6.2433705e00, 1.5403878e-17, 0]))
        npt.assert_almost_equal(tau[:, 20], np.array([-6.7256342e-10, 5.5025474e00, 1.3602434e-17, 0]))
        npt.assert_almost_equal(tau[:, -1], np.array([2.0377715e-09, 4.8358065e00, 3.7533411e-19, 0]))

        if isinstance(ode_solver, OdeSolver.IRK):
            # initial and final position
            npt.assert_almost_equal(q[:, 0], np.array((0, 0, 0, 0)), decimal=5)
            npt.assert_almost_equal(q[:, -1], np.array((0, 2, 0, 0)), decimal=5)

            # initial and final velocities
            npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)), decimal=5)
            npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)), decimal=5)

            # detailed cost values
            if n_threads == 1:
                npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 7067.851604540213)
        else:
            # initial and final position
            npt.assert_almost_equal(q[:, 0], np.array([-4.6916756e-15, 6.9977394e-16, -1.6087563e-06, 0]), decimal=5)
            npt.assert_almost_equal(q[:, -1], np.array([-4.6917018e-15, 2.0000000e00, 1.6091612e-06, 0]), decimal=5)

            # initial and final velocities
            npt.assert_almost_equal(qdot[:, 0], np.array([0, 0, 1.60839825e-06, 0]), decimal=5)
            npt.assert_almost_equal(qdot[:, -1], np.array([0, 0, 1.6094277e-06, 0]), decimal=5)

            # detailed cost values
            if n_threads == 1:
                npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 7067.851604540213)

    else:
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 7073.702785927464)

        # Check constraints
        g = np.array(sol.constraints)
        npt.assert_equal(g.shape, (246, 1))
        npt.assert_almost_equal(g, np.zeros((246, 1)))

        # Check some of the results
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        q, qdot, tau = states["q"], states["qdot"], controls["tau"]

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([-0.18786284, 6.98419368, -0.15203139, 0.0]))
        npt.assert_almost_equal(tau[:, 10], np.array([0.06658482, 6.24337052, -0.15203139, 0.0]))
        npt.assert_almost_equal(tau[:, 20], np.array([0.04534891, 5.50254736, -0.15203139, 0.0]))
        npt.assert_almost_equal(tau[:, -1], np.array([-0.14707919, 4.83580652, -0.15203139, 0.0]))

        # initial and final position
        npt.assert_almost_equal(
            q[:, 0], np.array([-3.45394141e-15, 6.99773966e-16, -3.49050491e-02, 0.00000000e00]), decimal=5
        )
        npt.assert_almost_equal(
            q[:, -1], np.array([-3.94794954e-15, 2.00000000e00, 2.22536671e-02, 0.00000000e00]), decimal=5
        )

        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=5)
        npt.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=5)

        # detailed cost values
        if n_threads == 1:
            npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 7073.70278592746)

    # simulate
    TestUtils.simulate(sol)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)

    tracemalloc.stop()
    global test_memory
    test_memory[f"external_forces-{ode_solver}-{use_sx}-{n_threads}-{phase_dynamics}-{use_point_of_applications}"] = [
        building_duration,
        mem_used,
        solving_duration,
    ]


@pytest.mark.parametrize("ode_solver_type", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK, OdeSolver.COLLOCATION])
def test_example_multiphase(ode_solver_type):
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver = ode_solver_type()
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=ode_solver_type != OdeSolver.IRK,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 106088.01707867868)

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver.is_direct_collocation:
        npt.assert_equal(g.shape, (2124, 1))
        npt.assert_almost_equal(g, np.zeros((2124, 1)))
    else:
        npt.assert_equal(g.shape, (444, 1))
        npt.assert_almost_equal(g, np.zeros((444, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # initial and final position
    npt.assert_almost_equal(states[0]["q"][:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(states[0]["q"][:, -1], np.array((2, 0, 0.0078695)))
    npt.assert_almost_equal(states[1]["q"][:, 0], np.array((2, 0, 0.0078695)))
    npt.assert_almost_equal(states[1]["q"][:, -1], np.array((1, 0, 0)))
    npt.assert_almost_equal(states[2]["q"][:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(states[2]["q"][:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    npt.assert_almost_equal(states[0]["qdot"][:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(states[0]["qdot"][:, -1], np.array((0, 0, 0)))
    npt.assert_almost_equal(states[1]["qdot"][:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(states[1]["qdot"][:, -1], np.array((0, 0, 0)))
    npt.assert_almost_equal(states[2]["qdot"][:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(states[2]["qdot"][:, -1], np.array((0, 0, 0)))

    # initial and final controls
    npt.assert_almost_equal(controls[0]["tau"][:, 0], np.array((1.42857142, 9.81, 0.01124212)))
    npt.assert_almost_equal(controls[0]["tau"][:, -1], np.array((-1.42857144, 9.81, -0.01124212)))
    npt.assert_almost_equal(controls[1]["tau"][:, 0], np.array((-0.22788183, 9.81, 0.01775688)))
    npt.assert_almost_equal(controls[1]["tau"][:, -1], np.array((0.2957136, 9.81, 0.285805)))
    npt.assert_almost_equal(controls[2]["tau"][:, 0], np.array((0.3078264, 9.81, 0.34001243)))
    npt.assert_almost_equal(controls[2]["tau"][:, -1], np.array((-0.36233407, 9.81, -0.58394606)))

    # simulate
    TestUtils.simulate(sol)

    # Test warm start
    if ode_solver_type == OdeSolver.COLLOCATION:
        # We don't have test value for this one
        return

    TestUtils.assert_warm_start(ocp, sol)

    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 19397.605252449728)
    npt.assert_almost_equal(sol.detailed_cost[1]["cost_value_weighted"], 0.30851703399819436)
    npt.assert_almost_equal(sol.detailed_cost[2]["cost_value_weighted"], 48129.27750487157)
    npt.assert_almost_equal(sol.detailed_cost[3]["cost_value_weighted"], 38560.82580432337)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"multiphase-{ode_solver}-{phase_dynamics}"] = [building_duration, solving_duration, mem_used]


@pytest.mark.parametrize("expand_dynamics", [True, False])
@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.IRK])
def test_contact_forces_inequality_greater_than_constraint(ode_solver, phase_dynamics, expand_dynamics):
    from bioptim.examples.getting_started import example_inequality_constraint as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    min_bound = 50

    if not expand_dynamics and ode_solver != OdeSolver.IRK:
        # There is no point testing that
        return
    if expand_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and ode_solver == OdeSolver.IRK:
        with pytest.raises(RuntimeError):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
                phase_time=0.1,
                n_shooting=10,
                min_bound=min_bound,
                max_bound=np.inf,
                mu=0.2,
                ode_solver=ode_solver(),
                phase_dynamics=phase_dynamics,
                expand_dynamics=expand_dynamics,
            )
        return

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.1,
        n_shooting=10,
        min_bound=min_bound,
        max_bound=np.inf,
        mu=0.2,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=expand_dynamics,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 0.19216241950659246)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (120, 1))
    npt.assert_almost_equal(g[:80], np.zeros((80, 1)))
    npt.assert_array_less(-g[80:100], -min_bound)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    npt.assert_almost_equal(q[:, -1], np.array((-0.027221, 0.02358599, -0.67794882, 0.67794882)))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((-0.53979971, 0.43468705, 1.38612634, -1.38612634)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((-33.50557304)))
    npt.assert_almost_equal(tau[:, -1], np.array((-29.43209257)))

    # simulate
    TestUtils.simulate(sol)

    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.19216241950659244)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"contact_forces_greater_than_constraint-{ode_solver}-{phase_dynamics}-{expand_dynamics}"] = [
        building_duration,
        solving_duration,
        mem_used,
    ]


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.IRK])
def test_contact_forces_inequality_lesser_than_constraint(ode_solver):
    from bioptim.examples.getting_started import example_inequality_constraint as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    max_bound = 75
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2contacts.bioMod",
        phase_time=0.1,
        n_shooting=10,
        min_bound=-np.inf,
        max_bound=max_bound,
        mu=0.2,
        ode_solver=ode_solver(),
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 0.2005516965424669)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (120, 1))
    npt.assert_almost_equal(g[:80], np.zeros((80, 1)))
    npt.assert_array_less(g[80:100], max_bound)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    npt.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    npt.assert_almost_equal(q[:, -1], np.array((-0.00902682, 0.00820596, -0.72560094, 0.72560094)))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((-0.18616011, 0.16512913, 0.49768751, -0.49768751)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((-24.36593641)))
    npt.assert_almost_equal(tau[:, -1], np.array((-24.36125297)))

    # simulate
    TestUtils.simulate(sol)

    # detailed cost values
    npt.assert_almost_equal(sol.detailed_cost[0]["cost_value_weighted"], 0.2005516965424669)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8])  # use_SX and IRK are not compatible
def test_multinode_objective(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import example_multinode_objective as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver = ode_solver()

    n_shooting = 20
    if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE:
        with pytest.raises(
            RuntimeError,
        ):
            ocp = ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                n_shooting=n_shooting,
                final_time=1,
                ode_solver=ode_solver,
                phase_dynamics=phase_dynamics,
                expand_dynamics=True,
            )
        return

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=n_shooting,
        final_time=1,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()
    sol.print_cost()

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # initial and final position
    npt.assert_almost_equal(states["q"][:, 0], np.array([0.0, 0.0]))
    npt.assert_almost_equal(states["q"][:, -1], np.array([0.0, 3.14]))
    # initial and final velocities
    npt.assert_almost_equal(states["qdot"][:, 0], np.array([0.0, 0.0]))
    npt.assert_almost_equal(states["qdot"][:, -1], np.array([0.0, 0.0]))

    if isinstance(ode_solver, OdeSolver.RK4):
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 488.05375155958615)

        # Check constraints
        g = np.array(sol.constraints)
        npt.assert_equal(g.shape, (80, 1))
        npt.assert_almost_equal(g, np.zeros((80, 1)))

        # initial and final controls
        npt.assert_almost_equal(controls["tau"][:, 0], np.array([6.49295131, 0.0]))
        npt.assert_almost_equal(controls["tau"][:, -1], np.array([-14.26800861, 0.0]))

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 475.44403901331214)

        # Check constraints
        g = np.array(sol.constraints)
        npt.assert_equal(g.shape, (80, 1))
        npt.assert_almost_equal(g, np.zeros((80, 1)))

        # initial and final controls
        npt.assert_almost_equal(controls["tau"][:, 0], np.array([5.84195684, 0.0]))
        npt.assert_almost_equal(controls["tau"][:, -1], np.array([-13.1269555, 0.0]))

    # Check that the output is what we expect
    weight = 10
    target = []
    fun = ocp.nlp[0].J_internal[0].weighted_function
    dt = sol.t_span()[0][-1]
    t_out = []
    x_out = np.ndarray((0, 1))
    u_out = np.ndarray((0, 1))
    p_out = []
    a_out = []
    d_out = []
    for i in range(n_shooting):
        x_out = np.vstack((x_out, np.concatenate([states[key][:, i] for key in states.keys()])[:, np.newaxis]))
        if i == n_shooting:
            u_out = np.vstack((u_out, []))
        else:
            u_out = np.vstack((u_out, np.concatenate([controls[key][:, i] for key in controls.keys()])[:, np.newaxis]))

    # Note that dt=1, because the multi-node objectives are treated as mayer terms
    out = fun[0](t_out, dt, x_out, u_out, p_out, a_out, d_out, weight, target)
    out_expected = sum2(sum1(controls["tau"] ** 2)) * dt * weight
    npt.assert_almost_equal(out, out_expected)

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"multinode_objective-{ode_solver}-{phase_dynamics}"] = [building_duration, solving_duration, mem_used]


@pytest.mark.parametrize("node", [*Node, 0, 3])
def test_multinode_constraints_wrong_nodes(node):
    multinode_constraints = MultinodeConstraintList()

    if node in (Node.START, Node.MID, Node.PENULTIMATE, Node.END) or isinstance(node, int):
        multinode_constraints.add(
            MultinodeConstraintFcn.STATES_EQUALITY,
            nodes_phase=(0, 0),
            nodes=(Node.START, node),
            sub_nodes=(0, 0),
            key="all",
        )
        with pytest.raises(ValueError, match=re.escape("Each of the nodes must have a corresponding nodes_phase")):
            multinode_constraints.add(
                MultinodeConstraintFcn.STATES_EQUALITY,
                nodes_phase=(0,),
                nodes=(Node.START, node),
                sub_nodes=(0, 0),
                key="all",
            )
    else:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Multinode penalties only works with Node.START, Node.MID, Node.PENULTIMATE, "
                "Node.END or a node index (int)."
            ),
        ):
            multinode_constraints.add(
                MultinodeConstraintFcn.STATES_EQUALITY,
                nodes_phase=(0, 0),
                nodes=(Node.START, node),
                sub_nodes=(0, 0),
                key="all",
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.IRK])
def test_multinode_constraints_too_much_constraints(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import example_multinode_constraints as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_obj = ode_solver
    ode_solver = ode_solver()
    if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE:
        with pytest.raises(
            RuntimeError, match="Multinode penalties cannot be used with PhaseDynamics.SHARED_DURING_THE_PHASE"
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
                n_shootings=(8, 8, 8),
                ode_solver=ode_solver,
                phase_dynamics=phase_dynamics,
                expand_dynamics=ode_solver_obj != OdeSolver.IRK,
            )
    else:
        ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
            n_shootings=(8, 8, 8),
            ode_solver=ode_solver,
            phase_dynamics=phase_dynamics,
            expand_dynamics=ode_solver_obj != OdeSolver.IRK,
        )


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_multinode_constraints(ode_solver):
    from bioptim.examples.getting_started import example_multinode_constraints as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shootings=(8, 10, 8),
        ode_solver=ode_solver,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()
    sol.print_cost()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 106577.60874445777)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (187, 1))
    npt.assert_almost_equal(g, np.zeros((187, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # initial and final position
    npt.assert_almost_equal(states[0]["q"][:, 0], np.array([1.0, 0.0, 0.0]))
    npt.assert_almost_equal(states[-1]["q"][:, -1], np.array([2.0, 0.0, 1.57]))
    # initial and final velocities
    npt.assert_almost_equal(states[0]["qdot"][:, 0], np.array([0.0, 0.0, 0.0]))
    npt.assert_almost_equal(states[-1]["qdot"][:, -1], np.array([0.0, 0.0, 0.0]))

    # equality Node.START phase 0 and 2
    npt.assert_almost_equal(states[0]["q"][:, 0], states[2]["q"][:, 0])

    # initial and final controls
    npt.assert_almost_equal(controls[0]["tau"][:, 0], np.array([1.32977862, 9.81, 0.0]))
    npt.assert_almost_equal(controls[-1]["tau"][:, -1], np.array([-1.2, 9.81, -1.884]))

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"multinode_constraints-{ode_solver}-{phase_dynamics}"] = [
        building_duration,
        solving_duration,
        mem_used,
    ]


def test_multistart():
    from bioptim.examples.getting_started import example_multistart as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)
    bio_model_path = [bioptim_folder + "/models/pendulum.bioMod"]
    final_time = [1]
    n_shooting = [5, 10]
    seed = [2, 1]
    combinatorial_parameters = {
        "bio_model_path": bio_model_path,
        "final_time": final_time,
        "n_shooting": n_shooting,
        "seed": seed,
    }
    save_folder = "./Solutions_test_folder"
    multi_start = ocp_module.prepare_multi_start(
        combinatorial_parameters=combinatorial_parameters,
        save_folder=save_folder,
    )
    tak = time.time()
    multi_start.solve()
    tok = time.time()

    with open(f"{save_folder}/pendulum_multi_start_random_states_5_2.pkl", "rb") as file:
        multi_start_0 = pickle.load(file)
    with open(f"{save_folder}/pendulum_multi_start_random_states_5_1.pkl", "rb") as file:
        multi_start_1 = pickle.load(file)
    with open(f"{save_folder}/pendulum_multi_start_random_states_10_2.pkl", "rb") as file:
        multi_start_2 = pickle.load(file)
    with open(f"{save_folder}/pendulum_multi_start_random_states_10_1.pkl", "rb") as file:
        multi_start_3 = pickle.load(file)

    # Delete the solutions
    shutil.rmtree(f"{save_folder}")

    npt.assert_almost_equal(
        np.concatenate((multi_start_0["q"], multi_start_0["qdot"])),
        np.array(
            [
                [0.0, -0.9, 0.29797487, -0.38806564, -0.47779319, 0.0],
                [0.0, 1.49880317, -2.51761362, -2.93013488, 1.52221264, 3.14],
                [0.0, 0.85313852, -19.827228, 17.92813608, 22.24092358, 0.0],
                [0.0, -26.41165363, 0.32962156, -27.31385448, -4.51620735, 0.0],
            ]
        ),
    )

    npt.assert_almost_equal(
        np.concatenate((multi_start_1["q"], multi_start_1["qdot"])),
        np.array(
            [
                [0.0, 1.32194696, -0.9, -0.9, -0.9, 0.0],
                [0.0, -1.94074114, -1.29725818, 0.48778547, -1.01543168, 3.14],
                [0.0, 23.75781921, -29.6951133, 10.71078955, -5.19589251, 0.0],
                [0.0, -18.96884288, 18.89633855, 29.42174252, -11.72290462, 0.0],
            ]
        ),
    )

    npt.assert_almost_equal(
        np.concatenate((multi_start_2["q"], multi_start_2["qdot"])),
        np.array(
            [
                [
                    0.00000000e00,
                    -9.00000000e-01,
                    2.97974867e-01,
                    -3.88065644e-01,
                    -4.77793187e-01,
                    -9.00000000e-01,
                    -9.00000000e-01,
                    7.15625798e-01,
                    -9.00000000e-01,
                    -9.00000000e-01,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    -4.59200384e00,
                    1.70627704e-01,
                    -3.96544560e00,
                    3.58562722e00,
                    4.44818472e00,
                    -7.24220374e-02,
                    4.35502007e00,
                    -5.28233073e00,
                    6.59243127e-02,
                    3.14000000e00,
                ],
                [
                    0.00000000e00,
                    -2.53507102e01,
                    -2.34262299e01,
                    6.07868704e00,
                    -1.72151737e01,
                    -2.46963310e01,
                    -1.75736793e01,
                    -9.43569280e00,
                    -2.02397204e00,
                    -1.87400258e01,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    3.29032823e-01,
                    -7.10674433e00,
                    1.84497854e01,
                    5.02681081e00,
                    -2.12184048e01,
                    1.26136419e01,
                    2.91886052e01,
                    5.25347819e-04,
                    2.44742674e01,
                    0.00000000e00,
                ],
            ]
        ),
    )

    npt.assert_almost_equal(
        np.concatenate((multi_start_3["q"], multi_start_3["qdot"])),
        np.array(
            [
                [0.0, 1.32194696, -0.9, -0.9, -0.9, -0.9, -0.9, -0.92663564, -0.61939515, 0.2329004, 0.0],
                [
                    0.0,
                    -3.71396256,
                    4.75156384,
                    -5.93902266,
                    2.14215791,
                    -1.0391785,
                    0.73751814,
                    -4.51903101,
                    -3.79376858,
                    3.77926771,
                    3.14,
                ],
                [
                    0.0,
                    12.08398633,
                    23.64922791,
                    24.7938679,
                    -26.07244114,
                    -28.96204213,
                    -20.74516657,
                    23.75939422,
                    -25.23661272,
                    -4.95695411,
                    0.0,
                ],
                [
                    0.0,
                    12.05599463,
                    -11.59149477,
                    11.71819889,
                    21.02515105,
                    -30.26684018,
                    15.71703084,
                    30.71604811,
                    15.59270793,
                    -13.79511083,
                    0.0,
                ],
            ]
        ),
    )

    combinatorial_parameters = {
        "bio_model_path": bio_model_path,
        "final_time": final_time,
        "n_shooting": n_shooting,
        "seed": seed,
    }
    with pytest.raises(ValueError, match="save_folder must be an str"):
        ocp_module.prepare_multi_start(
            combinatorial_parameters=combinatorial_parameters,
            save_folder=5,
        )

    with pytest.raises(ValueError, match="combinatorial_parameters must be a dictionary"):
        ocp_module.prepare_multi_start(
            combinatorial_parameters=[combinatorial_parameters],
            save_folder=save_folder,
        )
    # Delete the solutions
    shutil.rmtree(f"{save_folder}")

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"multistart"] = [building_duration, solving_duration, mem_used]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_example_variable_scaling(phase_dynamics):
    from bioptim.examples.getting_started import example_variable_scaling as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1 / 10,
        n_shooting=30,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    tak = time.time()
    sol = ocp.solve()
    tok = time.time()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 31609.83406760166)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (120, 1))
    npt.assert_almost_equal(g, np.zeros((120, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([0.0, 0.0]))
    npt.assert_almost_equal(q[:, -1], np.array([0.0, 3.14]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
    npt.assert_almost_equal(qdot[:, -1], np.array([0.0, 0.0]))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([-1000.00000999, 0.0]))
    npt.assert_almost_equal(tau[:, -1], np.array([-1000.00000999, 0.0]))

    # Execution times
    building_duration = tak - tik
    solving_duration = tok - tak

    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    mem_after = tracemalloc.take_snapshot()
    top_stats = mem_after.compare_to(mem_before, "lineno")
    mem_used = sum(stat.size_diff for stat in top_stats)
    tracemalloc.stop()

    global test_memory
    test_memory[f"variable_scaling-{phase_dynamics}"] = [building_duration, solving_duration, mem_used]


def test_memory_and_execution_time():

    if platform.system() == "Windows":
        pytest.skip("Tests are slower on Windows")

    ref = {
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK1'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.1513352394104,
            3.5536017417907715,
            598816,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK1'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.185523986816406,
            3.9750709533691406,
            559443,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK1'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.862371921539307,
            1.7231292724609375,
            497764,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK1'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.8641357421875,
            1.9456541538238525,
            536214,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK2'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.462843894958496,
            3.889923095703125,
            486645,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK2'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.252295017242432,
            4.149457931518555,
            553802,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK2'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.931426048278809,
            1.9224112033843994,
            485197,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK2'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.94077205657959,
            2.180896043777466,
            538165,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.500595808029175,
            4.094981908798218,
            456958,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-False-1-PhaseDynamics.ONE_PER_NODE": [
            24.683043956756592,
            4.163797855377197,
            883682,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.399058818817139,
            4.211014032363892,
            528664,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.973743915557861,
            2.2713370323181152,
            476722,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-True-1-PhaseDynamics.ONE_PER_NODE": [
            13.790527820587158,
            2.2725698947906494,
            877330,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.054003000259399,
            2.4681458473205566,
            520273,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK8'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.498960971832275,
            3.521271228790283,
            517774,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK8'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.61425518989563,
            3.708000659942627,
            540117,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.551395893096924,
            4.164907932281494,
            526344,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.98697304725647,
            4.516143798828125,
            572859,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.920191049575806,
            4.39845085144043,
            506707,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-False-1-PhaseDynamics.ONE_PER_NODE": [
            22.197954177856445,
            4.479645013809204,
            1140193,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.074331045150757,
            4.760502099990845,
            598489,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.3521199226379395,
            2.3463258743286133,
            559429,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-True-1-PhaseDynamics.ONE_PER_NODE": [
            12.132819890975952,
            2.38869309425354,
            1123733,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.332345724105835,
            2.570103168487549,
            567673,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.TRAPEZOIDAL'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.286965131759644,
            3.561800956726074,
            543829,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.TRAPEZOIDAL'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.503605127334595,
            4.053746938705444,
            581438,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.TRAPEZOIDAL'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.966175079345703,
            1.6531569957733154,
            507530,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.TRAPEZOIDAL'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.999427795410156,
            1.8715283870697021,
            556466,
        ],
        "custom_constraint_track_markers-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.820377111434937,
            3.551621913909912,
            581783,
        ],
        "custom_constraint_track_markers-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [
            26.142763137817383,
            3.560593843460083,
            882478,
        ],
        "custom_constraint_track_markers-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.959012985229492,
            3.8739609718322754,
            542643,
        ],
        "custom_constraint_track_markers-RK8 5 steps-PhaseDynamics.ONE_PER_NODE": [
            62.250470876693726,
            3.8979461193084717,
            890960,
        ],
        "custom_constraint_track_markers-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.014273881912231,
            3.500044107437134,
            547007,
        ],
        "custom_constraint_track_markers-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [
            27.963459014892578,
            3.5615458488464355,
            1125462,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.528355121612549,
            0.8745269775390625,
            316498,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT-True-PhaseDynamics.ONE_PER_NODE": [
            5.227646112442017,
            0.8703899383544922,
            441850,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5004818439483643,
            0.8465471267700195,
            311482,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT-False-PhaseDynamics.ONE_PER_NODE": [
            5.195713043212891,
            0.8620669841766357,
            439971,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5244569778442383,
            0.867851972579956,
            294191,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT-True-PhaseDynamics.ONE_PER_NODE": [
            5.169760704040527,
            0.8655142784118652,
            446137,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.500936269760132,
            0.8467919826507568,
            306315,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT-False-PhaseDynamics.ONE_PER_NODE": [
            5.213915824890137,
            0.878535270690918,
            445024,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.LINEAR-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.6062657833099365,
            0.888721227645874,
            313048,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.LINEAR-True-PhaseDynamics.ONE_PER_NODE": [
            5.187885046005249,
            0.8783941268920898,
            446858,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.LINEAR-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5217301845550537,
            0.8631987571716309,
            315358,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.LINEAR-False-PhaseDynamics.ONE_PER_NODE": [
            5.1612937450408936,
            0.8431801795959473,
            444895,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.EACH_FRAME-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.4971258640289307,
            0.8691582679748535,
            310477,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.EACH_FRAME-True-PhaseDynamics.ONE_PER_NODE": [
            5.190582990646362,
            0.8718810081481934,
            443745,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.EACH_FRAME-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5074141025543213,
            0.8524069786071777,
            309865,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.EACH_FRAME-False-PhaseDynamics.ONE_PER_NODE": [
            5.156826734542847,
            0.8647990226745605,
            445298,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.ALL_POINTS-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5335800647735596,
            1.0024340152740479,
            311591,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.ALL_POINTS-True-PhaseDynamics.ONE_PER_NODE": [
            5.205708742141724,
            1.0020380020141602,
            447697,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.ALL_POINTS-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5077292919158936,
            0.9837539196014404,
            315213,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.ALL_POINTS-False-PhaseDynamics.ONE_PER_NODE": [
            5.319723844528198,
            1.0774710178375244,
            445434,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.SPLINE-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5698020458221436,
            0.8854770660400391,
            306577,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.SPLINE-True-PhaseDynamics.ONE_PER_NODE": [
            5.157807111740112,
            0.8733980655670166,
            448234,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.SPLINE-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5160067081451416,
            0.9336130619049072,
            309463,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.SPLINE-False-PhaseDynamics.ONE_PER_NODE": [
            5.193099021911621,
            0.909325122833252,
            447463,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CUSTOM-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.368273973464966,
            0.8829538822174072,
            314000,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CUSTOM-True-PhaseDynamics.ONE_PER_NODE": [
            4.501805067062378,
            0.8706808090209961,
            443079,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CUSTOM-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.351344108581543,
            0.8531820774078369,
            312841,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CUSTOM-False-PhaseDynamics.ONE_PER_NODE": [
            4.756149053573608,
            0.882012128829956,
            441641,
        ],
        "cyclic_objective-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            4.184632062911987,
            1.5201780796051025,
            405713,
        ],
        "cyclic_objective-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [9.308022737503052, 1.5241172313690186, 510014],
        "cyclic_objective-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.524325132369995,
            1.6116359233856201,
            402263,
        ],
        "cyclic_objective-RK8 5 steps-PhaseDynamics.ONE_PER_NODE": [20.720319986343384, 1.6008939743041992, 512195],
        "cyclic_objective-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            4.204087018966675,
            1.5054340362548828,
            355717,
        ],
        "cyclic_objective-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [9.792128086090088, 1.5030150413513184, 538596],
        "cyclic_constraint-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            4.1645917892456055,
            1.4973511695861816,
            400954,
        ],
        "cyclic_constraint-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [9.404989957809448, 1.5142982006072998, 512543],
        "cyclic_constraint-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.3193039894104,
            1.5833299160003662,
            404135,
        ],
        "cyclic_constraint-RK8 5 steps-PhaseDynamics.ONE_PER_NODE": [20.810055017471313, 1.5870802402496338, 513597],
        "cyclic_constraint-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            4.216747999191284,
            1.8143420219421387,
            356431,
        ],
        "cyclic_constraint-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [10.667237997055054, 1.446087121963501, 540792],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            28.063942193984985,
            10.764989852905273,
            1285552,
        ],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-PhaseDynamics.ONE_PER_NODE": [
            71.56481981277466,
            10.783386945724487,
            2212663,
        ],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK8'>-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            33.94674825668335,
            12.605023860931396,
            1282232,
        ],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            27.74601697921753,
            10.12258791923523,
            1319276,
        ],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.ONE_PER_NODE": [
            73.12699604034424,
            10.62309718132019,
            2840879,
        ],
        "custom_dynamics-RK4 5 steps-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.345491886138916,
            3.575360059738159,
            579659,
        ],
        "custom_dynamics-RK4 5 steps-True-PhaseDynamics.ONE_PER_NODE": [23.48348593711853, 3.407442092895508, 812917],
        "custom_dynamics-RK4 5 steps-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.43447208404541,
            3.3900868892669678,
            540327,
        ],
        "custom_dynamics-RK4 5 steps-False-PhaseDynamics.ONE_PER_NODE": [
            25.146296739578247,
            3.4173080921173096,
            883591,
        ],
        "custom_dynamics-RK8 5 steps-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.66609525680542,
            3.62235689163208,
            590736,
        ],
        "custom_dynamics-RK8 5 steps-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.773294925689697,
            3.694208860397339,
            562779,
        ],
        "custom_dynamics-IRK legendre 4-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.442317724227905,
            3.5097222328186035,
            558782,
        ],
        "custom_dynamics-IRK legendre 4-True-PhaseDynamics.ONE_PER_NODE": [
            22.204365968704224,
            3.4947710037231445,
            923398,
        ],
        "custom_dynamics-IRK legendre 4-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            9.541068077087402,
            3.5327117443084717,
            548155,
        ],
        "custom_dynamics-IRK legendre 4-False-PhaseDynamics.ONE_PER_NODE": [
            26.964985847473145,
            3.5070571899414062,
            1147125,
        ],
        "external_forces-RK4 5 steps-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            6.517794132232666,
            2.3037779331207275,
            535800,
        ],
        "external_forces-RK8 5 steps-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            7.210792779922485,
            3.966912031173706,
            526336,
        ],
        "external_forces-RK4 5 steps-True-1-PhaseDynamics.ONE_PER_NODE-True": [
            15.120752811431885,
            2.271672010421753,
            914086,
        ],
        "external_forces-RK8 5 steps-True-1-PhaseDynamics.ONE_PER_NODE-True": [
            31.632735013961792,
            4.178122043609619,
            877876,
        ],
        "external_forces-RK4 5 steps-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            6.381742000579834,
            2.4875428676605225,
            615533,
        ],
        "external_forces-RK8 5 steps-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            6.973922967910767,
            4.05414080619812,
            568032,
        ],
        "external_forces-RK4 5 steps-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            10.248576879501343,
            3.7544772624969482,
            578738,
        ],
        "external_forces-RK8 5 steps-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            11.717979907989502,
            4.608597993850708,
            539600,
        ],
        "external_forces-IRK legendre 4-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            10.631656885147095,
            3.5057692527770996,
            533250,
        ],
        "external_forces-RK4 5 steps-False-1-PhaseDynamics.ONE_PER_NODE-True": [
            30.453665018081665,
            3.8437700271606445,
            880297,
        ],
        "external_forces-RK8 5 steps-False-1-PhaseDynamics.ONE_PER_NODE-True": [
            70.94570708274841,
            4.963971138000488,
            878600,
        ],
        "external_forces-IRK legendre 4-False-1-PhaseDynamics.ONE_PER_NODE-True": [
            33.11180520057678,
            3.9515271186828613,
            1139486,
        ],
        "external_forces-RK4 5 steps-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            10.23096513748169,
            3.809520959854126,
            577617,
        ],
        "external_forces-RK8 5 steps-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            11.896209955215454,
            3.952944755554199,
            571259,
        ],
        "external_forces-IRK legendre 4-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            10.347720861434937,
            3.8076350688934326,
            586735,
        ],
        "external_forces-RK4 5 steps-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            6.325876951217651,
            2.2698328495025635,
            569544,
        ],
        "external_forces-RK8 5 steps-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            6.936321020126343,
            3.838604688644409,
            532452,
        ],
        "external_forces-RK4 5 steps-True-1-PhaseDynamics.ONE_PER_NODE-False": [
            15.31987476348877,
            2.2700250148773193,
            903241,
        ],
        "external_forces-RK8 5 steps-True-1-PhaseDynamics.ONE_PER_NODE-False": [
            31.632611751556396,
            3.9104461669921875,
            877081,
        ],
        "external_forces-RK4 5 steps-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            6.359724998474121,
            2.4342191219329834,
            605774,
        ],
        "external_forces-RK8 5 steps-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            7.021723985671997,
            4.2617199420928955,
            565005,
        ],
        "external_forces-RK4 5 steps-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            10.2382652759552,
            3.723505735397339,
            577926,
        ],
        "external_forces-RK8 5 steps-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            11.715183019638062,
            4.596646070480347,
            531119,
        ],
        "external_forces-IRK legendre 4-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            10.297888994216919,
            3.495901107788086,
            551079,
        ],
        "external_forces-RK4 5 steps-False-1-PhaseDynamics.ONE_PER_NODE-False": [
            30.71595001220703,
            3.7435030937194824,
            881598,
        ],
        "external_forces-RK8 5 steps-False-1-PhaseDynamics.ONE_PER_NODE-False": [
            70.45185399055481,
            4.641111850738525,
            877991,
        ],
        "external_forces-IRK legendre 4-False-1-PhaseDynamics.ONE_PER_NODE-False": [
            33.052973985672,
            3.9504141807556152,
            1146288,
        ],
        "external_forces-RK4 5 steps-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            10.27266001701355,
            3.804931163787842,
            573289,
        ],
        "external_forces-RK8 5 steps-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            11.604282140731812,
            3.9582459926605225,
            580018,
        ],
        "external_forces-IRK legendre 4-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            10.62734317779541,
            3.843648910522461,
            588680,
        ],
        "multiphase-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [23.17929220199585, 8.365160942077637, 1115335],
        "multiphase-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [67.69533586502075, 8.916538000106812, 1919312],
        "multiphase-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [26.772330045700073, 9.42667007446289, 1129505],
        "multiphase-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            23.982249975204468,
            8.310612916946411,
            1140644,
        ],
        "multiphase-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [62.950451135635376, 8.694111824035645, 2463043],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            6.883507966995239,
            4.195178031921387,
            247754,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-PhaseDynamics.ONE_PER_NODE-True": [
            11.95946192741394,
            4.369907855987549,
            501848,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            6.821503162384033,
            4.1019127368927,
            406984,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            6.819267272949219,
            4.061187982559204,
            373534,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.ONE_PER_NODE-True": [
            12.595759868621826,
            5.469526052474976,
            562772,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.ONE_PER_NODE-False": [
            12.781190872192383,
            5.58987021446228,
            561468,
        ],
        "multinode_objective-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [7.24596095085144, 1.3100521564483643, 556603],
        "multinode_objective-RK8 5 steps-PhaseDynamics.ONE_PER_NODE": [16.051567792892456, 1.9029803276062012, 539030],
        "multinode_constraints-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.927126169204712,
            3.9404168128967285,
            678951,
        ],
        "multinode_constraints-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [
            23.453507900238037,
            3.9539811611175537,
            981493,
        ],
        "multinode_constraints-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            14.419713020324707,
            4.253767967224121,
            674833,
        ],
        "multinode_constraints-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.90890908241272,
            3.7429709434509277,
            690377,
        ],
        "multinode_constraints-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [
            24.7413969039917,
            3.9145472049713135,
            1225773,
        ],
        "multistart": [0.002456188201904297, 16.426876068115234, 380975],
        "variable_scaling-PhaseDynamics.SHARED_DURING_THE_PHASE": [5.885438919067383, 2.3393502235412598, 546399],
        "variable_scaling-PhaseDynamics.ONE_PER_NODE": [12.253346920013428, 2.323004961013794, 894848],
    }

    for key in ref.keys():
        npt.assert_array_less(test_memory[key][0], ref[key][0] * 3)
        npt.assert_array_less(test_memory[key][1], ref[key][1] * 3)
        npt.assert_array_less(test_memory[key][2], ref[key][2] * 3)
