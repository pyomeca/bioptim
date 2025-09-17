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
import os

from bioptim import (
    InterpolationType,
    OdeSolver,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    Node,
    ControlType,
    PhaseDynamics,
    SolutionMerge,
    DefectType,
    OrderingStrategy,
    Solver,
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
        OdeSolver.RK4,
        OdeSolver.RK8,
        OdeSolver.IRK,
        OdeSolver.COLLOCATION,
        OdeSolver.TRAPEZOIDAL,
    ],
)
@pytest.mark.parametrize(
    "defects_type", [DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS, DefectType.TAU_EQUALS_INVERSE_DYNAMICS]
)
@pytest.mark.parametrize("solver", [Solver.IPOPT, Solver.FATROP])
def test_pendulum(ode_solver, use_sx, n_threads, phase_dynamics, defects_type, solver):
    ordering_strategy = OrderingStrategy.TIME_MAJOR
    from bioptim.examples.getting_started import basic_ocp as ocp_module

    if platform.system() == "Windows":
        pytest.skip("These tests fail on CI for Windows")
    elif platform.system() == "Darwin" and (
        defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS or solver == Solver.FATROP
    ):
        pytest.skip("These tests fail on CI for MacOS")

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    # For reducing time phase_dynamics=PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if n_threads > 1 and phase_dynamics == PhaseDynamics.ONE_PER_NODE:
        pytest.skip("PhaseDynamics.ONE_PER_NODE is only tested with RK4 and COLLOCATION to reduce time")
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver not in (OdeSolver.RK4, OdeSolver.COLLOCATION):
        pytest.skip("PhaseDynamics.ONE_PER_NODE is only tested with RK4 and COLLOCATION to reduce time")
    if ode_solver == OdeSolver.RK8 and not use_sx:
        pytest.skip("OdeSolver.RK8 is only tested with use_sx=True")

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_obj = None
    if ode_solver == OdeSolver.COLLOCATION or ode_solver == OdeSolver.IRK:
        ode_solver_obj = ode_solver(defects_type=defects_type)
    else:
        if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
            ode_solver_obj = ode_solver()
        else:
            pytest.skip("There is no defect type for non COLLOCATION OdeSolvers.")

    if isinstance(ode_solver_obj, OdeSolver.CVODES):
        with pytest.raises(
            NotImplementedError,
            match=f"CVODES is not yet implemented",
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
                final_time=2,
                n_shooting=10,
                n_threads=n_threads,
                use_sx=use_sx,
                ode_solver=ode_solver_obj,
                phase_dynamics=phase_dynamics,
                expand_dynamics=False,
                ordering_strategy=ordering_strategy,
            )
        return

    if isinstance(ode_solver_obj, (OdeSolver.IRK, OdeSolver.CVODES)) and use_sx:
        with pytest.raises(
            RuntimeError,
            match=f"use_sx=True and OdeSolver.{ode_solver_obj.integrator.__name__} are not yet compatible",
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
                final_time=2,
                n_shooting=10,
                n_threads=n_threads,
                use_sx=use_sx,
                ode_solver=ode_solver_obj,
                phase_dynamics=phase_dynamics,
                expand_dynamics=False,
                ordering_strategy=ordering_strategy,
            )
        return
    elif isinstance(ode_solver_obj, OdeSolver.CVODES):
        with pytest.raises(
            RuntimeError,
            match=f"CVODES cannot be used with dynamics that depends on time",
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
                final_time=2,
                n_shooting=10,
                n_threads=n_threads,
                use_sx=use_sx,
                ode_solver=ode_solver_obj,
                phase_dynamics=phase_dynamics,
                expand_dynamics=False,
                ordering_strategy=ordering_strategy,
            )
        return

    if isinstance(ode_solver_obj, (OdeSolver.TRAPEZOIDAL)):
        control_type = ControlType.CONSTANT_WITH_LAST_NODE
    else:
        control_type = ControlType.CONSTANT

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=30,
        n_threads=n_threads,
        use_sx=use_sx,
        ode_solver=ode_solver_obj,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver not in (OdeSolver.IRK, OdeSolver.CVODES),
        control_type=control_type,
        ordering_strategy=ordering_strategy,
    )
    tak = time.time()  # Time after building, but before solving
    ocp.print(to_console=True, to_graph=False)

    if isinstance(ode_solver_obj, OdeSolver.CVODES):
        pytest.skip("The test is too long with CVODES")

    sol = ocp.solve(solver())
    tok = time.time()  # This after solving

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    if n_threads > 1:
        with pytest.raises(
            NotImplementedError, match="Computing detailed cost with n_threads > 1 is not implemented yet"
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
        if solver == Solver.IPOPT:
            npt.assert_almost_equal(f[0, 0], 65.8236055171619)
            # detailed cost values
            if detailed_cost is not None:
                npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 65.8236055171619)
            npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.5536468, -0.4129719])
        elif solver == Solver.FATROP:
            npt.assert_almost_equal(f[0, 0], 58.65307209221627)
            # detailed cost values
            if detailed_cost is not None:
                npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 58.65307209221627)
            npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.42226046, -0.2856726])
        else:
            raise RuntimeError("Unexpected solver")

    elif isinstance(ode_solver_obj, OdeSolver.COLLOCATION):
        if solver == Solver.IPOPT:
            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                optimal_cost = 51.03624140672109
                npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.00282762, 0.14317854])
            else:
                optimal_cost = 65.86214777650544
                npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.55457473, -0.41280843])
        elif solver == Solver.FATROP:
            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                optimal_cost = 46.66734569903588
                npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [-0.17805071, 0.32542015])
            else:
                optimal_cost = 65.86214777650544
                npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.55457473, -0.41280843])
        else:
            raise RuntimeError("Unexpected solver")

        npt.assert_almost_equal(f[0, 0], optimal_cost)
        # detailed cost values
        if detailed_cost is not None:
            npt.assert_almost_equal(detailed_cost["cost_value_weighted"], optimal_cost)

    elif isinstance(ode_solver_obj, OdeSolver.RK1):
        if solver == Solver.IPOPT:
            npt.assert_almost_equal(f[0, 0], 47.360621044913245)
            # detailed cost values
            if detailed_cost is not None:
                npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 47.360621044913245)
            npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.1463538, 0.0215651])
        elif solver == Solver.FATROP:
            npt.assert_almost_equal(f[0, 0], 48.86934816227037)
            # detailed cost values
            if detailed_cost is not None:
                npt.assert_almost_equal(detailed_cost["cost_value_weighted"], 48.86934816227037)
            npt.assert_almost_equal(sol.decision_states()["q"][15][:, 0], [0.27981801, -0.11551382])
        else:
            raise RuntimeError("Unexpected solver")

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
        if solver == Solver.IPOPT:
            npt.assert_almost_equal(tau[:, 0], np.array((5.40765381, 0)))
            npt.assert_almost_equal(tau[:, -1], np.array((-25.26494109, 0)))
        elif solver == Solver.FATROP:
            npt.assert_almost_equal(tau[:, 0], np.array((5.48109258, 0)))
            npt.assert_almost_equal(tau[:, -1], np.array((-23.15753116, 0)))
        else:
            raise RuntimeError("Unexpected solver")
    elif isinstance(ode_solver_obj, OdeSolver.COLLOCATION):
        if solver == Solver.IPOPT:
            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                npt.assert_almost_equal(tau[:, 0], np.array((5.71088029, 0)))
                npt.assert_almost_equal(tau[:, -1], np.array((-19.89491045, 0)))
            else:
                npt.assert_almost_equal(tau[:, 0], np.array((5.42317977, 0)))
                npt.assert_almost_equal(tau[:, -1], np.array((-25.26762264, 0)))
        elif solver == Solver.FATROP:
            if defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                npt.assert_almost_equal(tau[:, 0], np.array((5.78386568, 0)))
                npt.assert_almost_equal(tau[:, -1], np.array((-18.22245516, 0)))
            else:
                npt.assert_almost_equal(tau[:, 0], np.array((5.42317977, 0)))
                npt.assert_almost_equal(tau[:, -1], np.array((-25.26762264, 0)))
        else:
            raise RuntimeError("Unexpected solver")
    elif isinstance(ode_solver_obj, OdeSolver.RK1):
        if solver == Solver.IPOPT:
            npt.assert_almost_equal(tau[:, 0], np.array((5.498956, 0)))
            npt.assert_almost_equal(tau[:, -1], np.array((-17.6888209, 0)))
        elif solver == Solver.FATROP:
            npt.assert_almost_equal(tau[:, 0], np.array([5.38396582, 0]))
            npt.assert_almost_equal(tau[:, -1], np.array([-18.81942426, 0]))
        else:
            raise RuntimeError("Unexpected solver")
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

    if solver == Solver.IPOPT:
        global test_memory
        test_memory[f"pendulum-{ode_solver}-{use_sx}-{n_threads}-{phase_dynamics}"] = [
            building_duration,
            solving_duration,
            mem_used,
        ]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_custom_constraint_track_markers(ode_solver, phase_dynamics):
    from bioptim.examples.toy_examples.feature_examples import custom_constraint as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
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
    from bioptim.examples.toy_examples.feature_examples import custom_initial_guess as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver = ode_solver()

    np.random.seed(42)

    if interpolation == InterpolationType.ALL_POINTS and ode_solver.is_direct_shooting:
        with pytest.raises(ValueError, match="InterpolationType.ALL_POINTS must only be used with direct collocation"):
            _ = ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
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
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
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

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    np.random.seed(42)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
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

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    np.random.seed(42)
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
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
        pytest.skip("PhaseDynamics.ONE_PER_NODE is only tested with RK4 and IRK to reduce time")

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
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
        pytest.skip("PhaseDynamics.ONE_PER_NODE is only tested with RK4 and IRK to reduce time")

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
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
    from tests import test_utils_ocp as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        pytest.skip("PhaseDynamics.ONE_PER_NODE is only tested with RK4 and IRK to reduce time")

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
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
        pytest.skip("OdeSolver.IRK is only tested with use_sx=True")
    if n_threads == 2 and phase_dynamics == PhaseDynamics.ONE_PER_NODE:
        pytest.skip("PhaseDynamics.ONE_PER_NODE is only tested with RK4 and IRK to reduce time")

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_with_forces.bioMod",
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
        solving_duration,
        mem_used,
    ]


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver_type", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK, OdeSolver.COLLOCATION])
def test_example_multiphase(ode_solver_type, phase_dynamics):
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver_type in [OdeSolver.RK8, OdeSolver.COLLOCATION]:
        pytest.skip("PhaseDynamics.ONE_PER_NODE is only tested with RK4 and IRK to reduce time")

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver = ode_solver_type()
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
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
        pytest.skip("No warm start test for collocation yet")

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

    bioptim_folder = TestUtils.bioptim_folder()

    min_bound = 50

    if not expand_dynamics and ode_solver != OdeSolver.IRK:
        # There is no point testing that
        pytest.skip("PhaseDynamics.ONE_PER_NODE is only tested with RK4 and IRK to reduce time")
    if expand_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and ode_solver == OdeSolver.IRK:
        with pytest.raises(RuntimeError):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/2segments_4dof_2contacts.bioMod",
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
        biorbd_model_path=bioptim_folder + "/examples/models/2segments_4dof_2contacts.bioMod",
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

    bioptim_folder = TestUtils.bioptim_folder()

    max_bound = 75
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/2segments_4dof_2contacts.bioMod",
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
    from bioptim.examples.toy_examples.feature_examples import example_multinode_objective as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver = ode_solver()

    n_shooting = 20
    if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE:
        with pytest.raises(
            ValueError,
            match=(
                "Valid values for setting the cx is 0, 1 or 2. If you reach this error message, "
                "you probably tried to add more penalties than available in a multinode constraint. "
                "You can try to split the constraints into more penalties or use "
                "phase_dynamics=PhaseDynamics.ONE_PER_NODE"
            ),
        ):
            ocp = ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
                n_shooting=n_shooting,
                final_time=1,
                ode_solver=ode_solver,
                phase_dynamics=phase_dynamics,
                expand_dynamics=True,
            )
        return

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
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
            MultinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 0), nodes=(Node.START, node), key="all"
        )
        with pytest.raises(ValueError, match=re.escape("Each of the nodes must have a corresponding nodes_phase")):
            multinode_constraints.add(
                MultinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0,), nodes=(Node.START, node), key="all"
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
                MultinodeConstraintFcn.STATES_EQUALITY, nodes_phase=(0, 0), nodes=(Node.START, node), key="all"
            )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("too_much_constraints", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.IRK])
def test_multinode_constraints_too_much_constraints(ode_solver, too_much_constraints, phase_dynamics):
    from bioptim.examples.toy_examples.feature_examples import example_multinode_constraints as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_obj = ode_solver
    ode_solver = ode_solver()
    if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and too_much_constraints:
        with pytest.raises(
            ValueError,
            match="Valid values for setting the cx is 0, 1 or 2. If you reach this error message, you probably tried to "
            "add more penalties than available in a multinode constraint. You can try to split the constraints "
            "into more penalties or use phase_dynamics=PhaseDynamics.ONE_PER_NODE",
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
                n_shootings=(8, 8, 8),
                ode_solver=ode_solver,
                phase_dynamics=phase_dynamics,
                with_too_much_constraints=too_much_constraints,
                expand_dynamics=ode_solver_obj != OdeSolver.IRK,
            )
    else:
        ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
            n_shootings=(8, 8, 8),
            ode_solver=ode_solver,
            phase_dynamics=phase_dynamics,
            with_too_much_constraints=too_much_constraints,
            expand_dynamics=ode_solver_obj != OdeSolver.IRK,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_multinode_constraints(ode_solver, phase_dynamics):
    from bioptim.examples.toy_examples.feature_examples import example_multinode_constraints as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        pytest.skip("OdeSolver.RK8 is only tested with use_sx=True")

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube.bioMod",
        n_shootings=(8, 10, 8),
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
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

    bioptim_folder = TestUtils.bioptim_folder()
    bio_model_path = [bioptim_folder + "/examples/models/pendulum.bioMod"]
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
    from bioptim.examples.toy_examples.feature_examples import example_variable_scaling as ocp_module

    gc.collect()  # Force garbage collection
    time.sleep(0.1)  # Avoiding delay in memory (re)allocation
    tracemalloc.start()  # Start memory tracking
    mem_before = tracemalloc.take_snapshot()

    tik = time.time()  # Time before starting to build the problem

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/pendulum.bioMod",
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
            10.428200244903564,
            4.572763681411743,
            632152,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK1'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.172236204147339,
            3.9698519706726074,
            591139,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK1'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.906153917312622,
            1.7605631351470947,
            589874,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK1'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.54404616355896,
            1.9113569259643555,
            572287,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK2'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.150030136108398,
            3.7786338329315186,
            608129,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK2'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.18199110031128,
            4.087592124938965,
            590669,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK2'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.241022109985352,
            1.896014928817749,
            589008,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK2'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.338693857192993,
            2.1055331230163574,
            569263,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.423840999603271,
            4.139120101928711,
            473755,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-False-1-PhaseDynamics.ONE_PER_NODE": [
            25.06763982772827,
            4.058828115463257,
            888033,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.20091700553894,
            4.119539022445679,
            603531,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.502306938171387,
            2.2537460327148438,
            576834,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-True-1-PhaseDynamics.ONE_PER_NODE": [
            12.794831275939941,
            2.2078840732574463,
            870134,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.332290887832642,
            2.3827359676361084,
            600001,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK8'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.844088077545166,
            3.4072978496551514,
            560401,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK8'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.832592010498047,
            3.611886978149414,
            576190,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            18.43957829475403,
            4.037032842636108,
            899947,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.616436958312988,
            4.390933036804199,
            735225,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            8.078601360321045,
            8.792459964752197,
            689994,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-False-1-PhaseDynamics.ONE_PER_NODE": [
            17.752251625061035,
            8.70758056640625,
            1275880,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            8.119649887084961,
            6.599950790405273,
            796572,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.980520248413086,
            4.108591079711914,
            763235,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-True-1-PhaseDynamics.ONE_PER_NODE": [
            12.725889682769775,
            3.9272022247314453,
            1263727,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.COLLOCATION'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.875128746032715,
            4.187591075897217,
            783077,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.TRAPEZOIDAL'>-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.018677949905396,
            3.510016918182373,
            594962,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.TRAPEZOIDAL'>-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.05292797088623,
            3.9034368991851807,
            612351,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.TRAPEZOIDAL'>-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.240038156509399,
            1.6190829277038574,
            594868,
        ],
        "pendulum-<class 'bioptim.dynamics.ode_solvers.OdeSolver.TRAPEZOIDAL'>-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            6.330470085144043,
            1.867854118347168,
            642813,
        ],
        "custom_constraint_track_markers-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.673295974731445,
            3.388075113296509,
            609714,
        ],
        "custom_constraint_track_markers-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [
            26.097075939178467,
            3.3995869159698486,
            923073,
        ],
        "custom_constraint_track_markers-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            11.549690008163452,
            3.6156070232391357,
            645463,
        ],
        "custom_constraint_track_markers-RK8 5 steps-PhaseDynamics.ONE_PER_NODE": [
            68.34736394882202,
            3.636629104614258,
            920536,
        ],
        "custom_constraint_track_markers-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.76711106300354,
            3.313241958618164,
            707622,
        ],
        "custom_constraint_track_markers-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [
            28.626792907714844,
            3.40545916557312,
            1218876,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.1778297424316406,
            0.7433199882507324,
            417561,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT-True-PhaseDynamics.ONE_PER_NODE": [
            4.239842891693115,
            0.8284783363342285,
            441187,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.624359130859375,
            0.9851813316345215,
            405859,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT-False-PhaseDynamics.ONE_PER_NODE": [
            4.457371234893799,
            0.8162188529968262,
            440896,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.380051612854004,
            0.7899308204650879,
            404238,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT-True-PhaseDynamics.ONE_PER_NODE": [
            4.717679023742676,
            0.9147405624389648,
            435233,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.3846101760864258,
            0.9232211112976074,
            411426,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT-False-PhaseDynamics.ONE_PER_NODE": [
            4.601020812988281,
            0.8674907684326172,
            444399,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.LINEAR-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.927689552307129,
            1.1065387725830078,
            401641,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.LINEAR-True-PhaseDynamics.ONE_PER_NODE": [
            5.082271099090576,
            0.9821200370788574,
            440636,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.LINEAR-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            3.0338597297668457,
            0.9261083602905273,
            403861,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.LINEAR-False-PhaseDynamics.ONE_PER_NODE": [
            4.8421716690063477,
            0.8306384086608887,
            440277,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.EACH_FRAME-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.626190185546875,
            0.9510993957519531,
            400041,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.EACH_FRAME-True-PhaseDynamics.ONE_PER_NODE": [
            4.6212244033813477,
            0.8069467544555664,
            435017,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.EACH_FRAME-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.70949125289917,
            0.8343720436096191,
            408325,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.EACH_FRAME-False-PhaseDynamics.ONE_PER_NODE": [
            4.9596595764160156,
            0.8695292472839355,
            436745,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.ALL_POINTS-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.6180410385131836,
            1.0502195358276367,
            410309,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.ALL_POINTS-True-PhaseDynamics.ONE_PER_NODE": [
            5.576479434967041,
            2.1515202522277832,
            444541,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.ALL_POINTS-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            3.1530189514160156,
            1.218109130859375,
            403795,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.ALL_POINTS-False-PhaseDynamics.ONE_PER_NODE": [
            4.999852180480957,
            0.9397292137145996,
            442997,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.SPLINE-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5156712532043457,
            0.8472609519958496,
            440063,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.SPLINE-True-PhaseDynamics.ONE_PER_NODE": [
            4.89361047744751,
            0.8550405502319336,
            437511,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.SPLINE-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.6365089416503906,
            1.0550594329833984,
            402261,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.SPLINE-False-PhaseDynamics.ONE_PER_NODE": [
            4.780232906341553,
            0.906989574432373,
            439410,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CUSTOM-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.5539207458496094,
            0.9598684310913086,
            438701,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CUSTOM-True-PhaseDynamics.ONE_PER_NODE": [
            4.376351833343506,
            0.864567756652832,
            438701,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CUSTOM-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            2.6754093170166016,
            1.0262036323547363,
            4248252,
        ],
        "initial_guesses-COLLOCATION legendre 4-InterpolationType.CUSTOM-False-PhaseDynamics.ONE_PER_NODE": [
            4.361300468444824,
            0.8208799362182617,
            440940,
        ],
        "cyclic_objective-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            4.349828243255615,
            1.4332208633422852,
            394051,
        ],
        "cyclic_objective-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [9.301621198654175, 1.432386875152588, 489293],
        "cyclic_objective-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.420027017593384,
            1.5296180248260498,
            389368,
        ],
        "cyclic_objective-RK8 5 steps-PhaseDynamics.ONE_PER_NODE": [20.131958961486816, 1.5301339626312256, 479750],
        "cyclic_objective-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            4.4788501262664795,
            1.3940727710723877,
            412248,
        ],
        "cyclic_objective-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [10.172360181808472, 1.4182348251342773, 614057],
        "cyclic_constraint-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            4.389387130737305,
            1.4227328300476074,
            394725,
        ],
        "cyclic_constraint-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [9.25717806816101, 1.4185810089111328, 484761],
        "cyclic_constraint-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            5.414251089096069,
            1.5820820331573486,
            393900,
        ],
        "cyclic_constraint-RK8 5 steps-PhaseDynamics.ONE_PER_NODE": [24.86505699157715, 1.5264549255371094, 487561],
        "cyclic_constraint-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            4.60000205039978,
            1.390260934829712,
            410702,
        ],
        "cyclic_constraint-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [10.216149806976318, 1.4064080715179443, 608689],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            30.478795051574707,
            10.666538000106812,
            1483638,
        ],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-PhaseDynamics.ONE_PER_NODE": [
            71.24667596817017,
            10.562263011932373,
            2297333,
        ],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK8'>-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            34.5624041557312,
            12.113614797592163,
            1471071,
        ],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            31.68831205368042,
            10.003971099853516,
            1759293,
        ],
        "phase_transition-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.ONE_PER_NODE": [
            78.9703598022461,
            10.492695093154907,
            3059874,
        ],
        "custom_dynamics-RK4 5 steps-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.58127498626709,
            3.37709903717041,
            606574,
        ],
        "custom_dynamics-RK4 5 steps-True-PhaseDynamics.ONE_PER_NODE": [23.73125386238098, 3.392482042312622, 809222],
        "custom_dynamics-RK4 5 steps-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.532292127609253,
            3.378679037094116,
            611740,
        ],
        "custom_dynamics-RK4 5 steps-False-PhaseDynamics.ONE_PER_NODE": [
            28.305109977722168,
            3.4108550548553467,
            927952,
        ],
        "custom_dynamics-RK8 5 steps-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            11.493597269058228,
            3.5843007564544678,
            609811,
        ],
        "custom_dynamics-RK8 5 steps-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            11.700747013092041,
            3.7299020290374756,
            589937,
        ],
        "custom_dynamics-IRK legendre 4-True-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.9104483127594,
            3.4524967670440674,
            732398,
        ],
        "custom_dynamics-IRK legendre 4-True-PhaseDynamics.ONE_PER_NODE": [
            24.073041915893555,
            3.3761467933654785,
            1024986,
        ],
        "custom_dynamics-IRK legendre 4-False-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            10.821430206298828,
            3.313566207885742,
            703332,
        ],
        "custom_dynamics-IRK legendre 4-False-PhaseDynamics.ONE_PER_NODE": [
            29.340002059936523,
            3.4039158821105957,
            1216012,
        ],
        "external_forces-RK4 5 steps-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            6.897389888763428,
            2.2513530254364014,
            607401,
        ],
        "external_forces-RK8 5 steps-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            7.531352281570435,
            3.851998805999756,
            607869,
        ],
        "external_forces-RK4 5 steps-True-1-PhaseDynamics.ONE_PER_NODE-True": [
            15.906329154968262,
            2.292609930038452,
            964447,
        ],
        "external_forces-RK8 5 steps-True-1-PhaseDynamics.ONE_PER_NODE-True": [
            58.4593780040741,
            3.8535048961639404,
            922468,
        ],
        "external_forces-RK4 5 steps-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            7.105379819869995,
            2.4340322017669678,
            622783,
        ],
        "external_forces-RK8 5 steps-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            7.551738739013672,
            4.198095083236694,
            659147,
        ],
        "external_forces-RK4 5 steps-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            11.284422874450684,
            3.734930992126465,
            603286,
        ],
        "external_forces-RK8 5 steps-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            37.67650890350342,
            4.55831503868103,
            619707,
        ],
        "external_forces-IRK legendre 4-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            11.606570720672607,
            3.4060099124908447,
            706930,
        ],
        "external_forces-RK4 5 steps-False-1-PhaseDynamics.ONE_PER_NODE-True": [
            56.89438986778259,
            3.6793341636657715,
            928346,
        ],
        "external_forces-RK8 5 steps-False-1-PhaseDynamics.ONE_PER_NODE-True": [
            96.77864289283752,
            29.973921060562134,
            941629,
        ],
        "external_forces-IRK legendre 4-False-1-PhaseDynamics.ONE_PER_NODE-True": [
            35.51646900177002,
            3.9422879219055176,
            1220334,
        ],
        "external_forces-RK4 5 steps-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            17.36193585395813,
            3.757427930831909,
            678841,
        ],
        "external_forces-RK8 5 steps-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            12.601202011108398,
            4.02860689163208,
            669127,
        ],
        "external_forces-IRK legendre 4-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            11.716263055801392,
            3.723665952682495,
            763866,
        ],
        "external_forces-RK4 5 steps-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            6.879094839096069,
            2.2598249912261963,
            584587,
        ],
        "external_forces-RK8 5 steps-True-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            7.53236198425293,
            3.913083076477051,
            615358,
        ],
        "external_forces-RK4 5 steps-True-1-PhaseDynamics.ONE_PER_NODE-False": [
            15.84078598022461,
            2.304800033569336,
            954320,
        ],
        "external_forces-RK8 5 steps-True-1-PhaseDynamics.ONE_PER_NODE-False": [
            32.529654026031494,
            3.8880581855773926,
            925462,
        ],
        "external_forces-RK4 5 steps-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            6.912766933441162,
            2.419163942337036,
            631894,
        ],
        "external_forces-RK8 5 steps-True-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            7.492810964584351,
            4.099035978317261,
            653783,
        ],
        "external_forces-RK4 5 steps-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            11.323030948638916,
            3.7454001903533936,
            614744,
        ],
        "external_forces-RK8 5 steps-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            12.608556032180786,
            4.535145998001099,
            622750,
        ],
        "external_forces-IRK legendre 4-False-1-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            11.693609952926636,
            3.4200541973114014,
            707271,
        ],
        "external_forces-RK4 5 steps-False-1-PhaseDynamics.ONE_PER_NODE-False": [
            31.826242208480835,
            3.692452907562256,
            934066,
        ],
        "external_forces-RK8 5 steps-False-1-PhaseDynamics.ONE_PER_NODE-False": [
            72.00836086273193,
            4.5808892250061035,
            933251,
        ],
        "external_forces-IRK legendre 4-False-1-PhaseDynamics.ONE_PER_NODE-False": [
            35.6413209438324,
            3.8857109546661377,
            1217436,
        ],
        "external_forces-RK4 5 steps-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            11.262562036514282,
            3.7912240028381348,
            668504,
        ],
        "external_forces-RK8 5 steps-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            12.692394971847534,
            3.9266250133514404,
            674000,
        ],
        "external_forces-IRK legendre 4-False-2-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            11.90785002708435,
            3.7309491634368896,
            761460,
        ],
        "multiphase-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [25.86482834815979, 8.448194742202759, 1270804],
        "multiphase-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [62.09187197685242, 8.761482238769531, 2004362],
        "multiphase-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            29.005884885787964,
            9.155505180358887,
            1271537,
        ],
        "multiphase-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            26.476194143295288,
            8.287683725357056,
            1529573,
        ],
        "multiphase-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [118.62483716011047, 8.728019952774048, 2669836],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            7.514098882675171,
            4.180995941162109,
            318311,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.RK4'>-PhaseDynamics.ONE_PER_NODE-True": [
            37.526939153671265,
            4.229383945465088,
            489763,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.SHARED_DURING_THE_PHASE-True": [
            7.5234458446502686,
            4.092674970626831,
            484960,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.SHARED_DURING_THE_PHASE-False": [
            7.506089925765991,
            4.00847601890564,
            446034,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.ONE_PER_NODE-True": [
            38.98725414276123,
            5.3120410442352295,
            580952,
        ],
        "contact_forces_greater_than_constraint-<class 'bioptim.dynamics.ode_solvers.OdeSolver.IRK'>-PhaseDynamics.ONE_PER_NODE-False": [
            13.908345937728882,
            5.336513996124268,
            580055,
        ],
        "multinode_objective-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [32.83850407600403, 1.2930281162261963, 628298],
        "multinode_objective-RK8 5 steps-PhaseDynamics.ONE_PER_NODE": [16.471795320510864, 1.8922476768493652, 613241],
        "multinode_constraints-RK4 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            11.765421867370605,
            3.871987819671631,
            769962,
        ],
        "multinode_constraints-RK4 5 steps-PhaseDynamics.ONE_PER_NODE": [
            24.823140144348145,
            29.103562831878662,
            1032166,
        ],
        "multinode_constraints-RK8 5 steps-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            14.892894983291626,
            4.162814140319824,
            782765,
        ],
        "multinode_constraints-IRK legendre 4-PhaseDynamics.SHARED_DURING_THE_PHASE": [
            12.175392866134644,
            3.7508199214935303,
            833686,
        ],
        "multinode_constraints-IRK legendre 4-PhaseDynamics.ONE_PER_NODE": [
            52.144675970077515,
            3.8724920749664307,
            1299765,
        ],
        "multistart": [0.0025186538696289062, 17.739748239517212, 388847],
        "variable_scaling-PhaseDynamics.SHARED_DURING_THE_PHASE": [31.575006008148193, 2.302826166152954, 569048],
        "variable_scaling-PhaseDynamics.ONE_PER_NODE": [13.260251998901367, 2.27763295173645, 951887],
    }

    building_time = []
    solving_time = []
    RAM_usage = []
    for key in ref.keys():
        print(
            f"{key} : Building OCP time diff: {ref[key][0] - test_memory[key][0]} \tSolving OCP time diff: {ref[key][1] - test_memory[key][1]}\t Peak RAM diff: {ref[key][2] - test_memory[key][2]}"
        )
        building_time += [ref[key][0] - test_memory[key][0]]
        solving_time += [ref[key][1] - test_memory[key][1]]
        RAM_usage += [ref[key][2] - test_memory[key][2]]
    print(
        f"Means: {np.mean(np.array(building_time))}\t {np.mean(np.array(solving_time))}\t {np.mean(np.array(RAM_usage))}"
    )

    # # If the changes you have made in the code are expected to change the values, you can run the following to update them.
    # for key in ref.keys():
    #     print(f"{key} : [{test_memory[key][0]*10}, {test_memory[key][1]*10}, {test_memory[key][2]}]")

    factor = 5 if os.getenv("GITHUB_ACTIONS") == "true" else 3
    for key in ref.keys():
        npt.assert_array_less(test_memory[key][0], ref[key][0] * factor)
        npt.assert_array_less(test_memory[key][1], ref[key][1] * factor)
        npt.assert_array_less(test_memory[key][2], ref[key][2] * factor)
