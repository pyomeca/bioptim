"""
Test for file IO
"""

import pickle
import platform
import re
import shutil

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
        return

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
    ocp.print(to_console=True, to_graph=False)

    # the test is too long with CVODES
    if isinstance(ode_solver_obj, OdeSolver.CVODES):
        return

    sol = ocp.solve()

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
    return


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_custom_constraint_track_markers(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import custom_constraint as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    sol = ocp.solve()

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("random_init", [True, False])
@pytest.mark.parametrize("interpolation", [*InterpolationType])
@pytest.mark.parametrize("ode_solver", [OdeSolver.COLLOCATION])
def test_initial_guesses(ode_solver, interpolation, random_init, phase_dynamics):
    from bioptim.examples.getting_started import custom_initial_guess as ocp_module

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

    sol = ocp.solve()

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_cyclic_objective(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import example_cyclic_movement as ocp_module

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
    sol = ocp.solve()

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_cyclic_constraint(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import example_cyclic_movement as ocp_module

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
    sol = ocp.solve()

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_phase_transitions(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import custom_phase_transitions as ocp_module

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
    sol = ocp.solve()

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
    sol = ocp.solve()

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("problem_type_custom", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_custom_problem_type_and_dynamics(problem_type_custom, ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import custom_dynamics as ocp_module

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
    sol = ocp.solve()

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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("n_threads", [1, 2])
@pytest.mark.parametrize("use_sx", [True, False])
@pytest.mark.parametrize("use_point_of_applications", [True, False])
def test_example_external_forces(ode_solver, phase_dynamics, n_threads, use_sx, use_point_of_applications):
    from bioptim.examples.getting_started import example_external_forces as ocp_module

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
    sol = ocp.solve()

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver_type", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK, OdeSolver.COLLOCATION])
def test_example_multiphase(ode_solver_type, phase_dynamics):
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver_type in [OdeSolver.RK8, OdeSolver.COLLOCATION]:
        return

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver = ode_solver_type()
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_type != OdeSolver.IRK,
    )
    sol = ocp.solve()

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


@pytest.mark.parametrize("expand_dynamics", [True, False])
@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.IRK])
def test_contact_forces_inequality_greater_than_constraint(ode_solver, phase_dynamics, expand_dynamics):
    from bioptim.examples.getting_started import example_inequality_constraint as ocp_module

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
    sol = ocp.solve()

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

    bioptim_folder = TestUtils.module_folder(ocp_module)

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
    sol = ocp.solve()
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
    from bioptim.examples.getting_started import example_multinode_constraints as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

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
                biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
                n_shootings=(8, 8, 8),
                ode_solver=ode_solver,
                phase_dynamics=phase_dynamics,
                with_too_much_constraints=too_much_constraints,
                expand_dynamics=ode_solver_obj != OdeSolver.IRK,
            )
    else:
        ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
            n_shootings=(8, 8, 8),
            ode_solver=ode_solver,
            phase_dynamics=phase_dynamics,
            with_too_much_constraints=too_much_constraints,
            expand_dynamics=ode_solver_obj != OdeSolver.IRK,
        )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_multinode_constraints(ode_solver, phase_dynamics):
    from bioptim.examples.getting_started import example_multinode_constraints as ocp_module

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        return

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shootings=(8, 10, 8),
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    sol = ocp.solve()
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


def test_multistart():
    from bioptim.examples.getting_started import example_multistart as ocp_module

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
    multi_start.solve()

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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_example_variable_scaling(phase_dynamics):
    from bioptim.examples.getting_started import example_variable_scaling as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1 / 10,
        n_shooting=30,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    sol = ocp.solve()

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
