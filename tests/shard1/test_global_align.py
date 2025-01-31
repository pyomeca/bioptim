"""
Test for file IO
"""

import platform
import pytest

from bioptim import OdeSolver, PhaseDynamics, SolutionMerge
import numpy as np
import numpy.testing as npt

from tests.utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_segment_on_rt(ode_solver, phase_dynamics):
    from bioptim.examples.track import track_segment_on_rt as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        final_time=0.5,
        n_shooting=8,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 197120.95524154368)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (91, 1))
    npt.assert_almost_equal(g, np.zeros((91, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([0.30543155, 0, -1.57, -1.57]))
    npt.assert_almost_equal(q[:, -1], np.array([0.30543155, 0, 1.57, 1.57]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([0, 9.81, 66.98666900582079, 66.98666424580644]))
    npt.assert_almost_equal(tau[:, -1], np.array([-0, 9.81, -66.98666900582079, -66.98666424580644]))

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_marker_on_segment(ode_solver, phase_dynamics):
    from bioptim.examples.track import track_marker_on_segment as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        final_time=0.5,
        n_shooting=8,
        initialize_near_solution=True,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )

    np.random.seed(42)
    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.random.rand(105, 1),
        expected_v_f_g=[49.41640708093857, 380.05556849200684, 18.43946477408692],
        decimal=6,
    )
    if platform.system() == "Windows":
        return

    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], 42127.04677760122)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (88, 1))
    npt.assert_almost_equal(g, np.zeros((88, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([1, 0, 0, 0.46364761]))
    npt.assert_almost_equal(q[:, -1], np.array([2, 0, 1.57, 0.78539785]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([23.6216587, 12.2590703, 31.520697, 12.9472294]))
    npt.assert_almost_equal(tau[:, -1], np.array([-16.659525, 14.5872277, -36.1009998, 4.417834]))

    # simulate
    TestUtils.simulate(sol)
