"""
Test for file IO
"""

import platform
import pytest
import numpy as np
import numpy.testing as npt
from bioptim import OdeSolver, PhaseDynamics, SolutionMerge

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_symmetry_by_mapping(ode_solver, phase_dynamics):
    from bioptim.examples.toy_examples.symmetrical_torque_driven_ocp import symmetry_by_mapping as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    if ode_solver == OdeSolver.COLLOCATION or ode_solver == OdeSolver.IRK:
        with pytest.raises(
            NotImplementedError,
            match="COLLOCATION transcription is not compatible with mapping for states. Please note that concept of states mapping in already sketchy on it's own, but is particularly not appropriate for COLLOCATION transcriptions.",
        ):
            ocp = ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/examples/models/cubeSym.bioMod",
                ode_solver=ode_solver(),
                phase_dynamics=phase_dynamics,
                expand_dynamics=ode_solver != OdeSolver.IRK,
            )
    else:
        ocp = ocp_module.prepare_ocp(
            biorbd_model_path=bioptim_folder + "/examples/models/cubeSym.bioMod",
            ode_solver=ode_solver(),
            phase_dynamics=phase_dynamics,
            expand_dynamics=ode_solver != OdeSolver.IRK,
        )
        sol = ocp.solve()

        # Check objective function value
        f = np.array(sol.cost)
        npt.assert_equal(f.shape, (1, 1))
        npt.assert_almost_equal(f[0, 0], 216.56763999010334)

        # Check constraints
        g = np.array(sol.constraints)
        npt.assert_equal(g.shape, (186, 1))
        npt.assert_almost_equal(g, np.zeros((186, 1)))

        # Check some of the results
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        q, qdot, tau = states["q"], states["qdot"], controls["tau"]

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array((-0.2, -1.1797959, 0.20135792)))
        npt.assert_almost_equal(q[:, -1], np.array((0.2, -0.7797959, -0.20135792)))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
        npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((1.16129033, 1.16129033, -0.58458751)))
        npt.assert_almost_equal(tau[:, -1], np.array((-1.16129033, -1.16129033, 0.58458751)))

        # simulate
        TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_symmetry_by_constraint(ode_solver, phase_dynamics):
    from bioptim.examples.symmetrical_torque_driven_ocp import symmetry_by_constraint as ocp_module

    if platform.system() == "Darwin":
        pytest.skip("This test does not pass in one case on MacOS.")

    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.COLLOCATION:
        pytest.skip("For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests")

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cubeSym.bioMod",
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_almost_equal(f[0, 0], 216.567618843852)
        npt.assert_equal(g.shape, (300 * 5 + 36, 1))
        npt.assert_almost_equal(g, np.zeros((300 * 5 + 36, 1)))
    else:
        npt.assert_almost_equal(f[0, 0], 216.56763999010465)
        npt.assert_equal(g.shape, (336, 1))
        npt.assert_almost_equal(g, np.zeros((336, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((-0.2, -1.1797959, 0, 0.20135792, -0.20135792)))
    npt.assert_almost_equal(q[:, -1], np.array((0.2, -0.7797959, 0, -0.20135792, 0.20135792)))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((1.16129033, 1.16129033, 0, -0.58458751, 0.58458751)))
    npt.assert_almost_equal(tau[:, -1], np.array((-1.16129033, -1.16129033, 0, 0.58458751, -0.58458751)))

    # simulate
    TestUtils.simulate(sol)
