from bioptim import OdeSolver, PhaseDynamics, SolutionMerge
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_soft_contact(phase_dynamics):
    from bioptim.examples.torque_driven_ocp import example_soft_contact as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver = OdeSolver.RK8()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "examples/models/soft_contact_sphere.bioMod",
        final_time=0.37,
        n_shooting=37,
        n_threads=8 if phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE else 1,
        use_sx=False,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )

    ocp.print(to_console=True, to_graph=False)
    sol = ocp.solve()

    # Check that it converged
    assert sol.status == 0

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    if isinstance(ode_solver, OdeSolver.RK8):
        npt.assert_almost_equal(f[0, 0], 23.679065887950486)
    else:
        npt.assert_almost_equal(f[0, 0], 41.58259426)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (228, 1))
    npt.assert_almost_equal(g, np.zeros((228, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((0, 0, 0)), decimal=1)
    npt.assert_almost_equal(q[:, -1], np.array([0.05, 0.0933177, -0.6262446]))

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)), decimal=4)
    npt.assert_almost_equal(qdot[:, -1], np.array([2.03004523e-01, -1.74795966e-05, -2.53770131e00]))

    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([-0.16347455, 0.02123226, -13.25955361]))
    npt.assert_almost_equal(tau[:, -1], np.array([0.00862357, -0.00298151, -0.16425701]))
