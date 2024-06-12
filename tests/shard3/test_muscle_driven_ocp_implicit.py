"""
Test for file IO
"""

import os
import pytest

import numpy as np
import numpy.testing as npt
from bioptim import OdeSolver, DefectType, PhaseDynamics, SolutionMerge

from tests.utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_driven_ocp_implicit(ode_solver, phase_dynamics):
    from bioptim.examples.muscle_driven_ocp import static_arm as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver_obj = ode_solver(defects_type=DefectType.IMPLICIT)
    ocp = ocp_module.prepare_ocp(
        bioptim_folder + "/models/arm26.bioMod",
        final_time=0.1,
        n_shooting=5,
        weight=1,
        ode_solver=ode_solver_obj,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
        n_threads=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (20 * 5, 1))
        npt.assert_almost_equal(g, np.zeros((20 * 5, 1)), decimal=5)
    else:
        npt.assert_equal(g.shape, (20, 1))
        npt.assert_almost_equal(g, np.zeros((20, 1)), decimal=5)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau, mus = states["q"], states["qdot"], controls["tau"], controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        npt.assert_almost_equal(f[0, 0], 0.12644299285122357)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        npt.assert_almost_equal(q[:, -1], np.array([-0.19992522, 2.65885512]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        npt.assert_almost_equal(qdot[:, -1], np.array([-2.31428244, 14.18136079]))
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([0.00799548, 0.02025833]))
        npt.assert_almost_equal(tau[:, -1], np.array([0.00228284, 0.00281158]))
        npt.assert_almost_equal(
            mus[:, 0],
            np.array([7.16894627e-06, 6.03295877e-01, 3.37029458e-01, 1.08379096e-05, 1.14087059e-05, 3.66744423e-01]),
        )
        npt.assert_almost_equal(
            mus[:, -1],
            np.array([5.46688078e-05, 6.60548530e-03, 3.77595547e-03, 4.92828831e-04, 5.09444822e-04, 9.08082070e-03]),
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        npt.assert_almost_equal(f[0, 0], 0.12644297341855165)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        npt.assert_almost_equal(q[:, -1], np.array([-0.19992534, 2.65884909]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        npt.assert_almost_equal(qdot[:, -1], np.array([-2.3143106, 14.1812974]))
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([0.00799575, 0.02025812]))
        npt.assert_almost_equal(tau[:, -1], np.array([0.00228286, 0.00281158]))
        npt.assert_almost_equal(
            mus[:, 0],
            np.array([7.16887076e-06, 6.03293415e-01, 3.37026700e-01, 1.08380212e-05, 1.14088234e-05, 3.66740786e-01]),
        )
        npt.assert_almost_equal(
            mus[:, -1],
            np.array([5.4664028e-05, 6.5610959e-03, 3.7092411e-03, 4.6592962e-04, 4.8159442e-04, 9.0543847e-03]),
        )
    else:
        raise ValueError("Test not ready")

    # simulate
    TestUtils.simulate(sol, decimal_value=5)
