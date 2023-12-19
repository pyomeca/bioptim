"""
Test for file IO
"""
import os
import pytest

import numpy as np
from bioptim import OdeSolver, ControlType, PhaseDynamics

from tests.utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.IRK, OdeSolver.COLLOCATION, OdeSolver.TRAPEZOIDAL])
def test_muscle_driven_ocp(ode_solver, phase_dynamics):
    from bioptim.examples.muscle_driven_ocp import static_arm as ocp_module

    # For reducing time phase_dynamics=PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.COLLOCATION:
        return
    if ode_solver == OdeSolver.TRAPEZOIDAL:
        control_type = ControlType.LINEAR_CONTINUOUS
    else:
        control_type = ControlType.CONSTANT

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        bioptim_folder + "/models/arm26.bioMod",
        final_time=0.1,
        n_shooting=5,
        weight=1,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
        control_type=control_type,
        n_threads=1,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (20 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((20 * 5, 1)), decimal=5)
    else:
        np.testing.assert_equal(g.shape, (20, 1))
        np.testing.assert_almost_equal(g, np.zeros((20, 1)), decimal=5)

    # Check some of the results
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 0.1264429986075503)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.19992514, 2.65885447]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.31428464, 14.18136011]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00799549, 0.02025832]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([0.00228285, 0.00281159]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([7.16894451e-06, 6.03295625e-01, 3.37029285e-01, 1.08379171e-05, 1.14087135e-05, 3.66744227e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([5.46687138e-05, 6.60562511e-03, 3.77597977e-03, 4.92824218e-04, 5.09440179e-04, 9.08091234e-03]),
        )

    elif ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 0.12644299285122357)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.19992522, 2.65885512]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.31428244, 14.18136079]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00799548, 0.02025833]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([0.00228284, 0.00281158]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([7.16894627e-06, 6.03295877e-01, 3.37029458e-01, 1.08379096e-05, 1.14087059e-05, 3.66744423e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([5.46688078e-05, 6.60548530e-03, 3.77595547e-03, 4.92828831e-04, 5.09444822e-04, 9.08082070e-03]),
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 0.12644297341855165)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.19992534, 2.65884909]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-2.31430927, 14.18129464]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00799575, 0.02025812]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([0.00228286, 0.00281158]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([7.16887076e-06, 6.03293415e-01, 3.37026700e-01, 1.08380212e-05, 1.14088234e-05, 3.66740786e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([5.46652642e-05, 6.57077193e-03, 3.72595814e-03, 4.73887187e-04, 4.89821189e-04, 9.06067240e-03]),
        )
    elif ode_solver == OdeSolver.TRAPEZOIDAL:
        np.testing.assert_almost_equal(f[0, 0], 0.13299706974727432)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.24156091, 2.61667234]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.37135239, 16.36179822]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00236075, 0.01175397]))
        np.testing.assert_almost_equal(tau[:, -3], np.array([0.00096139, 0.00296023]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([1.64993088e-05, 3.49179013e-01, 2.05274808e-01, 2.00177858e-05, 2.12125215e-05, 2.17492272e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -3],
            np.array([0.00015523, 0.05732295, 0.02321138, 0.00036435, 0.00039923, 0.04455363]),
        )
    else:
        raise ValueError("Test not ready")

    # simulate
    TestUtils.simulate(sol, decimal_value=5)
