"""
Test for file IO
"""

import pytest

import numpy as np
import numpy.testing as npt
from bioptim import OdeSolver, ControlType, PhaseDynamics, SolutionMerge

from ..utils import TestUtils

def test_muscle_driven_ocp():
    from bioptim.examples.muscle_driven_with_contact import (
        contact_forces_inverse_dynamics_soft_contacts_muscle as ocp_module,
    )

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_4dof_2soft_contacts_1muscle.bioMod",
        phase_time=1,
        n_shooting=100,
    )
    sol = ocp.solve()

    # Check that it converged
    assert sol.status == 0

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f, np.array([[13845.89524153]]), decimal=5)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (7988, 1))
    npt.assert_almost_equal(g, np.zeros((7988, 1)), decimal=5)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    algebraic_states = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)
    q, qdot, tau_residual, mus, contact_forces = (
        states["q"],
        states["qdot"],
        controls["tau"],
        controls["muscles"],
        algebraic_states["soft_contact_forces"],
    )

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([0.  ,  1.  , -0.75,  0.75]))
    npt.assert_almost_equal(q[:, -1], np.array([1.29798477,  0.43475825, -0.74106543,  0.74666865]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array([0., 0., 0., 0.]))
    npt.assert_almost_equal(qdot[:, -1], np.array([-4.87558018,   4.61065044,  12.99614536, -13.70780252]))
    # initial and final controls
    npt.assert_almost_equal(tau_residual[:, 0], np.array([-31.57475842, -111.60946208,    8.29614935,   -0.21144813]))
    npt.assert_almost_equal(tau_residual[:, -1], np.array([0., 0., 0., 0.]))
    npt.assert_almost_equal(mus[:, 0], np.array([4.22584556e-06]))
    npt.assert_almost_equal(mus[:, -1], np.array([0.0]))
    # algebraic states (not the first and last since they are free)
    npt.assert_almost_equal(contact_forces[:, 1], np.array([0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
       0.,   0.      ,   0.      ,   0.      ,   0.      ,
         0.      , 0.]))
    npt.assert_almost_equal(contact_forces[:, -2], np.array([23.8539612,  0.       ,  0.       ,  0.       , 19.894077 ,
       25.889183 , 11.1186312,  0.       ,  0.       ,  0.       ,
        6.737669 ,  8.7657347]))

    # simulate
    TestUtils.simulate(sol, decimal_value=5)
