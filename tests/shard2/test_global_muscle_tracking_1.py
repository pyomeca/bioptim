"""
Test for file IO
"""
import os
import pytest
import platform

import numpy as np
from bioptim import OdeSolver, PhaseDynamics, BiorbdModel

from tests.utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_activation_no_residual_torque_and_markers_tracking(ode_solver, phase_dynamics):
    # Load muscle_activations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module

    if platform.system() == "Windows" and phase_dynamics == PhaseDynamics.ONE_PER_NODE:
        # This is a long test and CI is already long for Windows
        return

    # For reducing time phase_dynamics=False is skipped for redundant tests
    # and because test fails on CI
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver in (OdeSolver.RK4, OdeSolver.COLLOCATION):
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    bio_model = BiorbdModel(model_path)
    final_time = 0.1
    n_shooting = 5
    use_residual_torque = False

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_activations_ref = ocp_module.generate_data(
        bio_model, final_time, n_shooting, use_residual_torque=use_residual_torque
    )

    bio_model = BiorbdModel(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        bio_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: bio_model.nb_q, :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (20 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((20 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (20, 1))
        np.testing.assert_almost_equal(g, np.zeros((20, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus = sol.states["q"], sol.states["qdot"], sol.controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 4.162211328576168e-09)
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-4.35868770e-06, 3.99285825e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20478893, -0.9507116]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.34313410e-04, -8.73178582e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.43553142, -6.90717515]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([0.77133463, 0.02085465, 0.63363299, 0.74881884, 0.49851663, 0.22482276]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([0.44190476, 0.43398509, 0.61774548, 0.51315871, 0.650407, 0.60099513]),
        )
    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 4.145731569100745e-09)
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-3.74337403e-06,  4.00697373e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20480488, -0.95076061]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.17026419e-04, -9.75179756e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.43456688, -6.90997413]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([0.77133458, 0.02085464, 0.63363333, 0.74881816, 0.49851632,
       0.22482216]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([0.4418359 , 0.4340145 , 0.61776425, 0.5131385 , 0.65039449,
       0.60103605]),
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 4.148276544152576e-09)
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-4.41978433e-06, 4.05234428e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20478889, -0.95071163]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00014806, -0.00012115]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.43553511, -6.90717149]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([0.77133494, 0.02085459, 0.6336328, 0.74881914, 0.49851677, 0.22482304]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([0.44190501, 0.43398497, 0.61774539, 0.5131588, 0.65040706, 0.60099495]),
        )
    else:
        raise ValueError("Test not ready")

    # simulate
    TestUtils.simulate(sol, decimal_value=6)
