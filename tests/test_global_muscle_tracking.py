"""
Test for file IO
"""
import os
import pytest

import numpy as np
from bioptim import OdeSolver, Solver, BiorbdModel

from .utils import TestUtils


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_muscle_activations_and_states_tracking(ode_solver, n_threads, assume_phase_dynamics):
    # Load muscle_activations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
        return
    if n_threads > 1 and not assume_phase_dynamics:
        return

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    bio_model = BiorbdModel(model_path)
    final_time = 0.1
    n_shooting = 5
    use_residual_torque = True

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
        n_threads=n_threads,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    solver = Solver.IPOPT()
    # solver.set_maximum_iterations(10)
    sol = ocp.solve(solver)

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
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 8.776096413864758e-09)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-6.94616318e-06, 5.36043303e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20478789, -0.95071274]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([2.12185372e-04, -4.51998027e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.43557515, -6.90724245]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([3.10812296e-06, -8.10321473e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-9.47419953e-07, 3.09587412e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([0.77134219, 0.02085427, 0.6336279, 0.74882745, 0.49852058, 0.22483054]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([0.44191616, 0.43397999, 0.61774185, 0.51316252, 0.65040935, 0.60098744]),
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 4.15552736658107e-09)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-4.31570204e-06, 3.86467256e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20478891, -0.95071153]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.32006135e-04, -8.20933840e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.43553183, -6.90717365]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([2.13121740e-06, -5.61544104e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-8.44568436e-07, 2.61276733e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.77133409, 0.02085475, 0.63363328, 0.74881837, 0.49851642, 0.22482234])
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([0.44190465, 0.43398513, 0.61774549, 0.51315869, 0.65040699, 0.60099517]),
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 8.759278201846765e-09)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-7.00609088e-06, 5.41894006e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20478786, -0.95071277]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([2.25863939e-04, -7.89597284e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.43557883, -6.90723878]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([3.13930953e-06, -8.18582928e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-9.49304938e-07, 3.10696405e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([0.7713425, 0.02085421, 0.63362772, 0.74882775, 0.49852071, 0.22483082]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([0.44191641, 0.43397987, 0.61774176, 0.5131626, 0.65040941, 0.60098726]),
        )

    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_activation_no_residual_torque_and_markers_tracking(ode_solver, assume_phase_dynamics):
    # Load muscle_activations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module

    # For reducing time assume_phase_dynamics=False is skipped for redundant tests
    if not assume_phase_dynamics and ode_solver == OdeSolver.COLLOCATION:
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
        assume_phase_dynamics=assume_phase_dynamics,
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
        np.testing.assert_almost_equal(q[:, 0], np.array([-4.33718022e-06, 3.93914750e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20478893, -0.95071153]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.34857046e-04, -9.01607090e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.43553087, -6.90717431]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([0.77133446, 0.02085468, 0.63363308, 0.74881868, 0.49851656, 0.22482263]),
        )
        np.testing.assert_almost_equal(
            mus[:, -2],
            np.array([0.44190474, 0.43398509, 0.61774548, 0.51315871, 0.650407, 0.60099513]),
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

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)
