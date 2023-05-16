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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_excitation_with_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    bio_model = BiorbdModel(model_path)
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(bio_model, final_time, n_shooting)

    bio_model = BiorbdModel(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        bio_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: bio_model.nb_q, :].T,
        use_residual_torque=True,
        kin_data_to_track="markers",
        ode_solver=ode_solver(),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (50 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((50 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (50, 1))
        np.testing.assert_almost_equal(g, np.zeros((50, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus_states, tau, mus_controls = (
        sol.states["q"],
        sol.states["qdot"],
        sol.states["muscles"],
        sol.controls["tau"],
        sol.controls["muscles"],
    )

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 1.9423215393458834e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.0022161, -0.00062983]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20632374, -0.96266977]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-0.02544063, 1.11230153]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.23632381, -9.11739593]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.5193721, 0.50851183, 0.6051374, 0.43719123, 0.59329003, 0.59971324])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([6.66733699e-05, 6.40935259e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([5.67398982e-05, -5.00305009e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76677771, 0.02174135, 0.633964, 0.74879614, 0.49849973, 0.22512206])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.44112329, 0.43426359, 0.61784926, 0.51301095, 0.65031982, 0.60125901])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 1.942678347042154e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00221554, -0.00063043]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20632272, -0.96266609]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-0.02540291, 1.1120538]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.23629081, -9.11724989]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.5193701, 0.50851049, 0.60513389, 0.4371962, 0.59328742, 0.59971041])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([6.66635743e-05, 6.40977127e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([5.67472786e-05, -5.00632439e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76677035, 0.02197853, 0.6339604, 0.74878751, 0.49849609, 0.22513587])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.44112149, 0.43426245, 0.61784649, 0.51301426, 0.6503179, 0.60125632])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 1.9563634639504918e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00224285, -0.00055806]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.2062965, -0.96260434]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-0.02505696, 1.11102856]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.23472979, -9.11352557]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51939095, 0.50853479, 0.6051437, 0.4371827, 0.59328557, 0.59973123])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([6.72188259e-05, 5.01548712e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([5.74813746e-05, -5.17061496e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76676674, 0.02172467, 0.63396249, 0.74880157, 0.49850197, 0.22513888])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.44110908, 0.43426931, 0.6178526, 0.51300672, 0.65031742, 0.60126635])
        )

    else:
        raise ValueError("Test not ready")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_excitation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    bio_model = BiorbdModel(model_path)
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(bio_model, final_time, n_shooting)

    bio_model = BiorbdModel(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        bio_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: bio_model.nb_q, :].T,
        use_residual_torque=False,
        kin_data_to_track="markers",
        ode_solver=ode_solver(),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_equal(g.shape, (50 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((50 * 5, 1)), decimal=6)
    else:
        np.testing.assert_equal(g.shape, (50, 1))
        np.testing.assert_almost_equal(g, np.zeros((50, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus_states, mus_controls = (
        sol.states["q"],
        sol.states["qdot"],
        sol.states["muscles"],
        sol.controls["muscles"],
    )

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 1.9426861462787857e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00221826, -0.00062423]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20632271, -0.96266717]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-0.02535376, 1.11208698]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.2362817, -9.11728306]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51937086, 0.50851233, 0.60513767, 0.43719097, 0.59328989, 0.59971401])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76677683, 0.02174148, 0.63396384, 0.74879658, 0.49849991, 0.22512315])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.44112273, 0.43426381, 0.61784939, 0.51301078, 0.65031973, 0.6012593])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 1.9430426243279718e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00221771, -0.00062483]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20632169, -0.9626635]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-0.02531606, 1.11183926]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.23624873, -9.11713708]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51936886, 0.50851099, 0.60513415, 0.43719595, 0.59328728, 0.59971118])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76676948, 0.02197866, 0.63396025, 0.74878795, 0.49849627, 0.22513696])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.44112094, 0.43426268, 0.61784662, 0.5130141, 0.6503178, 0.60125662])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 1.956741750742022e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00224508, -0.00055229]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.20629545, -0.96260166]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-0.02496724, 1.11080593]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.23468676, -9.11340941]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51938967, 0.50853531, 0.60514397, 0.43718244, 0.59328542, 0.59973203])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76676586, 0.02172479, 0.63396233, 0.74880202, 0.49850216, 0.22514])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.44110851, 0.43426954, 0.61785274, 0.51300655, 0.65031733, 0.60126665])
        )

    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)
