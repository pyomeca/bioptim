"""
Test for file IO
"""

import platform

from bioptim import OdeSolver, Solver, MusclesBiorbdModel, PhaseDynamics, SolutionMerge
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_muscle_activations_and_states_tracking(ode_solver, n_threads, phase_dynamics):
    # Load muscle_activations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module

    # For reducing time phase_dynamics=PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.COLLOCATION:
        pytest.skip("Redundant test")
    if n_threads > 1 and phase_dynamics == PhaseDynamics.ONE_PER_NODE:
        pytest.skip("Redundant test")

    bioptim_folder = TestUtils.module_folder(ocp_module)

    # Define the problem
    use_residual_torque = True
    model_path = bioptim_folder + "/models/arm26.bioMod"
    bio_model = MusclesBiorbdModel(model_path, with_residual_torque=use_residual_torque)
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_activations_ref = ocp_module.generate_data(
        bio_model, final_time, n_shooting, use_residual_torque=use_residual_torque
    )

    bio_model = MusclesBiorbdModel(model_path, with_residual_torque=use_residual_torque)  # To allow for non free variable, the model must be reloaded
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
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )

    # Check the values which will be sent to the solver
    np.random.seed(42)
    match ode_solver:
        case OdeSolver.RK4:
            v_len = 65
            expected = [30.176497827705912, 216.82271917709082, 41.92662823234213]
        case OdeSolver.COLLOCATION:
            v_len = 145
            expected = [69.45381468487611, 200.49863362309583, 1010.2302612756722]
        case OdeSolver.IRK:
            v_len = 65
            expected = [30.176497827705912, 216.82271917709082, 63.204797772662936]
        case _:
            raise ValueError("Test not implemented")

    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.random.rand(v_len, 1),
        expected_v_f_g=expected,
        decimal=6,
    )
    if platform.system() == "Windows":
        return

    solver = Solver.IPOPT()
    # solver.set_maximum_iterations(10)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (20 * 5, 1))
        npt.assert_almost_equal(g, np.zeros((20 * 5, 1)), decimal=6)
    else:
        npt.assert_equal(g.shape, (20, 1))
        npt.assert_almost_equal(g, np.zeros((20, 1)), decimal=6)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau, mus = states["q"], states["qdot"], controls["tau"], controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        npt.assert_almost_equal(f[0, 0], 8.776096413864758e-09)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-6.94616318e-06, 5.36043303e-06]))
        npt.assert_almost_equal(q[:, -1], np.array([0.20478789, -0.95071274]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([2.12185372e-04, -4.51998027e-05]))
        npt.assert_almost_equal(qdot[:, -1], np.array([-0.43557515, -6.90724245]))
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([3.10812296e-06, -8.10321473e-06]))
        npt.assert_almost_equal(tau[:, -1], np.array([-9.47419953e-07, 3.09587412e-06]))
        npt.assert_almost_equal(
            mus[:, 0],
            np.array([0.77134219, 0.02085427, 0.6336279, 0.74882745, 0.49852058, 0.22483054]),
        )
        npt.assert_almost_equal(
            mus[:, -1],
            np.array([0.44191616, 0.43397999, 0.61774185, 0.51316252, 0.65040935, 0.60098744]),
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        npt.assert_almost_equal(f[0, 0], 4.15552736658107e-09)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-3.71213259e-06, 3.93204485e-06]))
        npt.assert_almost_equal(q[:, -1], np.array([0.20480484, -0.95076056]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([1.13930895e-04, -8.97973309e-05]))
        npt.assert_almost_equal(qdot[:, -1], np.array([-0.43456887, -6.90997078]))
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([2.05296197e-06, -5.46867080e-06]))
        npt.assert_almost_equal(tau[:, -1], np.array([-1.99157590e-08, 6.13726538e-08]))
        npt.assert_almost_equal(
            mus[:, 0], np.array([0.7713342, 0.02085471, 0.63363354, 0.74881783, 0.49851617, 0.22482186])
        )
        npt.assert_almost_equal(
            mus[:, -1],
            np.array([0.4418359, 0.4340145, 0.61776425, 0.5131385, 0.65039449, 0.60103605]),
        )

    elif ode_solver == OdeSolver.RK4:
        npt.assert_almost_equal(f[0, 0], 8.759278201846765e-09)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-7.00609088e-06, 5.41894006e-06]))
        npt.assert_almost_equal(q[:, -1], np.array([0.20478786, -0.95071277]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([2.25863939e-04, -7.89597284e-05]))
        npt.assert_almost_equal(qdot[:, -1], np.array([-0.43557883, -6.90723878]))
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([3.13930953e-06, -8.18582928e-06]))
        npt.assert_almost_equal(tau[:, -1], np.array([-9.49304938e-07, 3.10696405e-06]))
        npt.assert_almost_equal(
            mus[:, 0],
            np.array([0.7713425, 0.02085421, 0.63362772, 0.74882775, 0.49852071, 0.22483082]),
        )
        npt.assert_almost_equal(
            mus[:, -1],
            np.array([0.44191641, 0.43397987, 0.61774176, 0.5131626, 0.65040941, 0.60098726]),
        )

    else:
        raise ValueError("Test not implemented")

    # simulate
    TestUtils.simulate(sol, decimal_value=5)
