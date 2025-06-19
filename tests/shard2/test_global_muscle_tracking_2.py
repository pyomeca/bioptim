"""
Test for file IO
"""

from bioptim import OdeSolver, MusclesBiorbdModel, SolutionMerge
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_excitation_with_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    # Define the problem
    use_residual_torque = True
    with_excitation = True
    model_path = bioptim_folder + "/models/arm26.bioMod"
    bio_model = MusclesBiorbdModel(
        model_path, with_residual_torque=use_residual_torque, with_excitation=with_excitation
    )
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(bio_model, final_time, n_shooting)

    bio_model = MusclesBiorbdModel(
        model_path, with_residual_torque=use_residual_torque, with_excitation=with_excitation
    )  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        bio_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: bio_model.nb_q, :].T,
        use_residual_torque=use_residual_torque,
        kin_data_to_track="markers",
        ode_solver=ode_solver(),
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (50 * 5, 1))
        npt.assert_almost_equal(g, np.zeros((50 * 5, 1)), decimal=6)
    else:
        npt.assert_equal(g.shape, (50, 1))
        npt.assert_almost_equal(g, np.zeros((50, 1)), decimal=6)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, mus_states = states["q"], states["qdot"], states["muscles"]
    tau, mus_controls = controls["tau"], controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        npt.assert_almost_equal(f[0, 0], 1.9423215393458834e-05)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-0.0022161, -0.00062983]))
        npt.assert_almost_equal(q[:, -1], np.array([0.20632374, -0.96266977]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([-0.02544063, 1.11230153]))
        npt.assert_almost_equal(qdot[:, -1], np.array([0.23632381, -9.11739593]))
        # initial and final muscle state
        npt.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        npt.assert_almost_equal(
            mus_states[:, -1], np.array([0.5193721, 0.50851183, 0.6051374, 0.43719123, 0.59329003, 0.59971324])
        )
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([6.66733699e-05, 6.40935259e-06]))
        npt.assert_almost_equal(tau[:, -1], np.array([5.67398982e-05, -5.00305009e-05]))
        npt.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76677771, 0.02174135, 0.633964, 0.74879614, 0.49849973, 0.22512206])
        )
        npt.assert_almost_equal(
            mus_controls[:, -1], np.array([0.44112329, 0.43426359, 0.61784926, 0.51301095, 0.65031982, 0.60125901])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        npt.assert_almost_equal(f[0, 0], 1.376879930342943e-05)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-0.0014234, -0.00147485]))
        npt.assert_almost_equal(q[:, -1], np.array([0.20339005, -0.95861425]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([-0.04120767, 1.11166648]))
        npt.assert_almost_equal(qdot[:, -1], np.array([0.17457134, -8.99660355]))
        # initial and final muscle state
        npt.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        npt.assert_almost_equal(
            mus_states[:, -1], np.array([0.52076927, 0.50803185, 0.6049856, 0.43736942, 0.59338758, 0.59927582])
        )
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([4.63258794e-05, 2.39522172e-05]))
        npt.assert_almost_equal(tau[:, -1], np.array([-2.86456641e-08, 8.63101439e-08]))
        npt.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76819928, 0.02175646, 0.6339027, 0.74872788, 0.49847323, 0.22487671])
        )
        npt.assert_almost_equal(
            mus_controls[:, -1], np.array([0.44183311, 0.43401359, 0.61776037, 0.51314242, 0.65039128, 0.60103257])
        )

    elif ode_solver == OdeSolver.RK4:
        npt.assert_almost_equal(f[0, 0], 1.9563634639504918e-05)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-0.00224285, -0.00055806]))
        npt.assert_almost_equal(q[:, -1], np.array([0.2062965, -0.96260434]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([-0.02505696, 1.11102856]))
        npt.assert_almost_equal(qdot[:, -1], np.array([0.23472979, -9.11352557]))
        # initial and final muscle state
        npt.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        npt.assert_almost_equal(
            mus_states[:, -1], np.array([0.51939095, 0.50853479, 0.6051437, 0.4371827, 0.59328557, 0.59973123])
        )
        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array([6.72188259e-05, 5.01548712e-06]))
        npt.assert_almost_equal(tau[:, -1], np.array([5.74813746e-05, -5.17061496e-05]))
        npt.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76676674, 0.02172467, 0.63396249, 0.74880157, 0.49850197, 0.22513888])
        )
        npt.assert_almost_equal(
            mus_controls[:, -1], np.array([0.44110908, 0.43426931, 0.6178526, 0.51300672, 0.65031742, 0.60126635])
        )

    else:
        raise ValueError("Test not ready")

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_excitation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    bio_model = MusclesBiorbdModel(model_path, with_residual_torque=False, with_excitation=True)
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(10)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(bio_model, final_time, n_shooting)

    bio_model = MusclesBiorbdModel(
        model_path, with_residual_torque=False, with_excitation=True
    )  # To allow for non free variable, the model must be reloaded
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
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (50 * 5, 1))
        npt.assert_almost_equal(g, np.zeros((50 * 5, 1)), decimal=6)
    else:
        npt.assert_equal(g.shape, (50, 1))
        npt.assert_almost_equal(g, np.zeros((50, 1)), decimal=6)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, mus_states, mus_controls = states["q"], states["qdot"], states["muscles"], controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        npt.assert_almost_equal(f[0, 0], 1.9426861462787857e-05)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-0.00221826, -0.00062423]))
        npt.assert_almost_equal(q[:, -1], np.array([0.20632271, -0.96266717]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([-0.02535376, 1.11208698]))
        npt.assert_almost_equal(qdot[:, -1], np.array([0.2362817, -9.11728306]))
        # initial and final muscle state
        npt.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        npt.assert_almost_equal(
            mus_states[:, -1], np.array([0.51937086, 0.50851233, 0.60513767, 0.43719097, 0.59328989, 0.59971401])
        )
        # initial and final controls
        npt.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76677683, 0.02174148, 0.63396384, 0.74879658, 0.49849991, 0.22512315])
        )
        npt.assert_almost_equal(
            mus_controls[:, -1], np.array([0.44112273, 0.43426381, 0.61784939, 0.51301078, 0.65031973, 0.6012593])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        npt.assert_almost_equal(f[0, 0], 1.37697154e-05)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-0.00142406, -0.0014732]))
        npt.assert_almost_equal(q[:, -1], np.array([0.20338897, -0.95861153]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([-0.04117675, 1.11159817]))
        npt.assert_almost_equal(qdot[:, -1], np.array([0.17454593, -8.99653212]))
        # initial and final muscle state
        npt.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        npt.assert_almost_equal(
            mus_states[:, -1], np.array([0.52076916, 0.50803189, 0.60498562, 0.43736941, 0.59338757, 0.59927589])
        )
        # initial and final controls
        npt.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76819906, 0.02175649, 0.63390265, 0.74872802, 0.49847329, 0.22487699])
        )
        npt.assert_almost_equal(
            mus_controls[:, -1], np.array([0.44183311, 0.43401359, 0.61776037, 0.51314242, 0.65039128, 0.60103257])
        )

    elif ode_solver == OdeSolver.RK4:
        npt.assert_almost_equal(f[0, 0], 1.956741750742022e-05)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array([-0.00224508, -0.00055229]))
        npt.assert_almost_equal(q[:, -1], np.array([0.20629545, -0.96260166]))
        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array([-0.02496724, 1.11080593]))
        npt.assert_almost_equal(qdot[:, -1], np.array([0.23468676, -9.11340941]))
        # initial and final muscle state
        npt.assert_almost_equal(
            mus_states[:, 0], np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701, 0.22479665])
        )
        npt.assert_almost_equal(
            mus_states[:, -1], np.array([0.51938967, 0.50853531, 0.60514397, 0.43718244, 0.59328542, 0.59973203])
        )
        # initial and final controls
        npt.assert_almost_equal(
            mus_controls[:, 0], np.array([0.76676586, 0.02172479, 0.63396233, 0.74880202, 0.49850216, 0.22514])
        )
        npt.assert_almost_equal(
            mus_controls[:, -1], np.array([0.44110851, 0.43426954, 0.61785274, 0.51300655, 0.65031733, 0.60126665])
        )

    else:
        raise ValueError("Test not implemented")

    # simulate
    TestUtils.simulate(sol, decimal_value=6)
