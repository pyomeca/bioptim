"""
Test for file IO
"""
import os
import pytest

import numpy as np
import biorbd_casadi as biorbd
from bioptim import OdeSolver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_activations_and_states_tracking(ode_solver):
    # Load muscle_activations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.1
    n_shooting = 5
    use_residual_torque = True

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_activations_ref = ocp_module.generate_data(
        biorbd_model, final_time, n_shooting, use_residual_torque=use_residual_torque
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver(),
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
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        np.testing.assert_almost_equal(f[0, 0], 3.624795808383824e-08)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.26294409e-05, -5.94685627e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.10541975, -0.48577985]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00074118, -0.00036854]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.21473881, 7.26398638]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-3.19231945e-08, 1.78181204e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([2.55285701e-06, -5.12710950e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37451645, 0.95067812, 0.73199474, 0.59864193, 0.15601703, 0.15600089])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.4559321, 0.78521782, 0.19970124, 0.51419847, 0.59238012, 0.04656187])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 3.6846293820760475e-08)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.26294409e-05, -5.94685627e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.10541975, -0.48577985]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00074233, -0.00037249]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.21473503, 7.26397692]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-3.19231945e-08, 1.78181204e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([2.55285701e-06, -5.12710950e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37451633, 0.95067815, 0.73199481, 0.5986417, 0.15601682, 0.15600081])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.4559318, 0.78521793, 0.19970129, 0.51419838, 0.59238004, 0.04656203])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 3.624795808383824e-08)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.24603457e-05, -5.56567245e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.10542008, -0.48578046]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00071319, -0.00034956]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.21476386, 7.26402641]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([7.86364319e-08, 1.43718933e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([2.33336715e-06, -4.52483197e-06]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.3745183, 0.9506776, 0.7319939, 0.59864459, 0.15601947, 0.15600189])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45594578, 0.78521284, 0.19969902, 0.51420259, 0.5923839, 0.04655438])
        )

    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=5)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_activation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_activations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_activations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.1
    n_shooting = 5
    use_residual_torque = False

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_activations_ref = ocp_module.generate_data(
        biorbd_model, final_time, n_shooting, use_residual_torque=use_residual_torque
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver(),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 3.634248634056222e-08)

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
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.26502327e-05, -5.98498658e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.10541969, -0.48577983]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00074251, -0.00036937]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.21474217, 7.26398954]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37451604, 0.95067823, 0.73199494, 0.59864126, 0.15601641, 0.15600064])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45593194, 0.78521787, 0.19970125, 0.51419844, 0.5923801, 0.04656193])
        )
    elif ode_solver == OdeSolver.COLLOCATION:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.26434090e-05, -5.99992755e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.10541971, -0.48577986]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00074381, -0.00037358]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.21473839, 7.26398039]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37451604, 0.95067823, 0.73199495, 0.59864125, 0.1560164, 0.15600064])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45593167, 0.78521797, 0.1997013, 0.51419836, 0.59238002, 0.04656208])
        )

    elif ode_solver == OdeSolver.RK4:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.24679103e-05, -5.63685028e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.10542003, -0.48578047]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00071458, -0.00035055]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-4.21476717, 7.26402945]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.3745179, 0.95067771, 0.7319941, 0.59864394, 0.15601888, 0.15600164])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.45594564, 0.78521289, 0.19969903, 0.51420257, 0.59238388, 0.04655442])
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
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(biorbd_model, final_time, n_shooting)

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
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
        np.testing.assert_almost_equal(f[0, 0], 3.9377280548492226e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00351782, 0.01702219]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.14352637, -0.72030433]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.02984019, -3.91364352]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.67284629, 3.62405443]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51285729, 0.69943619, 0.40390569, 0.48032451, 0.53752346, 0.31437668])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([5.42775569e-05, -3.45713249e-04]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-2.73167136e-05, -3.83494902e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37743387, 0.95055777, 0.73174428, 0.60093014, 0.15924303, 0.15866534])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.4560975, 0.78519158, 0.19973384, 0.51408083, 0.59227422, 0.04659415])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 3.9378422266498184e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00351729, 0.01701928]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.14352497, -0.72030059]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.02972633, -3.91317111]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.6728683, 3.62413508]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51285285, 0.69943161, 0.40390586, 0.48032585, 0.53752527, 0.31437738])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([5.42926592e-05, -3.45716906e-04]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-2.72776735e-05, -3.84479459e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37744597, 0.95044549, 0.73173082, 0.60092211, 0.15932209, 0.15869578])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45609644, 0.78518702, 0.19973488, 0.51408246, 0.59227441, 0.04659677])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 3.9163147567423305e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00352334, 0.01700853]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.14350606, -0.72027301]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.02920952, -3.91032827]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.67351448, 3.62485659]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51283945, 0.6994339, 0.40390624, 0.48031161, 0.53750849, 0.31441088])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([5.44773721e-05, -3.45454293e-04]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-2.68029143e-05, -3.90467765e-05]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37740553, 0.95056685, 0.73174651, 0.60092669, 0.15924254, 0.15856357])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45609247, 0.7851955, 0.19973458, 0.51407787, 0.59227145, 0.04659596])
        )
    else:
        raise ValueError("Test not ready")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.IRK])
def test_muscle_excitation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(biorbd_model, final_time, n_shooting)

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
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
        np.testing.assert_almost_equal(f[0, 0], 3.939617534835209e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00352248, 0.01703644]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.1435249, -0.7202986]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.03023126, -3.91481759]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.67283616, 3.62412467]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51285401, 0.69943683, 0.40390633, 0.48032393, 0.53752275, 0.31437821])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37743017, 0.95055919, 0.73174445, 0.60093176, 0.15924552, 0.15866818])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45609693, 0.78519207, 0.19973399, 0.51408032, 0.59227376, 0.04659447])
        )

    elif ode_solver == OdeSolver.COLLOCATION:
        np.testing.assert_almost_equal(f[0, 0], 3.939731707680551e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00352196, 0.01703354]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.1435235, -0.72029486]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.03011751, -3.91434553]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.6728582, 3.62420546]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51284957, 0.69943225, 0.40390649, 0.48032527, 0.53752456, 0.31437891])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37744227, 0.95044691, 0.73173098, 0.60092373, 0.15932458, 0.15869862])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.45609587, 0.78518751, 0.19973503, 0.51408194, 0.59227395, 0.04659709])
        )

    elif ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(f[0, 0], 3.918210818142734e-05)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00352802, 0.01702281]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.14350458, -0.72026726]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([1.02960131, -3.91150408]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.67350467, 3.62492773]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.51283615, 0.69943454, 0.40390687, 0.48031102, 0.53750777, 0.31441242])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37740184, 0.95056827, 0.73174668, 0.60092831, 0.15924504, 0.15856629])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -2], np.array([0.4560919, 0.785196, 0.19973472, 0.51407736, 0.59227099, 0.04659628])
        )
    else:
        raise ValueError("Test not implemented")

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)

