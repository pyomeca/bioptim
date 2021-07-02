"""
Test for file IO
"""
import pytest
import numpy as np
import biorbd
from bioptim import OdeSolver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_muscle_activations_and_states_tracking(ode_solver):
    # Load muscle_activations_tracker
    bioptim_folder = TestUtils.bioptim_folder()
    tracker = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/muscle_activations_tracker.py")
    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 2
    n_shooting = 9
    use_residual_torque = True

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_activations_ref = tracker.generate_data(
        biorbd_model, final_time, n_shooting, use_residual_torque=use_residual_torque
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = tracker.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if isinstance(ode_solver, OdeSolver.RK8):
        np.testing.assert_almost_equal(f[0, 0], 6.340821289366818e-06)
    else:
        np.testing.assert_almost_equal(f[0, 0], 6.518854595660012e-06)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (36, 1))
    np.testing.assert_almost_equal(g, np.zeros((36, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if isinstance(ode_solver, OdeSolver.IRK):
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.11292790e-06, -9.98708184e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.49388008, -1.4492482]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-1.58428412e-04, -6.69634564e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.87809776, -2.64745571]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-6.89985946e-07, 8.85124432e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-7.38474471e-06, -5.12994471e-07]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37442763, 0.95074155, 0.73202163, 0.59858471, 0.15595214, 0.15596623])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.54685822, 0.18481451, 0.96949193, 0.77512584, 0.93948978, 0.89483523])
        )

    elif isinstance(ode_solver, OdeSolver.RK8):
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.20296925e-05, -1.42883927e-05]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.49387969, -1.44924798]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-6.75664553e-05, -1.59537195e-04]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.87809983, -2.64745432]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-1.56121402e-06, 1.32347911e-05]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-7.48770006e-06, -5.90970158e-07]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37438764, 0.95075245, 0.73203411, 0.59854825, 0.15591868, 0.15595168])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.5468589, 0.18481491, 0.96949149, 0.77512487, 0.93948887, 0.89483671])
        )

    else:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.1123547e-05, -1.2705707e-05]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.4938793, -1.4492479]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-9.0402027e-05, -1.3433204e-04]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.8780898, -2.6474401]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-1.1482641e-06, 1.1539847e-05]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-7.6255276e-06, -5.1947040e-07]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.3744008, 0.9507489, 0.7320295, 0.5985624, 0.1559316, 0.1559573])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.5468632, 0.184813, 0.969489, 0.7751258, 0.9394897, 0.8948353])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_muscle_activation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_activations_tracker
    bioptim_folder = TestUtils.bioptim_folder()
    tracker = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/muscle_activations_tracker.py")
    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 2
    n_shooting = 9
    use_residual_torque = False

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_activations_ref = tracker.generate_data(
        biorbd_model, final_time, n_shooting, use_residual_torque=use_residual_torque
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = tracker.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    if isinstance(ode_solver, OdeSolver.RK8):
        np.testing.assert_almost_equal(f[0, 0], 6.39401362889915e-06)
    else:
        np.testing.assert_almost_equal(f[0, 0], 6.5736277330517424e-06)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (36, 1))
    np.testing.assert_almost_equal(g, np.zeros((36, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus = sol.states["q"], sol.states["qdot"], sol.controls["muscles"]

    if isinstance(ode_solver, OdeSolver.IRK):
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.17149105e-06, -1.00592773e-05]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.49387979, -1.44924811]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-1.58831625e-04, -6.69127853e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.87809239, -2.64744482]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37442688, 0.95074176, 0.73202184, 0.59858414, 0.15595162, 0.155966])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.5468617, 0.18481307, 0.96948995, 0.77512646, 0.93949036, 0.89483428])
        )

    elif isinstance(ode_solver, OdeSolver.RK8):
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.20797525e-05, -1.44483833e-05]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.4938794, -1.44924789]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-6.79694854e-05, -1.59565906e-04]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.87809444, -2.64744336]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37438663, 0.95075271, 0.7320346, 0.5985466, 0.15591717, 0.15595102])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.5468624, 0.18481347, 0.9694895, 0.77512549, 0.93948945, 0.89483576])
        )

    else:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.1123547e-05, -1.2705707e-05]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.49387905, -1.4492478]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-9.07884121e-05, -1.34382832e-04]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.87808434, -2.64742889]))
        # initial and final controls
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37439988, 0.95074914, 0.73202991, 0.598561, 0.15593039, 0.15595677])
        )
        np.testing.assert_almost_equal(
            mus[:, -2], np.array([0.54686681, 0.18481157, 0.969487, 0.7751264, 0.9394903, 0.89483438])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_muscle_excitation_with_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    bioptim_folder = TestUtils.bioptim_folder()
    tracker = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/muscle_excitations_tracker.py")
    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.5
    n_shooting = 9

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = tracker.generate_data(biorbd_model, final_time, n_shooting)

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = tracker.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        use_residual_torque=True,
        kin_data_to_track="markers",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (90, 1))
    np.testing.assert_almost_equal(g, np.zeros((90, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus_states, tau, mus_controls = (
        sol.states["q"],
        sol.states["qdot"],
        sol.states["muscles"],
        sol.controls["tau"],
        sol.controls["muscles"],
    )

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 7.972968350373634e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00025738, 0.00155432]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08502663, -0.49682756]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0091607, -0.08174147]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13524112, -1.55868503]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54298176, 0.310865, 0.94645053, 0.7714009, 0.91816808, 0.88114152])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-2.91199924e-06, -1.34810801e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-5.50139682e-07, -4.73229437e-07]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37477829, 0.95063176, 0.73196614, 0.59867481, 0.1560593, 0.15600768])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.546718, 0.18485758, 0.96954554, 0.7751266, 0.93947678, 0.89481784])
        )

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 7.972968350373634e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00027084, 0.00158996]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08501423, -0.4967964]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00929371, -0.08205146]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13526892, -1.55864048]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54288542, 0.31087161, 0.94651896, 0.77142083, 0.91824438, 0.88120091])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-2.86811951e-06, -1.41200803e-06]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-4.91632371e-07, -5.53045415e-07]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37477774, 0.95063188, 0.73196591, 0.5986757, 0.15606055, 0.15600778])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54671643, 0.18485788, 0.96954568, 0.77512639, 0.93947659, 0.89481833])
        )

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 3.5086270922948964e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00019766, 0.00078078]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08521152, -0.49746311]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00915609, -0.07268497]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13455099, -1.56043294]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54337288, 0.31120388, 0.94682111, 0.77137861, 0.91864248, 0.88108659])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-4.62544381e-07, -9.75433210e-07]))
        np.testing.assert_almost_equal(tau[:, -2], np.array([-7.93739618e-07, 8.51675280e-07]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37458886, 0.95067258, 0.73198315, 0.59866926, 0.15604832, 0.15600496])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54673199, 0.18485512, 0.9695433, 0.77513005, 0.93947984, 0.89480958])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_muscle_excitation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    bioptim_folder = TestUtils.bioptim_folder()
    tracker = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/muscle_excitations_tracker.py")
    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.5
    n_shooting = 9

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = tracker.generate_data(biorbd_model, final_time, n_shooting)

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = tracker.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        use_residual_torque=False,
        kin_data_to_track="markers",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (90, 1))
    np.testing.assert_almost_equal(g, np.zeros((90, 1)), decimal=6)

    # Check some of the results
    q, qdot, mus_states, mus_controls = (
        sol.states["q"],
        sol.states["qdot"],
        sol.states["muscles"],
        sol.controls["muscles"],
    )

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 7.973265397440505e-07)

        # initial and final position
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00025737, 0.00155433]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08502664, -0.49682755]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00916055, -0.08174164]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13524117, -1.55868483]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54298176, 0.310865, 0.94645053, 0.7714009, 0.91816808, 0.88114152])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37477831, 0.95063176, 0.73196614, 0.59867481, 0.1560593, 0.15600768])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.546718, 0.18485758, 0.96954554, 0.7751266, 0.93947678, 0.89481784])
        )

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 7.973265397440505e-07)

        # initial and final position
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00027084, 0.00158998]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08501423, -0.49679638]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0092936, -0.08205169]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13526894, -1.55864023]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54288542, 0.31087161, 0.94651896, 0.77142083, 0.91824438, 0.88120091])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37477776, 0.95063188, 0.73196591, 0.5986757, 0.15606055, 0.15600778])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54671643, 0.18485788, 0.96954568, 0.77512639, 0.93947659, 0.89481833])
        )

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 3.5087093735149467e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00019764, 0.00078075]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08521155, -0.49746317]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.009156, -0.07268483]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13455129, -1.56043348]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54337289, 0.31120388, 0.94682111, 0.77137861, 0.91864248, 0.88108659])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37458887, 0.95067258, 0.73198315, 0.59866926, 0.15604832, 0.15600496])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54673199, 0.18485512, 0.9695433, 0.77513005, 0.93947984, 0.89480958])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)
