"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np
import biorbd

from bioptim import Data, OdeSolver
from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.IRK])
def test_muscle_activations_and_states_tracking(ode_solver):
    # Load muscle_activations_tracker
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "muscle_activations_tracker", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/muscle_activations_tracker.py"
    )
    muscle_activations_tracker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muscle_activations_tracker)

    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 2
    nb_shooting = 9
    use_residual_torque = True

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_activations_ref = muscle_activations_tracker.generate_data(
        biorbd_model, final_time, nb_shooting, use_residual_torque=use_residual_torque
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = muscle_activations_tracker.prepare_ocp(
        biorbd_model,
        final_time,
        nb_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 6.518854595660012e-06)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (36, 1))
    np.testing.assert_almost_equal(g, np.zeros((36, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-9.11292790e-06, -9.98708184e-06]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.49388008, -1.4492482]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-1.58428412e-04, -6.69634564e-05]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.87809776, -2.64745571]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-6.89985946e-07,  8.85124432e-06]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-7.38474471e-06, -5.12994471e-07]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37442763, 0.95074155, 0.73202163, 0.59858471, 0.15595214, 0.15596623])
        )
        np.testing.assert_almost_equal(
            mus[:, -1], np.array([0.54685822, 0.18481451, 0.96949193, 0.77512584, 0.93948978, 0.89483523])
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
        np.testing.assert_almost_equal(tau[:, -1], np.array([-7.6255276e-06, -5.1947040e-07]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.3744008, 0.9507489, 0.7320295, 0.5985624, 0.1559316, 0.1559573])
        )
        np.testing.assert_almost_equal(
            mus[:, -1], np.array([0.5468632, 0.184813, 0.969489, 0.7751258, 0.9394897, 0.8948353])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.IRK])
def test_muscle_activation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_activations_tracker
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "muscle_activations_tracker", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/muscle_activations_tracker.py"
    )
    muscle_activations_tracker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muscle_activations_tracker)

    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 2
    nb_shooting = 9
    use_residual_torque = False

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_activations_ref = muscle_activations_tracker.generate_data(
        biorbd_model, final_time, nb_shooting, use_residual_torque=use_residual_torque
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = muscle_activations_tracker.prepare_ocp(
        biorbd_model,
        final_time,
        nb_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :],
        use_residual_torque=use_residual_torque,
        kin_data_to_track="q",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 6.5736277330517424e-06)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (36, 1))
    np.testing.assert_almost_equal(g, np.zeros((36, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, mus = states["q"], states["q_dot"], controls["muscles"]

    if ode_solver == OdeSolver.IRK:
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
            mus[:, -1], np.array([0.5468617 , 0.18481307, 0.96948995, 0.77512646, 0.93949036, 0.89483428])
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
            mus[:, -1], np.array([0.54686681, 0.18481157, 0.969487, 0.7751264, 0.9394903, 0.89483438])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.IRK])
def test_muscle_excitation_with_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "muscle_excitations_tracker", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/muscle_excitations_tracker.py"
    )
    muscle_excitations_tracker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muscle_excitations_tracker)

    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.5
    nb_shooting = 9

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = muscle_excitations_tracker.generate_data(
        biorbd_model, final_time, nb_shooting
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = muscle_excitations_tracker.prepare_ocp(
        biorbd_model,
        final_time,
        nb_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        use_residual_torque=True,
        kin_data_to_track="markers",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (90, 1))
    np.testing.assert_almost_equal(g, np.zeros((90, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, mus_states, tau, mus_controls = (
        states["q"],
        states["q_dot"],
        states["muscles"],
        controls["tau"],
        controls["muscles"],
    )

    if ode_solver == OdeSolver.IRK:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 7.64035317916434e-09)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([3.38768030e-05, -1.00458812e-04]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08513496, -0.49730436]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00101809, -0.00148609]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13564738, -1.56108483]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.49815448, 0.460973, 0.50258145, 0.47532674, 0.48106099, 0.52567983])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54297843, 0.31086581, 0.94644988, 0.77140285, 0.91816972, 0.88113678])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-6.40167162e-08,  1.62285537e-07]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-1.68679431e-07,  2.67056112e-07]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37454151, 0.95068817, 0.7319911 , 0.59865476, 0.15602333, 0.15599978])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54671449, 0.18485798, 0.96954483, 0.77512852, 0.93947848, 0.89481325])
        )
    else:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 2.536225590351395e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([7.2597670e-05, -1.8666436e-04]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.0852988, -0.4977459]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-7.0744650e-05, -2.4131474e-02]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.1348011, -1.5615855]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.3838763, 0.4606279, 0.5100601, 0.4521726, 0.339458, 0.5174455])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.5433766, 0.3112462, 0.9468205, 0.7713799, 0.9186436, 0.8810834])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-2.6018243e-07, 3.4374659e-07]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-8.316368e-07, 1.345901e-06]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.3745365, 0.9506899, 0.731993, 0.598652, 0.1560212, 0.1559999])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.5467358, 0.1848544, 0.9695426, 0.7751313, 0.939481, 0.8948065])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # Simulate (for some reason the next step passes in Run but not in Debug and not in all run...)
    # with pytest.raises(AssertionError, match="Arrays are not almost equal to 7 decimals"):
    #     TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.IRK])
def test_muscle_excitation_no_residual_torque_and_markers_tracking(ode_solver):
    # Load muscle_excitations_tracker
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "muscle_excitations_tracker", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/muscle_excitations_tracker.py"
    )
    muscle_excitations_tracker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muscle_excitations_tracker)

    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.5
    nb_shooting = 9

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = muscle_excitations_tracker.generate_data(
        biorbd_model, final_time, nb_shooting
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = muscle_excitations_tracker.prepare_ocp(
        biorbd_model,
        final_time,
        nb_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        use_residual_torque=False,
        kin_data_to_track="markers",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (90, 1))
    np.testing.assert_almost_equal(g, np.zeros((90, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, mus_states, mus_controls = (states["q"], states["q_dot"], states["muscles"], controls["muscles"])

    if ode_solver == OdeSolver.IRK:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 7.641292697695181e-09)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([3.38780701e-05, -1.00463359e-04]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08513497, -0.49730438]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00101862, -0.00148706]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13564749, -1.56108504]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.4981557 , 0.4609733 , 0.50258196, 0.47532579, 0.48105234, 0.52567875])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54297843, 0.31086581, 0.94644988, 0.77140285, 0.91816972, 0.88113678])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37454151, 0.95068817, 0.7319911 , 0.59865476, 0.15602333, 0.15599978])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54671449, 0.18485798, 0.96954483, 0.77512852, 0.93947848, 0.89481325])
        )
    else:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 2.5364022572768427e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([7.2600707e-05, -1.8667459e-04]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.0852988, -0.497746]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-6.9251708e-05, -2.4134658e-02]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.1348016, -1.5615864]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.3838776, 0.4606282, 0.5100613, 0.4521704, 0.3394338, 0.5174442])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.5433766, 0.3112462, 0.9468205, 0.7713799, 0.9186436, 0.8810834])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.3745365, 0.9506899, 0.731993, 0.598652, 0.1560212, 0.1559999])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.5467358, 0.1848544, 0.9695426, 0.7751313, 0.939481, 0.8948065])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.IRK])
def test_muscle_activation_and_contacts_tracking(ode_solver):
    # Load muscle_activations_contact_tracker
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "muscle_activations_contact_tracker",
        str(PROJECT_FOLDER) + "/examples/muscle_driven_with_contact/muscle_activations_contacts_tracker.py",
    )
    muscle_activations_contact_tracker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muscle_activations_contact_tracker)

    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_with_contact/2segments_4dof_2contacts_1muscle.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.3
    nb_shooting = 10

    # Generate random data to fit
    contact_forces_ref = np.array(
        [
            [
                -81.76167127,
                -69.61586405,
                -35.68564618,
                -15.37939873,
                -15.79649488,
                -19.48643318,
                -24.83072827,
                -31.37006652,
                -38.72133782,
                -45.3732221,
            ],
            [
                51.10694314,
                52.00705138,
                57.14841462,
                63.25205608,
                65.99940832,
                67.33152066,
                66.99052864,
                64.46060997,
                59.00664793,
                50.29377455,
            ],
            [
                158.47794037,
                139.07750225,
                89.50719005,
                59.7699281,
                55.18121509,
                53.45748305,
                52.5388107,
                51.95213223,
                51.51348129,
                50.34932116,
            ],
        ]
    )
    muscle_activations_ref = np.array(
        [
            [
                0.49723853,
                0.49488324,
                0.50091057,
                0.51505782,
                0.53542531,
                0.56369329,
                0.60171651,
                0.64914307,
                0.70026122,
                0.47032099,
                0.47032099,
            ]
        ]
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = muscle_activations_contact_tracker.prepare_ocp(
        model_path, final_time, nb_shooting, muscle_activations_ref[:, :-1], contact_forces_ref, ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 7.06749952e-11)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (80, 1))
    np.testing.assert_almost_equal(g, np.zeros((80, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau, mus_controls = (states["q"], states["q_dot"], controls["tau"], controls["muscles"])

    if ode_solver == OdeSolver.IRK:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.27775865,  0.13016297, -0.12645823,  0.12645823]), decimal=2)
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-1.41984379,  0.1805142 ,  2.86254555, -2.86254555]), decimal=2)
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-1.26664911,   1.18000589,  -0.34622618, -54.82441356]), decimal=2)
        np.testing.assert_almost_equal(tau[:, -1], np.array([12.05861622,  19.79097096,   3.16521777, -21.36292473]), decimal=2)
        np.testing.assert_almost_equal(mus_controls[:, 0], np.array([0.49723861]))
        np.testing.assert_almost_equal(mus_controls[:, -1], np.array([0.47025091]))
    else:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.2778512, 0.1301747, -0.1262716, 0.1262716]), decimal=2)
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([-1.4213149, 0.1804316, 2.8654435, -2.8654435]), decimal=2)
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-1.2645016, 1.1780052, -0.3456392, -54.8244136]), decimal=2)
        np.testing.assert_almost_equal(tau[:, -1], np.array([12.0122738, 19.7590172, 3.1561865, -21.3647429]),
                                       decimal=2)
        np.testing.assert_almost_equal(mus_controls[:, 0], np.array([0.4972386]))
        np.testing.assert_almost_equal(mus_controls[:, -1], np.array([0.4702531]))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)
