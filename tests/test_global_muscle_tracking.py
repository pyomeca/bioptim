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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
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
    if ode_solver == OdeSolver.RK8:
        np.testing.assert_almost_equal(f[0, 0], 6.340821289366818e-06)
    else:
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
        np.testing.assert_almost_equal(tau[:, 0], np.array([-6.89985946e-07, 8.85124432e-06]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-7.38474471e-06, -5.12994471e-07]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37442763, 0.95074155, 0.73202163, 0.59858471, 0.15595214, 0.15596623])
        )
        np.testing.assert_almost_equal(
            mus[:, -1], np.array([0.54685822, 0.18481451, 0.96949193, 0.77512584, 0.93948978, 0.89483523])
        )

    elif ode_solver == OdeSolver.RK8:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-1.20296925e-05, -1.42883927e-05]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.49387969, -1.44924798]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([-6.75664553e-05, -1.59537195e-04]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.87809983, -2.64745432]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-1.56121402e-06, 1.32347911e-05]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-7.48770006e-06, -5.90970158e-07]))
        np.testing.assert_almost_equal(
            mus[:, 0], np.array([0.37438764, 0.95075245, 0.73203411, 0.59854825, 0.15591868, 0.15595168])
        )
        np.testing.assert_almost_equal(
            mus[:, -1], np.array([0.5468589, 0.18481491, 0.96949149, 0.77512487, 0.93948887, 0.89483671])
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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
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
    if ode_solver == OdeSolver.RK8:
        np.testing.assert_almost_equal(f[0, 0], 6.39401362889915e-06)
    else:
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
            mus[:, -1], np.array([0.5468617, 0.18481307, 0.96948995, 0.77512646, 0.93949036, 0.89483428])
        )

    elif ode_solver == OdeSolver.RK8:
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
            mus[:, -1], np.array([0.5468624, 0.18481347, 0.9694895, 0.77512549, 0.93948945, 0.89483576])
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


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
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
        np.testing.assert_almost_equal(f[0, 0], 7.972968350373634e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00026202, 0.00156288]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08504491, -0.49686238]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00924123, -0.08190647]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13519926, -1.55858184]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54298145, 0.31086508, 0.94645056, 0.77140088, 0.91816806, 0.88114157])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-2.88086841e-06, -1.36959531e-06]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-5.36515407e-07, -4.82461091e-07]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37477686, 0.95063206, 0.73196617, 0.5986751, 0.15605979, 0.15600775])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54671768, 0.18485764, 0.96954556, 0.77512657, 0.93947675, 0.89481789])
        )

    elif ode_solver == OdeSolver.RK8:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 7.972968350373634e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00027549, 0.00159852]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08503251, -0.49683122]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00937424, -0.08221644]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13522705, -1.5585373]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.5428851, 0.31087169, 0.94651899, 0.7714208, 0.91824435, 0.88120097])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-2.83698484e-06, -1.43349514e-06]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-4.77991578e-07, -5.62286935e-07]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37477631, 0.95063219, 0.73196594, 0.59867599, 0.15606103, 0.15600786])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.5467161, 0.18485794, 0.9695457, 0.77512636, 0.93947657, 0.89481839])
        )

    else:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 3.5086270922948964e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.0002031, 0.00078964]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08522694, -0.49749536]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00926103, -0.07289721]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13449917, -1.56033745]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54337182, 0.31124728, 0.94682117, 0.77137855, 0.91864243, 0.88108674])
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-4.21034161e-07, -1.00044103e-06]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-7.49737010e-07, 8.27732267e-07]))
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37458704, 0.95067302, 0.73198322, 0.5986696, 0.15604883, 0.15600538])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54673089, 0.1848553, 0.96954337, 0.77512999, 0.93947978, 0.89480973])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # with pytest.raises(AssertionError, match="Arrays are not almost equal to 7 decimals"):
    #     TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
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
        np.testing.assert_almost_equal(f[0, 0], 7.973265397440505e-07)

        # initial and final position
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00026201, 0.00156289]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08504492, -0.49686236]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0092411, -0.08190665]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.1351993, -1.55858163]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54298146, 0.31086508, 0.94645056, 0.77140088, 0.91816806, 0.88114157])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37477688, 0.95063206, 0.73196617, 0.5986751, 0.15605979, 0.15600775])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54671768, 0.18485764, 0.96954556, 0.77512657, 0.93947675, 0.89481789])
        )

    elif ode_solver == OdeSolver.RK8:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 7.973265397440505e-07)

        # initial and final position
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00027548, 0.00159855]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08503251, -0.4968312]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00937414, -0.08221669]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13522706, -1.55853703]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.5428851, 0.31087169, 0.94651899, 0.7714208, 0.91824435, 0.88120097])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37477632, 0.95063218, 0.73196593, 0.59867599, 0.15606103, 0.15600786])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.54671611, 0.18485793, 0.9695457, 0.77512636, 0.93947657, 0.89481839])
        )

    else:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 3.5087093735149467e-07)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([-0.00020308, 0.00078961]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.08522697, -0.49749541]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.00926096, -0.07289709]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13449945, -1.56033796]))
        # initial and final muscle state
        np.testing.assert_almost_equal(
            mus_states[:, 0], np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452])
        )
        np.testing.assert_almost_equal(
            mus_states[:, -1], np.array([0.54337182, 0.31124727, 0.94682117, 0.77137855, 0.91864243, 0.88108674])
        )
        # initial and final controls
        np.testing.assert_almost_equal(
            mus_controls[:, 0], np.array([0.37458705, 0.95067302, 0.73198322, 0.59866959, 0.15604883, 0.15600538])
        )
        np.testing.assert_almost_equal(
            mus_controls[:, -1], np.array([0.5467309, 0.1848553, 0.96954337, 0.77512999, 0.93947978, 0.89480973])
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
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
    final_time = 0.1
    nb_shooting = 5

    # Generate random data to fit
    np.random.seed(42)
    contact_forces_ref = np.random.rand(biorbd_model.nbContacts(), nb_shooting)
    muscle_activations_ref = np.random.rand(biorbd_model.nbMuscles(), nb_shooting + 1)

    ocp = muscle_activations_contact_tracker.prepare_ocp(
        model_path,
        final_time,
        nb_shooting,
        muscle_activations_ref[:, :-1],
        contact_forces_ref,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 1.2080146471135251)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau, mus_controls = (states["q"], states["q_dot"], controls["tau"], controls["muscles"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.01785865, -0.01749107, -0.8, 0.8]), decimal=5)
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.5199767, -0.535388, -1.49267023, 1.4926703]), decimal=5)
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([5.3773376, 127.6205162, -21.9933179, 1.3644034]), decimal=5)
    np.testing.assert_almost_equal(tau[:, -1], np.array([57.203734, 72.3153286, -7.4076227, 1.2641681]), decimal=5)
    np.testing.assert_almost_equal(mus_controls[:, 0], np.array([0.18722964]), decimal=5)
    np.testing.assert_almost_equal(mus_controls[:, -1], np.array([0.29591125]), decimal=5)

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)
