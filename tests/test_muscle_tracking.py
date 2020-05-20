"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np
import biorbd

from biorbd_optim import Data

# Load muscle_activations_tracker
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "muscle_activations_tracker", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/muscle_activations_tracker.py",
)
muscle_activations_tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(muscle_activations_tracker)

# Load muscle_excitations_tracker
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "muscle_excitations_tracker", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/muscle_excitations_tracker.py",
)
muscle_excitations_tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(muscle_excitations_tracker)

# Load muscle_activations_contact_tracker
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "muscle_activations_contact_tracker",
    str(PROJECT_FOLDER) + "/examples/muscle_driven_with_contact/muscle_activations_contacts_tracker.py",
)
muscle_activations_contact_tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(muscle_activations_contact_tracker)


def test_muscle_activations_and_states_tracking():
    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 2
    nb_shooting = 9

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_activations_ref = muscle_activations_tracker.generate_data(
        biorbd_model, final_time, nb_shooting
    )

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = muscle_activations_tracker.prepare_ocp(
        biorbd_model,
        final_time,
        nb_shooting,
        markers_ref,
        muscle_activations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        kin_data_to_track="q",
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 1.4506639252752042e-06)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (36, 1))
    np.testing.assert_almost_equal(g, np.zeros((36, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([-1.13043502e-05, -1.35629661e-05]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-0.49387966, -1.44924784]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([-8.66527631e-05, -1.31486656e-04]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.8780829, -2.6474387]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-1.55359644e-06, 1.26569700e-05]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-7.41845169e-06, -7.67568954e-07]))
    np.testing.assert_almost_equal(
        mus[:, 0], np.array([0.37439688, 0.95073361, 0.73203047, 0.59855246, 0.15592687, 0.15595739]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1], np.array([0.54685367, 0.18482085, 0.96945157, 0.77512036, 0.93947405, 0.89483397]),
    )


def test_muscle_excitation_and_markers_tracking():
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
        kin_data_to_track="markers",
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 3.254781346397887e-08)

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

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.00025253, -0.00087191]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.08534081, -0.49791254]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([-0.00934186, 0.01178825]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.13487912, -1.5622835]))
    # initial and final muscle state
    np.testing.assert_almost_equal(
        mus_states[:, 0], np.array([0.42904226, 0.46546383, 0.4994946, 0.48864119, 0.55152868, 0.54539094]),
    )
    np.testing.assert_almost_equal(
        mus_states[:, -1], np.array([0.54336665, 0.31127584, 0.94621301, 0.77129814, 0.91831282, 0.88088336]),
    )
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([5.12738229e-08, 1.22781041e-06]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-8.06344799e-07, 1.66654624e-06]))
    np.testing.assert_almost_equal(
        mus_controls[:, 0], np.array([0.37454014, 0.95025616, 0.73193708, 0.59861754, 0.15612896, 0.15609662]),
    )
    np.testing.assert_almost_equal(
        mus_controls[:, -1], np.array([0.54672713, 0.18492921, 0.96888485, 0.77504726, 0.93912423, 0.89459345]),
    )


def test_muscle_activation_and_contacts_tracking():
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
        model_path, final_time, nb_shooting, muscle_activations_ref[:, :-1].T, contact_forces_ref.T,
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
    q, qdot, tau, mus_controls = (
        states["q"],
        states["q_dot"],
        controls["tau"],
        controls["muscles"],
    )

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-0.28197616, 0.13068096, -0.1179598, 0.1179598]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-1.05077202, 0.12452694, 2.11625027, -2.11625027]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-9.97208713, 9.2899607, -2.72577237, -54.82441629]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([61.5351003, 13.60300419, 12.83275737, -21.52321688]))
    np.testing.assert_almost_equal(
        mus_controls[:, 0], np.array([0.49723965]),
    )
    np.testing.assert_almost_equal(
        mus_controls[:, -1], np.array([0.47037075]),
    )
