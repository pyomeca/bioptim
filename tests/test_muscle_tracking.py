"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np
import biorbd

from biorbd_optim import ProblemType

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
        show_online_optim=False,
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
    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])

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


# def test_muscle_excitation_and_markers_tracking():
#     # Define the problem
#     model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod"
#     biorbd_model = biorbd.Model(model_path)
#     final_time = 2
#     nb_shooting = 29
#
#     # Generate random data to fit
#     np.random.seed(42)
#     t, markers_ref, x_ref, muscle_activations_ref = muscle_excitations_tracker.generate_data(biorbd_model, final_time, nb_shooting)
#
#     biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
#     ocp = muscle_excitations_tracker.prepare_ocp(
#         biorbd_model,
#         final_time,
#         nb_shooting,
#         markers_ref,
#         muscle_activations_ref,
#         x_ref[: biorbd_model.nbQ(), :].T,
#         show_online_optim=False,
#         kin_data_to_track="markers",
#     )
#     sol = ocp.solve()
#
#     # Check objective function value
#     f = np.array(sol["f"])
#     np.testing.assert_equal(f.shape, (1, 1))
#     np.testing.assert_almost_equal(f[0, 0], 5.5018106033265635e-08)
#
#     # Check constraints
#     g = np.array(sol["g"])
#     np.testing.assert_equal(g.shape, (116, 1))
#     np.testing.assert_almost_equal(g, np.zeros((116, 1)), decimal=6)
#
#     # Check some of the results
#     q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])
#
#     # initial and final position
#     np.testing.assert_almost_equal(q[:, 0], np.array([2.08066097e-05, -6.73880595e-05]))
#     np.testing.assert_almost_equal(q[:, -1], np.array([-0.32841823, -1.77962274]))
#     # initial and final velocities
#     np.testing.assert_almost_equal(qdot[:, 0], np.array([-8.36173846e-05, 2.75315574e-04]))
#     np.testing.assert_almost_equal(qdot[:, -1], np.array([-0.01426175, -1.67709559]))
#     # initial and final controls
#     np.testing.assert_almost_equal(tau[:, 0], np.array([-2.30003787e-08, 1.55065042e-07]))
#     np.testing.assert_almost_equal(tau[:, -1], np.array([-1.51225186e-07, 8.64594768e-08]))
#     np.testing.assert_almost_equal(
#         mus[:, 0], np.array([0.37454857, 0.9505309, 0.73197187, 0.59864926, 0.15606728, 0.15604403]),
#     )
#     np.testing.assert_almost_equal(
#         mus[:, -1], np.array([0.04100615, 0.59088207, 0.67754577, 0.0172683, 0.51209316, 0.22652397]),
#     )
