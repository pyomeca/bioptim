"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import biorbd

from biorbd_optim import ShowResult


# Load graphs_one_phase
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/align_markers.py"
)
graphs_one_phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graphs_one_phase)


def test_plot_graphs_one_phase():
    ocp = graphs_one_phase.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod",
        number_shooting_points=30,
        final_time=2,
        show_online_optim=False,
    )
    sol = ocp.solve()

    plt = ShowResult(ocp, sol)
    plt.graphs(automatically_organize=False)


# Load graphs_one_phase
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_markers", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/muscle_excitations_tracker.py"
)
merged_graphs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merged_graphs)


def test_plot_merged_graphs():
    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.5
    nb_shooting = 9

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = merged_graphs.generate_data(biorbd_model, final_time, nb_shooting)

    biorbd_model = biorbd.Model(model_path)  # To allow for non free variable, the model must be reloaded
    ocp = merged_graphs.prepare_ocp(
        biorbd_model,
        final_time,
        nb_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        show_online_optim=False,
        kin_data_to_track="markers",
    )
    sol = ocp.solve()

    plt = ShowResult(ocp, sol)
    plt.graphs(automatically_organize=False)


# Load graphs_one_phase
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/multiphase_align_markers.py"
)
graphs_multi_phases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graphs_multi_phases)


def test_plot_graphs_multi_phases():
    ocp = graphs_multi_phases.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod"
    )
    sol = ocp.solve()

    plt = ShowResult(ocp, sol)
    plt.graphs(automatically_organize=False)
