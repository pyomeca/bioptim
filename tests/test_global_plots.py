"""
Test for file IO
"""
import io
import sys
import os
import pytest
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import biorbd
from bioptim import ShowResult, OptimalControlProgram


def test_plot_graphs_one_phase():
    # Load graphs_one_phase
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "track_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/track_markers_with_torque_actuators.py"
    )
    graphs_one_phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graphs_one_phase)

    ocp = graphs_one_phase.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod",
        n_shooting=30,
        final_time=2,
    )
    sol = ocp.solve()

    plt = ShowResult(ocp, sol)
    plt.graphs(automatically_organize=False)


def test_plot_merged_graphs():
    # Load graphs_one_phase
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "track_markers", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/muscle_excitations_tracker.py"
    )
    merged_graphs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merged_graphs)

    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 0.5
    n_shooting = 9

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = merged_graphs.generate_data(biorbd_model, final_time, n_shooting)

    biorbd_model = biorbd.Model(model_path)  # To prevent from non free variable, the model must be reloaded
    ocp = merged_graphs.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        use_residual_torque=True,
        kin_data_to_track="markers",
    )
    sol = ocp.solve()

    plt = ShowResult(ocp, sol)
    plt.graphs(automatically_organize=False)


def test_plot_graphs_multi_phases():
    # Load graphs_one_phase
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "track_markers", str(PROJECT_FOLDER) + "/examples/getting_started/example_multiphase.py"
    )
    graphs_multi_phases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graphs_multi_phases)

    ocp = graphs_multi_phases.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod"
    )
    sol = ocp.solve()

    plt = ShowResult(ocp, sol)
    plt.graphs(automatically_organize=False)


def test_add_new_plot():
    # Load graphs_one_phase
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "track_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/track_markers_with_torque_actuators.py"
    )
    graphs_one_phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graphs_one_phase)

    ocp = graphs_one_phase.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod",
        n_shooting=20,
        final_time=0.5,
    )
    sol = ocp.solve(solver_options={"max_iter": 1})

    # Saving/loading files reset the plot settings to normal
    save_name = "test_plot.bo"
    ocp.save(sol, save_name)

    # Test 1 - Working plot
    ocp.add_plot("My New Plot", lambda x, u, p: x[0:2, :])
    ShowResult(ocp, sol).graphs(automatically_organize=False)

    # Test 2 - Combine using combine_to is not allowed
    ocp, sol = OptimalControlProgram.load(save_name)
    with pytest.raises(RuntimeError):
        ocp.add_plot("My New Plot", lambda x, u, p: x[0:2, :], combine_to="NotAllowed")

    # Test 3 - Create a completely new plot
    ocp, sol = OptimalControlProgram.load(save_name)
    ocp.add_plot("My New Plot", lambda x, u, p: x[0:2, :])
    ocp.add_plot("My Second New Plot", lambda x, p, u: x[0:2, :])
    ShowResult(ocp, sol).graphs(automatically_organize=False)

    # Test 4 - Combine to the first using fig_name
    ocp, sol = OptimalControlProgram.load(save_name)
    ocp.add_plot("My New Plot", lambda x, u, p: x[0:2, :])
    ocp.add_plot("My New Plot", lambda x, u, p: x[0:2, :])
    ShowResult(ocp, sol).graphs(automatically_organize=False)

    # Delete the saved file
    os.remove(save_name)


def test_console_objective_functions():
    # Load graphs_one_phase
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "track_markers", str(PROJECT_FOLDER) + "/examples/getting_started/example_multiphase.py"
    )
    graphs_multi_phases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graphs_multi_phases)

    ocp = graphs_multi_phases.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/getting_started/cube.bioMod"
    )
    sol = ocp.solve()

    plt = ShowResult(ocp, sol)
    plt.graphs(automatically_organize=False)

    sol.constraints = np.array([range(sol.constraints.shape[0])]).T / 10
    captured_output = io.StringIO()  # Create StringIO object
    sys.stdout = captured_output  # and redirect stdout.
    plt.objective_functions()
    plt.constraints()
    expected_output = (
        "\n---- COST FUNCTION VALUES ----\n"
        "PHASE 0\n"
        "MINIMIZE_TORQUE: 1939.759593984963 (weighted 19397.6)\n\n"
        "PHASE 1\n"
        "MINIMIZE_TORQUE: 2887.6596407119023 (weighted 48127.7)\n\n"
        "PHASE 2\n"
        "MINIMIZE_TORQUE: 1927.9784849624061 (weighted 38559.6)\n\n"
        "Sum cost functions: 106085\n"
        "------------------------------\n\n"
        "--------- CONSTRAINTS ---------\n"
        "CONTINUITY 0: 1.5\n"
        "CONTINUITY 1: 5.1\n"
        "CONTINUITY 2: 8.7\n"
        "PHASE_TRANSITION 0->1: 12.3\n"
        "PHASE_TRANSITION 1->2: 15.9\n\n"
        "PHASE 0\n"
        "SUPERIMPOSE_MARKERS: 9.3\n"
        "SUPERIMPOSE_MARKERS: 10.2\n\n"
        "PHASE 1\n"
        "SUPERIMPOSE_MARKERS: 11.100000000000001\n\n"
        "PHASE 2\n"
        "SUPERIMPOSE_MARKERS: 12.0\n\n"
        "------------------------------\n"
    )

    sys.stdout = sys.__stdout__  # Reset redirect.
    assert captured_output.getvalue() == expected_output
