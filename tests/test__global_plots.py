"""
Test for file IO
"""
import io
import sys
import os
import pytest

from casadi import Function, MX
import numpy as np
import biorbd_casadi as biorbd
from bioptim import OptimalControlProgram, CostType, OdeSolver, Solver
from bioptim.limits.penalty import PenaltyOption

import matplotlib

matplotlib.use("Agg")


def test_plot_graphs_one_phase():
    # Load graphs_one_phase
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


def test_plot_merged_graphs():
    # Load graphs_one_phase
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

    biorbd_model = biorbd.Model(model_path)  # To prevent from free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        ode_solver=OdeSolver.RK4(),
        use_residual_torque=True,
        kin_data_to_track="markers",
    )
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


def test_plot_graphs_multi_phases():
    # Load graphs_one_phase
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(biorbd_model_path=bioptim_folder + "/models/cube.bioMod")
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


def test_add_new_plot():
    # Load graphs_one_phase
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=20,
        final_time=0.5,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1)
    sol = ocp.solve(solver)

    # Saving/loading files reset the plot settings to normal
    save_name = "test_plot.bo"
    ocp.save(sol, save_name)

    # Test 1 - Working plot
    ocp.add_plot("My New Plot", lambda t, x, u, p: x[0:2, :])
    sol.graphs(automatically_organize=False)

    # Test 2 - Combine using combine_to is not allowed
    ocp, sol = OptimalControlProgram.load(save_name)
    with pytest.raises(RuntimeError):
        ocp.add_plot("My New Plot", lambda t, x, u, p: x[0:2, :], combine_to="NotAllowed")

    # Test 3 - Create a completely new plot
    ocp, sol = OptimalControlProgram.load(save_name)
    ocp.add_plot("My New Plot", lambda t, x, u, p: x[0:2, :])
    ocp.add_plot("My Second New Plot", lambda t, x, p, u: x[0:2, :])
    sol.graphs(automatically_organize=False)

    # Test 4 - Combine to the first using fig_name
    ocp, sol = OptimalControlProgram.load(save_name)
    ocp.add_plot("My New Plot", lambda t, x, u, p: x[0:2, :])
    ocp.add_plot("My New Plot", lambda t, x, u, p: x[0:2, :])
    sol.graphs(automatically_organize=False)

    # Add the plot of objectives and constraints to this mess
    ocp.add_plot_penalty(CostType.ALL)
    sol.graphs(automatically_organize=False)

    # Delete the saved file
    os.remove(save_name)


def test_console_objective_functions():
    # Load graphs_one_phase
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(biorbd_model_path=bioptim_folder + "/models/cube.bioMod")
    sol = ocp.solve()
    ocp = sol.ocp  # We will override ocp with known and controlled values for the test

    sol.constraints = np.array([range(sol.constraints.shape[0])]).T / 10
    # Create some consistent answer
    sol.solver_time_to_optimize = 1.2345
    sol.real_time_to_optimize = 5.4321

    def override_penalty(pen: list[PenaltyOption]):
        for cmp, p in enumerate(pen):
            if p:
                name = p.name.replace("->", "_").replace(" ", "_")
                x = MX.sym("x", *p.weighted_function.sparsity_in("i0").shape)
                u = MX.sym("u", *p.weighted_function.sparsity_in("i1").shape)
                param = MX.sym("param", *p.weighted_function.sparsity_in("i2").shape)
                weight = MX.sym("weight", *p.weighted_function.sparsity_in("i3").shape)
                target = MX.sym("target", *p.weighted_function.sparsity_in("i4").shape)
                dt = MX.sym("dt", *p.weighted_function.sparsity_in("i5").shape)

                p.function = Function(name, [x, u, param], [np.array([range(cmp, len(p.rows) + cmp)]).T])
                p.function_non_threaded = p.function
                p.weighted_function = Function(
                    name, [x, u, param, weight, target, dt], [np.array([range(cmp + 1, len(p.rows) + cmp + 1)]).T]
                )
                p.weighted_function_non_threaded = p.weighted_function

    override_penalty(ocp.g_internal)  # Override constraints in the ocp
    override_penalty(ocp.g)  # Override constraints in the ocp
    override_penalty(ocp.J_internal)  # Override objectives in the ocp
    override_penalty(ocp.J)  # Override objectives in the ocp

    for nlp in ocp.nlp:
        override_penalty(nlp.g_internal)  # Override constraints in the nlp
        override_penalty(nlp.g)  # Override constraints in the nlp
        override_penalty(nlp.J_internal)  # Override objectives in the nlp
        override_penalty(nlp.J)  # Override objectives in the nlp

    captured_output = io.StringIO()  # Create StringIO object
    sys.stdout = captured_output  # and redirect stdout.
    sol.print()
    expected_output = (
        "Solver reported time: 1.2345 sec\n"
        "Real time: 5.4321 sec\n"
        "\n"
        "---- COST FUNCTION VALUES ----\n"
        "PHASE 0\n"
        "MINIMIZE_CONTROL:  60.00 (weighted 120.0)\n"
        "\n"
        "PHASE 1\n"
        "MINIMIZE_CONTROL:  90.00 (weighted 180.0)\n"
        "minimize_difference:  6.00 (weighted 9.0)\n"
        "\n"
        "PHASE 2\n"
        "MINIMIZE_CONTROL:  60.00 (weighted 120.0)\n"
        "\n"
        "Sum cost functions: 429.0\n"
        "------------------------------\n"
        "\n"
        "--------- CONSTRAINTS ---------\n"
        "PHASE 0\n"
        "CONTINUITY: 420.0\n"
        "PHASE_TRANSITION 0->1: 27.0\n"
        "SUPERIMPOSE_MARKERS: 6.0\n"
        "SUPERIMPOSE_MARKERS: 9.0\n"
        "\n"
        "PHASE 1\n"
        "CONTINUITY: 630.0\n"
        "PHASE_TRANSITION 1->2: 27.0\n"
        "SUPERIMPOSE_MARKERS: 6.0\n"
        "\n"
        "PHASE 2\n"
        "CONTINUITY: 420.0\n"
        "SUPERIMPOSE_MARKERS: 6.0\n"
        "\n"
        "------------------------------\n"
    )

    sys.stdout = sys.__stdout__  # Reset redirect.
    assert captured_output.getvalue() == expected_output
