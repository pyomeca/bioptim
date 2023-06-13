"""
Test for file IO
"""
import io
import sys
import os
import pytest

from casadi import Function, MX
import numpy as np
from bioptim import OptimalControlProgram, CostType, OdeSolver, Solver, RigidBodyDynamics, BiorbdModel
from bioptim.limits.penalty import PenaltyOption

import matplotlib

matplotlib.use("Agg")


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_plot_graphs_one_phase(assume_phase_dynamics):
    # Load graphs_one_phase
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


def test_plot_check_conditioning():
    # Load graphs check conditioning
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=False,
        assume_phase_dynamics=False,
    )
    ocp.check_conditioning()
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_plot_merged_graphs(assume_phase_dynamics):
    # Load graphs_one_phase
    from bioptim.examples.muscle_driven_ocp import muscle_excitations_tracker as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/arm26.bioMod"
    bio_model = BiorbdModel(model_path)
    final_time = 0.1
    n_shooting = 5

    # Generate random data to fit
    np.random.seed(42)
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(bio_model, final_time, n_shooting)

    bio_model = BiorbdModel(model_path)  # To prevent from free variable, the model must be reloaded
    ocp = ocp_module.prepare_ocp(
        bio_model,
        final_time,
        n_shooting,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: bio_model.nb_q, :].T,
        ode_solver=OdeSolver.RK4(),
        use_residual_torque=True,
        kin_data_to_track="markers",
        assume_phase_dynamics=assume_phase_dynamics,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1)
    sol = ocp.solve(solver)
    sol.graphs(automatically_organize=False)


def test_plot_graphs_multi_phases():
    # Load graphs_one_phase
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod", assume_phase_dynamics=False
    )
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
def test_add_new_plot(assume_phase_dynamics):
    # Load graphs_one_phase
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=20,
        final_time=0.5,
        assume_phase_dynamics=assume_phase_dynamics,
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


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize(
    "rigidbody_dynamics",
    [RigidBodyDynamics.ODE, RigidBodyDynamics.DAE_FORWARD_DYNAMICS, RigidBodyDynamics.DAE_INVERSE_DYNAMICS],
)
def test_plot_graphs_for_implicit_constraints(rigidbody_dynamics, assume_phase_dynamics):
    from bioptim.examples.getting_started import example_implicit_dynamics as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=5,
        final_time=1,
        rigidbody_dynamics=rigidbody_dynamics,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve()
    if sys.platform != "linux":
        sol.graphs(automatically_organize=False)


def test_implicit_example():
    from bioptim.examples.getting_started import example_implicit_dynamics as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    sol_implicit = ocp_module.solve_ocp(
        rigidbody_dynamics=RigidBodyDynamics.DAE_INVERSE_DYNAMICS,
        max_iter=1,
        model_path=bioptim_folder + "/models/pendulum.bioMod",
    )
    sol_semi_explicit = ocp_module.solve_ocp(
        rigidbody_dynamics=RigidBodyDynamics.DAE_FORWARD_DYNAMICS,
        max_iter=1,
        model_path=bioptim_folder + "/models/pendulum.bioMod",
    )
    sol_explicit = ocp_module.solve_ocp(
        rigidbody_dynamics=RigidBodyDynamics.ODE, max_iter=1, model_path=bioptim_folder + "/models/pendulum.bioMod"
    )
    ocp_module.prepare_plots(sol_implicit, sol_semi_explicit, sol_explicit)


def test_console_objective_functions():
    # Load graphs_one_phase
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod", assume_phase_dynamics=False
    )
    sol = ocp.solve()
    ocp = sol.ocp  # We will override ocp with known and controlled values for the test

    sol.constraints = np.array([range(sol.constraints.shape[0])]).T / 10
    # Create some consistent answer
    sol.solver_time_to_optimize = 1.2345
    sol.real_time_to_optimize = 5.4321

    def override_penalty(pen: list[PenaltyOption]):
        for cmp, p in enumerate(pen):
            if p:
                for node_index in p.node_idx:
                    nlp.states.node_index = node_index
                    nlp.states_dot.node_index = node_index
                    nlp.controls.node_index = node_index

                    name = (
                        p.name.replace("->", "_")
                        .replace(" ", "_")
                        .replace("(", "_")
                        .replace(")", "_")
                        .replace(",", "_")
                        .replace(":", "_")
                        .replace(".", "_")
                        .replace("__", "_")
                    )
                    x = MX.sym("x", *p.weighted_function[node_index].sparsity_in("i0").shape)
                    u = MX.sym("u", *p.weighted_function[node_index].sparsity_in("i1").shape)
                    param = MX.sym("param", *p.weighted_function[node_index].sparsity_in("i2").shape)
                    weight = MX.sym("weight", *p.weighted_function[node_index].sparsity_in("i3").shape)
                    target = MX.sym("target", *p.weighted_function[node_index].sparsity_in("i4").shape)
                    dt = MX.sym("dt", *p.weighted_function[node_index].sparsity_in("i5").shape)

                    p.function[node_index] = Function(
                        name, [x, u, param], [np.array([range(cmp, len(p.rows) + cmp)]).T]
                    )
                    p.function_non_threaded[node_index] = p.function[node_index]
                    p.weighted_function[node_index] = Function(
                        name, [x, u, param, weight, target, dt], [np.array([range(cmp + 1, len(p.rows) + cmp + 1)]).T]
                    )
                    p.weighted_function_non_threaded[node_index] = p.weighted_function[node_index]

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
    sol.print_cost()

    if assume_phase_dynamics:
        expected_output = (
            "Solver reported time: 1.2345 sec\n"
            "Real time: 5.4321 sec\n"
            "\n"
            "---- COST FUNCTION VALUES ----\n"
            "PHASE 0\n"
            "Lagrange.MINIMIZE_CONTROL: 120.0 (non weighted  60.00)\n"
            "\n"
            "PHASE 1\n"
            "MultinodeObjectiveFcn.CUSTOM: 6.0 (non weighted  3.00)\n"
            "Lagrange.MINIMIZE_CONTROL: 180.0 (non weighted  90.00)\n"
            "\n"
            "PHASE 2\n"
            "Lagrange.MINIMIZE_CONTROL: 120.0 (non weighted  60.00)\n"
            "\n"
            "Sum cost functions: 426.0\n"
            "------------------------------\n"
            "\n"
            "--------- CONSTRAINTS ---------\n"
            "PHASE 0\n"
            "ConstraintFcn.CONTINUITY: 420.0\n"
            "PhaseTransitionFcn.CONTINUOUS: 27.0\n"
            "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
            "ConstraintFcn.SUPERIMPOSE_MARKERS: 9.0\n"
            "\n"
            "PHASE 1\n"
            "ConstraintFcn.CONTINUITY: 630.0\n"
            "PhaseTransitionFcn.CONTINUOUS: 27.0\n"
            "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
            "\n"
            "PHASE 2\n"
            "ConstraintFcn.CONTINUITY: 420.0\n"
            "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
            "\n"
            "------------------------------\n"
        )
    else:
        expected_output = (
            "Solver reported time: 1.2345 sec\n"
            "Real time: 5.4321 sec\n"
            "\n"
            "---- COST FUNCTION VALUES ----\n"
            "PHASE 0\n"
            "Lagrange.MINIMIZE_CONTROL: 120.0 (non weighted  60.00)\n"
            "\n"
            "PHASE 1\n"
            "MultinodeObjectiveFcn.CUSTOM: 6.0 (non weighted  3.00)\n"
            "Lagrange.MINIMIZE_CONTROL: 180.0 (non weighted  90.00)\n"
            "\n"
            "PHASE 2\n"
            "Lagrange.MINIMIZE_CONTROL: 120.0 (non weighted  60.00)\n"
            "\n"
            "Sum cost functions: 426.0\n"
            "------------------------------\n"
            "\n"
            "--------- CONSTRAINTS ---------\n"
            "PHASE 0\n"
            "ConstraintFcn.CONTINUITY: 21.0\n"
            "ConstraintFcn.CONTINUITY: 27.0\n"
            "ConstraintFcn.CONTINUITY: 33.0\n"
            "ConstraintFcn.CONTINUITY: 39.0\n"
            "ConstraintFcn.CONTINUITY: 45.0\n"
            "ConstraintFcn.CONTINUITY: 51.0\n"
            "ConstraintFcn.CONTINUITY: 57.0\n"
            "ConstraintFcn.CONTINUITY: 63.0\n"
            "ConstraintFcn.CONTINUITY: 69.0\n"
            "ConstraintFcn.CONTINUITY: 75.0\n"
            "ConstraintFcn.CONTINUITY: 81.0\n"
            "ConstraintFcn.CONTINUITY: 87.0\n"
            "ConstraintFcn.CONTINUITY: 93.0\n"
            "ConstraintFcn.CONTINUITY: 99.0\n"
            "ConstraintFcn.CONTINUITY: 105.0\n"
            "ConstraintFcn.CONTINUITY: 111.0\n"
            "ConstraintFcn.CONTINUITY: 117.0\n"
            "ConstraintFcn.CONTINUITY: 123.0\n"
            "ConstraintFcn.CONTINUITY: 129.0\n"
            "ConstraintFcn.CONTINUITY: 135.0\n"
            "PhaseTransitionFcn.CONTINUOUS: 141.0\n"
            "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
            "ConstraintFcn.SUPERIMPOSE_MARKERS: 9.0\n"
            "\n"
            "PHASE 1\n"
            "ConstraintFcn.CONTINUITY: 21.0\n"
            "ConstraintFcn.CONTINUITY: 27.0\n"
            "ConstraintFcn.CONTINUITY: 33.0\n"
            "ConstraintFcn.CONTINUITY: 39.0\n"
            "ConstraintFcn.CONTINUITY: 45.0\n"
            "ConstraintFcn.CONTINUITY: 51.0\n"
            "ConstraintFcn.CONTINUITY: 57.0\n"
            "ConstraintFcn.CONTINUITY: 63.0\n"
            "ConstraintFcn.CONTINUITY: 69.0\n"
            "ConstraintFcn.CONTINUITY: 75.0\n"
            "ConstraintFcn.CONTINUITY: 81.0\n"
            "ConstraintFcn.CONTINUITY: 87.0\n"
            "ConstraintFcn.CONTINUITY: 93.0\n"
            "ConstraintFcn.CONTINUITY: 99.0\n"
            "ConstraintFcn.CONTINUITY: 105.0\n"
            "ConstraintFcn.CONTINUITY: 111.0\n"
            "ConstraintFcn.CONTINUITY: 117.0\n"
            "ConstraintFcn.CONTINUITY: 123.0\n"
            "ConstraintFcn.CONTINUITY: 129.0\n"
            "ConstraintFcn.CONTINUITY: 135.0\n"
            "ConstraintFcn.CONTINUITY: 141.0\n"
            "ConstraintFcn.CONTINUITY: 147.0\n"
            "ConstraintFcn.CONTINUITY: 153.0\n"
            "ConstraintFcn.CONTINUITY: 159.0\n"
            "ConstraintFcn.CONTINUITY: 165.0\n"
            "ConstraintFcn.CONTINUITY: 171.0\n"
            "ConstraintFcn.CONTINUITY: 177.0\n"
            "ConstraintFcn.CONTINUITY: 183.0\n"
            "ConstraintFcn.CONTINUITY: 189.0\n"
            "ConstraintFcn.CONTINUITY: 195.0\n"
            "PhaseTransitionFcn.CONTINUOUS: 201.0\n"
            "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
            "\n"
            "PHASE 2\n"
            "ConstraintFcn.CONTINUITY: 21.0\n"
            "ConstraintFcn.CONTINUITY: 27.0\n"
            "ConstraintFcn.CONTINUITY: 33.0\n"
            "ConstraintFcn.CONTINUITY: 39.0\n"
            "ConstraintFcn.CONTINUITY: 45.0\n"
            "ConstraintFcn.CONTINUITY: 51.0\n"
            "ConstraintFcn.CONTINUITY: 57.0\n"
            "ConstraintFcn.CONTINUITY: 63.0\n"
            "ConstraintFcn.CONTINUITY: 69.0\n"
            "ConstraintFcn.CONTINUITY: 75.0\n"
            "ConstraintFcn.CONTINUITY: 81.0\n"
            "ConstraintFcn.CONTINUITY: 87.0\n"
            "ConstraintFcn.CONTINUITY: 93.0\n"
            "ConstraintFcn.CONTINUITY: 99.0\n"
            "ConstraintFcn.CONTINUITY: 105.0\n"
            "ConstraintFcn.CONTINUITY: 111.0\n"
            "ConstraintFcn.CONTINUITY: 117.0\n"
            "ConstraintFcn.CONTINUITY: 123.0\n"
            "ConstraintFcn.CONTINUITY: 129.0\n"
            "ConstraintFcn.CONTINUITY: 135.0\n"
            "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
            "\n"
            "------------------------------\n"
        )
    sys.stdout = sys.__stdout__  # Reset redirect.
    assert captured_output.getvalue() == expected_output
