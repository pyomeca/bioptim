"""
Test for file IO
"""

import io
import matplotlib
import numpy as np
import os
import pytest
import sys
from casadi import Function, MX

from bioptim import CostType, OdeSolver, Solver, RigidBodyDynamics, BiorbdModel, PhaseDynamics
from bioptim.limits.penalty import PenaltyOption

matplotlib.use("Agg")


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_plot_graphs_one_phase(phase_dynamics):
    # Load graphs_one_phase
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_plot_check_conditioning(phase_dynamics):
    # Load graphs check conditioning
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=False,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    ocp.check_conditioning()
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_plot_check_conditioning_live(phase_dynamics):
    # Load graphs check conditioning
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=False,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    ocp.add_plot_check_conditioning()


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_plot_ipopt_output_live(phase_dynamics):
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=False,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    ocp.add_plot_ipopt_outputs()


def test_save_ipopt_output():
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=40,
    )
    path_to_results = "temporary_results"
    if path_to_results not in os.listdir(bioptim_folder):
        os.mkdir(os.path.join(bioptim_folder, path_to_results))
    result_file_name = "pendulum"
    nb_iter_save = 10
    ocp.save_intermediary_ipopt_iterations(path_to_results, result_file_name, nb_iter_save)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_plot_merged_graphs(phase_dynamics):
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
    t, markers_ref, x_ref, muscle_excitations_ref = ocp_module.generate_data(
        bio_model, final_time, n_shooting, use_sx=False
    )

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
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1)
    sol = ocp.solve(solver)
    sol.graphs(automatically_organize=False)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_plot_graphs_multi_phases(phase_dynamics):
    # Load graphs_one_phase
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    sol = ocp.solve()
    sol.graphs(automatically_organize=False)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE])
def test_add_new_plot(phase_dynamics):
    # Load graphs_one_phase
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=20,
        final_time=0.5,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    solver = Solver.IPOPT()
    solver.set_maximum_iterations(1)
    sol = ocp.solve(solver)

    # Test 1 - Working plot
    ocp.add_plot("My New Plot", lambda t0, phases_dt, node_idx, x, u, p, a, d: x[0:2, :])
    sol.graphs(automatically_organize=False)

    # Add the plot of objectives and constraints to this mess
    ocp.add_plot_penalty(CostType.ALL)
    sol.graphs(automatically_organize=False)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize(
    "rigidbody_dynamics",
    [RigidBodyDynamics.ODE, RigidBodyDynamics.DAE_FORWARD_DYNAMICS, RigidBodyDynamics.DAE_INVERSE_DYNAMICS],
)
def test_plot_graphs_for_implicit_constraints(rigidbody_dynamics, phase_dynamics):
    from bioptim.examples.getting_started import example_implicit_dynamics as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_shooting=5,
        final_time=1,
        rigidbody_dynamics=rigidbody_dynamics,
        phase_dynamics=phase_dynamics,
        expand_dynamics=False,
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


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_console_objective_functions(phase_dynamics):
    # Load graphs_one_phase
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
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
                    nlp.algebraic_states.node_index = node_index

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
                    t = MX.sym("t", *p.weighted_function[node_index].size_in("t"))
                    phases_dt = MX.sym("dt", *p.weighted_function[node_index].size_in("dt"))
                    x = MX.sym("x", *p.weighted_function[node_index].size_in("x"))
                    u = MX.sym("u", *p.weighted_function[node_index].size_in("u"))
                    if p.weighted_function[node_index].size_in("u") == (0, 0):
                        u = MX.sym("u", 3, 1)
                    param = MX.sym("param", *p.weighted_function[node_index].size_in("p"))
                    a = MX.sym("a", *p.weighted_function[node_index].size_in("a"))
                    d = MX.sym("d", *p.weighted_function[node_index].size_in("d"))
                    weight = MX.sym("weight", *p.weighted_function[node_index].size_in("weight"))
                    target = MX.sym("target", *p.weighted_function[node_index].size_in("target"))

                    p.function[node_index] = Function(
                        name, [t, phases_dt, x, u, param, a, d], [np.array([range(cmp, len(p.rows) + cmp)]).T]
                    )
                    p.function_non_threaded[node_index] = p.function[node_index]
                    p.weighted_function[node_index] = Function(
                        name,
                        [t, phases_dt, x, u, param, a, d, weight, target],
                        [np.array([range(cmp + 1, len(p.rows) + cmp + 1)]).T],
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

    expected_output = (
        "Solver reported time: 1.2345 sec\n"
        "Real time: 5.4321 sec\n"
        "\n"
        "---- COST FUNCTION VALUES ----\n"
        "PHASE 0\n"
        "Lagrange.MINIMIZE_CONTROL: 120.0 (non weighted  60.0)\n"
        "\n"
        "PHASE 1\n"
        "MultinodeObjectiveFcn.CUSTOM: 6.0 (non weighted  3.0)\n"
        "Lagrange.MINIMIZE_CONTROL: 180.0 (non weighted  90.0)\n"
        "\n"
        "PHASE 2\n"
        "Lagrange.MINIMIZE_CONTROL: 120.0 (non weighted  60.0)\n"
        "\n"
        "Sum cost functions: 426.0\n"
        "------------------------------\n"
        "\n"
        "--------- CONSTRAINTS ---------\n"
        "PHASE 0\n"
        "ConstraintFcn.STATE_CONTINUITY: 420.0\n"
        "PhaseTransitionFcn.CONTINUOUS: 27.0\n"
        "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
        "ConstraintFcn.SUPERIMPOSE_MARKERS: 9.0\n"
        "\n"
        "PHASE 1\n"
        "ConstraintFcn.STATE_CONTINUITY: 630.0\n"
        "PhaseTransitionFcn.CONTINUOUS: 27.0\n"
        "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
        "\n"
        "PHASE 2\n"
        "ConstraintFcn.STATE_CONTINUITY: 420.0\n"
        "ConstraintFcn.SUPERIMPOSE_MARKERS: 6.0\n"
        "\n"
        "------------------------------\n"
    )

    sys.stdout = sys.__stdout__  # Reset redirect.
    assert captured_output.getvalue() == expected_output
