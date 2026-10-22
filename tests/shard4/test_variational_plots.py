"""
Tests of the plots of the variational integrator: the states exist only at the nodes, so they must be drawn as one
line through the nodes (and not as one empty segment per interval, as for the integrated states of the other solvers).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from bioptim import OdeSolver, PlotType, SolutionMerge, Solver
from bioptim.examples.getting_started import basic_ocp
from bioptim.examples.toy_examples.discrete_mechanics_and_optimal_control import (
    example_variational_integrator_pendulum as pendulum_example,
    example_variational_integrator_with_holonomic_constraints_pendulum as holonomic_example,
)

from ..utils import TestUtils

N_SHOOTING = 5

EXAMPLES = {
    "pendulum": (pendulum_example, "pendulum.bioMod"),
    "holonomic": (holonomic_example, "pendulum_holonomic.bioMod"),
}


def _draw_variational_solution(name: str):
    """Solve a variational example for a few iterations and read back the lines of every axis of its figures"""
    example, model = EXAMPLES[name]
    ocp = example.prepare_ocp(
        bio_model_path=TestUtils.bioptim_folder() + "/examples/models/" + model,
        final_time=1,
        n_shooting=N_SHOOTING,
        use_sx=False,
    )
    solution = ocp.solve(Solver.IPOPT(show_online_optim=False, _print_level=0, _max_iter=10))

    plt.close("all")
    solution.graphs(show_now=False, automatically_organize=False)
    # The title of an axis is the name of the variable, followed by the name of its dof (e.g. q_Seg1_RotX-0)
    lines_per_axis = {
        ax.get_title(): [
            (np.array(line.get_xdata(), dtype=float), np.array(line.get_ydata(), dtype=float))
            for line in ax.get_lines()
        ]
        for number in plt.get_fignums()
        for ax in plt.figure(number).axes
    }
    plt.close("all")
    return solution, lines_per_axis


@pytest.fixture(scope="module", params=list(EXAMPLES))
def drawn(request):
    return _draw_variational_solution(request.param)


def _axes_of(lines_per_axis: dict, key: str) -> list:
    """The lines of the axes of a variable, in the order of its dof"""
    return [lines for title, lines in lines_per_axis.items() if title.startswith(f"{key}_")]


def _keys_to_check(solution) -> list:
    return [key for key in ("q", "lambdas") if key in solution.decision_states(to_merge=SolutionMerge.NODES)]


def test_variational_states_are_a_single_line_through_the_nodes(drawn):
    solution, lines_per_axis = drawn
    time = solution.decision_time(to_merge=SolutionMerge.NODES).squeeze()
    assert time.shape == (N_SHOOTING + 1,)

    keys = _keys_to_check(solution)
    assert "q" in keys
    for key in keys:
        axes = _axes_of(lines_per_axis, key)
        assert len(axes) == solution.decision_states(to_merge=SolutionMerge.NODES)[key].shape[0]
        for lines in axes:
            assert len(lines) == 1
            assert lines[0][0].shape == (N_SHOOTING + 1,)
            npt.assert_almost_equal(lines[0][0], time)


def test_variational_plotted_states_are_the_decision_states(drawn):
    solution, lines_per_axis = drawn
    states = solution.decision_states(to_merge=SolutionMerge.NODES)

    for key in _keys_to_check(solution):
        for dof, lines in enumerate(_axes_of(lines_per_axis, key)):
            # The first node is not checked here, see test_variational_first_node_is_plotted
            npt.assert_almost_equal(lines[0][1][1:], states[key][dof, 1:])


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4(), OdeSolver.COLLOCATION(polynomial_degree=3)])
def test_states_of_other_ode_solvers_are_still_integrated(ode_solver):
    ocp = basic_ocp.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=N_SHOOTING,
        ode_solver=ode_solver,
        use_sx=False,
    )
    assert ocp.nlp[0].plot["q_states"].type == PlotType.INTEGRATED


def test_variational_first_node_is_plotted(drawn):
    solution, lines_per_axis = drawn
    states = solution.decision_states(to_merge=SolutionMerge.NODES)

    for key in _keys_to_check(solution):
        for dof, lines in enumerate(_axes_of(lines_per_axis, key)):
            npt.assert_almost_equal(lines[0][1], states[key][dof, :])
