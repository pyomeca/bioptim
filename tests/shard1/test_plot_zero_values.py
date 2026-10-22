"""
Check that the plots draw the values of the solution when they are exactly zero. Only nodes without any data (e.g., the
controls of the last node of ControlType.CONSTANT) are expected to be NaN.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

from bioptim import (
    BoundsList,
    ControlType,
    InterpolationType,
    Solver,
    SolutionMerge,
)
from bioptim.examples.getting_started import basic_ocp
from bioptim.examples.toy_examples.holonomic_constraints import two_pendulums_algebraic

from ..utils import TestUtils

# A pendulum without gravity and without mesh: at rest, it stays exactly at zero
PENDULUM_WITHOUT_GRAVITY = """version 4

gravity 0 0 0

segment Seg1
    translations	y
    rotations	x
    ranges
        -1 5
        -2*pi 2*pi
    mass 1
    inertia
        0.0391  0.0000	0.0000
        0.0000	0.0335	-0.0032
        0.0000	-0.0032	0.0090
    com  -0.0005 0.0688 -0.9542
endsegment
"""

N_SHOOTING = 3
N_STEPS = 5  # Number of integration steps of the default OdeSolver.RK4


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _solve_pendulum_started_at_rest(tmp_path, control_type: ControlType):
    """
    Solve a pendulum without gravity which starts at rest, with a null torque during the first interval and a unit
    torque afterward. The states are therefore exactly zero during the whole first interval.
    """
    model_path = tmp_path / "pendulum_without_gravity.bioMod"
    model_path.write_text(PENDULUM_WITHOUT_GRAVITY)

    ocp = basic_ocp.prepare_ocp(
        str(model_path), final_time=1, n_shooting=N_SHOOTING, use_sx=False, control_type=control_type
    )

    x_bounds = BoundsList()
    x_bounds["q"] = [-10, -10], [10, 10]
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = [-50, -50], [50, 50]
    x_bounds["qdot"][:, 0] = 0

    # The first interval is at rest: the torque is null at the first node (and at the second one if interpolated)
    n_controls = N_SHOOTING if control_type == ControlType.CONSTANT else N_SHOOTING + 1
    first_nonzero_node = 1 if control_type == ControlType.CONSTANT else 2
    tau = np.zeros((2, n_controls))
    tau[0, first_nonzero_node:] = 1
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=tau, max_bound=tau, interpolation=InterpolationType.EACH_FRAME)
    ocp.update_bounds(x_bounds=x_bounds, u_bounds=u_bounds)

    solver = Solver.IPOPT()
    solver.set_print_level(0)
    solver.set_maximum_iterations(20)
    sol = ocp.solve(solver)
    assert sol.status == 0
    return sol


def _get_figure(figures, name: str):
    return [fig for fig in figures if fig.get_label() == name][0]


CONTROL_TYPES = [ControlType.CONSTANT, ControlType.CONSTANT_WITH_LAST_NODE, ControlType.LINEAR_CONTINUOUS]


@pytest.mark.parametrize("control_type", CONTROL_TYPES)
def test_zero_valued_states_are_plotted(tmp_path, control_type):
    sol = _solve_pendulum_started_at_rest(tmp_path, control_type)
    figures = sol.graphs(show_now=False, automatically_organize=False)

    # States (PlotType.INTEGRATED): one line per interval, the first one is exactly zero
    for key in ("q", "qdot"):
        expected = sol.stepwise_states(to_merge=SolutionMerge.NODES)[key]
        assert not expected[:, : N_STEPS + 1].any()
        assert expected[:, N_STEPS + 1 :].any()
        for i_dof, axis in enumerate(_get_figure(figures, f"{key}_states").axes):
            for i_interval, line in enumerate(axis.lines[:N_SHOOTING]):
                npt.assert_almost_equal(
                    line.get_ydata(), expected[i_dof, i_interval * (N_STEPS + 1) : (i_interval + 1) * (N_STEPS + 1)]
                )


@pytest.mark.parametrize("control_type", CONTROL_TYPES)
def test_zero_valued_controls_are_plotted(tmp_path, control_type):
    sol = _solve_pendulum_started_at_rest(tmp_path, control_type)
    figures = sol.graphs(show_now=False, automatically_organize=False)

    # One array per node, the plot draws the first column of each of them
    expected = np.array([tau_node[:, 0] for tau_node in sol.decision_controls()["tau"]]).T[0, :]
    assert expected[0] == 0 and expected[-1] == 1

    # Controls (PlotType.STEP for the constant ones, PlotType.PLOT for the linear ones), the first node is zero
    if control_type == ControlType.CONSTANT:
        # There is no control at the last node: that is the only missing data
        expected = np.concatenate((expected, [np.nan]))
    npt.assert_almost_equal(_get_figure(figures, "tau_controls").axes[0].lines[0].get_ydata(), expected)


def test_nodes_without_data_are_not_plotted():
    # With ControlType.CONSTANT, there is no control at the last node, the plot function receives an empty array
    ocp = basic_ocp.prepare_ocp(
        TestUtils.bioptim_folder() + "/examples/models/pendulum.bioMod",
        final_time=1,
        n_shooting=N_SHOOTING,
        use_sx=False,
        control_type=ControlType.CONSTANT,
    )
    plot_function = ocp.nlp[0].plot["tau_controls"].function
    x = np.ones((4, 1))

    out = plot_function(0, [0.3], N_SHOOTING, x, np.ndarray((0, 1)), np.empty((0, 1)), np.ndarray((0, 1)), [])
    assert out.shape == (2, 1)
    assert np.isnan(out).all()

    # Exactly zero controls are data
    out = plot_function(0, [0.3], 0, x, np.zeros((2, 1)), np.empty((0, 1)), np.ndarray((0, 1)), [])
    npt.assert_almost_equal(out, np.zeros((2, 1)))


def test_zero_valued_algebraic_states_are_plotted():
    ocp = two_pendulums_algebraic.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/two_pendulums.bioMod",
        n_shooting=N_SHOOTING,
        final_time=1,
    )
    nlp = ocp.nlp[0]
    plot_function = nlp.plot["q_v_algebraic"].function
    n_q_v = nlp.algebraic_states["q_v"].shape
    x = np.zeros((nlp.states.shape, 1))
    u = np.zeros((nlp.controls.shape, 1))

    # Zero algebraic states are data
    a = np.zeros((nlp.algebraic_states.shape, 2))
    npt.assert_almost_equal(plot_function(0, [0.3], 0, x, u, np.empty((0, 1)), a, []), np.zeros((n_q_v, 2)))

    # Non zero algebraic states are unchanged
    a = np.arange(1, 2 * nlp.algebraic_states.shape + 1, dtype=float).reshape(nlp.algebraic_states.shape, 2)
    npt.assert_almost_equal(plot_function(0, [0.3], 0, x, u, np.empty((0, 1)), a, []), a)

    # An empty array has no data
    out = plot_function(0, [0.3], 0, x, u, np.empty((0, 1)), np.ndarray((0, 1)), [])
    assert out.shape == (n_q_v, 1)
    assert np.isnan(out).all()
