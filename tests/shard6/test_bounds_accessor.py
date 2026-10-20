import numpy as np
import numpy.testing as npt
from bioptim import BoundsList, InterpolationType, VariableScalingList
from bioptim.examples.toy_examples.torque_driven_ocp import torque_activation_driven
from bioptim.optimization.bound_vector import _dispatch_control_bounds
from tests.utils import TestUtils


def test_accessors_on_bounds_option():
    x_min = [-100] * 6
    x_max = [100] * 6
    x_bounds = BoundsList()
    x_bounds.add("my_key", min_bound=x_min, max_bound=x_max, interpolation=InterpolationType.CONSTANT)
    x_bounds["my_key"][:3] = 0
    x_bounds["my_key"].min[3:] = -10
    x_bounds["my_key"].max[1:3] = 10

    # Check min and max have the right value
    npt.assert_almost_equal(x_bounds["my_key"].min[:], np.array([[0], [0], [0], [-10], [-10], [-10]]))
    npt.assert_almost_equal(x_bounds["my_key"].max[:], np.array([[0], [10], [10], [100], [100], [100]]))


def test_accessors_on_bounds_option_multidimensional():
    x_min = [[-100, -50, 0] for i in range(6)]
    x_max = [[100, 150, 200] for i in range(6)]
    x_bounds = BoundsList()
    x_bounds.add(
        "my_key",
        min_bound=x_min,
        max_bound=x_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    x_bounds["my_key"][:3, 0] = 0
    x_bounds["my_key"].min[1:5, 1:] = -10
    x_bounds["my_key"].max[1:5, 1:] = 10

    # Check min and max have the right value
    npt.assert_almost_equal(
        x_bounds["my_key"].min[:],
        np.array([[0, -50, 0], [0, -10, -10], [0, -10, -10], [-100, -10, -10], [-100, -10, -10], [-100, -50, 0]]),
    )
    npt.assert_almost_equal(
        x_bounds["my_key"].max[:],
        np.array([[0, 150, 200], [0, 10, 10], [0, 10, 10], [100, 10, 10], [100, 10, 10], [100, 150, 200]]),
    )


def test_dispatch_control_bounds_keeps_scaled_bounds_in_solver_order():
    ocp = torque_activation_driven.prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/2segments_2dof_2contacts.bioMod",
        n_shooting=2,
        final_time=1,
        expand_dynamics=False,
    )
    nlp = ocp.nlp[0]

    u_bounds = BoundsList()
    u_bounds.add(
        "residual_tau",
        min_bound=[20.0, -60.0],
        max_bound=[60.0, 100.0],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    u_bounds.add(
        "tau",
        min_bound=[-10.0, -30.0],
        max_bound=[10.0, 50.0],
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    u_scaling = VariableScalingList()
    u_scaling.add("residual_tau", scaling=[10.0, 20.0])
    u_scaling.add("tau", scaling=[2.0, 10.0])

    lower, upper = _dispatch_control_bounds(nlp, nlp.controls, u_bounds, u_scaling)

    assert len(lower) == nlp.n_controls_nodes
    assert len(upper) == nlp.n_controls_nodes
    for node_lower, node_upper in zip(lower, upper):
        npt.assert_allclose(node_lower[:, 0], [-5.0, -3.0, 2.0, -3.0])
        npt.assert_allclose(node_upper[:, 0], [5.0, 5.0, 6.0, 5.0])
