import numpy as np
import os
import pytest

from casadi import MX, Function, jacobian

from bioptim import (
    BiorbdModelHolonomic,
    Solver,
)
from biorbd import marker_index

from .utils import TestUtils


def generate_constraint_functions(model, marker_1: str, marker_2: str, index: slice = slice(0, 3)):
    # symbolic variables to create the functions
    q_sym = MX.sym("q", model.nb_q, 1)
    q_dot_sym = MX.sym("q_dot", model.nb_qdot, 1)
    q_ddot_sym = MX.sym("q_ddot", model.nb_qdot, 1)

    # symbolic markers in global frame
    marker_1_sym = model.marker(q_sym, index=marker_index(model.model, marker_1))
    marker_2_sym = model.marker(q_sym, index=marker_index(model.model, marker_2))

    # the constraint is the distance between the two markers, set to zero
    constraint = (marker_1_sym - marker_2_sym)[index]
    # the jacobian of the constraint
    constraint_jacobian = jacobian(constraint, q_sym)

    constraint_func = Function(
        "holonomic_constraint",
        [q_sym],
        [constraint],
        ["q"],
        ["holonomic_constraint"],
    ).expand()

    constraint_jacobian_func = Function(
        "holonomic_constraint_jacobian",
        [q_sym],
        [constraint_jacobian],
        ["q"],
        ["holonomic_constraint_jacobian"],
    ).expand()

    # the double derivative of the constraint
    constraint_double_derivative = (
        constraint_jacobian_func(q_sym) @ q_ddot_sym + constraint_jacobian_func(q_dot_sym) @ q_dot_sym
    )

    constraint_double_derivative_func = Function(
        "holonomic_constraint_double_derivative",
        [q_sym, q_dot_sym, q_ddot_sym],
        [constraint_double_derivative],
        ["q", "q_dot", "q_ddot"],
        ["holonomic_constraint_double_derivative"],
    ).expand()

    return constraint_func, constraint_jacobian_func, constraint_double_derivative_func


def test_model_holonomic():
    # TODO: Change
    model_path = "/home/mickaelbegon/Documents/Stage_Amandine/bioptim/bioptim/examples/torque_driven_ocp/models/triple_pendulum.bioMod"

    model = BiorbdModelHolonomic(model_path)

    with pytest.raises(
        ValueError,
        match="The sum of the number of dependent and independent joints should be equal to the number of DoF of the"
        " model",
    ):
        model.set_dependencies([1], [2])

    with pytest.raises(
        ValueError,
        match="You registered 1 as both dependent and independent",
    ):
        model.set_dependencies([1, 2], [1])

    with pytest.raises(
        ValueError,
        match="Joint index 3 is not a valid joint index since the model has 3 DoF",
    ):
        model.set_dependencies([1, 2], [3])

    model.set_dependencies([1, 2], [0])

    np.testing.assert_equal(model.nb_independent_joints, 1)
    np.testing.assert_equal(model.nb_dependent_joints, 2)

    y_constraint_fun, y_constraint_jac_fun, y_constraint_double_derivative_fun = generate_constraint_functions(
        model, "marker_1", "marker_6", index=slice(1, 2)
    )
    model.add_holonomic_constraint(y_constraint_fun, y_constraint_jac_fun, y_constraint_double_derivative_fun)
    z_constraint_fun, z_constraint_jac_fun, z_constraint_double_derivative_fun = generate_constraint_functions(
        model, "marker_1", "marker_6", index=slice(2, 3)
    )
    model.add_holonomic_constraint(z_constraint_fun, z_constraint_jac_fun, z_constraint_double_derivative_fun)

    np.testing.assert_equal(model.nb_holonomic_constraints, 2)

    # symbolic variables
    q = MX([1, 2, 3])
    q_dot = MX([4, 5, 6])
    q_ddot = MX([7, 8, 9])
    tau = MX([10, 11, 12])
    TestUtils.assert_equal(model.holonomic_constraints(q), [-0.70317549, 0.5104801])
    TestUtils.assert_equal(
        model.holonomic_constraints_jacobian(q),
        [[-0.5104801, 0.02982221, -0.96017029], [-0.70317549, 0.13829549, 0.2794155]],
    )
    TestUtils.assert_equal(model.holonomic_constraints_derivative(q, q_dot), [-7.65383105, -0.44473154])
    TestUtils.assert_equal(model.holonomic_constraints_double_derivative(q, q_dot, q_ddot), [10.23374996, -11.73729905])
    TestUtils.assert_equal(model.constrained_forward_dynamics(q, q_dot, tau), [-5.18551845, -3.01921376, 25.79451813])
    TestUtils.assert_equal(
        model.partitioned_mass_matrix(q),
        [
            [2.87597472e00, 4.60793003e-01, 3.36615631e-01],
            [4.60793003e-01, 9.99942366e-01, -2.88168107e-05],
            [3.36615631e-01, -2.88168107e-05, 9.54331080e-01],
        ],
    )
    TestUtils.assert_equal(model.partitioned_non_linear_effect(q, q_dot), [88.75493352, 4.13246046, -10.90514929])
    TestUtils.assert_equal(model.partitioned_q(q), [1.0, 2.0, 3.0])
    TestUtils.assert_equal(model.partitioned_qdot(q_dot), [4.0, 5.0, 6.0])
    TestUtils.assert_equal(model.partitioned_tau(tau), [10.0, 11.0, 12.0])
    TestUtils.assert_equal(
        model.partitioned_constrained_jacobian(q),
        [[-0.5104801, 0.02982221, -0.96017029], [-0.70317549, 0.13829549, 0.2794155]],
    )
    # TODO: partitioned_forward_dynamics, coupling matrix, biais_vector => expand=False

    u = MX(TestUtils.to_array(q[model._independent_joint_index]))
    v = MX(TestUtils.to_array(q[model._dependent_joint_index]))
    TestUtils.assert_equal(model.q_from_u_and_v(u, v), q)

    # TODO: compute_v_from_u, compute_v_from_u_numeric


@pytest.mark.parametrize("use_sx", [False, True])
def test_example_two_pendulums(use_sx):
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.holonomic_constraints import two_pendulums

    bioptim_folder = os.path.dirname(two_pendulums.__file__)

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/two_pendulums.bioMod", n_shooting=10, final_time=1, use_sx=use_sx
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    q = np.zeros((4, 10))
    u_end = sol.states["u"][-1, :]
    v_end = model.compute_v_from_u_numeric(u_end, v_init=np.zeros(2)).toarray()
    q_end = model.q_from_u_and_v(u_end[:, np.newaxis], v_end).toarray().squeeze()

    np.testing.assert_almost_equal(
        q_end,
        [-1.54, -0.999525830605479, -0.030791459082466, 0.0],
        decimal=6,
    )
