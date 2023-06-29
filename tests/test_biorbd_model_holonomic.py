import numpy as np
import os
import pytest

from casadi import DM, MX, Function

from bioptim import (
    BiorbdModelHolonomic,
    Solver,
)
from bioptim.examples.discrete_mechanics_and_optimal_control.holonomic_constraints import HolonomicConstraintFcn

from .utils import TestUtils


def test_model_holonomic():
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = bioptim_folder + "/models/triple_pendulum.bioMod"
    model = BiorbdModelHolonomic(biorbd_model_path)

    with pytest.raises(
        ValueError,
        match="The sum of the number of dependent and independent joints should be equal to the number of DoF of the"
        " model",
    ):
        model.set_dependencies([1], [2])

    with pytest.raises(
        ValueError,
        match="Joint 1 is both dependant and independent. You need to specify this index in "
        "only one of these arguments: dependent_joint_index: independent_joint_index.",
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

    (
        y_constraint_fun,
        y_constraint_jac_fun,
        y_constraint_double_derivative_fun,
    ) = HolonomicConstraintFcn.superimpose_markers(model, "marker_1", "marker_6", index=slice(1, 2))
    model.add_holonomic_constraint(y_constraint_fun, y_constraint_jac_fun, y_constraint_double_derivative_fun)
    (
        z_constraint_fun,
        z_constraint_jac_fun,
        z_constraint_double_derivative_fun,
    ) = HolonomicConstraintFcn.superimpose_markers(model, "marker_1", "marker_6", index=slice(2, 3))
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

    q_u = MX(TestUtils.to_array(q[model._independent_joint_index]))
    qdot_u = MX(TestUtils.to_array(q_dot[model._independent_joint_index]))
    q_v = MX(TestUtils.to_array(q[model._dependent_joint_index]))
    q_u_sym = MX.sym("q_u_sym", model.nb_independent_joints, 1)
    qdot_u_sym = MX.sym("q_u_sym", model.nb_independent_joints, 1)
    tau_sym = MX.sym("q_u_sym", model.nb_tau, 1)

    partitioned_forward_dynamics_func = Function(
        "partitioned_forward_dynamics",
        [q_u_sym, qdot_u_sym, tau_sym],
        [model.partitioned_forward_dynamics(q_u_sym, qdot_u_sym, tau_sym, q_v_init=q_v)],
    )
    TestUtils.assert_equal(partitioned_forward_dynamics_func(q_u, qdot_u, tau), -6.3109, expand=False)
    TestUtils.assert_equal(model.coupling_matrix(q), [5.79509793, -0.35166415], expand=False)
    TestUtils.assert_equal(model.biais_vector(q, q_dot), [-27.03137348, -23.97095718], expand=False)
    TestUtils.assert_equal(model.state_from_partition(q_u, q_v), q)

    compute_v_from_u_func = Function("compute_q_v", [q_u_sym], [model.compute_q_v(q_u_sym, q_v_init=q_v)])
    TestUtils.assert_equal(compute_v_from_u_func(q_u), [2 * np.pi / 3, 2 * np.pi / 3], expand=False)
    compute_q_func = Function("compute_q", [q_u_sym], [model.compute_q_v(q_u_sym, q_v_init=q_v)])
    TestUtils.assert_equal(compute_q_func(q_u), [2.0943951, 2.0943951], expand=False)
    TestUtils.assert_equal(model.compute_qdot_v(q, qdot_u), [23.18039172, -1.4066566], expand=False)
    TestUtils.assert_equal(model.compute_qdot(q, qdot_u), [4.0, 23.18039172, -1.4066566], expand=False)

    np.testing.assert_almost_equal(
        model.compute_q_v_numeric(DM([0.0]), q_v_init=[1, 1]).toarray().squeeze(),
        np.array([2 * np.pi / 3, 2 * np.pi / 3]),
        decimal=6,
    )


def test_example_two_pendulums():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.holonomic_constraints import two_pendulums

    bioptim_folder = os.path.dirname(two_pendulums.__file__)

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/two_pendulums.bioMod", n_shooting=10, final_time=1
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    np.testing.assert_almost_equal(
        sol.states["q_u"],
        [
            [
                1.54,
                1.34874433,
                0.98862654,
                0.64450762,
                0.34444899,
                0.02992554,
                -0.31997814,
                -0.69642723,
                -1.09920309,
                -1.44582515,
                -1.54,
            ],
            [
                1.54,
                1.76965698,
                2.14098023,
                2.45361809,
                2.69040682,
                2.74257871,
                2.58671659,
                2.22277978,
                1.62756889,
                0.82149671,
                0.0,
            ],
        ],
        decimal=6,
    )


def test_example_three_bars():
    """Test the holonomic_constraints/two_pendulums example"""
    from bioptim.examples.holonomic_constraints import three_bars

    bioptim_folder = os.path.dirname(three_bars.__file__)

    # --- Prepare the ocp --- #
    ocp, model = three_bars.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/three_bars.bioMod", n_shooting=10, final_time=1
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    np.testing.assert_almost_equal(
        sol.states["q_u"],
        [
            [
                1.57079633,
                1.82998567,
                2.03581609,
                1.83253539,
                1.31518337,
                0.58224633,
                -0.25291553,
                -0.9673269,
                -1.2836074,
                -1.46862711,
                -1.57079633,
            ],
            [
                0.0,
                -0.63962958,
                -1.2567877,
                -1.40743311,
                -1.28285506,
                -0.85200181,
                -0.16962037,
                0.24751359,
                -0.08058608,
                -0.20147857,
                0.0,
            ],
            [
                0.0,
                0.08562447,
                -0.32607879,
                -1.08655335,
                -1.96180706,
                -2.62757206,
                -2.71394484,
                -2.5955768,
                -1.8961505,
                -1.40873655,
                -1.57079633,
            ],
        ],
        decimal=6,
    )
