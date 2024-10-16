import os
from casadi import MX, Function

from bioptim import (
    ControlType,
    HolonomicConstraintsFcn,
    QuadratureRule,
    VariationalBiorbdModel,
)

from tests.utils import TestUtils


def test_variational_model():
    from bioptim.examples.discrete_mechanics_and_optimal_control import (
        example_variational_integrator_pendulum as ocp_module,
    )

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = bioptim_folder + "/models/pendulum.bioMod"
    model = VariationalBiorbdModel(biorbd_model_path)

    q = MX([3.0, 4.0])
    qdot = MX([1.0, 2.0])

    TestUtils.assert_equal(model.lagrangian()(q, qdot), -4.34239126)

    time_step = MX(0.5)

    TestUtils.assert_equal(model.discrete_lagrangian(q, qdot, time_step), 1.1629533)
    model_mid_point = VariationalBiorbdModel(biorbd_model_path, discrete_approximation=QuadratureRule.MIDPOINT)
    TestUtils.assert_equal(model_mid_point.discrete_lagrangian(q, qdot, time_step), -4.49869015)
    model_right = VariationalBiorbdModel(biorbd_model_path, discrete_approximation=QuadratureRule.RECTANGLE_RIGHT)
    TestUtils.assert_equal(model_right.discrete_lagrangian(q, qdot, time_step), 1.88558012)
    model_left = VariationalBiorbdModel(biorbd_model_path, discrete_approximation=QuadratureRule.RECTANGLE_LEFT)
    TestUtils.assert_equal(model_left.discrete_lagrangian(q, qdot, time_step), 0.44032649)

    control0 = MX([10.0, 5.0])
    control1 = MX([3.0, 9.0])

    TestUtils.assert_equal(model.control_approximation(control0, control1, time_step), [2.5, 1.25])
    model_linear = VariationalBiorbdModel(biorbd_model_path, control_type=ControlType.LINEAR_CONTINUOUS)
    TestUtils.assert_equal(model_linear.control_approximation(control0, control1, time_step), [1.625, 1.75])
    model_linear_left = VariationalBiorbdModel(
        biorbd_model_path,
        control_type=ControlType.LINEAR_CONTINUOUS,
        control_discrete_approximation=QuadratureRule.RECTANGLE_LEFT,
    )
    TestUtils.assert_equal(
        model_linear_left.control_approximation(
            control0,
            control1,
            time_step,
        ),
        [2.5, 1.25],
    )
    model_linear_right = VariationalBiorbdModel(
        biorbd_model_path,
        control_type=ControlType.LINEAR_CONTINUOUS,
        control_discrete_approximation=QuadratureRule.RECTANGLE_RIGHT,
    )
    TestUtils.assert_equal(
        model_linear_right.control_approximation(
            control0,
            control1,
            time_step,
        ),
        [0.75, 2.25],
    )

    q_prev = MX([1.0, 2.0])
    q_cur = MX([3.0, 4.0])
    q_next = MX([5.0, 6.0])
    control_prev = MX([7.0, 8.0])
    control_cur = MX([9.0, 10.0])
    control_next = MX([11.0, 12.0])

    time_step_sym = MX.sym("time_step", 1, 1)
    q_prev_sym = MX.sym("q_prev", model.nb_q, 1)
    q_cur_sym = MX.sym("q_cur", model.nb_q, 1)
    q_next_sym = MX.sym("q_next", model.nb_q, 1)
    control_prev_sym = MX.sym("control_prev", model.nb_tau, 1)
    control_cur_sym = MX.sym("control_cur", model.nb_tau, 1)
    control_next_sym = MX.sym("control_next", model.nb_tau, 1)
    qdot_sym = MX.sym("qdot", model.nb_qdot, 1)

    discrete_ele = Function(
        "discrete_euler_lagrange_equations",
        [time_step_sym, q_prev_sym, q_cur_sym, q_next_sym, control_prev_sym, control_cur_sym, control_next_sym],
        [
            model.discrete_euler_lagrange_equations(
                time_step_sym, q_prev_sym, q_cur_sym, q_next_sym, control_prev_sym, control_cur_sym, control_next_sym
            )
        ],
    )
    TestUtils.assert_equal(
        discrete_ele(time_step, q_prev, q_cur, q_next, control_prev, control_cur, control_next),
        [1.2098695, 11.60944499],
    )
    compute_initial_states = Function(
        "compute_initial_states",
        [time_step_sym, q_cur_sym, qdot_sym, q_next_sym, control_cur_sym, control_next_sym],
        [
            model.compute_initial_states(
                time_step_sym, q_cur_sym, qdot_sym, q_next_sym, control_cur_sym, control_next_sym
            )
        ],
    )
    TestUtils.assert_equal(
        compute_initial_states(time_step, q_cur, qdot, q_next, control_cur, control_next), [-2.62083655, 4.24192777]
    )
    compute_final_states = Function(
        "compute_final_states",
        [time_step_sym, q_prev_sym, q_cur_sym, qdot_sym, control_prev_sym, control_cur_sym],
        [model.compute_final_states(time_step_sym, q_prev_sym, q_cur_sym, qdot_sym, control_prev_sym, control_cur_sym)],
    )
    TestUtils.assert_equal(
        compute_final_states(time_step, q_prev, q_cur, qdot, control_prev, control_cur), [3.83070605, 7.36751722]
    )

    biorbd_model_path = bioptim_folder + "/models/pendulum_holonomic.bioMod"
    holonomic_model = VariationalBiorbdModel(biorbd_model_path)
    (
        constraints_func,
        constraints_jacobian_func,
        constraints_double_derivative_func,
    ) = HolonomicConstraintsFcn.superimpose_markers(holonomic_model, marker_1="marker_1", index=slice(2, 3))
    holonomic_model._add_holonomic_constraint(
        constraints_func, constraints_jacobian_func, constraints_double_derivative_func
    )

    q_prev = MX([1.0, 2.0, 3.0])
    q_cur = MX([3.0, 4.0, 5.0])
    q_next = MX([5.0, 6.0, 7.0])
    control_prev = MX([7.0, 8.0, 9.0])
    control_cur = MX([9.0, 10.0, 11.0])
    control_next = MX([11.0, 12.0, 13.0])
    qdot = MX([1.0, 2.0, 3.0])
    lambdas = MX([1.0])

    q_prev_sym = MX.sym("q_prev", holonomic_model.nb_q, 1)
    q_cur_sym = MX.sym("q_cur", holonomic_model.nb_q, 1)
    q_next_sym = MX.sym("q_next", holonomic_model.nb_q, 1)
    control_prev_sym = MX.sym("control_prev", holonomic_model.nb_tau, 1)
    control_cur_sym = MX.sym("control_cur", holonomic_model.nb_tau, 1)
    control_next_sym = MX.sym("control_next", holonomic_model.nb_tau, 1)
    qdot_sym = MX.sym("qdot", holonomic_model.nb_qdot, 1)
    lambdas_sym = MX.sym("lambda", holonomic_model.nb_holonomic_constraints, 1)

    holonomic_discrete_constraints_jacobian = Function(
        "holonomic_discrete_constraints_jacobian",
        [time_step_sym, q_cur_sym],
        [holonomic_model.discrete_holonomic_constraints_jacobian(time_step, q_cur_sym)],
    )
    TestUtils.assert_equal(holonomic_discrete_constraints_jacobian(time_step, q_cur), [0.0, 0.5, 0.0])
    discrete_ele = Function(
        "discrete_euler_lagrange_equations",
        [
            time_step_sym,
            q_prev_sym,
            q_cur_sym,
            q_next_sym,
            control_prev_sym,
            control_cur_sym,
            control_next_sym,
            lambdas_sym,
        ],
        [
            holonomic_model.discrete_euler_lagrange_equations(
                time_step_sym,
                q_prev_sym,
                q_cur_sym,
                q_next_sym,
                control_prev_sym,
                control_cur_sym,
                control_next_sym,
                lambdas_sym,
            )
        ],
    )
    TestUtils.assert_equal(
        discrete_ele(time_step, q_prev, q_cur, q_next, control_prev, control_cur, control_next, lambdas),
        [0.7429345, -2.12943972, 14.76794345, 6.0],
    )
    compute_initial_states = Function(
        "compute_initial_states",
        [time_step_sym, q_cur_sym, qdot_sym, q_next_sym, control_cur_sym, control_next_sym, lambdas_sym],
        [
            holonomic_model.compute_initial_states(
                time_step_sym, q_cur_sym, qdot_sym, q_next_sym, control_cur_sym, control_next_sym, lambdas_sym
            )
        ],
    )
    TestUtils.assert_equal(
        compute_initial_states(time_step, q_cur, qdot, q_next, control_cur, control_next, lambdas),
        [-1.76170126, -4.45551976, 5.87787293, 6.0, 4.0, 2.0],
    )
    compute_final_states = Function(
        "compute_final_states",
        [time_step_sym, q_prev_sym, q_cur_sym, qdot_sym, control_prev_sym, control_cur_sym, lambdas_sym],
        [
            holonomic_model.compute_final_states(
                time_step_sym, q_prev_sym, q_cur_sym, qdot_sym, control_prev_sym, control_cur_sym, lambdas_sym
            )
        ],
    )
    TestUtils.assert_equal(
        compute_final_states(time_step, q_prev, q_cur, qdot, control_prev, control_cur, lambdas),
        [2.50463576, 2.32608004, 8.89007052, 2.0],
    )
