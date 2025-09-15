import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

from bioptim import (
    OptimalControlProgram,
    DynamicsOptions,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    OdeSolver,
    TorqueBiorbdModel,
    ControlType,
    InterpolationType,
    ObjectiveList,
    PhaseDynamics,
    Solution,
)
from bioptim.optimization.optimization_vector import OptimizationVectorHelper

from ..utils import TestUtils

FILE_LOCATION = os.path.dirname(os.path.abspath(__file__))


def prepare_ocp(
    biorbd_model_path,
    ode_solver,
    control_type,
    n_shooting,
    interpolation_type,
    q_min_bounds,
    q_max_bounds,
    q_init,
    qdot_min_bounds,
    qdot_max_bounds,
    qdot_init,
    tau_min,
    tau_max,
    tau_init,
    min_time: bool = False,
) -> OptimalControlProgram:
    bio_model = TorqueBiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    if min_time:
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, min_bound=0.195, max_bound=195)

    # DynamicsOptions
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
    )

    x_bounds = BoundsList()
    x_bounds.add("q", min_bound=q_min_bounds, max_bound=q_max_bounds, interpolation=interpolation_type)
    x_bounds.add("qdot", min_bound=qdot_min_bounds, max_bound=qdot_max_bounds, interpolation=interpolation_type)

    x_init = InitialGuessList()
    x_init.add("q", q_init, interpolation=interpolation_type)
    x_init.add("qdot", qdot_init, interpolation=interpolation_type)

    # Define control path bounds
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=tau_min, max_bound=tau_max, interpolation=interpolation_type)

    u_init = InitialGuessList()
    u_init.add("tau", tau_init, interpolation=interpolation_type)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        phase_time=1,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        control_type=control_type,
    )


@pytest.mark.parametrize("min_time", [True, False])
def test_vector_layout_rk4_shared(min_time):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 10
    min_bounds = [-i * 1 for i in range(n_shooting + 1)]
    max_bounds = [i * 1 for i in range(n_shooting + 1)]

    q_min_bounds = np.vstack((min_bounds, np.array(min_bounds) * 2))
    q_max_bounds = np.vstack((max_bounds, np.array(max_bounds) * 2))

    min_bounds = [-i * 0.1 for i in range(n_shooting + 1)]
    max_bounds = [i * 0.1 for i in range(n_shooting + 1)]

    qdot_min_bounds = np.vstack((min_bounds, np.array(min_bounds) * 2))
    qdot_max_bounds = np.vstack((max_bounds, np.array(max_bounds) * 2))

    q_init = np.vstack(([i * 0.01 for i in range(n_shooting + 1)], [i * 0.02 for i in range(n_shooting + 1)]))
    qdot_init = np.vstack([[i * 0.012 for i in range(n_shooting + 1)], [i * 0.024 for i in range(n_shooting + 1)]])

    tau_min = qdot_min_bounds[:, :-1] * 1 / 3
    tau_max = qdot_max_bounds[:, :-1] * 1 / 3

    init = [i * 0.00015 for i in range(n_shooting)]
    tau_init = np.vstack((init, np.array(init) * 2))

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        ode_solver=OdeSolver.RK4(),
        control_type=ControlType.CONSTANT,
        n_shooting=10,
        interpolation_type=InterpolationType.EACH_FRAME,
        q_min_bounds=q_min_bounds,
        q_max_bounds=q_max_bounds,
        q_init=q_init,
        qdot_min_bounds=qdot_min_bounds,
        qdot_max_bounds=qdot_max_bounds,
        qdot_init=qdot_init,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_init=tau_init,
        min_time=min_time,
    )
    v_sym = OptimizationVectorHelper.vector(ocp)

    f = io.StringIO()
    with redirect_stdout(f):
        print(v_sym)

    output = f.getvalue().strip()

    expected_output = (
        "vertcat(dt_phase0, X_scaled_0_0, X_scaled_0_1, X_scaled_0_2, X_scaled_0_3, X_scaled_0_4, "
        "X_scaled_0_5, X_scaled_0_6, X_scaled_0_7, X_scaled_0_8, X_scaled_0_9, X_scaled_0_10, "
        "U_scaled_0_0, U_scaled_0_1, U_scaled_0_2, U_scaled_0_3, U_scaled_0_4, U_scaled_0_5, U_scaled_0_6, "
        "U_scaled_0_7, U_scaled_0_8, U_scaled_0_9)"
    )
    assert (
        output == expected_output
    ), f"The output does not match what was expected.\nExpected:{expected_output}\nGot: {output}"

    v_bounds = OptimizationVectorHelper.bounds_vectors(ocp)

    v_bounds_min_expected = np.load(FILE_LOCATION + "/v_bounds_min.npy", allow_pickle=False)
    v_bounds_max_expected = np.load(FILE_LOCATION + "/v_bounds_max.npy", allow_pickle=False)

    if min_time:
        v_bounds_min_expected[0] = 0.195 / n_shooting
        v_bounds_max_expected[0] = 195.0 / n_shooting

    np.testing.assert_almost_equal(v_bounds[0], v_bounds_min_expected)
    np.testing.assert_almost_equal(v_bounds[1], v_bounds_max_expected)

    v_init = OptimizationVectorHelper.init_vector(ocp)

    v_init_expected = np.load(FILE_LOCATION + "/v_init.npy", allow_pickle=False)

    if min_time:
        v_init_expected[0] = 1.0 / n_shooting * np.ones((1, 1))

    np.testing.assert_almost_equal(v_init, v_init_expected)


@pytest.mark.parametrize("duplicate_starting_point", [False, True])
def test_vector_layout_collocation(duplicate_starting_point):
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 10
    intermediate_point = 4
    actual_intermediate_point = intermediate_point

    if duplicate_starting_point:
        actual_intermediate_point = intermediate_point + 1
    total_points = n_shooting * (actual_intermediate_point + 1)

    min_bounds = [-i * 1 - 1 for i in range(total_points + 1)]
    max_bounds = [i * 1 + 1 for i in range(total_points + 1)]

    q_min_bounds = np.vstack((min_bounds, np.array(min_bounds) * 2))
    q_max_bounds = np.vstack((max_bounds, np.array(max_bounds) * 2))

    min_bounds = [-i * 0.1 - 1 for i in range(total_points + 1)]
    max_bounds = [i * 0.1 + 1 for i in range(total_points + 1)]

    qdot_min_bounds = np.vstack((min_bounds, np.array(min_bounds) * 2))
    qdot_max_bounds = np.vstack((max_bounds, np.array(max_bounds) * 2))

    q_init = np.vstack(
        ([i * 0.01 + 0.1111 for i in range(total_points + 1)], [i * 0.02 for i in range(total_points + 1)])
    )
    qdot_init = np.vstack(
        [[i * 0.012 + 0.1112 for i in range(total_points + 1)], [i * 0.024 for i in range(total_points + 1)]]
    )

    tau_min = qdot_min_bounds[:, :n_shooting] * 1 / 3
    tau_max = qdot_max_bounds[:, :n_shooting] * 1 / 3

    init = [i * 0.00015 + 0.0123 for i in range(n_shooting)]
    tau_init = np.vstack((init, np.array(init) * 2))

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        ode_solver=OdeSolver.COLLOCATION(
            polynomial_degree=intermediate_point, duplicate_starting_point=duplicate_starting_point
        ),
        control_type=ControlType.CONSTANT,
        n_shooting=10,
        interpolation_type=InterpolationType.ALL_POINTS,
        q_min_bounds=q_min_bounds,
        q_max_bounds=q_max_bounds,
        q_init=q_init,
        qdot_min_bounds=qdot_min_bounds,
        qdot_max_bounds=qdot_max_bounds,
        qdot_init=qdot_init,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_init=tau_init,
    )
    v_sym = OptimizationVectorHelper.vector(ocp)

    f = io.StringIO()
    with redirect_stdout(f):
        print(v_sym)

    output = f.getvalue().strip()

    expected_output = "vertcat(dt_phase0, vec(X_scaled_0_0), vec(X_scaled_0_1), vec(X_scaled_0_2), vec(X_scaled_0_3), vec(X_scaled_0_4), vec(X_scaled_0_5), vec(X_scaled_0_6), vec(X_scaled_0_7), vec(X_scaled_0_8), vec(X_scaled_0_9), X_scaled_0_10, U_scaled_0_0, U_scaled_0_1, U_scaled_0_2, U_scaled_0_3, U_scaled_0_4, U_scaled_0_5, U_scaled_0_6, U_scaled_0_7, U_scaled_0_8, U_scaled_0_9)"
    assert (
        output == expected_output
    ), f"The output does not match what was expected.\nExpected:{expected_output}\nGot: {output}"

    v_bounds = OptimizationVectorHelper.bounds_vectors(ocp)

    file_suffix = "_starting_point" if duplicate_starting_point else ""

    v_bounds_min_expected = np.load(FILE_LOCATION + f"/v_bounds_min_collocation{file_suffix}.npy", allow_pickle=False)
    v_bounds_max_expected = np.load(FILE_LOCATION + f"/v_bounds_max_collocation{file_suffix}.npy", allow_pickle=False)

    np.testing.assert_almost_equal(v_bounds[0], v_bounds_min_expected)
    np.testing.assert_almost_equal(v_bounds[1], v_bounds_max_expected)

    v_init = OptimizationVectorHelper.init_vector(ocp)

    v_init_expected = np.load(FILE_LOCATION + f"/v_init_collocation{file_suffix}.npy", allow_pickle=False)
    np.testing.assert_almost_equal(v_init, v_init_expected)


def test_vector_layout_linear_continuous():
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 10
    min_bounds = [-i * 1 for i in range(n_shooting + 1)]
    max_bounds = [i * 1 for i in range(n_shooting + 1)]

    q_min_bounds = np.vstack((min_bounds, np.array(min_bounds) * 2))
    q_max_bounds = np.vstack((max_bounds, np.array(max_bounds) * 2))

    min_bounds = [-i * 0.1 for i in range(n_shooting + 1)]
    max_bounds = [i * 0.1 for i in range(n_shooting + 1)]

    qdot_min_bounds = np.vstack((min_bounds, np.array(min_bounds) * 2))
    qdot_max_bounds = np.vstack((max_bounds, np.array(max_bounds) * 2))

    q_init = np.vstack(([i * 0.01 for i in range(n_shooting + 1)], [i * 0.02 for i in range(n_shooting + 1)]))
    qdot_init = np.vstack([[i * 0.012 for i in range(n_shooting + 1)], [i * 0.024 for i in range(n_shooting + 1)]])

    tau_min = qdot_min_bounds * 1 / 3
    tau_max = qdot_max_bounds * 1 / 3

    init = [i * 0.00015 for i in range(n_shooting + 1)]
    tau_init = np.vstack((init, np.array(init) * 2))

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        ode_solver=OdeSolver.RK4(),
        control_type=ControlType.LINEAR_CONTINUOUS,
        n_shooting=10,
        interpolation_type=InterpolationType.EACH_FRAME,
        q_min_bounds=q_min_bounds,
        q_max_bounds=q_max_bounds,
        q_init=q_init,
        qdot_min_bounds=qdot_min_bounds,
        qdot_max_bounds=qdot_max_bounds,
        qdot_init=qdot_init,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_init=tau_init,
    )

    v_sym = OptimizationVectorHelper.vector(ocp)

    f = io.StringIO()
    with redirect_stdout(f):
        print(v_sym)

    output = f.getvalue().strip()

    expected_output = (
        "vertcat(dt_phase0, X_scaled_0_0, X_scaled_0_1, X_scaled_0_2, X_scaled_0_3, X_scaled_0_4, "
        "X_scaled_0_5, X_scaled_0_6, X_scaled_0_7, X_scaled_0_8, X_scaled_0_9, X_scaled_0_10, "
        "U_scaled_0_0, U_scaled_0_1, U_scaled_0_2, U_scaled_0_3, U_scaled_0_4, U_scaled_0_5, U_scaled_0_6, "
        "U_scaled_0_7, U_scaled_0_8, U_scaled_0_9, U_scaled_0_10)"
    )
    assert (
        output == expected_output
    ), f"The output does not match what was expected.\nExpected:{expected_output}\nGot: {output}"

    v_bounds = OptimizationVectorHelper.bounds_vectors(ocp)

    v_bounds_min_expected = np.load(FILE_LOCATION + "/v_bounds_min.npy", allow_pickle=False)
    v_bounds_max_expected = np.load(FILE_LOCATION + "/v_bounds_max.npy", allow_pickle=False)

    v_bounds_min_expected = np.vstack((v_bounds_min_expected, tau_min[:, -1:]))
    v_bounds_max_expected = np.vstack((v_bounds_max_expected, tau_max[:, -1:]))

    np.testing.assert_almost_equal(v_bounds[0], v_bounds_min_expected)
    np.testing.assert_almost_equal(v_bounds[1], v_bounds_max_expected)

    v_init = OptimizationVectorHelper.init_vector(ocp)

    v_init_expected = np.load(FILE_LOCATION + "/v_init.npy", allow_pickle=False)
    v_init_expected = np.vstack((v_init_expected, tau_init[:, -1:]))

    np.testing.assert_almost_equal(v_init, v_init_expected)

    sol = Solution.from_vector(ocp, v_init)
    sol.decision_states()
    sol.decision_controls()


def test_vector_layout_linear_continuous_reconstruct():
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    n_shooting = 10
    min_bounds = [-i * 1 for i in range(n_shooting + 1)]
    max_bounds = [i * 1 for i in range(n_shooting + 1)]

    q_min_bounds = np.vstack((min_bounds, np.array(min_bounds) * 2))
    q_max_bounds = np.vstack((max_bounds, np.array(max_bounds) * 2))

    min_bounds = [-i * 0.1 for i in range(n_shooting + 1)]
    max_bounds = [i * 0.1 for i in range(n_shooting + 1)]

    qdot_min_bounds = np.vstack((min_bounds, np.array(min_bounds) * 2))
    qdot_max_bounds = np.vstack((max_bounds, np.array(max_bounds) * 2))

    q_init = np.vstack(([i * 0.01 for i in range(n_shooting + 1)], [i * 0.02 for i in range(n_shooting + 1)]))
    qdot_init = np.vstack([[i * 0.012 for i in range(n_shooting + 1)], [i * 0.024 for i in range(n_shooting + 1)]])

    tau_min = qdot_min_bounds * 1 / 3
    tau_max = qdot_max_bounds * 1 / 3

    init = [i * 0.00015 for i in range(n_shooting + 1)]
    tau_init = np.vstack((init, np.array(init) * 2))

    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        ode_solver=OdeSolver.RK4(),
        control_type=ControlType.LINEAR_CONTINUOUS,
        n_shooting=10,
        interpolation_type=InterpolationType.EACH_FRAME,
        q_min_bounds=q_min_bounds,
        q_max_bounds=q_max_bounds,
        q_init=q_init,
        qdot_min_bounds=qdot_min_bounds,
        qdot_max_bounds=qdot_max_bounds,
        qdot_init=qdot_init,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_init=tau_init,
    )

    v_init = OptimizationVectorHelper.init_vector(ocp)

    sol = Solution.from_vector(ocp, v_init)

    sol_states = sol.decision_states()
    sol_controls = sol.decision_controls()

    np.load(FILE_LOCATION + "/sol_states.npy", allow_pickle=False)
    np.load(FILE_LOCATION + "/sol_states_dot.npy", allow_pickle=False)
    np.load(FILE_LOCATION + "/sol_controls.npy", allow_pickle=False)

    np.testing.assert_almost_equal(
        np.hstack([*sol_states["q"]]), np.load(FILE_LOCATION + "/sol_states.npy", allow_pickle=False)
    )
    np.testing.assert_almost_equal(
        np.hstack([*sol_states["qdot"]]), np.load(FILE_LOCATION + "/sol_states_dot.npy", allow_pickle=False)
    )
    np.testing.assert_almost_equal(
        np.hstack([*sol_controls["tau"]]), np.load(FILE_LOCATION + "/sol_controls.npy", allow_pickle=False)
    )

    dt = ocp.dt_parameter_initial_guess.init
    states_init = ocp.nlp[0].x_init
    controls_init = ocp.nlp[0].u_init
    param_init = ocp.parameter_init
    algebraic_init = ocp.nlp[0].a_init

    sol_reconstructed = Solution.from_initial_guess(ocp, [dt, states_init, controls_init, param_init, algebraic_init])
    sol_states = sol_reconstructed.decision_states()
    sol_controls = sol_reconstructed.decision_controls()

    np.testing.assert_almost_equal(
        np.hstack([*sol_states["q"]]),
        np.load(FILE_LOCATION + "/sol_states.npy", allow_pickle=False),
    )
    np.testing.assert_almost_equal(
        np.hstack([*sol_states["qdot"]]),
        np.load(FILE_LOCATION + "/sol_states_dot.npy", allow_pickle=False),
    )
    np.testing.assert_almost_equal(
        np.hstack([*sol_controls["tau"]]),
        np.load(FILE_LOCATION + "/sol_controls.npy", allow_pickle=False),
    )


def test_parameters():
    from bioptim.examples.getting_started import custom_parameters as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    target_g = np.zeros((3, 1))
    target_g[2] = -9.81
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=3,
        n_shooting=10,
        optim_gravity=True,
        optim_mass=True,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=target_g,
        target_m=20,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )

    v_sym = OptimizationVectorHelper.vector(ocp)
    f = io.StringIO()
    with redirect_stdout(f):
        print(v_sym)

    output = f.getvalue().strip()

    expected_output = (
        "vertcat(dt_phase0, X_scaled_0_0, X_scaled_0_1, X_scaled_0_2, X_scaled_0_3, X_scaled_0_4, "
        "X_scaled_0_5, X_scaled_0_6, X_scaled_0_7, X_scaled_0_8, X_scaled_0_9, X_scaled_0_10, "
        "U_scaled_0_0, U_scaled_0_1, U_scaled_0_2, U_scaled_0_3, U_scaled_0_4, U_scaled_0_5, U_scaled_0_6, "
        "U_scaled_0_7, U_scaled_0_8, U_scaled_0_9, gravity_xyz, mass)"
    )
    assert (
        output == expected_output
    ), f"The output does not match what was expected.\nExpected:{expected_output}\nGot: {output}"

    v_bounds = OptimizationVectorHelper.bounds_vectors(ocp)

    v_bounds_min_expected = np.load(FILE_LOCATION + "/v_bounds_min_parameters.npy", allow_pickle=False)
    v_bounds_max_expected = np.load(FILE_LOCATION + "/v_bounds_max_parameters.npy", allow_pickle=False)

    np.testing.assert_almost_equal(v_bounds[0], v_bounds_min_expected)
    np.testing.assert_almost_equal(v_bounds[1], v_bounds_max_expected)

    v_init = OptimizationVectorHelper.init_vector(ocp)

    v_init_expected = np.load(FILE_LOCATION + "/v_init_parameters.npy", allow_pickle=False)
    np.testing.assert_almost_equal(v_init, v_init_expected)


def test_vector_layout_multiple_phases():
    from bioptim.examples.getting_started import example_multiphase as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        long_optim=False,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics=False,
    )

    v_sym = OptimizationVectorHelper.vector(ocp)
    f = io.StringIO()
    with redirect_stdout(f):
        print(v_sym)
    output = f.getvalue().strip()

    expected_output = (
        "vertcat(dt_phase0, dt_phase1, dt_phase2, X_scaled_0_0, X_scaled_0_1, X_scaled_0_2, X_scaled_0_3, "
        "X_scaled_0_4, X_scaled_0_5, X_scaled_0_6, X_scaled_0_7, X_scaled_0_8, X_scaled_0_9, X_scaled_0_10, "
        "X_scaled_0_11, X_scaled_0_12, X_scaled_0_13, X_scaled_0_14, X_scaled_0_15, X_scaled_0_16, X_scaled_0_17, "
        "X_scaled_0_18, X_scaled_0_19, X_scaled_0_20, X_scaled_1_0, X_scaled_1_1, X_scaled_1_2, X_scaled_1_3, "
        "X_scaled_1_4, X_scaled_1_5, X_scaled_1_6, X_scaled_1_7, X_scaled_1_8, X_scaled_1_9, X_scaled_1_10, "
        "X_scaled_1_11, X_scaled_1_12, X_scaled_1_13, X_scaled_1_14, X_scaled_1_15, X_scaled_1_16, X_scaled_1_17, "
        "X_scaled_1_18, X_scaled_1_19, X_scaled_1_20, X_scaled_1_21, X_scaled_1_22, X_scaled_1_23, X_scaled_1_24, "
        "X_scaled_1_25, X_scaled_1_26, X_scaled_1_27, X_scaled_1_28, X_scaled_1_29, X_scaled_1_30, X_scaled_2_0, "
        "X_scaled_2_1, X_scaled_2_2, X_scaled_2_3, X_scaled_2_4, X_scaled_2_5, X_scaled_2_6, X_scaled_2_7, "
        "X_scaled_2_8, X_scaled_2_9, X_scaled_2_10, X_scaled_2_11, X_scaled_2_12, X_scaled_2_13, X_scaled_2_14, "
        "X_scaled_2_15, X_scaled_2_16, X_scaled_2_17, X_scaled_2_18, X_scaled_2_19, X_scaled_2_20, "
        "U_scaled_0_0, U_scaled_0_1, U_scaled_0_2, U_scaled_0_3, U_scaled_0_4, U_scaled_0_5, U_scaled_0_6, "
        "U_scaled_0_7, U_scaled_0_8, U_scaled_0_9, U_scaled_0_10, U_scaled_0_11, U_scaled_0_12, "
        "U_scaled_0_13, U_scaled_0_14, U_scaled_0_15, U_scaled_0_16, U_scaled_0_17, U_scaled_0_18, "
        "U_scaled_0_19, U_scaled_1_0, U_scaled_1_1, U_scaled_1_2, U_scaled_1_3, U_scaled_1_4, "
        "U_scaled_1_5, U_scaled_1_6, U_scaled_1_7, U_scaled_1_8, U_scaled_1_9, U_scaled_1_10, "
        "U_scaled_1_11, U_scaled_1_12, U_scaled_1_13, U_scaled_1_14, U_scaled_1_15, U_scaled_1_16, "
        "U_scaled_1_17, U_scaled_1_18, U_scaled_1_19, U_scaled_1_20, U_scaled_1_21, U_scaled_1_22, "
        "U_scaled_1_23, U_scaled_1_24, U_scaled_1_25, U_scaled_1_26, U_scaled_1_27, U_scaled_1_28, "
        "U_scaled_1_29, U_scaled_2_0, U_scaled_2_1, U_scaled_2_2, U_scaled_2_3, U_scaled_2_4, "
        "U_scaled_2_5, U_scaled_2_6, U_scaled_2_7, U_scaled_2_8, U_scaled_2_9, U_scaled_2_10, "
        "U_scaled_2_11, U_scaled_2_12, U_scaled_2_13, U_scaled_2_14, U_scaled_2_15, U_scaled_2_16, "
        "U_scaled_2_17, U_scaled_2_18, U_scaled_2_19)"
    )

    assert (
        output == expected_output
    ), f"The output does not match what was expected.\nExpected:{expected_output}\nGot: {output}"

    v_bounds = OptimizationVectorHelper.bounds_vectors(ocp)

    v_bounds_min_expected = np.load(FILE_LOCATION + "/v_bounds_min_multiple_phases.npy", allow_pickle=False)
    v_bounds_max_expected = np.load(FILE_LOCATION + "/v_bounds_max_multiple_phases.npy", allow_pickle=False)

    np.testing.assert_almost_equal(v_bounds[0], v_bounds_min_expected)
    np.testing.assert_almost_equal(v_bounds[1], v_bounds_max_expected)

    v_init = OptimizationVectorHelper.init_vector(ocp)
    v_init_expected = np.load(FILE_LOCATION + "/v_init_multiple_phases.npy", allow_pickle=False)
    np.testing.assert_almost_equal(v_init, v_init_expected)


def test_vector_layout_algebraic_states():
    """Test the holonomic_constraints/two_pendulums_algebraic example"""
    from bioptim.examples.holonomic_constraints import two_pendulums_algebraic

    bioptim_folder = TestUtils.module_folder(two_pendulums_algebraic)

    # --- Prepare the ocp --- #
    ocp, model = two_pendulums_algebraic.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/two_pendulums.bioMod",
        n_shooting=5,
        final_time=1,
        expand_dynamics=False,
    )

    v_sym = OptimizationVectorHelper.vector(ocp)

    f = io.StringIO()
    with redirect_stdout(f):
        print(v_sym)
    output = f.getvalue().strip()

    expected_output = (
        "vertcat(dt_phase0, vec(X_scaled_0_0), vec(X_scaled_0_1), vec(X_scaled_0_2), vec(X_scaled_0_3), "
        "vec(X_scaled_0_4), X_scaled_0_5, "
        "U_scaled_0_0, U_scaled_0_1, U_scaled_0_2, U_scaled_0_3, U_scaled_0_4, "
        "vec(A_scaled_0_0), vec(A_scaled_0_1), vec(A_scaled_0_2), vec(A_scaled_0_3), "
        "vec(A_scaled_0_4), A_scaled_0_5)"
    )

    assert (
        output == expected_output
    ), f"The output does not match what was expected.\nExpected:{expected_output}\nGot: {output}"

    v_bounds = OptimizationVectorHelper.bounds_vectors(ocp)

    v_bounds_min_expected = np.load(FILE_LOCATION + "/v_bounds_min_algebraic_states.npy", allow_pickle=False)
    v_bounds_max_expected = np.load(FILE_LOCATION + "/v_bounds_max_algebraic_states.npy", allow_pickle=False)
    np.testing.assert_almost_equal(v_bounds[0], v_bounds_min_expected)
    np.testing.assert_almost_equal(v_bounds[1], v_bounds_max_expected)

    v_init = OptimizationVectorHelper.init_vector(ocp)
    v_init_expected = np.load(FILE_LOCATION + "/v_init_algebraic_states.npy", allow_pickle=False)
    np.testing.assert_almost_equal(v_init, v_init_expected)
