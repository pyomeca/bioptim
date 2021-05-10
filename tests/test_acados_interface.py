"""
Test for file IO.
It tests the results of an optimal control problem with acados regarding the proper functioning of :
- the handling of mayer and lagrange obj
"""
import os
import shutil
import pytest

import biorbd
import numpy as np
from bioptim import (
    Axis,
    Solver,
    ObjectiveList,
    ObjectiveFcn,
    Bounds,
    QAndQDotBounds,
    OdeSolver,
    ConstraintList,
    ConstraintFcn,
    Node,
    MovingHorizonEstimator,
    Dynamics,
    DynamicsFcn,
    InitialGuess,
    InterpolationType,
)

from .utils import TestUtils


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_no_obj(cost_type):
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/acados/cube.py")
    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/cube.bioMod",
        n_shooting=10,
        tf=2,
    )

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_mayer(cost_type):
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/acados/cube.py")

    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/cube.bioMod",
        n_shooting=10,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0], target=np.array([[1.0]]).T)
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    q = sol.states["q"]
    np.testing.assert_almost_equal(q[0, -1], 1.0)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_several_mayer(cost_type):
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/acados/cube.py")
    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/cube.bioMod",
        n_shooting=10,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0, 1], target=np.array([[1.0, 2.0]]).T)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[2], target=np.array([[3.0]]))
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    q = sol.states["q"]
    np.testing.assert_almost_equal(q[0, -1], 1.0)
    np.testing.assert_almost_equal(q[1, -1], 2.0)
    np.testing.assert_almost_equal(q[2, -1], 3.0)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_lagrange(cost_type):
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/acados/cube.py")
    n_shooting = 10
    target = np.expand_dims(np.arange(0, n_shooting + 1), axis=0)
    target[0, -1] = n_shooting - 2
    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/cube.bioMod",
        n_shooting=n_shooting,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=10, index=[0], target=target)
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    q = sol.states["q"]
    np.testing.assert_almost_equal(q[0, :], target[0, :].squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_lagrange_and_one_mayer(cost_type):
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/acados/cube.py")
    n_shooting = 10
    target = np.expand_dims(np.arange(0, n_shooting + 1), axis=0)
    target[0, -1] = n_shooting - 2
    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/cube.bioMod",
        n_shooting=n_shooting,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, weight=10, index=[0], target=target)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0], target=target[:, -1:])
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    q = sol.states["q"]
    np.testing.assert_almost_equal(q[0, :], target[0, :].squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_control_lagrange_and_state_mayer(cost_type):
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/acados/cube.py")
    n_shooting = 10
    target = np.array([[2]])
    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/cube.bioMod",
        n_shooting=n_shooting,
        tf=2,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_ALL_CONTROLS,
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, index=[0], target=target, weight=1000)
    ocp.update_objectives(objective_functions)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"cost_type": cost_type})

    # Check end state value
    q = sol.states["q"]
    np.testing.assert_almost_equal(q[0, -1], target.squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("solver", [Solver.ACADOS, Solver.IPOPT])
def test_acados_mhe(solver):
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    pendulum = TestUtils.load_module(root_folder + "mhe.py")
    biorbd_model = biorbd.Model(root_folder + "cart_pendulum.bioMod")
    nq = biorbd_model.nbQ()
    torque_max = 5  # Give a bit of slack on the max torque

    n_cycles = 5 if solver == Solver.ACADOS else 1
    n_frame_by_cycle = 20
    window_len = 5
    window_duration = 0.2

    final_time = window_duration / window_len * n_cycles * n_frame_by_cycle
    x_init = np.zeros((nq * 2, window_len + 1))
    u_init = np.zeros((nq, window_len))

    target_q, _, _, _ = pendulum.generate_data(
        biorbd_model, final_time, [0, np.pi / 2, 0, 0], torque_max, n_cycles * n_frame_by_cycle, 0
    )
    target = pendulum.states_to_markers(biorbd_model, target_q)

    def update_functions(mhe, t, _):
        def target_func(i: int):
            return target[:, :, i : i + window_len + 1]

        mhe.update_objectives_target(target=target_func(t), list_index=0)
        return t < n_frame_by_cycle * n_cycles - window_len - 1

    biorbd_model = biorbd.Model(root_folder + "cart_pendulum.bioMod")
    sol = pendulum.prepare_mhe(
        biorbd_model=biorbd_model,
        window_len=window_len,
        window_duration=window_duration,
        max_torque=torque_max,
        x_init=x_init,
        u_init=u_init,
    ).solve(update_functions, **pendulum.get_solver_options(solver))

    # Clean test folder
    if solver == Solver.ACADOS:
        # Compare the position on the first few frames (only ACADOS, since IPOPT is not precise with current options)
        np.testing.assert_almost_equal(
            sol.states["q"][:, : -2 * window_len], target_q[:nq, : -3 * window_len - 1], decimal=3
        )
        os.remove(f"./acados_ocp.json")
        shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_options(cost_type):
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/acados/pendulum.py")
    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/pendulum.bioMod",
        final_time=3,
        n_shooting=12,
    )

    tol = [1e-1, 1e-0, 1e1]
    iter = []
    for i in range(3):
        solver_options = {"nlp_solver_tol_stat": tol[i], "cost_type": cost_type}
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=solver_options)
        iter += [sol.iterations]

    # Check that tol impacted convergence
    np.testing.assert_array_less(iter[1], iter[0])
    np.testing.assert_array_less(iter[2], iter[1])

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_fail_external():
    bioptim_folder = TestUtils.bioptim_folder()
    pendulum = TestUtils.load_module(bioptim_folder + "/examples/acados/pendulum.py")
    ocp = pendulum.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/pendulum.bioMod",
        final_time=1,
        n_shooting=2,
    )

    solver_options = {"cost_type": "EXTERNAL"}

    with pytest.raises(RuntimeError, match="EXTERNAL is not interfaced yet, please use NONLINEAR_LS"):
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=solver_options)


def test_acados_fail_lls():
    bioptim_folder = TestUtils.bioptim_folder()
    arm = TestUtils.load_module(bioptim_folder + "/examples/acados/static_arm.py")
    ocp = arm.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/arm26.bioMod",
        final_time=1,
        n_shooting=2,
        use_sx=True,
    )

    solver_options = {"cost_type": "LINEAR_LS"}

    with pytest.raises(
        RuntimeError, match="SUPERIMPOSE_MARKERS is an incompatible objective term with LINEAR_LS cost type"
    ):
        sol = ocp.solve(solver=Solver.ACADOS, solver_options=solver_options)


@pytest.mark.parametrize("problem_type_custom", [True, False])
def test_acados_custom_dynamics(problem_type_custom):
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_dynamics.py")
    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/cube.bioMod",
        problem_type_custom=problem_type_custom,
        ode_solver=OdeSolver.RK4(),
        use_sx=True,
    )
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)
    ocp.update_constraints(constraints)
    sol = ocp.solve(solver=Solver.ACADOS)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((2, 0, 0)), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.81, 2.27903226)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 9.81, -2.27903226)))


def test_acados_one_parameter():
    bioptim_folder = TestUtils.bioptim_folder()
    parameters = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_parameters.py")
    ocp = parameters.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        n_shooting=100,
        optim_gravity=True,
        optim_mass=False,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=np.array([0, 0, -9.81]),
        target_m=20,
        use_sx=True,
    )
    model = ocp.nlp[0].model
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, index=[0, 1], target=np.array([[0, 3.14]]).T, weight=100000)
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, index=[2, 3], target=np.array([[0, 0]]).T, weight=100)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, index=1, weight=10)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, index=[2, 3], weight=0.000000010)
    ocp.update_objectives(objectives)

    # Path constraint
    x_bounds = QAndQDotBounds(model)
    x_bounds[[0, 1, 2, 3], 0] = 0
    u_bounds = Bounds([-300] * model.nbQ(), [300] * model.nbQ())
    ocp.update_bounds(x_bounds, u_bounds)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"nlp_solver_tol_eq": 1e-3})

    # Check some of the results
    q, qdot, tau, gravity = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.parameters["gravity_xyz"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)), decimal=6)

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)), decimal=6)

    # parameters
    np.testing.assert_almost_equal(gravity[-1, :], np.array([-9.81]), decimal=6)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_several_parameter():
    bioptim_folder = TestUtils.bioptim_folder()
    parameters = TestUtils.load_module(bioptim_folder + "/examples/getting_started/custom_parameters.py")
    ocp = parameters.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/getting_started/pendulum.bioMod",
        final_time=2,
        n_shooting=100,
        optim_gravity=True,
        optim_mass=True,
        min_g=np.array([-1, -1, -10]),
        max_g=np.array([1, 1, -5]),
        min_m=10,
        max_m=30,
        target_g=np.array([0, 0, -9.81]),
        target_m=20,
        use_sx=True,
    )
    model = ocp.nlp[0].model
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, index=[0, 1], target=np.array([[0, 3.14]]).T, weight=100000)
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, index=[2, 3], target=np.array([[0, 0]]).T, weight=100)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, index=1, weight=10)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, index=[2, 3], weight=0.000000010)
    ocp.update_objectives(objectives)

    # Path constraint
    x_bounds = QAndQDotBounds(model)
    x_bounds[[0, 1, 2, 3], 0] = 0
    u_bounds = Bounds([-300] * model.nbQ(), [300] * model.nbQ())
    ocp.update_bounds(x_bounds, u_bounds)

    sol = ocp.solve(solver=Solver.ACADOS, solver_options={"nlp_solver_tol_eq": 1e-3})

    # Check some of the results
    q, qdot, tau, gravity, mass = (
        sol.states["q"],
        sol.states["qdot"],
        sol.controls["tau"],
        sol.parameters["gravity_xyz"],
        sol.parameters["mass"],
    )

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)), decimal=6)

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)), decimal=6)

    # parameters
    np.testing.assert_almost_equal(gravity[-1, :], np.array([-9.81]), decimal=6)
    np.testing.assert_almost_equal(mass, np.array([[20]]), decimal=6)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_one_end_constraints():
    bioptim_folder = TestUtils.bioptim_folder()
    cube = TestUtils.load_module(bioptim_folder + "/examples/acados/cube.py")
    ocp = cube.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/acados/cube.bioMod",
        n_shooting=10,
        tf=2,
    )

    model = ocp.nlp[0].model
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, index=0, weight=100, target=np.array([[1]]))
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=100)
    ocp.update_objectives(objective_functions)

    # Path constraint
    x_bounds = QAndQDotBounds(model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[0, 0] = 0
    ocp.update_bounds(x_bounds=x_bounds)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 0)), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((2.72727272, 9.81, 0)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -1], np.array((-2.72727272, 9.81, 0)), decimal=6)


def test_acados_constraints_all():
    bioptim_folder = TestUtils.bioptim_folder()
    track = TestUtils.load_module(bioptim_folder + "/examples/track/track_marker_on_segment.py")
    ocp = track.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/track/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
        constr=False,
        use_sx=True,
    )

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL, marker_idx=1, segment_idx=2, axis=(Axis.X)
    )
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.8385190835, 0, 0, -0.212027938]), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((0.8385190835, 0, 1.57, -0.212027938)), decimal=6)

    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.81, 2.27903226, 0)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 9.81, -2.27903226, 0)), decimal=6)


def test_acados_constraints_end_all():
    bioptim_folder = TestUtils.bioptim_folder()
    track = TestUtils.load_module(bioptim_folder + "/examples/track/track_marker_on_segment.py")
    ocp = track.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/track/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
        constr=False,
        use_sx=True,
    )

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=5)
    constraints.add(
        ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL, marker_idx=1, segment_idx=2, axis=(Axis.X)
    )
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, 0], np.array([2, 0, 0, -0.139146705]), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57, -0.139146705)), decimal=6)

    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.81, 2.27903226, 0)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 9.81, -2.27903226, 0)), decimal=6)


@pytest.mark.parametrize("failing", ["u_bounds", "x_bounds"])
def test_acados_bounds_not_implemented(failing):
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    biorbd_model = biorbd.Model(root_folder + "cart_pendulum.bioMod")
    nq = biorbd_model.nbQ()
    ntau = biorbd_model.nbGeneralizedTorque()

    n_cycles = 3
    window_len = 5
    window_duration = 0.2
    x_init = InitialGuess(np.zeros((nq * 2, 1)), interpolation=InterpolationType.CONSTANT)
    u_init = InitialGuess(np.zeros((ntau, 1)), interpolation=InterpolationType.CONSTANT)
    if failing == "u_bounds":
        x_bounds = Bounds(np.zeros((nq * 2, 1)), np.zeros((nq * 2, 1)))
        u_bounds = Bounds(np.zeros((ntau, 1)),  np.zeros((ntau, 1)), interpolation=InterpolationType.CONSTANT)
    elif failing == "x_bounds":
        x_bounds = Bounds(np.zeros((nq * 2, 1)), np.zeros((nq * 2, 1)), interpolation=InterpolationType.CONSTANT)
        u_bounds = Bounds(np.zeros((ntau, 1)), np.zeros((ntau, 1)))
    else:
        raise ValueError("Wrong value for failing")

    mhe = MovingHorizonEstimator(
        biorbd_model,
        Dynamics(DynamicsFcn.TORQUE_DRIVEN),
        window_len,
        window_duration,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        n_threads=4,
    )

    def update_functions(mhe, t, _):
        return t < n_cycles

    with pytest.raises(
            NotImplementedError,
            match=f"ACADOS must declare an InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT for the {failing}"):
        mhe.solve(update_functions, Solver.ACADOS)
