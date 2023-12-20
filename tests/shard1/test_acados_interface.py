"""
Test for file IO.
It tests the results of an optimal control problem with acados regarding the proper functioning of :
- the handling of mayer and lagrange obj
"""
import os
import shutil
import pytest
from sys import platform

import numpy as np
from bioptim import (
    BiorbdModel,
    Axis,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    ConstraintList,
    ConstraintFcn,
    Node,
    MovingHorizonEstimator,
    Dynamics,
    DynamicsFcn,
    InitialGuessList,
    InterpolationType,
    Solver,
    BoundsList,
    PhaseDynamics,
    SolutionMerge,
)

from tests.utils import TestUtils


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_no_obj(cost_type):
    if platform == "win32":
        return

    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_mayer(cost_type):
    if platform == "win32":
        return

    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod", n_shooting=10, tf=2, expand_dynamics=True
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0], target=np.array([[1.0]]).T)
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    np.testing.assert_almost_equal(q[0, -1], 1.0)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_mayer_first_node(cost_type):
    if platform == "win32":
        return

    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, node=Node.START, key="q", index=[0], target=np.array([[1.0]]).T
    )
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    np.testing.assert_almost_equal(q[0, 0], 0.999999948505021)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_several_mayer(cost_type):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0, 1], target=np.array([[1.0, 2.0]]).T)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[2], target=np.array([[3.0]]))
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    np.testing.assert_almost_equal(q[0, -1], 1.0)
    np.testing.assert_almost_equal(q[1, -1], 2.0)
    np.testing.assert_almost_equal(q[2, -1], 3.0)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_lagrange(cost_type):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_shooting = 10
    target = np.expand_dims(np.arange(0, n_shooting + 1), axis=0)
    target[0, -1] = n_shooting - 2
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=n_shooting,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="q",
        node=Node.ALL,
        weight=10,
        index=[0],
        target=target,
        multi_thread=False,
    )
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    np.testing.assert_almost_equal(q[0, :], target[0, :].squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_one_lagrange_and_one_mayer(cost_type):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_shooting = 10
    target = np.expand_dims(np.arange(0, n_shooting + 1), axis=0)
    target[0, -1] = n_shooting - 2
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=n_shooting,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="q",
        node=Node.ALL,
        weight=10,
        index=[0],
        target=target,
        multi_thread=False,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0], target=target[:, -1:], multi_thread=False
    )
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    np.testing.assert_almost_equal(q[0, :], target[0, :].squeeze(), decimal=6)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_control_lagrange_and_state_mayer(cost_type):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    n_shooting = 10
    target = np.array([[2]])
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=n_shooting,
        tf=2,
        expand_dynamics=True,
    )
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", multi_thread=False)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=[0], target=target, weight=1000, multi_thread=False
    )
    ocp.update_objectives(objective_functions)

    solver = Solver.ACADOS()
    solver.set_cost_type(cost_type)
    sol = ocp.solve(solver=solver)

    # Check end state value
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]
    np.testing.assert_almost_equal(q[0, -1], target.squeeze())

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


@pytest.mark.parametrize("cost_type", ["LINEAR_LS", "NONLINEAR_LS"])
def test_acados_options(cost_type):
    if platform == "win32" or platform == "darwin":
        print("Tests for ACADOS options on Windows and Mac are skipped")
        return

    from bioptim.examples.acados import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=0.6,
        n_shooting=200,
        expand_dynamics=True,
    )

    tols = [1e-1, 1e1]
    iter = []
    for tol in tols:
        solver = Solver.ACADOS()
        solver.set_cost_type(cost_type)
        solver.set_nlp_solver_tol_stat(tol)
        sol = ocp.solve(solver=solver)
        iter += [sol.iterations]

    # Check that tol impacted convergence
    for i in range(len(tols) - 1):
        np.testing.assert_array_less(iter[i + 1], iter[i])

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_fail_external():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.acados import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=2,
        expand_dynamics=True,
    )

    solver = Solver.ACADOS()
    solver.set_cost_type("EXTERNAL")

    with pytest.raises(RuntimeError, match="EXTERNAL is not interfaced yet, please use NONLINEAR_LS"):
        sol = ocp.solve(solver=solver)


def test_acados_fail_lls():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.acados import static_arm as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/arm26.bioMod",
        final_time=1,
        n_shooting=2,
        use_sx=True,
        expand_dynamics=True,
    )

    solver = Solver.ACADOS()
    solver.set_cost_type("LINEAR_LS")

    with pytest.raises(
        RuntimeError, match="SUPERIMPOSE_MARKERS is an incompatible objective term with LINEAR_LS cost type"
    ):
        sol = ocp.solve(solver=solver)


@pytest.mark.parametrize("problem_type_custom", [True, False])
def test_acados_custom_dynamics(problem_type_custom):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.getting_started import custom_dynamics as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        problem_type_custom=problem_type_custom,
        ode_solver=OdeSolver.RK4(),
        use_sx=True,
        expand_dynamics=True,
    )
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")
    ocp.update_constraints(constraints)
    sol = ocp.solve(solver=Solver.ACADOS())

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

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
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.getting_started import custom_parameters as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
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
        expand_dynamics=True,
    )
    model = ocp.nlp[0].model
    objectives = ObjectiveList()
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, key="q", target=np.array([[0, 3.14]]).T, weight=100000)
    objectives.add(ObjectiveFcn.Mayer.TRACK_STATE, key="qdot", target=np.array([[0, 0]]).T, weight=100)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=1, weight=10, multi_thread=False)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=0.000000010, multi_thread=False)
    ocp.update_objectives(objectives)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    u_bounds = BoundsList()
    u_bounds["tau"] = [-300] * model.nb_q, [300] * model.nb_q

    ocp.update_bounds(x_bounds, u_bounds)

    solver = Solver.ACADOS()
    solver.set_nlp_solver_tol_eq(1e-3)
    sol = ocp.solve(solver=solver)

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    gravity = sol.parameters["gravity_xyz"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)), decimal=6)

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)), decimal=6)

    # parameters
    np.testing.assert_almost_equal(gravity[-1, :], np.array([-9.80995]), decimal=4)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_several_parameter():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.getting_started import custom_parameters as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
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
        expand_dynamics=True,
    )
    model = ocp.nlp[0].model
    objectives = ObjectiveList()
    objectives.add(
        ObjectiveFcn.Mayer.TRACK_STATE, key="q", target=np.array([[0, 3.14]]).T, weight=100000, multi_thread=False
    )
    objectives.add(
        ObjectiveFcn.Mayer.TRACK_STATE, key="qdot", target=np.array([[0, 0]]).T, weight=100, multi_thread=False
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=1, weight=10, multi_thread=False)
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=0.000000010, multi_thread=False)
    ocp.update_objectives(objectives)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    u_bounds = BoundsList()
    u_bounds["tau"] = [-300] * model.nb_q, [300] * model.nb_q

    ocp.update_bounds(x_bounds, u_bounds)

    solver = Solver.ACADOS()
    solver.set_nlp_solver_tol_eq(1e-3)
    sol = ocp.solve(solver=solver)

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    gravity, mass = sol.parameters["gravity_xyz"], sol.parameters["mass"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)), decimal=6)

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)), decimal=6)

    # parameters
    np.testing.assert_almost_equal(gravity[-1, :], np.array([-9.80996]), decimal=4)
    np.testing.assert_almost_equal(mass, np.array([[20]]), decimal=6)

    # Clean test folder
    os.remove(f"./acados_ocp.json")
    shutil.rmtree(f"./c_generated_code/")


def test_acados_one_end_constraints():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.acados import cube as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=10,
        tf=2,
        expand_dynamics=True,
    )

    model = ocp.nlp[0].model
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_STATE, index=0, key="q", weight=100, target=np.array([[1]]), multi_thread=False
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)
    ocp.update_objectives(objective_functions)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = model.bounds_from_ranges("q")
    x_bounds["q"][1:, [0, -1]] = 0
    x_bounds["qdot"] = model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    x_bounds["q"][0, 0] = 0

    ocp.update_bounds(x_bounds=x_bounds)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS())

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((2.72727272, 9.81, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-2.72727272, 9.81, 0)))


def test_acados_constraints_all():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.track import track_marker_on_segment as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
        constr=False,
        use_sx=True,
        expand_dynamics=True,
    )

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL, marker="m1", segment="seg_rt", axis=Axis.X
    )
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS())

    # Check some results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, 0], np.array([2.28988221, 0, 0, 2.95087911e-01]), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((2.28215749, 0, 1.57, 6.62470772e-01)), decimal=6)

    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.04483914, 9.90739842, 2.24951691, 0.78496612)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -1], np.array((0.15945561, 10.03978178, -2.36075327, 0.07267697)), decimal=6)


def test_acados_constraints_end_all():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.track import track_marker_on_segment as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube_and_line.bioMod",
        n_shooting=30,
        final_time=2,
        initialize_near_solution=True,
        constr=False,
        use_sx=True,
        expand_dynamics=True,
    )

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m5")
    constraints.add(
        ConstraintFcn.TRACK_MARKER_WITH_SEGMENT_AXIS, node=Node.ALL_SHOOTING, marker="m1", segment="seg_rt", axis=Axis.X
    )
    ocp.update_constraints(constraints)

    sol = ocp.solve(solver=Solver.ACADOS())

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # final position
    np.testing.assert_almost_equal(q[:, 0], np.array([2.01701330, 0, 0, 3.20057865e-01]), decimal=6)
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57, 7.85398168e-01)), decimal=6)

    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0, 0, 0]), decimal=6)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0, 0, 0, 0]), decimal=6)

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.04648408, 9.88616194, 2.24285498, 0.864213)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -1], np.array((0.19389194, 9.99905781, -2.37713652, -0.19858311)), decimal=6)


def test_acados_phase_dynamics_reject():
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return

    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=10,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
        expand_dynamics=True,
    )

    with pytest.raises(RuntimeError, match=f"ACADOS necessitate phase_dynamics==PhaseDynamics.SHARED_DURING_THE_PHASE"):
        ocp.solve(solver=Solver.ACADOS())


@pytest.mark.parametrize("failing", ["u_bounds", "x_bounds"])
def test_acados_bounds_not_implemented(failing):
    if platform == "win32":
        print("Test for ACADOS on Windows is skipped")
        return
    root_folder = TestUtils.bioptim_folder() + "/examples/moving_horizon_estimation/"
    bio_model = BiorbdModel(root_folder + "models/cart_pendulum.bioMod")

    nq = bio_model.nb_q
    ntau = bio_model.nb_tau

    n_cycles = 3
    window_len = 5
    window_duration = 0.2
    x_init = InitialGuessList()
    x_init["final"] = np.zeros((nq * 2, 1))
    u_init = InitialGuessList()
    u_init["final"] = np.zeros((ntau, 1))
    if failing == "u_bounds":
        x_bounds = BoundsList()
        x_bounds.add("q", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)))
        x_bounds.add("qdot", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)))
        u_bounds = BoundsList()
        u_bounds.add(
            "tau",
            min_bound=np.zeros((ntau, 1)),
            max_bound=np.zeros((ntau, 1)),
            interpolation=InterpolationType.CONSTANT,
        )
    elif failing == "x_bounds":
        x_bounds = BoundsList()
        x_bounds.add(
            "q", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)), interpolation=InterpolationType.CONSTANT
        )
        x_bounds.add(
            "qdot", min_bound=np.zeros((nq, 1)), max_bound=np.zeros((nq, 1)), interpolation=InterpolationType.CONSTANT
        )
        u_bounds = BoundsList()
        u_bounds.add("tau", min_bound=np.zeros((ntau, 1)), max_bound=np.zeros((ntau, 1)))
    else:
        raise ValueError("Wrong value for failing")

    mhe = MovingHorizonEstimator(
        bio_model,
        Dynamics(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True),
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
        match=f"ACADOS must declare an InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT for the {failing}",
    ):
        mhe.solve(update_functions, Solver.ACADOS())
