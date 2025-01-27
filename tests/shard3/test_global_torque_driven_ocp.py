"""
Test for file IO
"""

import platform

from bioptim import (
    OdeSolver,
    ConstraintList,
    ConstraintFcn,
    Node,
    Solver,
    TorqueBiorbdModel,
    PhaseDynamics,
    SolutionMerge,
)
from bioptim.models.biorbd.viewer_utils import _prepare_tracked_markers_for_animation
import numpy.testing as npt
import numpy as np
import pytest

from ..utils import TestUtils


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
@pytest.mark.parametrize("actuator_type", [None, 2])
def test_track_markers(ode_solver, actuator_type, phase_dynamics):
    # Load track_markers
    from bioptim.examples.toy_examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        return

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_with_actuators.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=actuator_type,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=19767.53312569522)

    # Check constraints
    g = np.array(sol.constraints)
    if not actuator_type:
        npt.assert_equal(g.shape, (186, 1))
    else:
        npt.assert_equal(g.shape, (366, 1))
    npt.assert_almost_equal(g[:186], np.zeros((186, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
    npt.assert_almost_equal(tau[:, -1], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_markers_changing_constraints(ode_solver, phase_dynamics):
    # Load track_markers
    from bioptim.examples.toy_examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        return

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_with_actuators.bioMod",
        n_shooting=30,
        final_time=2,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Add a new constraint and reoptimize
    new_constraints = ConstraintList()
    new_constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="m0", second_marker="m2", list_index=2
    )
    ocp.update_constraints(new_constraints)
    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=20370.211697123825)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (189, 1))
    npt.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((4.2641129, 9.81, 2.27903226)))
    npt.assert_almost_equal(tau[:, -1], np.array((1.36088709, 9.81, -2.27903226)))

    # simulate
    TestUtils.simulate(sol)

    # Replace constraints and reoptimize
    new_constraints = ConstraintList()
    new_constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m2", list_index=0
    )
    new_constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.MID, first_marker="m0", second_marker="m3", list_index=2
    )
    ocp.update_constraints(new_constraints)
    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=31670.93770220887)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (189, 1))
    npt.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((2, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((-5.625, 21.06, 2.2790323)))
    npt.assert_almost_equal(tau[:, -1], np.array((-5.625, 21.06, -2.27903226)))

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_markers_with_actuators(ode_solver, phase_dynamics):
    # Load track_markers
    from bioptim.examples.toy_examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    # For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped for redundant tests
    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.RK8:
        return

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/cube_with_actuators.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=1,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=204.18087334169184)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (186, 1))
    npt.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    npt.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((0.2140175, 0.981, 0.3360075)))
    npt.assert_almost_equal(tau[:, -1], np.array((-0.2196496, 0.981, -0.3448498)))

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.IRK, OdeSolver.COLLOCATION])
def test_track_marker_2D_pendulum(ode_solver, phase_dynamics):
    # Load muscle_activations_contact_tracker
    from bioptim.examples.toy_examples.torque_driven_ocp import track_markers_2D_pendulum as ocp_module

    if phase_dynamics == PhaseDynamics.ONE_PER_NODE and ode_solver == OdeSolver.COLLOCATION:
        pytest.skip(
            "For reducing time phase_dynamics == PhaseDynamics.ONE_PER_NODE is skipped as it is a redundant tests"
        )

    bioptim_folder = TestUtils.bioptim_folder()

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/examples/models/pendulum.bioMod"
    bio_model = TorqueBiorbdModel(model_path)

    final_time = 2
    n_shooting = 30

    # Generate data to fit
    np.random.seed(42)
    markers_ref = np.random.rand(3, 2, n_shooting + 1)
    tau_ref = np.random.rand(2, n_shooting)

    if isinstance(ode_solver, OdeSolver.IRK):
        tau_ref = tau_ref * 5

    ocp = ocp_module.prepare_ocp(
        bio_model,
        final_time,
        n_shooting,
        markers_ref,
        tau_ref,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )

    # Check the values which will be sent to the solver
    np.random.seed(42)
    match ode_solver_orig:
        case OdeSolver.COLLOCATION:
            v_len = 665
            expected = [329.58704584455836, 45.86799945455372, 76.30823619468703]
        case OdeSolver.IRK:
            v_len = 185
            expected = [87.49523141142917, 194.20847154483175, 4027.416142481593]
        case _:
            raise ValueError("Test not implemented")

    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.random.rand(v_len, 1),
        expected_v_f_g=expected,
        decimal=6,
    )
    if platform.system() == "Windows":
        return

    bioptim_folder = TestUtils.module_folder(ocp_module)

    ode_solver_orig = ode_solver
    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/models/pendulum.bioMod"
    bio_model = TorqueBiorbdModel(model_path)

    final_time = 2
    n_shooting = 30

    # Generate data to fit
    np.random.seed(42)
    markers_ref = np.random.rand(3, 2, n_shooting + 1)
    tau_ref = np.random.rand(2, n_shooting)

    if isinstance(ode_solver, OdeSolver.IRK):
        tau_ref = tau_ref * 5

    ocp = ocp_module.prepare_ocp(
        bio_model,
        final_time,
        n_shooting,
        markers_ref,
        tau_ref,
        ode_solver=ode_solver,
        expand_dynamics=ode_solver_orig != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    if isinstance(ode_solver, OdeSolver.IRK):
        npt.assert_equal(g.shape, (n_shooting * 4, 1))
        npt.assert_almost_equal(g, np.zeros((n_shooting * 4, 1)))

        # Check objective function value
        TestUtils.assert_objective_value(sol=sol, expected_value=47.19432362677269)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array((0, 0)))
        npt.assert_almost_equal(q[:, -1], np.array((0.60535943, 1.02166394)))

        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        npt.assert_almost_equal(qdot[:, -1], np.array((1.40706186, 0.50344619)))

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((2.11556124, 1.8670448)))
        npt.assert_almost_equal(tau[:, -1], np.array((1.15750458, 4.66081778)))

    else:
        npt.assert_equal(g.shape, (n_shooting * 4 * 5, 1))
        npt.assert_almost_equal(g, np.zeros((n_shooting * 4 * 5, 1)))

        # Check objective function value
        TestUtils.assert_objective_value(sol=sol, expected_value=40.92496227517089)

        # initial and final position
        npt.assert_almost_equal(q[:, 0], np.array((0, 0)))
        npt.assert_almost_equal(q[:, -1], np.array((0.38321534, 0.46333863)))

        # initial and final velocities
        npt.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        npt.assert_almost_equal(qdot[:, -1], np.array((0.91690887, -0.4747445)))

        # initial and final controls
        npt.assert_almost_equal(tau[:, 0], np.array((1.13684199, 0.43000804)))
        npt.assert_almost_equal(tau[:, -1], np.array((0.21069152, 0.95289673)))

    # simulate
    TestUtils.simulate(sol)

    # testing that preparing tracked markers for animation properly works
    tracked_markers = _prepare_tracked_markers_for_animation(sol.ocp.nlp, n_shooting)
    npt.assert_equal(tracked_markers[0].shape, (3, 2, n_shooting + 1))
    npt.assert_equal(tracked_markers[0][0, :, :], np.zeros((2, n_shooting + 1)))
    npt.assert_almost_equal(tracked_markers[0][1:, :, 0], np.array([[0.82873751, 0.5612772], [0.22793516, 0.24205527]]))
    npt.assert_almost_equal(tracked_markers[0][1:, :, 5], np.array([[0.80219698, 0.02541913], [0.5107473, 0.36778313]]))
    npt.assert_almost_equal(
        tracked_markers[0][1:, :, -1], np.array([[0.76078505, 0.11005192], [0.98565045, 0.65998405]])
    )

    # testing that preparing tracked markers for animation properly works
    tracked_markers = _prepare_tracked_markers_for_animation(sol.ocp.nlp, None)
    npt.assert_equal(tracked_markers[0].shape, (3, 2, n_shooting + 1))
    npt.assert_equal(tracked_markers[0][0, :, :], np.zeros((2, n_shooting + 1)))
    npt.assert_almost_equal(tracked_markers[0][1:, :, 0], np.array([[0.82873751, 0.5612772], [0.22793516, 0.24205527]]))
    npt.assert_almost_equal(tracked_markers[0][1:, :, 5], np.array([[0.80219698, 0.02541913], [0.5107473, 0.36778313]]))
    npt.assert_almost_equal(
        tracked_markers[0][1:, :, -1], np.array([[0.76078505, 0.11005192], [0.98565045, 0.65998405]])
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE])
def test_example_quaternions(phase_dynamics):
    from bioptim.examples.toy_examples.torque_driven_ocp import example_quaternions as ocp_module

    if platform.system() == "Windows":
        pytest.skip("This OCP does not converge on Windows.")

    bioptim_folder = TestUtils.bioptim_folder()

    # Define the problem
    model_path = bioptim_folder + "/examples/models/trunk_and_2arm_quaternion.bioMod"
    final_time = 0.25
    n_shooting = 6

    ocp = ocp_module.prepare_ocp(
        model_path,
        n_shooting,
        final_time,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    sol = ocp.solve()
    assert sol.status == 0  # The optimization converged

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=4.899532845500326)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (162, 1))
    npt.assert_almost_equal(g[:160], np.zeros((160, 1)), decimal=6)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q_roots, q_joints, qdot_roots, qdot_joints, tau_joints = (
        states["q_roots"],
        states["q_joints"],
        states["qdot_roots"],
        states["qdot_joints"],
        controls["tau_joints"],
    )

    # initial and final position
    npt.assert_almost_equal(q_roots[:, 0], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    npt.assert_almost_equal(
        q_joints[:, 0], np.array([0.0, -0.9999875, 0.0, 0.0, 0.9999875, 0.0, 0.00499998, 0.00499998])
    )
    npt.assert_almost_equal(
        q_roots[:, -1],
        np.array([0.00475187, 0.00384924, 0.0326082, -0.07208434, -0.00475046, 0.18357191]),
    )

    npt.assert_almost_equal(
        q_joints[:, -1],
        np.array([-0.58188362, -0.46984388, 0.05948255, 0.99209049, 0.10564735, -0.06755123, 0.66115052, 0.0056508]),
    )

    # initial and final velocities
    npt.assert_almost_equal(
        qdot_roots[:, 0],
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    npt.assert_almost_equal(
        qdot_joints[:, 0],
        np.array([-0.09999995, 0.09999997, -0.09996661, -0.09999998, -0.09999998, -0.09998458]),
    )
    npt.assert_almost_equal(
        qdot_roots[:, -1],
        np.array([0.00377627, 0.00054556, 0.02098995, -0.01712139, -0.00534792, 0.07725791]),
    )

    npt.assert_almost_equal(
        qdot_joints[:, -1],
        np.array([0.39154646, 0.30091969, 0.93232889, 0.34127671, 0.64430008, 1.00960836]),
    )

    # initial and final controls
    npt.assert_almost_equal(
        tau_joints[:, 0],
        np.array([-0.01994007, 0.03368354, -0.00069303, -0.05415661, -0.04957538, -0.00121142]),
        decimal=6,
    )
    npt.assert_almost_equal(
        tau_joints[:, -1],
        np.array([2.40238785e-03, 2.21611332e-03, 3.07280623e-04, -1.10481106e-03, 2.89865386e-04, -4.19860530e-05]),
        decimal=6,
    )

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_phase_transition_uneven_variable_number_by_bounds(phase_dynamics):
    # Load phase_transition_uneven_variable_number_by_bounds
    from bioptim.examples.toy_examples.torque_driven_ocp import (
        phase_transition_uneven_variable_number_by_bounds as ocp_module,
    )

    bioptim_folder = TestUtils.bioptim_folder()

    # Define the problem
    biorbd_model_path_with_translations = bioptim_folder + "/examples/models/double_pendulum_with_translations.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path_with_translations=biorbd_model_path_with_translations,
        n_shooting=(10, 10),
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(10)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (170, 1))
    npt.assert_equal(sol.status, 1)  # Did not converge, therefore the constraints won't be zero


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
def test_phase_transition_uneven_variable_number_by_mapping(phase_dynamics):
    # Load phase_transition_uneven_variable_number_by_mapping
    from bioptim.examples.toy_examples.torque_driven_ocp import (
        phase_transition_uneven_variable_number_by_mapping as ocp_module,
    )

    bioptim_folder = TestUtils.bioptim_folder()

    # Define the problem
    biorbd_model_path = bioptim_folder + "/examples/models/double_pendulum.bioMod"
    biorbd_model_path_with_translations = bioptim_folder + "/examples/models/double_pendulum_with_translations.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        biorbd_model_path_with_translations=biorbd_model_path_with_translations,
        n_shooting=(10, 10),
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )
    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=-12397.11475053)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (126, 1))
    npt.assert_almost_equal(g[:44], np.zeros((44, 1)), decimal=6)
    npt.assert_almost_equal(g[44], np.array([0.45544143]), decimal=6)  # Time constraint with min / max bounds phase 0
    npt.assert_almost_equal(g[45:-1], np.zeros((80, 1)), decimal=6)
    npt.assert_almost_equal(g[-1], np.array([1.00939513]), decimal=6)  # Time constraint with min / max bounds phase 1

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # initial and final position
    npt.assert_almost_equal(states[0]["q"][:, 0], np.array([3.14, 0.0]))
    npt.assert_almost_equal(states[0]["q"][:, -1], np.array([7.93025703, 0.31520724]))
    npt.assert_almost_equal(states[1]["q"][:, 0], np.array([0.0, 0.0, 7.93025703, 0.31520724]))
    npt.assert_almost_equal(states[1]["q"][:, -1], np.array([-0.2593021, 10.0000001, 9.49212256, -0.1382893]))

    # initial and final velocities
    npt.assert_almost_equal(states[0]["qdot"][:, 0], np.array([1.89770078, 18.62453707]))
    npt.assert_almost_equal(states[0]["qdot"][:, -1], np.array([16.56293494, -16.83711551]))
    npt.assert_almost_equal(states[1]["qdot"][:, 0], np.array([0.0, 0.0, 16.56293494, -16.83711551]))
    npt.assert_almost_equal(states[1]["qdot"][:, -1], np.array([-1.28658849, 6.05426872, -0.20069993, 1.56293712]))

    # initial and final controls
    npt.assert_almost_equal(controls[0]["tau"][:, 0], np.array([-0.01975067]))
    npt.assert_almost_equal(controls[0]["tau"][:, -1], np.array([-0.12304145]))
    npt.assert_almost_equal(controls[1]["tau"][:, 0], np.array([3.21944836]))
    npt.assert_almost_equal(controls[1]["tau"][:, -1], np.array([-0.01901175]))


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.IRK])
def test_torque_activation_driven(ode_solver, phase_dynamics):
    # Load track_markers
    from bioptim.examples.toy_examples.torque_driven_ocp import torque_activation_driven as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/examples/models/2segments_2dof_2contacts.bioMod",
        n_shooting=30,
        final_time=2,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
        expand_dynamics=ode_solver != OdeSolver.IRK,
    )
    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=0.04880295023323905, decimal=3)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (120, 1))
    npt.assert_almost_equal(g, np.zeros((120, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array((-0.75, 0.75)))
    npt.assert_almost_equal(q[:, -1], np.array((3.0, 0.75)))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array((0.0, 0.0)))
    npt.assert_almost_equal(qdot[:, -1], np.array((0.0, 0.0)))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array((-0.2256539, 0.0681475)), decimal=3)
    npt.assert_almost_equal(tau[:, -1], np.array((-0.0019898, -0.0238914)), decimal=3)

    # simulate
    TestUtils.simulate(sol, decimal_value=4)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE])
def test_example_multi_biorbd_model(phase_dynamics):
    # Load example_multi_biorbd_model
    from bioptim.examples.toy_examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()
    biorbd_model_path = bioptim_folder + "/examples/models/triple_pendulum.bioMod"
    biorbd_model_path_modified_inertia = bioptim_folder + "/examples/models/triple_pendulum_modified_inertia.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        biorbd_model_path_modified_inertia=biorbd_model_path_modified_inertia,
        n_shooting=20,
        phase_dynamics=phase_dynamics,
        expand_dynamics=True,
    )

    np.random.seed(42)
    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.random.rand(313, 1),
        expected_v_f_g=[154.4724783298145, 21.353112388854846, 60.207690783556835],
        decimal=6,
    )


def test_example_minimize_segment_velocity():
    from bioptim.examples.toy_examples.torque_driven_ocp import example_minimize_segment_velocity as ocp_module

    bioptim_folder = TestUtils.bioptim_folder()

    # Define the problem

    biorbd_model_path = bioptim_folder + "/examples/models/triple_pendulum.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        n_shooting=5,
        expand_dynamics=True,
    )

    # Check the values which will be sent to the solver
    np.random.seed(42)
    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.random.rand(52, 1),
        expected_v_f_g=[24.04091267073873, 805.0958650566107, 7.321219099616136],
        decimal=6,
    )
    if platform.system() == "Windows":
        return

    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=41.40771798838792)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (30, 1))
    npt.assert_almost_equal(g, np.zeros((30, 1)), decimal=6)

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # initial and final position
    npt.assert_almost_equal(states["q"][:, 0], np.array([0.0, 0.0, 0.0]), decimal=6)
    npt.assert_almost_equal(states["q"][:, -1], np.array([0.0, 3.14159265, -3.14159265]), decimal=6)
    # initial and final velocities
    npt.assert_almost_equal(
        states["qdot"][:, 0],
        np.array([0.0, 0.0, 0.0]),
        decimal=6,
    )
    npt.assert_almost_equal(
        states["qdot"][:, -1],
        np.array([2.69457617, 0.25126143, -2.3535264]),
        decimal=6,
    )
    # initial and final controls
    npt.assert_almost_equal(controls["tau"][:, 0], np.array([-2.4613488, 3.70379261, -0.99483388]), decimal=6)
    npt.assert_almost_equal(controls["tau"][:, -1], np.array([0.80156395, 0.82773623, 0.35042046]), decimal=6)
