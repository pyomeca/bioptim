"""
Test for file IO
"""

import platform

from bioptim import (
    TorqueBiorbdModel,
    OptimalControlProgram,
    DynamicsOptions,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    ControlType,
    OdeSolver,
    OdeSolverBase,
    Node,
    PhaseDynamics,
    SolutionMerge,
)
from casadi import Function
import numpy as np
import numpy.testing as npt
import pytest

from tests.utils import TestUtils


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    marker_velocity_or_displacement: str,
    marker_in_first_coordinates_system: bool,
    control_type: ControlType,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    Prepare an ocp that targets some marker velocities, either by finite differences or by jacobian

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    final_time: float
        The time of the final node
    n_shooting: int
        The number of shooting points
    marker_velocity_or_displacement: str
        which type of tracking: finite difference ('disp') or by jacobian ('velo')
    marker_in_first_coordinates_system: bool
        If the marker to track should be expressed in the global or local reference frame
    control_type: ControlType
        The type of controls
    ode_solver: OdeSolverBase
        The ode solver to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = TorqueBiorbdModel(biorbd_model_path)

    # Add objective functions
    if marker_in_first_coordinates_system:
        # Marker should follow this segment (0 velocity when compare to this one)
        coordinates_system_idx = 0
    else:
        # Marker should be static in global reference frame
        coordinates_system_idx = None

    objective_functions = ObjectiveList()
    if marker_velocity_or_displacement == "disp":
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_MARKERS,
            derivative=True,
            reference_jcs=coordinates_system_idx,
            marker_index=7,
            weight=1000,
        )
    elif marker_velocity_or_displacement == "velo":
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_VELOCITY, node=Node.ALL, marker_index=7, weight=1000
        )
    else:
        raise RuntimeError(
            f"Wrong choice of marker_velocity_or_displacement, actual value is "
            f"{marker_velocity_or_displacement}, should be 'velo' or 'disp'."
        )
    # Make sure the segments actually moves (in order to test the relative speed objective)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", node=Node.ALL, index=[2, 3], weight=-1)

    # Dynamics
    dynamics = DynamicsOptions(
        expand_dynamics=not isinstance(ode_solver, OdeSolver.IRK),
        phase_dynamics=phase_dynamics,
        ode_solver=ode_solver,
    )

    # Path constraint
    nq = bio_model.nb_q
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = [-10] * bio_model.nb_qdot, [10] * bio_model.nb_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = 1.5, 1.5, 0.0, 0.0
    x_init["qdot"] = 0.7, 0.7, 0.6, 0.6

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        control_type=control_type,
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_displacement_global(ode_solver, phase_dynamics):
    # Load track_and_minimize_marker_velocity
    ode_solver = ode_solver()
    ocp = prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="disp",
        marker_in_first_coordinates_system=False,
        control_type=ControlType.CONSTANT,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=-143.5854887928483)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (40, 1))
    npt.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array([0.37791617, 3.70167396, 10.0, 10.0]), decimal=2)
    npt.assert_almost_equal(qdot[:, -1], np.array([0.37675299, -3.40771446, 10.0, 10.0]), decimal=2)
    # initial and final controls
    npt.assert_almost_equal(
        tau[:, 0], np.array([-4.52595667e-02, 9.25475333e-01, -4.34001849e-08, -9.24667407e01]), decimal=2
    )
    npt.assert_almost_equal(
        tau[:, -1], np.array([4.42976253e-02, 1.40077846e00, -7.28864793e-13, 9.24667396e01]), decimal=2
    )

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_displacement_RT(ode_solver, phase_dynamics):
    # Load track_and_minimize_marker_velocity
    ode_solver = ode_solver()
    ocp = prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="disp",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.CONSTANT,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )

    # Check the values which will be sent to the solver
    np.random.seed(42)
    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.random.rand(69, 1),
        expected_v_f_g=[31.736865760272735, 735.9774884772594, 14.88489768158775],
        decimal=6,
    )
    if platform.system() == "Windows":
        return

    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=-200.80194174353494)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (40, 1))
    npt.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([0.07221334, -0.4578082, -3.00436948, 1.57079633]))
    npt.assert_almost_equal(q[:, -1], np.array([0.05754807, -0.43931116, 2.99563057, 1.57079633]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array([5.17192208, 2.3422717, 10.0, -10.0]))
    npt.assert_almost_equal(qdot[:, -1], np.array([-3.56965109, -4.36318589, 10.0, 10.0]))
    # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([-3.21817755e01, 1.55202948e01, 7.42730542e-13, 2.61513401e-08]))
    npt.assert_almost_equal(tau[:, -1], np.array([-1.97981112e01, -9.89876772e-02, 4.34033234e-08, 2.61513636e-08]))

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_velocity(ode_solver, phase_dynamics):
    # Load track_and_minimize_marker_velocity
    ode_solver = ode_solver()
    ocp = prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="velo",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.CONSTANT,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )

    # Check the values which will be sent to the solver
    np.random.seed(42)
    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.random.rand(69, 1),
        expected_v_f_g=[31.736865760272735, 90.85915472423895, 14.88489768158775],
        decimal=6,
    )
    if platform.system() == "Windows":
        return

    sol = ocp.solve()

    # Check objective function value
    TestUtils.assert_objective_value(sol=sol, expected_value=-80.20048585400944)

    # Check constraints
    g = np.array(sol.constraints)
    npt.assert_equal(g.shape, (40, 1))
    npt.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[:, 0], np.array([7.18708669e-01, -4.45703930e-01, -3.14159262e00, 0]))
    npt.assert_almost_equal(q[:, -1], np.array([1.08646846e00, -3.86731175e-01, 3.14159262e00, 0]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[:, 0], np.array([3.78330878e-01, 3.70214281, 10, 0]))
    npt.assert_almost_equal(qdot[:, -1], np.array([3.77168521e-01, -3.40782793, 10, 0]))
    # # initial and final controls
    npt.assert_almost_equal(tau[:, 0], np.array([-4.52216174e-02, 9.25170010e-01, 0, 0]))
    npt.assert_almost_equal(tau[:, -1], np.array([4.4260355e-02, 1.4004583, 0, 0]))

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK, OdeSolver.COLLOCATION])
def test_track_and_minimize_marker_velocity_linear_controls(ode_solver, phase_dynamics):

    if platform.system() == "Windows" and ode_solver == OdeSolver.IRK:
        pytest.skip("Skipping as it does not pass")

    # Load track_and_minimize_marker_velocity
    ocp = prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="velo",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.LINEAR_CONTINUOUS,
        ode_solver=ode_solver(),
        phase_dynamics=phase_dynamics,
    )
    sol = ocp.solve()

    # Make sure it converged
    assert sol.status == 0

    # Check constraints
    g = np.array(sol.constraints)
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_equal(g.shape, (200, 1))
        npt.assert_almost_equal(g, np.zeros((200, 1)))
    else:
        npt.assert_equal(g.shape, (40, 1))
        npt.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]

    # initial and final position
    npt.assert_almost_equal(q[2:, 0], np.array([-3.14159264, 0]))
    npt.assert_almost_equal(q[2:, -1], np.array([3.14159264, 0]))
    # initial and final velocities
    npt.assert_almost_equal(qdot[2:, 0], np.array([10, 0]))
    npt.assert_almost_equal(qdot[2:, -1], np.array([10, 0]))
    # initial and final controls
    if ode_solver == OdeSolver.COLLOCATION:
        npt.assert_almost_equal(tau[2:, 0], np.array([-3.44506583, 0]), decimal=5)
        npt.assert_almost_equal(tau[2:, -1], np.array([3.44506583, 0]), decimal=5)
    else:
        if platform.system() == "Linux" and ode_solver == OdeSolver.RK8:
            pass
        else:
            npt.assert_almost_equal(tau[2:, 0], np.array([-8.495542, 0]), decimal=5)
            npt.assert_almost_equal(tau[2:, -1], np.array([8.495541, 0]), decimal=5)

    # simulate
    TestUtils.simulate(sol)
