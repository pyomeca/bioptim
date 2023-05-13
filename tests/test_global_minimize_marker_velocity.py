"""
Test for file IO
"""
import pytest
import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    ControlType,
    OdeSolver,
    OdeSolverBase,
    Node,
)

from .utils import TestUtils


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    marker_velocity_or_displacement: str,
    marker_in_first_coordinates_system: bool,
    control_type: ControlType,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    assume_phase_dynamics: bool = True,
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
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

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
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Path constraint
    nq = bio_model.nb_q
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0].min[nq:, :] = -10
    x_bounds[0].max[nq:, :] = 10

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([1.5, 1.5, 0.0, 0.0, 0.7, 0.7, 0.6, 0.6])

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        control_type=control_type,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_displacement_global(ode_solver, assume_phase_dynamics):
    # Load track_and_minimize_marker_velocity
    ode_solver = ode_solver()
    ocp = prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="disp",
        marker_in_first_coordinates_system=False,
        control_type=ControlType.CONSTANT,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -143.5854887928483)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.37791617, 3.70167396, 10.0, 10.0]), decimal=2)
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.37675299, -3.40771446, 10.0, 10.0]), decimal=2)
    # initial and final controls
    np.testing.assert_almost_equal(
        tau[:, 0], np.array([-4.52595667e-02, 9.25475333e-01, -4.34001849e-08, -9.24667407e01]), decimal=2
    )
    np.testing.assert_almost_equal(
        tau[:, -2], np.array([4.42976253e-02, 1.40077846e00, -7.28864793e-13, 9.24667396e01]), decimal=2
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_displacement_RT(ode_solver, assume_phase_dynamics):
    # Load track_and_minimize_marker_velocity
    ode_solver = ode_solver()
    ocp = prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="disp",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.CONSTANT,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -200.80194174353494)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07221334, -0.4578082, -3.00436948, 1.57079633]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.05754807, -0.43931116, 2.99563057, 1.57079633]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([5.17192208, 2.3422717, 10.0, -10.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([-3.56965109, -4.36318589, 10.0, 10.0]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-3.21817755e01, 1.55202948e01, 7.42730542e-13, 2.61513401e-08]))
    np.testing.assert_almost_equal(
        tau[:, -2], np.array([-1.97981112e01, -9.89876772e-02, 4.34033234e-08, 2.61513636e-08])
    )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_velocity(ode_solver, assume_phase_dynamics):
    # Load track_and_minimize_marker_velocity
    ode_solver = ode_solver()
    ocp = prepare_ocp(
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="velo",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.CONSTANT,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -80.20048585400944)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([7.18708669e-01, -4.45703930e-01, -3.14159262e00, 0]))
    np.testing.assert_almost_equal(q[:, -1], np.array([1.08646846e00, -3.86731175e-01, 3.14159262e00, 0]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([3.78330878e-01, 3.70214281, 10, 0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([3.77168521e-01, -3.40782793, 10, 0]))
    # # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-4.52216174e-02, 9.25170010e-01, 0, 0]))
    np.testing.assert_almost_equal(tau[:, -2], np.array([4.4260355e-02, 1.4004583, 0, 0]))

    # # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_velocity_linear_controls(ode_solver, assume_phase_dynamics):
    # Load track_and_minimize_marker_velocity
    if ode_solver == OdeSolver.IRK:
        ode_solver = ode_solver()
        with pytest.raises(
            NotImplementedError, match="ControlType.LINEAR_CONTINUOUS ControlType not implemented yet with COLLOCATION"
        ):
            prepare_ocp(
                biorbd_model_path=TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod",
                n_shooting=5,
                final_time=1,
                marker_velocity_or_displacement="velo",
                marker_in_first_coordinates_system=True,
                control_type=ControlType.LINEAR_CONTINUOUS,
                ode_solver=ode_solver,
                assume_phase_dynamics=assume_phase_dynamics,
            )
    else:
        ode_solver = ode_solver()
        ocp = prepare_ocp(
            biorbd_model_path=TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod",
            n_shooting=5,
            final_time=1,
            marker_velocity_or_displacement="velo",
            marker_in_first_coordinates_system=True,
            control_type=ControlType.LINEAR_CONTINUOUS,
            ode_solver=ode_solver,
            assume_phase_dynamics=assume_phase_dynamics,
        )
        sol = ocp.solve()

        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (40, 1))
        np.testing.assert_almost_equal(g, np.zeros((40, 1)))

        # Check some of the results
        q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

        # initial and final position
        np.testing.assert_almost_equal(q[2:, 0], np.array([-3.14159264, 0]))
        np.testing.assert_almost_equal(q[2:, -1], np.array([3.14159264, 0]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[2:, 0], np.array([10, 0]))
        np.testing.assert_almost_equal(qdot[2:, -1], np.array([10, 0]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[2:, 0], np.array([-8.495542, 0]), decimal=5)
        np.testing.assert_almost_equal(tau[2:, -1], np.array([8.495541, 0]), decimal=5)

        # save and load
        TestUtils.save_and_load(sol, ocp, False)

        # simulate
        TestUtils.simulate(sol)
