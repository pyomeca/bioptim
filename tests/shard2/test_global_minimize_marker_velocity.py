"""
Test for file IO
"""

import platform

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
    PhaseDynamics,
    SolutionMerge,
)
from bioptim.interfaces.ipopt_interface import IpoptInterface
from bioptim.interfaces.interface_utils import _shake_tree_for_penalties
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
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node

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
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        expand_dynamics=not isinstance(ode_solver, OdeSolver.IRK),
        phase_dynamics=phase_dynamics,
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
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        control_type=control_type,
        ode_solver=ode_solver,
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.SHARED_DURING_THE_PHASE, PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_displacement_global(ode_solver, phase_dynamics):
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
        phase_dynamics=phase_dynamics,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], -143.5854887928483)

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
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="disp",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.CONSTANT,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )

    # Check the values which will be sent to the solver
    interface = IpoptInterface(ocp=ocp)
    v = interface.ocp.variables_vector
    v_bounds = interface.ocp.bounds_vectors
    f = _shake_tree_for_penalties(interface.ocp, interface.dispatch_obj_func(), v, v_bounds, False)
    g = _shake_tree_for_penalties(interface.ocp, interface.dispatch_bounds()[0], v, v_bounds, False)

    np.random.seed(42)
    values = Function("v", [v], [v, f, g])(np.random.rand(*v.size()))
    npt.assert_almost_equal(
        np.array(values[0])[:, 0],
        [
            0.3745401188,
            0.9507143064,
            0.7319939418,
            0.5986584842,
            0.1560186404,
            0.1559945203,
            0.0580836122,
            0.8661761458,
            0.6011150117,
            0.7080725778,
            0.0205844943,
            0.9699098522,
            0.8324426408,
            0.2123391107,
            0.1818249672,
            0.1834045099,
            0.304242243,
            0.5247564316,
            0.4319450186,
            0.2912291402,
            0.6118528947,
            0.1394938607,
            0.2921446485,
            0.3663618433,
            0.4560699842,
            0.7851759614,
            0.1996737822,
            0.5142344384,
            0.5924145689,
            0.0464504127,
            0.6075448519,
            0.1705241237,
            0.065051593,
            0.9488855373,
            0.9656320331,
            0.8083973481,
            0.3046137692,
            0.097672114,
            0.6842330265,
            0.4401524937,
            0.1220382348,
            0.4951769101,
            0.0343885211,
            0.9093204021,
            0.2587799816,
            0.6625222844,
            0.3117110761,
            0.5200680212,
            0.5467102793,
            0.1848544555,
            0.9695846278,
            0.7751328234,
            0.9394989416,
            0.8948273504,
            0.5978999788,
            0.921874235,
            0.0884925021,
            0.1959828624,
            0.0452272889,
            0.3253303308,
            0.3886772897,
            0.2713490318,
            0.8287375092,
            0.3567533267,
            0.2809345097,
            0.5426960832,
            0.140924225,
            0.8021969808,
            0.0745506437,
        ],
        decimal=6,
    )
    npt.assert_almost_equal(
        np.array(values[1])[:, 0],
        [
            2.1777249117e-01,
            0.0000000000e00,
            3.1860711148e01,
            5.1534589288e-02,
            0.0000000000e00,
            7.5929602420e00,
            6.6851278077e00,
            0.0000000000e00,
            3.0211593975e00,
            6.4478285409e01,
            0.0000000000e00,
            5.2564740149e01,
            2.5922438169e01,
            0.0000000000e00,
            2.0108682174e02,
            -1.5005222310e-01,
            -7.2267851469e-02,
            -6.7274428469e-03,
            -1.8512668480e-02,
            -2.6844200044e-02,
            -4.1599966101e-02,
            -5.8156953519e-03,
            -8.4634194998e-04,
            -3.8746843549e-02,
            -2.9786661528e-03,
            -5.4094149330e-02,
            -5.9778425908e-02,
        ],
        decimal=6,
    )
    npt.assert_almost_equal(
        np.array(values[2])[:, 0],
        [
            -2.7753772179e-01,
            -5.4621786250e-01,
            1.8251348234e-01,
            5.3741101918e-01,
            1.9373699237e-02,
            1.8918244295e00,
            -8.3779820059e-01,
            -4.8477255710e-01,
            -2.4368051531e-01,
            5.5923753133e-01,
            -7.3379909864e-01,
            -2.8320804471e-01,
            -2.5181072011e-01,
            1.9527396856e00,
            -1.4175135644e-03,
            1.3412924085e-01,
            2.2860110038e-01,
            -9.5404711969e-02,
            1.4322632294e-01,
            -1.1842586850e-01,
            -1.3224002042e-01,
            2.2683547456e00,
            -2.6090378576e-01,
            -4.6875384917e-01,
            1.4899251268e-01,
            8.2407453035e-01,
            2.5292301843e-01,
            -3.0642980848e-01,
            -3.0481050684e-03,
            1.8729406728e00,
            1.9827770471e-01,
            7.9973992202e-04,
            -4.8409697161e-01,
            -8.7470860176e-01,
            -3.1513844007e-03,
            -7.1732447416e-02,
            4.5631095372e-01,
            1.5612932046e00,
            -8.0523868713e-02,
            4.0976191576e-01,
        ],
        decimal=6,
    )

    if platform.system() == "Windows":
        return

    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], -200.80194174353494)

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
        biorbd_model_path=TestUtils.bioptim_folder() + "/examples/track/models/cube_and_line.bioMod",
        n_shooting=5,
        final_time=1,
        marker_velocity_or_displacement="velo",
        marker_in_first_coordinates_system=True,
        control_type=ControlType.CONSTANT,
        ode_solver=ode_solver,
        phase_dynamics=phase_dynamics,
    )
    sol = ocp.solve()

    # Prior optimization checks
    nlp = ocp.ocp_solver.nlp
    v = nlp["x"]
    dummy_objectives = np.array(Function("x", [v], [nlp["f"]])(np.arange(v.numel())))
    np.testing.assert_almost_equal(dummy_objectives, np.array([[288936.4]]))

    dummy_constraints = np.sum(np.array(Function("x", [v], [nlp["g"]])(np.arange(v.numel()))))
    np.testing.assert_almost_equal(dummy_constraints, -16.609000000000044)

    limits = ocp.ocp_solver.limits
    min_bounds = np.sum(limits["lbx"])
    np.testing.assert_almost_equal(min_bounds, -2278.274333882308)
    max_bounds = np.sum(limits["ubx"])
    np.testing.assert_almost_equal(max_bounds, 2288.274333882308)
    min_constraints = np.sum(limits["lbg"])
    np.testing.assert_almost_equal(min_constraints, 0.0)
    max_constraints = np.sum(limits["ubg"])
    np.testing.assert_almost_equal(max_constraints, 0.0)
    x0 = np.sum(limits["x0"])
    np.testing.assert_almost_equal(x0, 33.8)

    # Check objective function value
    f = np.array(sol.cost)
    npt.assert_equal(f.shape, (1, 1))
    npt.assert_almost_equal(f[0, 0], -80.20048585400944)

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
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_and_minimize_marker_velocity_linear_controls(ode_solver, phase_dynamics):
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
                phase_dynamics=phase_dynamics,
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
            phase_dynamics=phase_dynamics,
        )
        sol = ocp.solve()

        # Check constraints
        g = np.array(sol.constraints)
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
        npt.assert_almost_equal(tau[2:, 0], np.array([-8.495542, 0]), decimal=5)
        npt.assert_almost_equal(tau[2:, -1], np.array([8.495541, 0]), decimal=5)

        # simulate
        TestUtils.simulate(sol)
