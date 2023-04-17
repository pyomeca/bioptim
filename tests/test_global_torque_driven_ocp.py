"""
Test for file IO
"""
import os
import pytest
import numpy as np
import biorbd_casadi as biorbd
from bioptim import OdeSolver, ConstraintList, ConstraintFcn, Node, DefectType, Solver, BiorbdModel
from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
@pytest.mark.parametrize("actuator_type", [None, 2])
def test_track_markers(ode_solver, actuator_type):
    # Load track_markers
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=actuator_type,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19767.53312569522)

    # Check constraints
    g = np.array(sol.constraints)
    if not actuator_type:
        np.testing.assert_equal(g.shape, (186, 1))
    else:
        np.testing.assert_equal(g.shape, (366, 1))
    np.testing.assert_almost_equal(g[:186], np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_markers_changing_constraints(ode_solver):
    # Load track_markers
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        ode_solver=ode_solver,
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
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 20370.211697123825)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (189, 1))
    np.testing.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((4.2641129, 9.81, 2.27903226)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((1.36088709, 9.81, -2.27903226)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

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
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 31670.93770220887)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (189, 1))
    np.testing.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-5.625, 21.06, 2.2790323)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-5.625, 21.06, -2.27903226)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_markers_with_actuators(ode_solver):
    # Load track_markers
    from bioptim.examples.torque_driven_ocp import track_markers_with_torque_actuators as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/cube.bioMod",
        n_shooting=30,
        final_time=2,
        actuator_type=1,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 204.18087334169184)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.2140175, 0.981, 0.3360075)))
    np.testing.assert_almost_equal(tau[:, -2], np.array((-0.2196496, 0.981, -0.3448498)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_track_marker_2D_pendulum(ode_solver):
    # Load muscle_activations_contact_tracker
    from bioptim.examples.torque_driven_ocp import track_markers_2D_pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/models/pendulum.bioMod"
    bio_model = BiorbdModel(model_path)

    final_time = 2
    n_shooting = 30

    # Generate data to fit
    np.random.seed(42)
    markers_ref = np.random.rand(3, 2, n_shooting + 1)
    tau_ref = np.random.rand(2, n_shooting)

    if isinstance(ode_solver, OdeSolver.IRK):
        tau_ref = tau_ref * 5

    ocp = ocp_module.prepare_ocp(bio_model, final_time, n_shooting, markers_ref, tau_ref, ode_solver=ode_solver)
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (n_shooting * 4, 1))
    np.testing.assert_almost_equal(g, np.zeros((n_shooting * 4, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 290.6751231)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0.64142484, 2.85371719)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((3.46921861, 3.24168308)))

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((9.11770196, -13.83677175)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((1.16836132, 4.77230548)))

    elif isinstance(ode_solver, OdeSolver.RK8):
        pass

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 281.8560713312711)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0.8367364, 3.37533055)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((3.2688391, 3.88242643)))

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((6.93890241, -12.76433504)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((0.13156876, 0.93749913)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.IRK, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("defects_type", [DefectType.EXPLICIT, DefectType.IMPLICIT])
def test_track_marker_2D_pendulum(ode_solver, defects_type):
    # Load muscle_activations_contact_tracker
    from bioptim.examples.torque_driven_ocp import track_markers_2D_pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    # Define the problem
    model_path = bioptim_folder + "/models/pendulum.bioMod"
    bio_model = BiorbdModel(model_path)

    final_time = 2
    n_shooting = 30

    # Generate data to fit
    np.random.seed(42)
    markers_ref = np.random.rand(3, 2, n_shooting + 1)
    tau_ref = np.random.rand(2, n_shooting)

    if isinstance(ode_solver, OdeSolver.IRK):
        tau_ref = tau_ref * 5

    ocp = ocp_module.prepare_ocp(bio_model, final_time, n_shooting, markers_ref, tau_ref, ode_solver=ode_solver)
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    if isinstance(ode_solver, OdeSolver.IRK):
        np.testing.assert_equal(g.shape, (n_shooting * 4, 1))
        np.testing.assert_almost_equal(g, np.zeros((n_shooting * 4, 1)))

        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 290.6751231)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0.64142484, 2.85371719)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((3.46921861, 3.24168308)))

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((9.11770196, -13.83677175)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((1.16836132, 4.77230548)))

    else:
        np.testing.assert_equal(g.shape, (n_shooting * 4 * 5, 1))
        np.testing.assert_almost_equal(g, np.zeros((n_shooting * 4 * 5, 1)))

        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 281.8462122624288)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(q[:, -1], np.array((0.8390514, 3.3819348)))

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((3.2598235, 3.8800289)))

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((6.8532419, -12.1810791)))
        np.testing.assert_almost_equal(tau[:, -2], np.array((0.1290981, 0.9345706)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


def test_trampo_quaternions():
    # Load trampo_quaternion
    from bioptim.examples.torque_driven_ocp import trampo_quaternions as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    model_path = bioptim_folder + "/models/TruncAnd2Arm_Quaternion.bioMod"
    final_time = 0.25
    n_shooting = 5

    ocp = ocp_module.prepare_ocp(model_path, n_shooting, final_time)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -41.491609816961535)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (130, 1))
    np.testing.assert_almost_equal(g, np.zeros((130, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(
        q[:, 0], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    )
    np.testing.assert_almost_equal(
        q[:, -1],
        np.array(
            [
                3.14159267,
                3.14159267,
                3.14159267,
                -0.78539816,
                0.6154797,
                -0.07516336,
                0.23662774,
                -0.69787559,
                0.23311438,
                0.22930573,
                0.62348603,
                0.38590688,
                0.63453499,
                0.64012494,
            ]
        ),
    )

    # initial and final velocities
    np.testing.assert_almost_equal(
        qdot[:, 0],
        np.array(
            [
                12.56193009,
                12.5198592,
                13.67105918,
                -2.66942572,
                2.64460582,
                -2.16473217,
                2.89069185,
                -4.74193932,
                4.88561749,
                4.18495164,
                5.12235989,
                1.65628252,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        qdot[:, -1],
        np.array(
            [
                12.59374119,
                12.65603932,
                11.46119531,
                -4.11706327,
                1.84777845,
                1.92003246,
                -1.99624566,
                -7.67384307,
                0.97705102,
                -0.0532827,
                7.28333747,
                2.68097813,
            ]
        ),
    )

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.zeros((12,)), decimal=6)
    np.testing.assert_almost_equal(tau[:, -2], np.zeros((12,)), decimal=6)

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=6)


def test_phase_transition_uneven_variable_number_by_bounds():
    # Load phase_transition_uneven_variable_number_by_bounds
    from bioptim.examples.torque_driven_ocp import phase_transition_uneven_variable_number_by_bounds as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path_withTranslations = bioptim_folder + "/models/double_pendulum_with_translations.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path_withTranslations=biorbd_model_path_withTranslations,
        n_shooting=(10, 10),
    )

    solver = Solver.IPOPT()
    solver.set_maximum_iterations(10)
    sol = ocp.solve(solver)

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (170, 1))
    np.testing.assert_equal(sol.status, 1)  # Did not converge, therefore the constraints won't be zero


def test_phase_transition_uneven_variable_number_by_mapping():
    # Load phase_transition_uneven_variable_number_by_mapping
    from bioptim.examples.torque_driven_ocp import phase_transition_uneven_variable_number_by_mapping as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/double_pendulum.bioMod"
    biorbd_model_path_withTranslations = bioptim_folder + "/models/double_pendulum_with_translations.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        biorbd_model_path_withTranslations=biorbd_model_path_withTranslations,
        n_shooting=(10, 10),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -12397.11475053)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (126, 1))
    np.testing.assert_almost_equal(g[:44], np.zeros((44, 1)), decimal=6)
    np.testing.assert_almost_equal(
        g[44], np.array([0.45544143]), decimal=6
    )  # Time constraint with min / max bounds phase 0
    np.testing.assert_almost_equal(g[45:-1], np.zeros((80, 1)), decimal=6)
    np.testing.assert_almost_equal(
        g[-1], np.array([1.00939513]), decimal=6
    )  # Time constraint with min / max bounds phase 1

    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array([3.14, 0.0]))
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array([7.93025703, 0.31520724]))
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array([0.0, 0.0, 7.93025703, 0.31520724]))
    np.testing.assert_almost_equal(states[1]["q"][:, -1], np.array([-0.2593021, 10.0000001, 9.49212256, -0.1382893]))

    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array([1.89770078, 18.62453707]))
    np.testing.assert_almost_equal(states[0]["qdot"][:, -1], np.array([16.56293494, -16.83711551]))
    np.testing.assert_almost_equal(states[1]["qdot"][:, 0], np.array([0.0, 0.0, 16.56293494, -16.83711551]))
    np.testing.assert_almost_equal(
        states[1]["qdot"][:, -1], np.array([-1.28658849, 6.05426872, -0.20069993, 1.56293712])
    )
    # initial and final controls
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array([-0.01975067]))
    np.testing.assert_almost_equal(controls[0]["tau"][:, -2], np.array([-0.12304145]))
    np.testing.assert_almost_equal(controls[1]["tau"][:, 0], np.array([3.21944836]))
    np.testing.assert_almost_equal(controls[1]["tau"][:, -2], np.array([-0.01901175]))


def test_multi_model_by_mapping():
    """
    This test is highly sensitive, do not change anything please.
    """

    # Load multi_model_by_mapping
    from bioptim.examples.torque_driven_ocp import multi_model_by_mapping as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/double_pendulum.bioMod"
    biorbd_model_path_modified_inertia = bioptim_folder + "/models/double_pendulum_modified_inertia.bioMod"

    with pytest.raises(NotImplementedError, match="Mapping over phases is broken"):
        ocp = ocp_module.prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            biorbd_model_path_modified_inertia=biorbd_model_path_modified_inertia,
            n_shooting=(5, 5),
        )
    return  # TODO: when it is not broken anymore, the following results should be good

    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)
    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate
    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array([-3.14159265, 0.0]), decimal=6)
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array([3.04159296, 0.0]), decimal=3)
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array([-3.14159265, 0.0]), decimal=6)
    np.testing.assert_almost_equal(states[1]["q"][:, -1], np.array([3.04159271, 0.0]), decimal=6)
    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array([11.47768245, 26.16790572]), decimal=6)
    np.testing.assert_almost_equal(states[0]["qdot"][:, -1], np.array([11.52232512, 26.06343438]), decimal=6)
    np.testing.assert_almost_equal(states[1]["qdot"][:, 0], np.array([10.54030594, 28.30202101]), decimal=6)
    np.testing.assert_almost_equal(states[1]["qdot"][:, -1], np.array([10.594124, 28.17553337]), decimal=6)
    # initial and final controls
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array([0.01906557]), decimal=6)
    np.testing.assert_almost_equal(controls[0]["tau"][:, -2], np.array([-0.00619146]), decimal=6)
    np.testing.assert_equal(controls[1], {})


def test_multi_model_by_constraint():
    # Load multi_model_by_constraint
    from bioptim.examples.torque_driven_ocp import multi_model_by_constraint as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/double_pendulum.bioMod"
    biorbd_model_path_modified_inertia = bioptim_folder + "/models/double_pendulum_modified_inertia.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        biorbd_model_path_modified_inertia=biorbd_model_path_modified_inertia,
        n_shooting=(5, 5),
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.51241813)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (52, 1))
    np.testing.assert_almost_equal(g, np.zeros((52, 1)), decimal=6)

    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array([-3.14159265, 0.0]), decimal=6)
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array([3.04159276, 0.0]), decimal=6)
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array([-3.14159265, 0.0]), decimal=6)
    np.testing.assert_almost_equal(states[1]["q"][:, -1], np.array([3.04159811, 0.0]), decimal=4)
    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array([-0.00633747, -0.12756957]), decimal=6)
    np.testing.assert_almost_equal(states[0]["qdot"][:, -1], np.array([0.34507652, 0.59111588]), decimal=6)
    np.testing.assert_almost_equal(states[1]["qdot"][:, 0], np.array([-0.00659825, -0.13014626]), decimal=5)
    np.testing.assert_almost_equal(states[1]["qdot"][:, -1], np.array([0.34576491, 0.58886914]), decimal=6)
    # initial and final controls
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array([0.00146709]), decimal=6)
    np.testing.assert_almost_equal(controls[0]["tau"][:, -2], np.array([0.01132175]), decimal=6)
    np.testing.assert_almost_equal(controls[1]["tau"][:, 0], np.array([0.00146709]), decimal=6)
    np.testing.assert_almost_equal(controls[1]["tau"][:, -2], np.array([0.01132175]), decimal=6)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_torque_activation_driven(ode_solver):
    # Load track_markers
    from bioptim.examples.torque_driven_ocp import torque_activation_driven as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ode_solver = ode_solver()

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/2segments_2dof_2contacts.bioMod",
        n_shooting=30,
        final_time=2,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.04880295023323905, decimal=3)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (120, 1))
    np.testing.assert_almost_equal(g, np.zeros((120, 1)))

    # Check some of the results
    q, qdot, tau = sol.states["q"], sol.states["qdot"], sol.controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((-0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((3.0, 0.75)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0.0, 0.0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0.0, 0.0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-0.2256539, 0.0681475)), decimal=3)
    np.testing.assert_almost_equal(tau[:, -2], np.array((-0.0019898, -0.0238914)), decimal=3)

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, decimal_value=4)


def test_example_multi_biorbd_model():
    # Load example_multi_biorbd_model
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem

    biorbd_model_path = bioptim_folder + "/models/triple_pendulum.bioMod"
    biorbd_model_path_modified_inertia = bioptim_folder + "/models/triple_pendulum_modified_inertia.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        biorbd_model_path_modified_inertia=biorbd_model_path_modified_inertia,
        n_shooting=20,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 10.697019532108447)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (240, 1))
    np.testing.assert_almost_equal(g, np.zeros((240, 1)), decimal=6)

    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate

    # initial and final position
    np.testing.assert_almost_equal(
        states["q"][:, 0], np.array([-3.14159265, 0.0, 0.0, -3.14159265, 0.0, 0.0]), decimal=6
    )
    np.testing.assert_almost_equal(
        states["q"][:, -1], np.array([3.05279505, 0.0, 0.0, 3.04159266, 0.0, 0.0]), decimal=6
    )
    # initial and final velocities
    np.testing.assert_almost_equal(
        states["qdot"][:, 0],
        np.array([15.68385811, -31.25068304, 19.2317873, 15.63939216, -31.4159265, 19.91541457]),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        states["qdot"][:, -1],
        np.array([15.90689541, -30.54499528, 16.03701393, 15.96682325, -30.89799758, 16.70457477]),
        decimal=6,
    )
    # initial and final controls
    np.testing.assert_almost_equal(controls["tau"][:, 0], np.array([-0.48437131, 0.0249894, 0.38051993]), decimal=6)
    np.testing.assert_almost_equal(controls["tau"][:, -2], np.array([-0.00235227, -0.02192184, -0.00709896]), decimal=6)


def test_example_minimize_JCS_velocity():
    # Load example_multi_biorbd_model
    from bioptim.examples.torque_driven_ocp import example_minimize_JCS_velocity as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem

    biorbd_model_path = bioptim_folder + "/models/triple_pendulum.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        n_shooting=5,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 34.94553532235115)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (30, 1))
    np.testing.assert_almost_equal(g, np.zeros((30, 1)), decimal=6)

    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate

    # initial and final position
    np.testing.assert_almost_equal(states["q"][:, 0], np.array([0.0, 0.0, 0.0]), decimal=6)
    np.testing.assert_almost_equal(states["q"][:, -1], np.array([0.0, 3.14159265, -3.14159265]), decimal=6)
    # initial and final velocities
    np.testing.assert_almost_equal(
        states["qdot"][:, 0],
        np.array([0.0, 0.0, 0.0]),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        states["qdot"][:, -1],
        np.array([3.87226896, -0.86304605, -4.365753]),
        decimal=6,
    )
    # initial and final controls
    np.testing.assert_almost_equal(controls["tau"][:, 0], np.array([-1.70961892, 3.45695861, 0.10722993]), decimal=6)
    np.testing.assert_almost_equal(controls["tau"][:, -2], np.array([-0.65879374, 0.46223097, -0.7662934]), decimal=6)
