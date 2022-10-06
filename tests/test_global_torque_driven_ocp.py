"""
Test for file IO
"""
import os
import pytest

import numpy as np
import biorbd_casadi as biorbd
from bioptim import OdeSolver, ConstraintList, ConstraintFcn, Node

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
    biorbd_model = biorbd.Model(model_path)

    final_time = 2
    n_shooting = 30

    # Generate data to fit
    np.random.seed(42)
    markers_ref = np.random.rand(3, 2, n_shooting + 1)
    tau_ref = np.random.rand(2, n_shooting)

    if isinstance(ode_solver, OdeSolver.IRK):
        tau_ref = tau_ref * 5

    ocp = ocp_module.prepare_ocp(biorbd_model, final_time, n_shooting, markers_ref, tau_ref, ode_solver=ode_solver)
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
    # Load multi_model_by_mapping
    from bioptim.examples.torque_driven_ocp import phase_transition_uneven_variable_number_by_bounds as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path_withTranslations = bioptim_folder + "/models/double_pendulum_with_translations.bioMod"

    ocp = ocp_module.prepare_ocp(biorbd_model_path_withTranslations=biorbd_model_path_withTranslations)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 2742.4301647067896)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (90, 1))
    np.testing.assert_almost_equal(g, np.zeros((90, 1)), decimal=6)

    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array([0.0, 0.0, 3.14, 1.31632191]))
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array([0.0, 0.0, 6.27999994, -1.26241947]))
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array([0.0, 0.0, 6.27999994, -1.26241947]))
    np.testing.assert_almost_equal(states[1]["q"][:, -1], np.array([-0.32257863, -4.89339993, 9.4783899, -0.82639208]))
    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array([0.0, 0.0, 1.46325183, -5.62025695]))
    np.testing.assert_almost_equal(states[0]["qdot"][:, -1], np.array([0.0, 0.0, -0.81745942, 9.76461884]))
    np.testing.assert_almost_equal(states[1]["qdot"][:, 0], np.array([0.0, 0.0, -0.81745942, 9.76461884]))
    np.testing.assert_almost_equal(
        states[1]["qdot"][:, -1], np.array([24.25885769, -7.72698493, 22.04825645, -16.55976218])
    )
    # initial and final controls
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array([2.24465939, 0.93382179, 2.70624195]))
    np.testing.assert_almost_equal(controls[0]["tau"][:, -2], np.array([-10.2828808, 10.79561267, 11.30408553]))
    np.testing.assert_almost_equal(controls[1]["tau"][:, 0], np.array([5.93898959]))
    np.testing.assert_almost_equal(controls[1]["tau"][:, -2], np.array([0.71101929]))


def test_phase_transition_uneven_variable_number_by_mapping():
    # Load multi_model_by_constraint
    from bioptim.examples.torque_driven_ocp import phase_transition_uneven_variable_number_by_mapping as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/double_pendulum.bioMod"
    biorbd_model_path_withTranslations = bioptim_folder + "/models/double_pendulum_with_translations.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path, biorbd_model_path_withTranslations=biorbd_model_path_withTranslations
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -1627.4268622107388)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (66, 1))
    np.testing.assert_almost_equal(g, np.zeros((66, 1)), decimal=6)

    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array([3.14, 0.0]))
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array([8.34553412, -1.49476117]))
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array([0.0, 0.0, 8.35220747, -1.49768318]))
    np.testing.assert_almost_equal(states[1]["q"][:, -1], np.array([4.3615878, 2.14480639, 10.79222634, 1.43970003]))
    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array([-0.69668687, 0.52746273]))
    np.testing.assert_almost_equal(states[0]["qdot"][:, -1], np.array([2.14064262, 14.77599501]))
    np.testing.assert_almost_equal(states[1]["qdot"][:, 0], np.array([0.0, 0.0, 2.14317197, 14.77871693]))
    np.testing.assert_almost_equal(
        states[1]["qdot"][:, -1], np.array([2.59087355, 22.38161183, 16.20960498, 3.48132539])
    )
    # initial and final controls
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array([-0.00444966]))
    np.testing.assert_almost_equal(controls[0]["tau"][:, -2], np.array([-2.18577026]))
    np.testing.assert_almost_equal(controls[1]["tau"][:, 0], np.array([0.6362266]))
    np.testing.assert_almost_equal(controls[1]["tau"][:, -2], np.array([0.59226538]))


def test_multi_model_by_mapping():
    # Load multi_model_by_mapping
    from bioptim.examples.torque_driven_ocp import multi_model_by_mapping as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/double_pendulum.bioMod"
    biorbd_model_path_modified_inertia = bioptim_folder + "/models/double_pendulum_modified_inertia.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path, biorbd_model_path_modified_inertia=biorbd_model_path_modified_inertia
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 7.443574590553702e-06)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (44, 1))
    np.testing.assert_almost_equal(g, np.zeros((44, 1)), decimal=6)

    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array([-3.14159265, 0.0]))
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array([3.14261123, 0.0]))
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array([-3.14159265, 0.0]))
    np.testing.assert_almost_equal(states[1]["q"][:, -1], np.array([3.1387283, 0.0]))
    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array([0.02032024, -0.11683541]))
    np.testing.assert_almost_equal(states[0]["qdot"][:, -1], np.array([0.39250662, 0.36244284]))
    np.testing.assert_almost_equal(states[1]["qdot"][:, 0], np.array([0.05343674, -0.10060115]))
    np.testing.assert_almost_equal(states[1]["qdot"][:, -1], np.array([0.33994252, 0.28596524]))
    # initial and final controls
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array([7.10563304e-10]))
    np.testing.assert_almost_equal(controls[0]["tau"][:, -2], np.array([-5.12573519e-08]))
    np.testing.assert_equal(controls[1], {})


def test_multi_model_by_constraint():
    # Load multi_model_by_constraint
    from bioptim.examples.torque_driven_ocp import multi_model_by_constraint as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    # Define the problem
    biorbd_model_path = bioptim_folder + "/models/double_pendulum.bioMod"
    biorbd_model_path_modified_inertia = bioptim_folder + "/models/double_pendulum_modified_inertia.bioMod"

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=biorbd_model_path, biorbd_model_path_modified_inertia=biorbd_model_path_modified_inertia
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 6.649006745212458e-06)

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (56, 1))
    np.testing.assert_almost_equal(g, np.zeros((56, 1)), decimal=6)

    # Check some of the results
    states, controls, states_no_intermediate = sol.states, sol.controls, sol.states_no_intermediate

    # initial and final position
    np.testing.assert_almost_equal(states[0]["q"][:, 0], np.array([-3.14159265e00, -1.52694304e-16]))
    np.testing.assert_almost_equal(states[0]["q"][:, -1], np.array([3.09877448e00, -1.88600092e-16]))
    np.testing.assert_almost_equal(states[1]["q"][:, 0], np.array([-3.14159265e00, -1.17038074e-16]))
    np.testing.assert_almost_equal(states[1]["q"][:, -1], np.array([3.06719675e00, 1.51871101e-17]))
    # initial and final velocities
    np.testing.assert_almost_equal(states[0]["qdot"][:, 0], np.array([0.19084639, -0.01976167]))
    np.testing.assert_almost_equal(states[0]["qdot"][:, -1], np.array([0.26783255, 0.35826908]))
    np.testing.assert_almost_equal(states[1]["qdot"][:, 0], np.array([0.3576423, 0.10950598]))
    np.testing.assert_almost_equal(states[1]["qdot"][:, -1], np.array([0.03454933, 0.32656719]))
    # initial and final controls
    np.testing.assert_almost_equal(controls[0]["tau"][:, 0], np.array([1.77451039e-07]))
    np.testing.assert_almost_equal(controls[0]["tau"][:, -2], np.array([4.01610057e-07]))
    np.testing.assert_almost_equal(controls[1]["tau"][:, 0], np.array([1.7745104e-07]))
    np.testing.assert_almost_equal(controls[1]["tau"][:, -2], np.array([4.01610057e-07]))
