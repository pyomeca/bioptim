"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np
import biorbd

from biorbd_optim import Data, OdeSolver, Constraint, Instant
from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_align_markers(ode_solver):
    # Load align_markers
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/align_markers.py"
    )
    align_markers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_markers)

    ocp = align_markers.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod",
        number_shooting_points=30,
        final_time=2,
        use_actuators=False,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19767.53312569522)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


def test_align_markers_changing_constraints():
    # Load align_markers
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/align_markers.py"
    )
    align_markers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_markers)

    ocp = align_markers.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod",
        number_shooting_points=30,
        final_time=2,
    )
    sol = ocp.solve()

    # Add a new constraint and reoptimize
    ocp.add_constraint(
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.MID, "first_marker_idx": 0, "second_marker_idx": 2,}
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 20370.211697123825)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (189, 1))
    np.testing.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((4.2641129, 9.81, 2.27903226)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((1.36088709, 9.81, -2.27903226)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # Replace constraints and reoptimize
    ocp.modify_constraint(
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker_idx": 0, "second_marker_idx": 2,}, 0
    )
    ocp.modify_constraint(
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.MID, "first_marker_idx": 0, "second_marker_idx": 3,}, 2
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 31670.93770220887)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (189, 1))
    np.testing.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-5.625, 21.06, 2.2790323)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-5.625, 21.06, -2.27903226)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_align_markers_with_actuators(ode_solver):
    # Load align_markers
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/align_markers.py"
    )
    align_markers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align_markers)

    ocp = align_markers.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod",
        number_shooting_points=30,
        final_time=2,
        use_actuators=True,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 204.18087334169184)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0.2140175, 0.981, 0.3360075)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-0.2196496, 0.981, -0.3448498)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_multiphase_align_markers(ode_solver):
    # Load multiphase_align_markers
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "multiphase_align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/multiphase_align_markers.py"
    )
    multiphase_align_markers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multiphase_align_markers)

    ocp = multiphase_align_markers.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 106084.82631762947)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (444, 1))
    np.testing.assert_almost_equal(g, np.zeros((444, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"], concatenate=False)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[0][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[0][:, -1], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[1][:, 0], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[1][:, -1], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[2][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[2][:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[0][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[0][:, -1], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[1][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[1][:, -1], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[2][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[2][:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[0][:, 0], np.array((1.42857142, 9.81, 0)))
    np.testing.assert_almost_equal(tau[0][:, -1], np.array((-1.42857144, 9.81, 0)))
    np.testing.assert_almost_equal(tau[1][:, 0], np.array((-0.2322581, 9.81, 0.0)))
    np.testing.assert_almost_equal(tau[1][:, -1], np.array((0.2322581, 9.81, -0.0)))
    np.testing.assert_almost_equal(tau[2][:, 0], np.array((0.35714285, 9.81, 0.56071428)))
    np.testing.assert_almost_equal(tau[2][:, -1], np.array((-0.35714285, 9.81, -0.56071428)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_external_forces(ode_solver):
    # Load external_forces
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "external_forces", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/external_forces.py"
    )
    external_forces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external_forces)

    ocp = external_forces.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube_with_forces.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 9875.88768746912)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (246, 1))
    np.testing.assert_almost_equal(g, np.zeros((246, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 2, 0, 0)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.71322593, 0, 0)))
    np.testing.assert_almost_equal(tau[:, 10], np.array((0, 7.71100122, 0, 0)))
    np.testing.assert_almost_equal(tau[:, 20], np.array((0, 5.70877651, 0, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 3.90677425, 0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


def test_track_marker_2D_pendulum():
    # Load muscle_activations_contact_tracker
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "track_markers_2D_pendulum", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/track_markers_2D_pendulum.py",
    )
    track_markers_2D_pendulum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(track_markers_2D_pendulum)

    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/getting_started/pendulum.bioMod"
    biorbd_model = biorbd.Model(model_path)
    final_time = 3
    nb_shooting = 20

    # Generate data to fit
    markers_ref = np.zeros((3, 2, nb_shooting + 1))
    markers_ref[1, :, :] = np.array(
        [
            [
                0.0,
                0.59994037,
                1.41596094,
                1.78078534,
                2.00186844,
                2.21202252,
                2.44592581,
                2.66008025,
                2.52186711,
                1.70750629,
                1.01558254,
                0.80411881,
                0.93002487,
                1.10917845,
                0.99843605,
                0.67973254,
                0.50780818,
                0.28087458,
                -0.72163361,
                -0.76304319,
                0.0,
            ],
            [
                0.1,
                0.39821962,
                0.84838182,
                1.09895944,
                1.30730217,
                1.56395927,
                1.89764787,
                2.28282596,
                2.53658449,
                2.38557865,
                2.00463841,
                1.75824184,
                1.65985162,
                1.54219021,
                1.25554824,
                0.87993452,
                0.59982138,
                0.30228811,
                -0.38347237,
                -0.48508912,
                -0.09840722,
            ],
        ]
    )

    markers_ref[2, 1, :] = np.array(
        [
            -1.0,
            -0.98453478,
            -0.8293696,
            -0.73831799,
            -0.72634543,
            -0.7681237,
            -0.84225371,
            -0.931493,
            -1.00487979,
            -0.74176672,
            -0.17823715,
            0.31567267,
            0.69090731,
            0.90691831,
            0.97154172,
            0.98484474,
            1.00076649,
            1.0047594,
            0.94638627,
            0.96578545,
            1.000158,
        ]
    )

    tau_ref = np.zeros((2, nb_shooting))
    tau_ref[0, :] = np.array(
        [
            26.50837752,
            -13.00256609,
            -4.73822352,
            0.98400741,
            3.31060529,
            3.53663986,
            1.04020674,
            -12.72188939,
            -23.25758642,
            2.81968664,
            9.13976837,
            4.01638109,
            -5.72934928,
            -9.29116278,
            1.38256926,
            7.10636934,
            -8.65483649,
            -25.85422034,
            77.77873644,
            -34.37165499,
        ]
    )

    ocp = track_markers_2D_pendulum.prepare_ocp(biorbd_model, final_time, nb_shooting, markers_ref, tau_ref)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 3.15)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (80, 1))
    np.testing.assert_almost_equal(g, np.zeros((80, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 3.14)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((26.5083775, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-34.3716550, 0)))
