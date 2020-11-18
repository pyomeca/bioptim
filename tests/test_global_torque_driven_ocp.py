"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np
import biorbd

from bioptim import Data, OdeSolver, ConstraintList, Constraint, Node
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

    # simulate
    TestUtils.simulate(sol, ocp)


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
    new_constraints = ConstraintList()
    new_constraints.add(Constraint.ALIGN_MARKERS, node=Node.MID, first_marker_idx=0, second_marker_idx=2, list_index=2)
    ocp.update_constraints(new_constraints)
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

    # simulate
    TestUtils.simulate(sol, ocp)

    # Replace constraints and reoptimize
    new_constraints = ConstraintList()
    new_constraints.add(
        Constraint.ALIGN_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=2, list_index=0
    )
    new_constraints.add(Constraint.ALIGN_MARKERS, node=Node.MID, first_marker_idx=0, second_marker_idx=3, list_index=2)
    ocp.update_constraints(new_constraints)
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

    # simulate
    TestUtils.simulate(sol, ocp)


def test_align_markers_with_actuators():
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

    # simulate
    TestUtils.simulate(sol, ocp)


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
    sol, obj = ocp.solve(return_objectives=True)

    # Check return_objectives
    np.testing.assert_almost_equal(
        obj[0],
        np.array(
            [
                [
                    982.76916324,
                    978.69883706,
                    975.08076934,
                    971.91496008,
                    969.20140929,
                    966.94011697,
                    965.13108311,
                    963.77430771,
                    962.86979078,
                    962.41753231,
                    962.41753231,
                    962.86979077,
                    963.7743077,
                    965.1310831,
                    966.94011696,
                    969.20140929,
                    971.91496009,
                    975.08076936,
                    978.6988371,
                    982.76916331,
                ]
            ]
        ),
    )
    np.testing.assert_almost_equal(
        obj[1],
        np.array(
            [
                [
                    1604.83406353,
                    1604.71433092,
                    1604.60315064,
                    1604.5005227,
                    1604.40644708,
                    1604.3209238,
                    1604.24395284,
                    1604.17553422,
                    1604.11566792,
                    1604.06435395,
                    1604.02159231,
                    1603.987383,
                    1603.96172602,
                    1603.94462137,
                    1603.93606904,
                    1603.93606904,
                    1603.94462137,
                    1603.96172603,
                    1603.98738301,
                    1604.02159232,
                    1604.06435396,
                    1604.11566793,
                    1604.17553422,
                    1604.24395285,
                    1604.32092379,
                    1604.40644707,
                    1604.50052267,
                    1604.6031506,
                    1604.71433086,
                    1604.83406345,
                ]
            ]
        ),
    )
    np.testing.assert_almost_equal(
        obj[2],
        np.array(
            [
                [
                    1933.56103058,
                    1931.79812144,
                    1930.23109109,
                    1928.85993953,
                    1927.68466677,
                    1926.7052728,
                    1925.92175762,
                    1925.33412124,
                    1924.94236365,
                    1924.74648485,
                    1924.74648485,
                    1924.94236364,
                    1925.33412123,
                    1925.92175761,
                    1926.70527279,
                    1927.68466676,
                    1928.85993954,
                    1930.23109111,
                    1931.79812149,
                    1933.56103067,
                ]
            ]
        ),
    )

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

    # simulate
    with pytest.raises(AssertionError, match="Arrays are not almost equal to 7 decimals"):
        TestUtils.simulate(sol, ocp)


def test_external_forces():
    # Load external_forces
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "external_forces", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/external_forces.py"
    )
    external_forces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external_forces)

    ocp = external_forces.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube_with_forces.bioMod",
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

    # simulate
    TestUtils.simulate(sol, ocp)


def test_track_marker_2D_pendulum():
    # Load muscle_activations_contact_tracker
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "track_markers_2D_pendulum", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/track_markers_2D_pendulum.py"
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
    np.testing.assert_almost_equal(f[0, 0], 0)

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

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


def test_trampo_quaternions():
    # Load trampo_quaternion
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "trampo_quaternions",
        str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/trampo_quaternions.py",
    )
    trampo_quaternions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trampo_quaternions)

    # Define the problem
    model_path = str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/TruncAnd2Arm_Quaternion.bioMod"
    final_time = 0.25
    nb_shooting = 5

    ocp = trampo_quaternions.prepare_ocp(model_path, nb_shooting, final_time)
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], -41.491609816961535)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (130, 1))
    np.testing.assert_almost_equal(
        g,
        np.array(
            [
                [-2.62869726e-11],
                [-2.59978705e-11],
                [-4.68713957e-11],
                [-7.22507609e-11],
                [6.04819528e-11],
                [5.10899975e-10],
                [3.54277866e-08],
                [1.47388822e-07],
                [2.76794123e-08],
                [5.62982064e-08],
                [-1.05090022e-07],
                [2.02544745e-08],
                [1.50100370e-07],
                [9.03570907e-08],
                [-1.33880640e-09],
                [-1.48755586e-09],
                [-2.29075248e-09],
                [-2.99122663e-09],
                [3.34152594e-09],
                [2.07377712e-08],
                [-3.93602084e-09],
                [-6.26294314e-08],
                [7.85837972e-09],
                [3.33429790e-08],
                [-2.90390112e-09],
                [4.12724077e-10],
                [-5.35571587e-12],
                [3.77520237e-12],
                [4.38094006e-12],
                [-7.72493181e-13],
                [-3.12296855e-11],
                [-3.95635608e-12],
                [-1.05131891e-09],
                [1.52380329e-08],
                [2.24447717e-11],
                [-1.63395921e-10],
                [-5.72631709e-09],
                [-6.14008504e-10],
                [-1.25496301e-08],
                [-5.51599033e-09],
                [-2.75368173e-10],
                [1.35714551e-10],
                [1.30859767e-10],
                [-1.28282301e-10],
                [-1.31088629e-09],
                [6.71719014e-10],
                [-4.43085968e-09],
                [1.01520671e-08],
                [-1.01466322e-08],
                [-5.75423542e-09],
                [2.76743517e-10],
                [-3.61604069e-09],
                [-3.26538796e-12],
                [5.21938048e-12],
                [5.92281779e-12],
                [4.31099600e-13],
                [-3.27530780e-11],
                [1.72692416e-12],
                [-9.25327218e-10],
                [1.50035209e-08],
                [-4.03308265e-10],
                [-3.72501890e-10],
                [-5.91380533e-09],
                [-5.09477224e-10],
                [-1.28129521e-08],
                [-5.35508871e-09],
                [-1.90380156e-10],
                [1.84304128e-10],
                [2.01880290e-10],
                [-3.19055615e-11],
                [-1.34893363e-09],
                [5.37814016e-10],
                [-5.33846300e-10],
                [8.35767588e-09],
                [-9.56881263e-09],
                [-5.56975732e-09],
                [-8.59005533e-10],
                [-3.53006502e-09],
                [-8.01581024e-13],
                [6.03117556e-12],
                [6.84563517e-12],
                [1.71107573e-12],
                [-3.20352633e-11],
                [6.41613151e-12],
                [-7.36204347e-10],
                [1.45070870e-08],
                [-9.15970327e-10],
                [-5.83527560e-10],
                [-6.16792484e-09],
                [-4.00933266e-10],
                [-1.27920323e-08],
                [-5.25297139e-09],
                [-9.00550745e-11],
                [2.09417372e-10],
                [2.46619170e-10],
                [6.06714401e-11],
                [-1.29745592e-09],
                [4.13145851e-10],
                [2.37002595e-09],
                [6.36820452e-09],
                [-8.36713410e-09],
                [-5.64461144e-09],
                [-2.15799734e-09],
                [-3.31499672e-09],
                [1.76303416e-12],
                [6.24700291e-12],
                [7.05746572e-12],
                [2.60091948e-12],
                [-2.93295388e-11],
                [1.16103932e-11],
                [-5.20448004e-10],
                [1.37706937e-08],
                [-1.50919800e-09],
                [-7.92144378e-10],
                [-6.48559595e-09],
                [-3.03825201e-10],
                [-1.24819112e-08],
                [-5.19425047e-09],
                [1.43227652e-11],
                [2.12847517e-10],
                [2.61523248e-10],
                [1.30651323e-10],
                [-1.16898646e-09],
                [3.63143182e-10],
                [4.07144851e-09],
                [4.31356673e-09],
                [-6.69863010e-09],
                [-5.99857852e-09],
                [-3.62082897e-09],
                [-2.98583003e-09],
            ]
        ),
    )

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(
        q[:, 0],
        np.array(
            [
                1.81406193,
                1.91381625,
                2.01645623,
                -0.82692043,
                0.22763972,
                -0.12271756,
                0.01240349,
                -0.2132477,
                -0.02276448,
                -0.02113187,
                0.25834635,
                0.02304512,
                0.16975019,
                0.28942782,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        q[:, -1],
        np.array(
            [
                3.14159265,
                3.14159265,
                3.14159265,
                -0.78539815,
                0.6154797,
                0.10983897,
                0.03886357,
                -0.7291664,
                0.09804026,
                0.1080609,
                0.77553818,
                0.05670268,
                0.67616132,
                0.61939341,
            ]
        ),
    )

    # initial and final velocities
    np.testing.assert_almost_equal(
        qdot[:, 0],
        np.array(
            [
                5.29217272,
                4.89048559,
                5.70619178,
                0.14746316,
                1.53632472,
                0.99292273,
                0.939669,
                0.93500575,
                0.6618465,
                0.71874771,
                1.1402524,
                0.84452034,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        qdot[:, -1],
        np.array(
            [
                5.33106829,
                4.93221086,
                3.29404834,
                0.19267326,
                1.55927187,
                0.8793173,
                0.9413292,
                0.52604828,
                1.27993251,
                1.25250626,
                1.39280633,
                1.13948993,
            ]
        ),
    )

    # initial and final controls
    np.testing.assert_almost_equal(
        tau[:, 0],
        np.array(
            [
                3.90118808e-12,
                1.37003760e-12,
                2.36150621e-12,
                -3.06473544e-12,
                -1.97308650e-11,
                -2.15361747e-10,
                6.08890889e-10,
                5.34191006e-10,
                -3.06474069e-08,
                -3.11346548e-10,
                1.92411028e-10,
                -1.36507313e-08,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        tau[:, -1],
        np.array(
            [
                4.16625669e-12,
                2.78293338e-12,
                1.88352712e-13,
                -9.54764334e-12,
                -2.45115795e-11,
                -2.66890608e-10,
                6.44153858e-10,
                4.51256175e-10,
                -2.48333899e-08,
                -4.08869506e-10,
                2.70999557e-10,
                -8.94752188e-09,
            ]
        ),
    )
