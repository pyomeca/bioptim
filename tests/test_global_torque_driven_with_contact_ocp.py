"""
Test for file IO.
It tests the results of an optimal control problem with torque_driven_with_contact problem type regarding the proper functioning of :
- the maximize/minimize_predicted_height_CoM objective
- the contact_forces_inequality constraint
- the non_slipping constraint
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from bioptim import Data, OdeSolver
from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
@pytest.mark.parametrize(
    "objective_name", ["MINIMIZE_PREDICTED_COM_HEIGHT", "MINIMIZE_COM_POSITION", "MINIMIZE_COM_VELOCITY"]
)
@pytest.mark.parametrize("com_constraints", [False, True])
def test_maximize_predicted_height_CoM(ode_solver, objective_name, com_constraints):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "maximize_predicted_height_CoM",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/maximize_predicted_height_CoM.py",
    )
    maximize_predicted_height_CoM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maximize_predicted_height_CoM)

    ocp = maximize_predicted_height_CoM.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        number_shooting_points=20,
        use_actuators=False,
        ode_solver=ode_solver,
        objective_name=objective_name,
        com_constraints=com_constraints,
    )
    sol = ocp.solve()

    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))

    # Check constraints
    g = np.array(sol["g"])
    if com_constraints:
        np.testing.assert_equal(g.shape, (244, 1))
    else:
        np.testing.assert_equal(g.shape, (160, 1))
        np.testing.assert_almost_equal(g, np.zeros((160, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    # initial velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))

    if objective_name == "MINIMIZE_PREDICTED_COM_HEIGHT":
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 0.7592028279017864)

        # final position
        np.testing.assert_almost_equal(q[:, -1], np.array((0.1189651, -0.0904378, -0.7999996, 0.7999996)))
        # final velocities
        np.testing.assert_almost_equal(qdot[:, -1], np.array((1.2636414, -1.3010929, -3.6274687, 3.6274687)))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-22.1218282)))
        np.testing.assert_almost_equal(tau[:, -1], np.array(0.2653957))
    elif objective_name == "MINIMIZE_COM_POSITION":
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 0.458575464873056)

        # final position
        np.testing.assert_almost_equal(q[:, -1], np.array((0.1189651, -0.0904378, -0.7999996, 0.7999996)))
        # final velocities
        np.testing.assert_almost_equal(qdot[:, -1], np.array((1.24525494, -1.28216182, -3.57468814, 3.57468814)))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-21.96213697)))
        np.testing.assert_almost_equal(tau[:, -1], np.array(-0.22120207))
    elif objective_name == "MINIMIZE_COM_VELOCITY":
        # Check objective function value
        np.testing.assert_almost_equal(f[0, 0], 0.4709888694097001)

        # final position
        np.testing.assert_almost_equal(q[:, -1], np.array((0.1189652, -0.09043785, -0.79999979, 0.79999979)))
        # final velocities
        np.testing.assert_almost_equal(qdot[:, -1], np.array((1.26103572, -1.29841047, -3.61998944, 3.61998944)))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-22.18008227)))
        np.testing.assert_almost_equal(tau[:, -1], np.array(-0.02280469))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_maximize_predicted_height_CoM_with_actuators(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "maximize_predicted_height_CoM",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/maximize_predicted_height_CoM.py",
    )
    maximize_predicted_height_CoM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maximize_predicted_height_CoM)

    ocp = maximize_predicted_height_CoM.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        number_shooting_points=20,
        use_actuators=True,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.21850679397314332)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (160, 1))
    np.testing.assert_almost_equal(g, np.zeros((160, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    if ode_solver == OdeSolver.IRK:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
        np.testing.assert_almost_equal(q[:, -1], np.array((-0.2393758, 0.0612086, -0.0006739, 0.0006739)))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-4.87675667e-01, 3.28672149e-04, 9.75351556e-01, -9.75351556e-01))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-0.5509092)))
        np.testing.assert_almost_equal(tau[:, -1], np.array(-0.00506117))

    elif ode_solver == OdeSolver.RK8:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
        np.testing.assert_almost_equal(q[:, -1], np.array((-0.23937581, 0.06120861, -0.00067392, 0.00067392)))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-4.87675528e-01, 3.28670915e-04, 9.75351279e-01, -9.75351279e-01))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-0.55090931)))
        np.testing.assert_almost_equal(tau[:, -1], np.array(-0.00506117))

    else:
        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
        np.testing.assert_almost_equal(q[:, -1], np.array((-0.2393758, 0.0612086, -0.0006739, 0.0006739)))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-4.8768219e-01, 3.2867302e-04, 9.7536459e-01, -9.7536459e-01))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-0.550905)))
        np.testing.assert_almost_equal(tau[:, -1], np.array(-0.0050623))

        # save and load
        TestUtils.save_and_load(sol, ocp, False)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_contact_forces_inequality_GREATER_THAN_constraint(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "contact_forces_inequality_constraint",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/contact_forces_inequality_constraint.py",
    )
    contact_forces_inequality_GREATER_THAN_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contact_forces_inequality_GREATER_THAN_constraint)

    min_bound = 50
    ocp = contact_forces_inequality_GREATER_THAN_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        number_shooting_points=10,
        min_bound=min_bound,
        max_bound=np.inf,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.14525621569048172)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    if ode_solver == OdeSolver.IRK:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (100, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(-g[80:], -min_bound)
        expected_pos_g = np.array(
            [
                [50.76334043],
                [51.42154006],
                [57.79496471],
                [64.29700748],
                [67.01987853],
                [68.32305222],
                [67.91820667],
                [65.26711376],
                [59.57312581],
                [50.1847888],
                [160.1560585],
                [141.16683648],
                [85.1060599],
                [56.33412288],
                [53.32765464],
                [52.21769321],
                [51.63001946],
                [51.2579451],
                [50.98768816],
                [50.21989568],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_pos_g)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
        np.testing.assert_almost_equal(q[:, -1], np.array((-0.34054772, 0.1341555, -0.00054332, 0.00054332)))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-2.01096899e00, 1.09261741e-03, 4.02193851e00, -4.02193851e00))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-54.17110048)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-15.69344349)))

    elif ode_solver == OdeSolver.RK8:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (100, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(-g[80:], -min_bound)
        expected_pos_g = np.array(
            [
                [50.76332058],
                [51.42161439],
                [57.79492106],
                [64.29699128],
                [67.01987366],
                [68.32304868],
                [67.91820215],
                [65.26710733],
                [59.57311721],
                [50.18478458],
                [160.15615544],
                [141.16646975],
                [85.10634138],
                [56.33416239],
                [53.32764628],
                [52.21768768],
                [51.63001903],
                [51.25794807],
                [50.98769249],
                [50.21989225],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_pos_g)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
        np.testing.assert_almost_equal(q[:, -1], np.array((-0.34054772, 0.1341555, -0.00054332, 0.00054332)))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-2.01096914e00, 1.09259198e-03, 4.02193881e00, -4.02193881e00))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-54.17113441)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-15.69344229)))

    else:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (100, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(-g[80:], -min_bound)
        expected_pos_g = np.array(
            [
                [50.76491919],
                [51.42493119],
                [57.79007374],
                [64.29551934],
                [67.01905769],
                [68.3225625],
                [67.91793917],
                [65.26700138],
                [59.57311867],
                [50.18463134],
                [160.14834799],
                [141.15361769],
                [85.13345729],
                [56.33535022],
                [53.32684286],
                [52.21679255],
                [51.62923106],
                [51.25728666],
                [50.9871531],
                [50.21972377],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_pos_g)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
        np.testing.assert_almost_equal(q[:, -1], np.array((-0.34054748, 0.1341555, -0.0005438, 0.0005438)))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.01097559, 1.09352001e-03, 4.02195175, -4.02195175)))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-54.1684018)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-15.69338332)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_contact_forces_inequality_LESSER_THAN_constraint(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "contact_forces_inequality_constraint",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/contact_forces_inequality_constraint.py",
    )
    contact_forces_inequality_LESSER_THAN_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contact_forces_inequality_LESSER_THAN_constraint)

    max_bound = 100
    ocp = contact_forces_inequality_LESSER_THAN_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        number_shooting_points=10,
        min_bound=-np.inf,
        max_bound=max_bound,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.14525619649247054)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(
        q[:, -1], np.array((-3.40655617e-01, 1.34155544e-01, -3.27530886e-04, 3.27530886e-04))
    )

    if ode_solver == OdeSolver.IRK:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (100, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(g[80:], max_bound)
        expected_non_zero_g = np.array(
            [
                [63.27209168],
                [63.02302254],
                [62.13840892],
                [60.38286495],
                [57.31035211],
                [52.1969189],
                [43.95984323],
                [31.14447074],
                [12.4527049],
                [-6.20139005],
                [99.0646825],
                [98.87878575],
                [98.64638238],
                [98.3478478],
                [97.94940411],
                [97.3880652],
                [96.53094583],
                [95.03988984],
                [91.72272481],
                [77.29740256],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_non_zero_g)

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-2.86544932e00, 9.38791617e-04, 5.73089895e00, -5.73089895e00))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-32.78911887)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-25.1705709)))

    elif ode_solver == OdeSolver.RK8:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (100, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(g[80:], max_bound)
        expected_non_zero_g = np.array(
            [
                [63.27213612],
                [63.02308136],
                [62.13849861],
                [60.38301138],
                [57.31059542],
                [52.19731543],
                [43.96044403],
                [31.14516578],
                [12.45211],
                [-6.22371884],
                [99.06446547],
                [98.87852555],
                [98.64607279],
                [98.34748387],
                [97.94898697],
                [97.38761773],
                [96.53056621],
                [95.0399696],
                [91.72523713],
                [77.32555225],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_non_zero_g)

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array((-2.86560719e00, 9.38625648e-04, 5.73121471e00, -5.73121471e00))
        )
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-32.78904291)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-25.18042329)))

    else:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14525619649247054)

        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (100, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(g[80:], max_bound)
        expected_non_zero_g = np.array(
            [
                [63.27237842],
                [63.02339946],
                [62.13898369],
                [60.38380769],
                [57.31193141],
                [52.19952395],
                [43.9638679],
                [31.14938032],
                [12.45022537],
                [-6.35179034],
                [99.06328211],
                [98.87711942],
                [98.64440005],
                [98.34550037],
                [97.94667107],
                [97.38505013],
                [96.52820867],
                [95.03979128],
                [91.73734926],
                [77.48803304],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_non_zero_g)

        # Check some of the results
        states, controls = Data.get_data(ocp, sol["x"])
        q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
        np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.86650427, 9.38827988e-04, 5.73300901, -5.73300901)))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-32.78862874)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-25.23729156)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_non_slipping_constraint(ode_solver):
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "non_slipping_constraint",
        str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/non_slipping_constraint.py",
    )
    non_slipping_constraint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(non_slipping_constraint)

    ocp = non_slipping_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.6,
        number_shooting_points=10,
        mu=0.005,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.23984490846250128)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.02364845, 0.01211471, -0.44685185, 0.44685185)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-0.08703131, 0.04170362, 0.1930144, -0.1930144)))

    if ode_solver == OdeSolver.IRK:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (120, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(-g[80:], 0)
        expected_pos_g = np.array(
            [
                [8.74337995e01],
                [8.74671258e01],
                [8.75687834e01],
                [8.77422815e01],
                [8.79913159e01],
                [8.83197846e01],
                [8.87318042e01],
                [8.92317303e01],
                [8.98241984e01],
                [9.05145023e01],
                [4.63475930e01],
                [4.63130361e01],
                [4.62075073e01],
                [4.60271955e01],
                [4.57680917e01],
                [4.54259739e01],
                [4.49963905e01],
                [4.44746352e01],
                [4.38556794e01],
                [4.31334131e01],
                [1.33775343e00],
                [6.04899683e-05],
                [1.33773204e00],
                [6.95785710e-05],
                [1.33768173e00],
                [8.11784388e-05],
                [1.33759829e00],
                [9.64764544e-05],
                [1.33747653e00],
                [1.17543268e-04],
                [1.33730923e00],
                [1.48352207e-04],
                [1.33708435e00],
                [1.97600315e-04],
                [1.33677502e00],
                [2.88636405e-04],
                [1.33628619e00],
                [5.12590351e-04],
                [1.33466928e00],
                [1.80987563e-03],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_pos_g)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-14.33813755)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-13.21317457)))

    elif ode_solver == OdeSolver.RK8:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (120, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(-g[80:], 0)
        expected_pos_g = np.array(
            [
                [8.74337995e01],
                [8.74671258e01],
                [8.75687834e01],
                [8.77422815e01],
                [8.79913159e01],
                [8.83197846e01],
                [8.87318042e01],
                [8.92317303e01],
                [8.98241983e01],
                [9.05145021e01],
                [4.63475930e01],
                [4.63130361e01],
                [4.62075073e01],
                [4.60271956e01],
                [4.57680917e01],
                [4.54259739e01],
                [4.49963906e01],
                [4.44746352e01],
                [4.38556795e01],
                [4.31334132e01],
                [1.33775343e00],
                [6.04899707e-05],
                [1.33773204e00],
                [6.95785768e-05],
                [1.33768173e00],
                [8.11784366e-05],
                [1.33759829e00],
                [9.64764593e-05],
                [1.33747653e00],
                [1.17543270e-04],
                [1.33730923e00],
                [1.48352215e-04],
                [1.33708435e00],
                [1.97600321e-04],
                [1.33677502e00],
                [2.88636410e-04],
                [1.33628619e00],
                [5.12590375e-04],
                [1.33466928e00],
                [1.80987583e-03],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_pos_g)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-14.33813755)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-13.2131746)))

    else:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (120, 1))
        np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
        np.testing.assert_array_less(-g[80:], 0)
        expected_pos_g = np.array(
            [
                [8.74337995e01],
                [8.74671258e01],
                [8.75687834e01],
                [8.77422814e01],
                [8.79913157e01],
                [8.83197844e01],
                [8.87318039e01],
                [8.92317298e01],
                [8.98241976e01],
                [9.05145013e01],
                [4.63475930e01],
                [4.63130361e01],
                [4.62075073e01],
                [4.60271956e01],
                [4.57680919e01],
                [4.54259742e01],
                [4.49963909e01],
                [4.44746357e01],
                [4.38556802e01],
                [4.31334141e01],
                [1.33775343e00],
                [6.04899894e-05],
                [1.33773204e00],
                [6.95785950e-05],
                [1.33768173e00],
                [8.11784641e-05],
                [1.33759829e00],
                [9.64764869e-05],
                [1.33747653e00],
                [1.17543301e-04],
                [1.33730923e00],
                [1.48352248e-04],
                [1.33708435e00],
                [1.97600363e-04],
                [1.33677502e00],
                [2.88636453e-04],
                [1.33628619e00],
                [5.12590377e-04],
                [1.33466928e00],
                [1.80987419e-03],
            ]
        )
        np.testing.assert_almost_equal(g[80:], expected_pos_g)

        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array((-14.33813755)))
        np.testing.assert_almost_equal(tau[:, -1], np.array((-13.21317493)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)
