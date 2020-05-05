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

from biorbd_optim import ProblemType, OdeSolver

# Load symmetry
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "maximize_predicted_height_CoM",
    str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/maximize_predicted_height_CoM.py",
)
maximize_predicted_height_CoM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maximize_predicted_height_CoM)

spec = importlib.util.spec_from_file_location(
    "contact_forces_inequality_constraint", str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/contact_forces_inequality_constraint.py",
)
contact_forces_inequality_constraint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contact_forces_inequality_constraint)

spec = importlib.util.spec_from_file_location(
    "non_slipping_constraint", str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/non_slipping_constraint.py",
)
non_slipping_constraint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(non_slipping_constraint)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_maximize_predicted_height_CoM(ode_solver):
    ocp = maximize_predicted_height_CoM.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        number_shooting_points=20,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.2370021957829777)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (160, 1))
    # np.testing.assert_almost_equal(g, np.zeros((160, 1)))

    # Check some of the results
    q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0., 0., -0.5, 0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.2191102,  0.0607841, -0.0412168,  0.0412168)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-0.3689301,  0.0152147,  0.7384873, -0.7384873)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-27.2604552)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-0.2636804)))


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_contact_forces_inequality_constraint(ode_solver):
    ocp = contact_forces_inequality_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        number_shooting_points=10
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.1452562070438664)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (100, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)), decimal=6)
    idx_neg_value_of_g = [0, 1, 3, 4, 5, 7, 8, 9, 11, 15, 19, 23, 27, 31, 35, 39, 40, 43, 44, 47, 48, 49, 51, 52, 53, 55, 56, 57, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 73, 75, 77, 79]
    for idx in idx_neg_value_of_g:
        assert(g[idx] <= 0)
    idx_pos_value_of_g = [2, 6, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 41, 42, 45, 46, 50, 54, 58, 62, 66, 70, 72, 74, 76, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    for idx in idx_pos_value_of_g:
        assert(g[idx] >= 0)

    # Check some of the results
    q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0., 0., -0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-3.4081740e-01,  1.3415556e-01, -3.9566794e-06,  3.9566794e-06)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.0922416e+00,  8.2778459e-06,  4.1844833e+00, -4.1844833e+00)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-70.1205906)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-17.5866089)))

@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_non_slipping_constraint(ode_solver):
    ocp = non_slipping_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.6,
        number_shooting_points=10,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.21820306105763151)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (110, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
    idx_neg_value_of_g = [2, 6, 10, 14, 18, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 49, 50, 53, 54, 58, 62, 66, 70, 75, 79]
    for idx in idx_neg_value_of_g:
        assert(g[idx] <= 0)
    idx_pos_value_of_g = [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 27, 31, 35, 39, 43, 47, 48, 51, 52, 55, 56, 57, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 74, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    for idx in idx_pos_value_of_g:
        assert(g[idx] >= 0)

    # Check some of the results
    q, qdot, tau = ProblemType.get_data_from_V(ocp, sol["x"])

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0. ,  0. , -0.5,  0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-2.3970871e-01,  6.1208704e-02, -8.1394939e-06,  8.1394939e-06)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-7.4541514e-01,  5.9274858e-06,  1.4908303e+00, -1.4908303e+00)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-19.7138885)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-4.2970368)))
