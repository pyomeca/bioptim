"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np

from tests import Utils
from biorbd_optim import Data

# Load static_arm
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "static_arm", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/static_arm.py",
)
static_arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_arm)

# Load static_arm_with_contact
spec = importlib.util.spec_from_file_location(
    "static_arm_with_contact", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/static_arm_with_contact.py",
)
static_arm_with_contact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_arm_with_contact)

# Load contact_forces_inequality_constraint_muscle_excitations
spec = importlib.util.spec_from_file_location(
    "contact_forces_inequality_constraint_muscle_excitations",
    str(PROJECT_FOLDER)
    + "/examples/muscle_driven_with_contact/contact_forces_inequality_constraint_muscle_excitations.py",
)
contact_forces_inequality_constraint_muscle_excitations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contact_forces_inequality_constraint_muscle_excitations)


def test_muscle_driven_ocp():
    ocp = static_arm.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod", final_time=2, number_shooting_points=10
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.12145862454010191)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([-0.95746379, 3.09503322]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.34608528, -0.42902204]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([0.00458156, 0.00423722]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00134814, 0.00312362]))
    np.testing.assert_almost_equal(
        mus[:, 0],
        np.array([6.86190724e-06, 1.68975654e-01, 8.70683950e-02, 2.47147957e-05, 2.53934274e-05, 8.45479390e-02]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1],
        np.array([1.96717613e-02, 3.42406388e-05, 3.29456098e-05, 8.61728932e-03, 8.57918458e-03, 7.07096066e-03]),
    )

    # save and load
    Utils.save_and_load(sol, ocp, False)


def test_muscle_with_contact_driven_ocp():
    ocp = static_arm_with_contact.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26_with_contact.bioMod",
        final_time=2,
        number_shooting_points=10,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.12145850795787627)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (60, 1))
    np.testing.assert_almost_equal(g, np.zeros((60, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0, 0.07, 1.4]))
    np.testing.assert_almost_equal(q[:, -1], np.array([0.00806796, -0.95744612, 3.09501746]))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0.0, 0.0]))
    np.testing.assert_almost_equal(qdot[:, -1], np.array([0.00060479, 0.34617159, -0.42910107]))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-4.67191123e-05, 4.58080354e-03, 4.23701953e-03]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-1.19345958e-05, -1.34810157e-03, 3.12378179e-03]))
    np.testing.assert_almost_equal(
        mus[:, 0],
        np.array([6.86280305e-06, 1.68961894e-01, 8.70635867e-02, 2.47160155e-05, 2.53946780e-05, 8.45438966e-02]),
    )
    np.testing.assert_almost_equal(
        mus[:, -1],
        np.array([1.96721725e-02, 3.42398501e-05, 3.29454916e-05, 8.61757459e-03, 8.57946846e-03, 7.07152302e-03]),
    )

    # save and load
    Utils.save_and_load(sol, ocp, False)


def test_muscle_excitation_with_contact_driven_ocp():
    boundary = 50
    ocp = contact_forces_inequality_constraint_muscle_excitations.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_with_contact/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        number_shooting_points=10,
        direction="GREATER_THAN",
        boundary=boundary,
        show_online_optim=False,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.14525619)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (110, 1))
    np.testing.assert_almost_equal(g[:90], np.zeros((90, 1)))
    np.testing.assert_array_less(-g[90:], -boundary)
    expected_pos_g = np.array(
        [
            [51.5673555],
            [52.82179693],
            [57.5896514],
            [62.60246484],
            [65.13414631],
            [66.29498636],
            [65.77592127],
            [62.98288508],
            [57.0934291],
            [50.47918162],
            [156.22933663],
            [135.96633458],
            [89.93755291],
            [63.57705684],
            [57.59613028],
            [55.17020948],
            [53.83337907],
            [52.95213608],
            [52.20317604],
            [50.57048159],
        ]
    )
    np.testing.assert_almost_equal(g[90:], expected_pos_g)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, mus_states, tau, mus_controls = (
        states["q"],
        states["q_dot"],
        states["muscles"],
        controls["tau"],
        controls["muscles"],
    )

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
    np.testing.assert_almost_equal(
        q[:, -1], np.array([-3.40710032e-01, 1.34155565e-01, -2.18684502e-04, 2.18684502e-04])
    )
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(
        qdot[:, -1], np.array([-2.01607708e00, 4.40761528e-04, 4.03215433e00, -4.03215433e00])
    )
    # initial and final muscle state
    np.testing.assert_almost_equal(
        mus_states[:, 0], np.array([0.5]),
    )
    np.testing.assert_almost_equal(
        mus_states[:, -1], np.array([0.54388439]),
    )
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array([-54.04429218]))
    np.testing.assert_almost_equal(tau[:, -1], np.array([-26.70770378]))
    np.testing.assert_almost_equal(
        mus_controls[:, 0], np.array([0.47810392]),
    )
    np.testing.assert_almost_equal(
        mus_controls[:, -1], np.array([0.42519766]),
    )

    # save and load
    Utils.save_and_load(sol, ocp, False)
