"""
Test for file IO
"""
import importlib.util
from pathlib import Path
import pytest
import numpy as np

from bioptim import Data, OdeSolver
from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.IRK])
def test_muscle_driven_ocp(ode_solver):
    # Load static_arm
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "static_arm", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/static_arm.py"
    )
    static_arm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(static_arm)

    ocp = static_arm.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26.bioMod",
        final_time=2,
        number_shooting_points=10,
        weight=1,
        ode_solver=ode_solver,
    )
    sol, obj = ocp.solve(return_objectives=True)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]

    if ode_solver == OdeSolver.IRK:
        # Check return_objectives
        np.testing.assert_almost_equal(
            obj[0],
            np.array(
                [
                    [
                        5.85907876e-06,
                        4.90787480e-06,
                        4.11445097e-06,
                        3.23249314e-06,
                        2.32017681e-06,
                        1.53145423e-06,
                        9.52003920e-07,
                        5.97894049e-07,
                        4.50363195e-07,
                        1.11557294e-07,
                    ],
                    [
                        9.60459127e-03,
                        7.74907187e-03,
                        5.60335998e-03,
                        2.97901541e-03,
                        1.03718035e-03,
                        2.17889033e-04,
                        2.73537061e-05,
                        1.22144173e-05,
                        1.78454284e-05,
                        5.89515242e-06,
                    ],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.16237622e-01],
                ]
            ),
        )

        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14351611580879933)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.94511299, 3.07048865]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.41149114, -0.55863385]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00147561, 0.00520749]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00027953, 0.00069257]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.29029533e-06, 1.64976642e-01, 1.00004898e-01, 4.01974257e-06, 4.13014984e-06, 1.03945583e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.25940361e-03, 3.21754460e-05, 3.12984790e-05, 2.00725054e-03, 1.99993619e-03, 1.81725854e-03]),
        )
    else:
        # Check return_objectives
        np.testing.assert_almost_equal(
            obj[0],
            np.array(
                [
                    [
                        5.86278160e-06,
                        4.90908874e-06,
                        4.11442651e-06,
                        3.23179182e-06,
                        2.31914537e-06,
                        1.53018849e-06,
                        9.50482800e-07,
                        5.96421781e-07,
                        4.49664434e-07,
                        1.11563008e-07,
                    ],
                    [
                        9.60988809e-03,
                        7.74655413e-03,
                        5.59633777e-03,
                        2.97348197e-03,
                        1.03531097e-03,
                        2.17878176e-04,
                        2.73789409e-05,
                        1.21584248e-05,
                        1.78095282e-05,
                        5.90847984e-06,
                    ],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.16237866e-01],
                ]
            ),
        )

        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14350464848810182)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.9451058, 3.0704789]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.4115254, -0.5586797]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.0014793, 0.0052082]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-0.0002795, 0.0006926]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.2869218e-06, 1.6503522e-01, 1.0002514e-01, 4.0190181e-06, 4.1294041e-06, 1.0396051e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.2599283e-03, 3.2188697e-05, 3.1307377e-05, 2.0121186e-03, 2.0048373e-03, 1.8235679e-03]),
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.IRK])
def test_muscle_activations_with_contact_driven_ocp(ode_solver):
    # Load static_arm_with_contact
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "static_arm_with_contact", str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/static_arm_with_contact.py"
    )
    static_arm_with_contact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(static_arm_with_contact)

    ocp = static_arm_with_contact.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_ocp/arm26_with_contact.bioMod",
        final_time=2,
        number_shooting_points=10,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    if ode_solver == OdeSolver.IRK:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14351397970185203)

        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (60, 1))
        np.testing.assert_almost_equal(g, np.zeros((60, 1)), decimal=6)

        # Check some of the results
        states, controls = Data.get_data(ocp, sol["x"])
        q, qdot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0, 0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.0081671, -0.94509584, 3.07047323]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.00093981, 0.41157421, -0.55870943]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-3.49332839e-07, 1.47494809e-03, 5.20721575e-03]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-2.72476211e-06, -2.79524486e-04, 6.92600551e-04]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.29081617e-06, 1.64961906e-01, 9.99986809e-02, 4.01995665e-06, 4.13036938e-06, 1.03940164e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.25988708e-03, 3.21882769e-05, 3.13076618e-05, 2.01160287e-03, 2.00431774e-03, 1.82289866e-03]),
        )

    else:
        # Check objective function value
        f = np.array(sol["f"])
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.1435025030068162)

        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (60, 1))
        np.testing.assert_almost_equal(g, np.zeros((60, 1)), decimal=6)

        # Check some of the results
        states, controls = Data.get_data(ocp, sol["x"])
        q, qdot, tau, mus = states["q"], states["q_dot"], controls["tau"], controls["muscles"]

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0, 0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.0081671, -0.9450881, 3.0704626]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.0009398, 0.4116121, -0.5587618]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-3.9652660e-07, 1.4785825e-03, 5.2079505e-03]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-2.7248808e-06, -2.7952503e-04, 6.9262306e-04]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.2873915e-06, 1.6502014e-01, 1.0001872e-01, 4.0192359e-06, 4.1296273e-06, 1.0395487e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.2599697e-03, 3.2187363e-05, 3.1307175e-05, 2.0116712e-03, 2.0043861e-03, 1.8230214e-03]),
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.IRK])
def test_muscle_excitation_with_contact_driven_ocp(ode_solver):
    # Load contact_forces_inequality_constraint_muscle_excitations
    PROJECT_FOLDER = Path(__file__).parent / ".."
    spec = importlib.util.spec_from_file_location(
        "contact_forces_inequality_constraint_muscle_excitations",
        str(PROJECT_FOLDER)
        + "/examples/muscle_driven_with_contact/contact_forces_inequality_constraint_muscle_excitations.py",
    )
    contact_forces_inequality_constraint_muscle_excitations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contact_forces_inequality_constraint_muscle_excitations)

    boundary = 50
    ocp = contact_forces_inequality_constraint_muscle_excitations.prepare_ocp(
        str(PROJECT_FOLDER) + "/examples/muscle_driven_with_contact/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        number_shooting_points=10,
        min_bound=boundary,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.14525619)

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, mus_states, tau, mus_controls = (
        states["q"],
        states["q_dot"],
        states["muscles"],
        controls["tau"],
        controls["muscles"],
    )

    if ode_solver == OdeSolver.IRK:
        # Check constraints
        g = np.array(sol["g"])
        np.testing.assert_equal(g.shape, (110, 1))
        np.testing.assert_almost_equal(g[:90], np.zeros((90, 1)), decimal=6)
        np.testing.assert_array_less(-g[90:], -boundary)
        expected_pos_g = np.array(
            [
                [51.5414325],
                [52.77742181],
                [57.57780262],
                [62.62940016],
                [65.1683722],
                [66.33551167],
                [65.82614885],
                [63.06016376],
                [57.23683342],
                [50.47124118],
                [156.35594176],
                [136.1362431],
                [89.86994764],
                [63.41325331],
                [57.493027],
                [55.09716611],
                [53.77813649],
                [52.90987628],
                [52.19502561],
                [50.56093511],
            ]
        )
        np.testing.assert_almost_equal(g[90:], expected_pos_g)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
        np.testing.assert_almost_equal(
            q[:, -1], np.array([-3.40708085e-01, 1.34155553e-01, -2.22589697e-04, 2.22589697e-04])
        )
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array([-2.01858700e00, 4.49316671e-04, 4.03717411e00, -4.03717411e00])
        )
        # initial and final muscle state
        np.testing.assert_almost_equal(mus_states[:, 0], np.array([0.5]))
        np.testing.assert_almost_equal(mus_states[:, -1], np.array([0.52946019]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-54.08860398]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-26.70209712]))
        np.testing.assert_almost_equal(mus_controls[:, 0], np.array([0.48071638]))
        np.testing.assert_almost_equal(mus_controls[:, -1], np.array([0.40159522]))
    else:
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
        np.testing.assert_almost_equal(mus_states[:, 0], np.array([0.5]))
        np.testing.assert_almost_equal(mus_states[:, -1], np.array([0.54388439]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-54.04429218]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-26.70770378]))
        np.testing.assert_almost_equal(mus_controls[:, 0], np.array([0.47810392]))
        np.testing.assert_almost_equal(mus_controls[:, -1], np.array([0.42519766]))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol, ocp, decimal_value=5)
