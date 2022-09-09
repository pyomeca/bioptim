import warnings
import os
from sys import platform
import pytest

import numpy as np
from bioptim import Shooting, OdeSolver, SolutionIntegrator, Solver


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_time(
        ode_solver: OdeSolver
):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        ode_solver=ode_solver(),
    )
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)

    sol = ocp.solve(solver=solver)
    if ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(sol.time.shape, (11,))
        np.testing.assert_almost_equal(sol.time[0], 0)
        np.testing.assert_almost_equal(sol.time[-1], 2)
        np.testing.assert_almost_equal(sol.time[4], 0.8)
    else:
        np.testing.assert_almost_equal(sol.time.shape, (51,))
        np.testing.assert_almost_equal(sol.time[0], 0)
        np.testing.assert_almost_equal(sol.time[-1], 2)
        np.testing.assert_almost_equal(sol.time[4], 0.18611363115940527)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_time_multiphase(
        ode_solver: OdeSolver
):
    # Load slider
    from bioptim.examples.torque_driven_ocp import slider as ocp_module
    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/slider.bioMod",
        ode_solver=ode_solver(),
        phase_time=(0.2, 0.3, 0.5),
        n_shooting=(3, 4, 5),
    )

    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)

    sol = ocp.solve(solver=solver)

    np.testing.assert_almost_equal(len(sol.time), 3)
    if ode_solver == OdeSolver.RK4:
        np.testing.assert_almost_equal(len(sol.time[0]), 4)
        np.testing.assert_almost_equal(len(sol.time[1]), 5)
        np.testing.assert_almost_equal(len(sol.time[2]), 6)
        np.testing.assert_almost_equal(sol.time[0][0], 0)
        np.testing.assert_almost_equal(sol.time[0][-1], 0.2)
        np.testing.assert_almost_equal(sol.time[1][0], 0.2)
        np.testing.assert_almost_equal(sol.time[1][-1], 0.5)
        np.testing.assert_almost_equal(sol.time[2][0], 0.5)
        np.testing.assert_almost_equal(sol.time[2][-1], 1)
        np.testing.assert_almost_equal(sol.time[2][3], 0.8)

    else:
        np.testing.assert_almost_equal(len(sol.time[0]), 16)
        np.testing.assert_almost_equal(len(sol.time[1]), 21)
        np.testing.assert_almost_equal(len(sol.time[2]), 26)
        np.testing.assert_almost_equal(sol.time[0][0], 0)
        np.testing.assert_almost_equal(sol.time[0][-1], 0.2)
        np.testing.assert_almost_equal(sol.time[1][0], 0.2)
        np.testing.assert_almost_equal(sol.time[1][-1], 0.5)
        np.testing.assert_almost_equal(sol.time[2][0], 0.5)
        np.testing.assert_almost_equal(sol.time[2][-1], 1)
        np.testing.assert_almost_equal(sol.time[2][3], 0.5669990521792428)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("merge_phase", [True, False])
@pytest.mark.parametrize("keep_intermediate_points", [True, False])
@pytest.mark.parametrize("shooting_type", [
    Shooting.SINGLE,
    Shooting.SINGLE_DISCONTINUOUS_PHASE,
    Shooting.MULTIPLE
])
def test_generate_time(
        ode_solver: OdeSolver,
        merge_phase: bool,
        keep_intermediate_points: bool,
        shooting_type: Shooting
):
    # Load slider
    from bioptim.examples.torque_driven_ocp import slider as ocp_module
    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/slider.bioMod",
        ode_solver=ode_solver(),
        phase_time=(0.2, 0.3, 0.5),
        n_shooting=(3, 4, 5),
    )

    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)

    sol = ocp.solve(solver=solver)

    time = sol._generate_ocp_time(
        shooting_type=shooting_type,
        keep_intermediate_points=keep_intermediate_points,
        merge_phases=merge_phase,
    )

    print(time)
    if shooting_type == Shooting.SINGLE:
        if merge_phase:

            np.testing.assert_almost_equal(time[0], 0)
            np.testing.assert_almost_equal(time[-1], 1)
            if keep_intermediate_points:
                np.testing.assert_equal(time.shape, (61,))
                if ode_solver == OdeSolver.RK4:
                    np.testing.assert_almost_equal(time[4], 0.05333333333333334)
                else:
                    np.testing.assert_almost_equal(time[4], 0.06203787705313508)
            else:
                np.testing.assert_equal(time.shape, (13,))
                np.testing.assert_almost_equal(time[2], 0.13333333333333333)
                np.testing.assert_almost_equal(time[2], 0.13333333333333333)
        else:
            np.testing.assert_equal(len(time), 3)
            np.testing.assert_almost_equal(time[0][0], 0)
            np.testing.assert_almost_equal(time[-1][-1], 1)
            if keep_intermediate_points:
                if ode_solver == OdeSolver.RK4:
                    np.testing.assert_almost_equal(time[0][4], 0.05333333333333334)
                else:
                    np.testing.assert_almost_equal(time[0][4], 0.06203787705313508)

    elif shooting_type == Shooting.SINGLE_DISCONTINUOUS_PHASE:

        if merge_phase:
            np.testing.assert_almost_equal(time[0], 0)
            np.testing.assert_almost_equal(time[-1], 1)
            if keep_intermediate_points:
                np.testing.assert_equal(time.shape, (63,))
                if ode_solver == OdeSolver.RK4:
                    np.testing.assert_almost_equal(time[4], 0.05333333333333334)
                    np.testing.assert_almost_equal(time[30], 0.41000000000000003)
                else:
                    np.testing.assert_almost_equal(time[4], 0.06203787705313508)
                    np.testing.assert_almost_equal(time[30], 0.419792611684777)
            else:
                np.testing.assert_equal(time.shape, (15,))
                np.testing.assert_almost_equal(time[2], 0.13333333333333333)
                np.testing.assert_almost_equal(time[10], 0.6)

    elif shooting_type == Shooting.MULTIPLE:
        if merge_phase:
            np.testing.assert_almost_equal(time[0], 0)
            np.testing.assert_almost_equal(time[-1], 1)
            if keep_intermediate_points:
                np.testing.assert_equal(time.shape, (63,))
                if ode_solver == OdeSolver.RK4:
                    np.testing.assert_almost_equal(time[4], 0.05333333333333334)
                    np.testing.assert_almost_equal(time[30], 0.41000000000000003)
                    np.testing.assert_almost_equal(time[56], 0.8800000000000001)
                else:
                    np.testing.assert_almost_equal(time[4], 0.06203787705313508)
                    np.testing.assert_almost_equal(time[30], 0.419792611684777)
                    np.testing.assert_almost_equal(time[56], 0.8930568155797027)
            else:
                np.testing.assert_equal(time.shape, (15,))
                np.testing.assert_almost_equal(time[2], 0.13333333333333333)
                np.testing.assert_almost_equal(time[10], 0.6)

        else:
            np.testing.assert_almost_equal(time[0][0][0], 0)
            np.testing.assert_almost_equal(time[-1][-1][-1], 1)
            if keep_intermediate_points:
                np.testing.assert_equal(time[0].shape, (3, 6))
                np.testing.assert_equal(time[1].shape, (4, 6))
                np.testing.assert_equal(time[2].shape, (5, 6))
                if ode_solver == OdeSolver.RK4:
                    np.testing.assert_almost_equal(time[0][0][4], 0.05333333333333334)
                    np.testing.assert_almost_equal(time[1][0][4], 0.26)
                    np.testing.assert_almost_equal(time[2][0][4], 0.5800000000000001)
                else:
                    np.testing.assert_almost_equal(time[0][0][4], 0.06203787705313508)
                    np.testing.assert_almost_equal(time[1][0][4], 0.269792611684777)
                    np.testing.assert_almost_equal(time[2][0][4], 0.5930568155797027)
            else:
                np.testing.assert_equal(time[0].shape, (3, 2))
                np.testing.assert_equal(time[1].shape, (4, 2))
                np.testing.assert_equal(time[2].shape, (5, 2))
                np.testing.assert_almost_equal(time[0][0][1], 0.06666666666666667)
                np.testing.assert_almost_equal(time[1][0][1], 0.275)
                np.testing.assert_almost_equal(time[2][0][1], 0.6)
