import os

import pytest
import numpy as np
from bioptim import Shooting, OdeSolver, SolutionIntegrator, Solver, ControlType


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_time(ode_solver, assume_phase_dynamics):
    # Load pendulum
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=2,
        n_shooting=10,
        ode_solver=ode_solver(),
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=False,
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


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
def test_time_multiphase(ode_solver, assume_phase_dynamics):
    # Load slider
    from bioptim.examples.torque_driven_ocp import slider as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/slider.bioMod",
        ode_solver=ode_solver(),
        phase_time=(0.2, 0.3, 0.5),
        n_shooting=(3, 4, 5),
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=False,
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


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("merge_phase", [True, False])
@pytest.mark.parametrize("keep_intermediate_points", [True, False])
@pytest.mark.parametrize("shooting_type", [Shooting.SINGLE, Shooting.SINGLE_DISCONTINUOUS_PHASE, Shooting.MULTIPLE])
def test_generate_time(ode_solver, merge_phase, keep_intermediate_points, shooting_type, assume_phase_dynamics):
    # Load slider
    from bioptim.examples.torque_driven_ocp import slider as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/slider.bioMod",
        ode_solver=ode_solver(),
        phase_time=(0.2, 0.3, 0.5),
        n_shooting=(3, 4, 5),
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=False,
    )

    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(0)
    solver.set_print_level(0)

    sol = ocp.solve(solver=solver)

    time = sol._generate_time(
        shooting_type=shooting_type,
        keep_intermediate_points=keep_intermediate_points,
        merge_phases=merge_phase,
    )

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
                np.testing.assert_equal(time[0].shape, (4,))
                np.testing.assert_equal(time[1].shape, (5,))
                np.testing.assert_equal(time[2].shape, (6,))
                if ode_solver == OdeSolver.RK4:
                    np.testing.assert_almost_equal(time[0][0][4], 0.05333333333333334)
                    np.testing.assert_almost_equal(time[1][0][4], 0.26)
                    np.testing.assert_almost_equal(time[2][0][4], 0.5800000000000001)
                else:
                    np.testing.assert_almost_equal(time[0][0][4], 0.06203787705313508)
                    np.testing.assert_almost_equal(time[1][0][4], 0.269792611684777)
                    np.testing.assert_almost_equal(time[2][0][4], 0.5930568155797027)
            else:
                np.testing.assert_equal(time[0].shape, (4,))
                np.testing.assert_equal(time[1].shape, (5,))
                np.testing.assert_equal(time[2].shape, (6,))
                np.testing.assert_almost_equal(time[0][0][1], 0.06666666666666667)
                np.testing.assert_almost_equal(time[1][0][1], 0.275)
                np.testing.assert_almost_equal(time[2][0][1], 0.6)


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("merge_phase", [True, False])
@pytest.mark.parametrize("keep_intermediate_points", [True, False])
@pytest.mark.parametrize("shooting_type", [Shooting.SINGLE, Shooting.SINGLE_DISCONTINUOUS_PHASE, Shooting.MULTIPLE])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
def test_generate_integrate(
    ode_solver,
    merge_phase,
    keep_intermediate_points,
    shooting_type,
    integrator,
    assume_phase_dynamics,
):
    # Load slider
    from bioptim.examples.torque_driven_ocp import slider as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/slider.bioMod",
        ode_solver=ode_solver(),
        phase_time=(0.2, 0.3, 0.5),
        n_shooting=(3, 4, 5),
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=False,
    )

    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(100)
    solver.set_print_level(0)
    sol = ocp.solve(solver=solver)

    if shooting_type == Shooting.MULTIPLE and keep_intermediate_points is False:
        with pytest.raises(
            ValueError,
            match="shooting_type=Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously."
            "When using multiple shooting, the intermediate points should be kept",
        ):
            sol.integrate(
                shooting_type=shooting_type,
                merge_phases=merge_phase,
                keep_intermediate_points=keep_intermediate_points,
                integrator=integrator,
            )
    elif ode_solver == OdeSolver.COLLOCATION and integrator == SolutionIntegrator.OCP:
        with pytest.raises(
            ValueError,
            match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
            "we cannot use the SolutionIntegrator.OCP.\n"
            "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
            " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
        ):
            sol.integrate(
                shooting_type=shooting_type,
                merge_phases=merge_phase,
                keep_intermediate_points=keep_intermediate_points,
                integrator=integrator,
            )

    else:
        integrated_sol = sol.integrate(
            shooting_type=shooting_type,
            merge_phases=merge_phase,
            keep_intermediate_points=keep_intermediate_points,
            integrator=integrator,
        )

        merged_sol = sol.merge_phases()

        np.testing.assert_equal(merged_sol.time.shape, merged_sol.states["q"][0, :].shape)
        if merge_phase:
            np.testing.assert_almost_equal(integrated_sol.time.shape, integrated_sol.states["q"][0, :].shape)
        else:
            for t, state in zip(integrated_sol.time, integrated_sol.states):
                np.testing.assert_almost_equal(t.shape, state["q"][0, :].shape)

        if shooting_type == Shooting.SINGLE and merge_phase is False:
            np.testing.assert_almost_equal(integrated_sol.states[0]["q"][0, -1], integrated_sol.states[1]["q"][0, 0])
            np.testing.assert_almost_equal(integrated_sol.states[1]["q"][0, -1], integrated_sol.states[2]["q"][0, 0])

        import matplotlib.pyplot as plt

        plt.figure()

        plt.plot(merged_sol.time, merged_sol.states["q"][0, :], label="merged", marker=".")
        if merge_phase:
            plt.plot(
                integrated_sol.time,
                integrated_sol.states["q"][0, :],
                label="integrated by bioptim",
                marker=".",
                alpha=0.5,
                markersize=5,
            )
        else:
            for t, state in zip(integrated_sol.time, integrated_sol.states):
                plt.plot(t[:, np.newaxis], state["q"].T, label="integrated by bioptim", marker=".")

        plt.legend()
        plt.vlines(0.2, -1, 1, color="black", linestyle="--")
        plt.vlines(0.5, -1, 1, color="black", linestyle="--")

        plt.title(
            f"keep_intermediate={keep_intermediate_points},\n"
            f" merged={merge_phase},\n"
            f" ode_solver={ode_solver},\n"
            f" integrator={integrator},\n"
        )
        plt.rcParams["axes.titley"] = 1.0  # y is in axes-relative coordinates.
        plt.rcParams["axes.titlepad"] = -20
        # plt.show()


@pytest.mark.parametrize("assume_phase_dynamics", [True, False])
@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.COLLOCATION])
@pytest.mark.parametrize("merge_phase", [True, False])
@pytest.mark.parametrize("keep_intermediate_points", [True, False])
@pytest.mark.parametrize("shooting_type", [Shooting.SINGLE, Shooting.SINGLE_DISCONTINUOUS_PHASE, Shooting.MULTIPLE])
@pytest.mark.parametrize("integrator", [SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45])
def test_generate_integrate_linear_continuous(
    ode_solver,
    merge_phase,
    keep_intermediate_points,
    shooting_type,
    integrator,
    assume_phase_dynamics,
):
    # Load slider
    from bioptim.examples.torque_driven_ocp import slider as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    if ode_solver == OdeSolver.COLLOCATION:
        with pytest.raises(
            NotImplementedError, match="ControlType.LINEAR_CONTINUOUS ControlType not implemented yet with COLLOCATION"
        ):
            ocp_module.prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/slider.bioMod",
                ode_solver=ode_solver(),
                phase_time=(0.2, 0.3, 0.5),
                n_shooting=(3, 4, 5),
                control_type=ControlType.LINEAR_CONTINUOUS,
                assume_phase_dynamics=assume_phase_dynamics,
                expand_dynamics=False,
            )
        return
    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/slider.bioMod",
        ode_solver=ode_solver(),
        phase_time=(0.2, 0.3, 0.5),
        n_shooting=(3, 4, 5),
        control_type=ControlType.LINEAR_CONTINUOUS,
        assume_phase_dynamics=assume_phase_dynamics,
        expand_dynamics=False,
    )

    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_maximum_iterations(100)
    solver.set_print_level(0)
    sol = ocp.solve(solver=solver)

    if shooting_type == Shooting.MULTIPLE and keep_intermediate_points is False:
        with pytest.raises(
            ValueError,
            match="shooting_type=Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously."
            "When using multiple shooting, the intermediate points should be kept",
        ):
            sol.integrate(
                shooting_type=shooting_type,
                merge_phases=merge_phase,
                keep_intermediate_points=keep_intermediate_points,
                integrator=integrator,
            )
    elif ode_solver == OdeSolver.COLLOCATION and integrator == SolutionIntegrator.OCP:
        with pytest.raises(
            ValueError,
            match="When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
            "and one uses the SolutionIntegrator.OCP, we must use the shooting_type=Shooting.MULTIPLE.\n"
            "Or, we can use one of the SolutionIntegrator provided by scipy to any Shooting such as"
            " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
        ):
            sol.integrate(
                shooting_type=shooting_type,
                merge_phases=merge_phase,
                keep_intermediate_points=keep_intermediate_points,
                integrator=integrator,
            )

    else:
        integrated_sol = sol.integrate(
            shooting_type=shooting_type,
            merge_phases=merge_phase,
            keep_intermediate_points=keep_intermediate_points,
            integrator=integrator,
        )

        merged_sol = sol.merge_phases()

        np.testing.assert_equal(merged_sol.time.shape, merged_sol.states["q"][0, :].shape)
        if merge_phase:
            np.testing.assert_almost_equal(integrated_sol.time.shape, integrated_sol.states["q"][0, :].shape)
        else:
            for t, state in zip(integrated_sol.time, integrated_sol.states):
                np.testing.assert_almost_equal(t.shape, state["q"][0, :].shape)

        if shooting_type == Shooting.SINGLE and merge_phase is False:
            np.testing.assert_almost_equal(integrated_sol.states[0]["q"][0, -1], integrated_sol.states[1]["q"][0, 0])
            np.testing.assert_almost_equal(integrated_sol.states[1]["q"][0, -1], integrated_sol.states[2]["q"][0, 0])

        import matplotlib.pyplot as plt

        plt.figure()

        plt.plot(merged_sol.time, merged_sol.states["q"][0, :], label="merged", marker=".")
        if merge_phase:
            plt.plot(
                integrated_sol.time,
                integrated_sol.states["q"][0, :],
                label="integrated by bioptim",
                marker=".",
                alpha=0.5,
                markersize=5,
            )
        else:
            for t, state in zip(integrated_sol.time, integrated_sol.states):
                plt.plot(t[:, np.newaxis], state["q"].T, label="integrated by bioptim", marker=".")

        # plt.legend()
        # plt.vlines(0.2, -1, 1, color="black", linestyle="--")
        # plt.vlines(0.5, -1, 1, color="black", linestyle="--")
        #
        # plt.title(f"keep_intermediate={keep_intermediate_points},\n"
        #           f" merged={merge_phase},\n"
        #           f" ode_solver={ode_solver},\n"
        #           f" integrator={integrator},\n"
        #           )
        # plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
        # plt.rcParams['axes.titlepad'] = -20
        # plt.show()
