"""
This example uses the data from the balanced pendulum example to generate the data to track.
When it optimizes the program, contrary to the vanilla pendulum, it tracks the values instead of 'knowing' that
it is supposed to balance the pendulum. It is designed to show how to track marker and kinematic data.

Note that the final node is not tracked.
"""
import os
import pytest

import numpy as np

from casadi import MX, SX, vertcat, sin
from bioptim import (
    BiorbdModel,
    BoundsList,
    ConfigureProblem,
    ControlType,
    DynamicsEvaluation,
    DynamicsFunctions,
    DynamicsList,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    NonLinearProgram,
)


def time_dynamic(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    stochastic_variables: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, s)

    Parameters
    ----------
    time: MX | SX
        The time of the system
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    stochastic_variables: MX | SX
        The stochastic variables of the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls) * (sin(time) * time.ones(nlp.model.nb_tau) * 10)

    # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = nlp.model.forward_dynamics(q, qdot, tau)

    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, time_dynamic)


def prepare_ocp(
    biorbd_model_path: str,
    n_phase: int,
    ode_solver: OdeSolverBase,
    control_type: ControlType,
    minimize_time: bool,
    use_sx: bool,
    assume_phase_dynamics: bool = False,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    n_phase: int
        The number of phases
    ode_solver: OdeSolverBase
        The ode solver to use
    control_type: ControlType
        The type of the controls
    minimize_time: bool,
        Add a minimized time objective
    use_sx: bool
        If the ocp should be built with SX. Please note that ACADOS requires SX
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = (
        [BiorbdModel(biorbd_model_path)]
        if n_phase == 1
        else [BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path)]
    )
    final_time = [1] * n_phase
    n_shooting = [50 if isinstance(ode_solver, OdeSolver.IRK) else 30] * n_phase

    # Add objective functions
    objective_functions = ObjectiveList()
    for i in range(len(bio_model)):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=i)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, weight=0.01, phase=i)
        if minimize_time:
            objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, phase=i)

    # Dynamics
    dynamics = DynamicsList()
    expand = not isinstance(ode_solver, OdeSolver.IRK)
    for i in range(len(bio_model)):
        dynamics.add(custom_configure, dynamic_function=time_dynamic, phase=i, expand=expand)

    # Define states path constraint
    x_bounds = BoundsList()
    if n_phase == 1:
        x_bounds["q"] = bio_model[0].bounds_from_ranges("q")
        x_bounds["q"][:, [0, -1]] = 0
        x_bounds["q"][-1, -1] = 3.14
        x_bounds["qdot"] = bio_model[0].bounds_from_ranges("qdot")
        x_bounds["qdot"][:, [0, -1]] = 0
    else:
        x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
        x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
        x_bounds[0]["q"][:, [0, -1]] = 0
        x_bounds[0]["q"][-1, -1] = 3.14
        x_bounds[1]["q"][:, -1] = 0

        x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
        x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
        x_bounds[0]["qdot"][:, [0, -1]] = 0
        x_bounds[1]["qdot"][:, -1] = 0

    # Define control path constraint
    n_tau = bio_model[0].nb_tau
    u_bounds = BoundsList()
    u_bounds_tau = [[-100] * n_tau, [100] * n_tau]  # Limit the strength of the pendulum to (-100 to 100)...
    u_bounds_tau[0][1] = 0  # ...but remove the capability to actively rotate
    u_bounds_tau[1][1] = 0  # ...but remove the capability to actively rotate
    for i in range(len(bio_model)):
        u_bounds.add("tau", min_bound=u_bounds_tau[0], max_bound=u_bounds_tau[1], phase=i)

    x_init = InitialGuessList()
    u_init = InitialGuessList()

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        control_type=control_type,
        use_sx=use_sx,
        assume_phase_dynamics=assume_phase_dynamics,
    )


@pytest.mark.parametrize("n_phase", [1, 2])
@pytest.mark.parametrize(
    "integrator",
    [
        OdeSolver.IRK,
        OdeSolver.RK4,
        OdeSolver.COLLOCATION,
        OdeSolver.TRAPEZOIDAL,
    ],
)
@pytest.mark.parametrize("control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS])
@pytest.mark.parametrize("minimize_time", [True, False])
@pytest.mark.parametrize("use_sx", [False, True])
def test_time_dependent_problem(n_phase, integrator, control_type, minimize_time, use_sx):
    """
    Firstly, it solves the getting_started/pendulum.py example.
    It then creates and solves this ocp and show the results.
    """
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    if integrator == OdeSolver.IRK and use_sx:
        with pytest.raises(
            NotImplementedError,
            match="use_sx=True and OdeSolver.IRK are not yet compatible",
        ):
            ocp = prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                n_phase=n_phase,
                ode_solver=integrator(),
                control_type=control_type,
                minimize_time=minimize_time,
                use_sx=use_sx,
            )

    elif integrator == OdeSolver.TRAPEZOIDAL and control_type == ControlType.CONSTANT:
        with pytest.raises(
            RuntimeError,
            match="TRAPEZOIDAL cannot be used with piece-wise constant controls, please use ControlType.CONSTANT_WITH_LAST_NODE or ControlType.LINEAR_CONTINUOUS instead.",
        ):
            ocp = prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                n_phase=n_phase,
                ode_solver=integrator(),
                control_type=control_type,
                minimize_time=minimize_time,
                use_sx=use_sx,
            )

    elif integrator in (OdeSolver.COLLOCATION, OdeSolver.IRK) and control_type == ControlType.LINEAR_CONTINUOUS:
        with pytest.raises(
            NotImplementedError, match="ControlType.LINEAR_CONTINUOUS ControlType not implemented yet with COLLOCATION"
        ):
            ocp = prepare_ocp(
                biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
                n_phase=n_phase,
                ode_solver=integrator(),
                control_type=control_type,
                minimize_time=minimize_time,
                use_sx=use_sx,
            )

    # --- Solve the program --- #
    else:
        ocp = prepare_ocp(
            biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
            n_phase=n_phase,
            ode_solver=integrator(),
            control_type=control_type,
            minimize_time=minimize_time,
            use_sx=use_sx,
        )
        sol = ocp.solve()

        if integrator is OdeSolver.IRK:
            if minimize_time:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.8061379831798005)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -0.03247033387511758)
                        np.testing.assert_almost_equal(sol.time[-1], 0.7525479246782548)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.8064202084554257)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -0.03158472640622656)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.7505899476414)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -1.6823897986794019)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -1.6725884947936083)
                        np.testing.assert_almost_equal(sol.time[1][-1], 1.2182068093439873)
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.7426487623278059)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -0.11218601435037819)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.7426487623279004)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -0.11218601435004287)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 1.7170907081856484)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -1.3961928072352583)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.RK4:
            if minimize_time:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 9.0456172436041)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -4.017990807857719)
                        np.testing.assert_almost_equal(sol.time[-1], 0.8590128886362812)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.6035534523730542)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 11.024665900300825)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.7387280726931472)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -2.004325773700408)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -1.6519393514650482)
                        np.testing.assert_almost_equal(sol.time[1][-1], 1.3272163599484914)
                elif control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], -0.3392063188703631)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 2.9862617380186545)
                        np.testing.assert_almost_equal(sol.time[-1], 0.6615905912195619)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.4071960379641867)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 3.816939990893976)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.7323392725426761)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 7.0156443600333205)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.048756917809465394)
                        np.testing.assert_almost_equal(sol.time[1][-1], 1.4205875278838858)
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.4054079577887415)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 2.2210055455266042)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.4787784694832532)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -1.118653718167358)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -1.5658712082429582)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 0.11748321399843228)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)
                elif control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.1584600044201273)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 2.3058806527015654)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.15846000442016803)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 2.305880652701225)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -1.7691580840192065)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 0.005506940029476814)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.COLLOCATION:
            if minimize_time:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.5969550797208588)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 8.403138437314404)
                        np.testing.assert_almost_equal(sol.time[-1], 0.7162941229409554)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.49318250188879587)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 8.964059202431075)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.759023435059598)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -1.874918945886359, decimal=6)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -2.7233090441206573)
                        np.testing.assert_almost_equal(sol.time[1][-1], 1.430716792777341)
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.20540828181779627)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.5314201112799963)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.20540828116882126)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.5314201138799158)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -1.3688983057043562)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 0.11688816680719695)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.TRAPEZOIDAL:
            if minimize_time:
                if control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        if use_sx:  # Awkward behavior of SX not giving the same result as MX
                            np.testing.assert_almost_equal(sol.controls["tau"][0][10], 1.4387867097106664e-05)
                            np.testing.assert_almost_equal(sol.controls["tau"][0][20], 4.152749205255539e-05)
                            np.testing.assert_almost_equal(sol.time[-1], 1.2808173674288864e-10)
                        else:
                            np.testing.assert_almost_equal(sol.controls["tau"][0][10], 26.71726475821808)
                            np.testing.assert_almost_equal(sol.controls["tau"][0][20], 6.889550807490503)
                            np.testing.assert_almost_equal(sol.time[-1], 1.2808198950794705e-10)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.8593782322287159)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -3.679698945048069)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.8703699395814744)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 1.641852245067451)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -1.693859152701225)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.5755591693627116)
            else:
                if control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.7953894180833663)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -2.127110340642236)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.7953894180833854)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -2.127110340642238)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -1.3807528747673066)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -2.764254286944976)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)
