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
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.42994886542407745)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.0730095610094896)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][30], 11.322644274010235)
                        np.testing.assert_almost_equal(sol.time[-1], 0.5265564592305236)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.4453822590767244)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.08127975416241208)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][30], 10.783410413546257)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.5328521750270141)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 19.71398866443379)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -8.494581035760161)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][30], -1.117190309770819)
                        np.testing.assert_almost_equal(sol.time[1][-1], 0.7684344048027503)
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 1.0627380359954306)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -0.2085296357750012)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][30], -0.3255947333266582)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 1.06273809537391)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -0.20852991499656517)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][30], -0.3255925208864834)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.6751956398396105)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 0.2693023405309375)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][30], -0.557922378301458)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.RK4:
            if minimize_time:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 10.535298506860732)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -2.5276115661911267)
                        np.testing.assert_almost_equal(sol.time[-1], 0.4853354514411001)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.7300793797747384)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 12.328835008857212)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.5988532853060373)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 1.6473004406083955)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 5.787359865809643)
                        np.testing.assert_almost_equal(sol.time[1][-1], 0.8310646891355389)
                elif control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        if use_sx:  # Awkward behavior of SX not giving the same result as MX
                            np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.5288791823419771)
                            np.testing.assert_almost_equal(sol.controls["tau"][0][20], 18.347404279470123)
                            np.testing.assert_almost_equal(sol.time[-1], 0.5122303221905166)
                        else:
                            np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.5288791756539257)
                            np.testing.assert_almost_equal(sol.controls["tau"][0][20], 18.34740379501906)
                            np.testing.assert_almost_equal(sol.time[-1], 0.5122303215108029)
                    else:
                        if use_sx:  # Awkward behavior of SX not giving the same result as MX
                            np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.5158427378040261)
                            np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 16.891397209314942)
                            np.testing.assert_almost_equal(sol.time[0][-1], 0.5218641453532176)
                            np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 5.07655599497984)
                            np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 3.8243809447103487)
                            np.testing.assert_almost_equal(sol.time[1][-1], 0.7603930004675463)
                        else:
                            np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.5499325747810617)
                            np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 15.600053989731066)
                            np.testing.assert_almost_equal(sol.time[0][-1], 0.5326454304377053)
                            np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -13.873289688946695)
                            np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -14.970529786924043)
                            np.testing.assert_almost_equal(sol.time[1][-1], 0.8420270644169306)
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.5139094577012504)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.30702401389985223)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.36302489459617565)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.5619574586283279)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.39333341058217297)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.18690010862175288)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)
                elif control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.6323325534859385)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.20780290226142156)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.6931500418424145)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.10071649020791093)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.39912562629699866)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.2628253034251759)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.COLLOCATION:
            if minimize_time:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.277936975867381)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 5.129630718448575)
                        np.testing.assert_almost_equal(sol.time[-1], 0.5046784032465458)
                    else:
                        if use_sx:  # Awkward behavior of SX not giving the same result as MX
                            np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.38243230552609514)
                            np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -1.5926093734442652)
                            np.testing.assert_almost_equal(sol.time[0][-1], 0.5830594892396639)
                            np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -5.307691181037627)
                            np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.0010755539775450678)
                            np.testing.assert_almost_equal(sol.time[1][-1], 1.0283004173524801)
                        else:
                            np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.31718053752742986)
                            np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.6974492990131168)
                            np.testing.assert_almost_equal(sol.time[0][-1], 0.5999658199171516)
                            np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -8.056163776372038)
                            np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.0024829060821942037)
                            np.testing.assert_almost_equal(sol.time[1][-1], 1.046108398309169)
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.3125420865819661)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.339682333868206)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.4338677935898688)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.11953219689658545)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.32205855052780985)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.2268512813479941)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.TRAPEZOIDAL:
            if minimize_time:
                if control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        if use_sx:  # Awkward behavior of SX not giving the same result as MX
                            np.testing.assert_almost_equal(sol.controls["tau"][0][10], -1.4620815821959086)
                            np.testing.assert_almost_equal(sol.controls["tau"][0][20], -6.207177719873385)
                            np.testing.assert_almost_equal(sol.time[-1], 0.6188543569139974)
                        else:
                            np.testing.assert_almost_equal(sol.controls["tau"][0][10], 19.79146328766602)
                            np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.2268976397399745)
                            np.testing.assert_almost_equal(sol.time[-1], 0.5656036202933684)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.6680888144935364)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 1.2309073456939035)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.7768173672113625)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 4.3591737603853336)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.18140833455869917)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.2257495170567285)
            else:
                if control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.8003065798999568)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -1.5700439680018332)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.8003065798989261)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -1.5700439680074494)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -0.007831939511758989)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.11333547011240175)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)
