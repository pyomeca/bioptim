"""
This example uses the data from the balanced pendulum example to generate the data to track.
When it optimizes the program, contrary to the vanilla pendulum, it tracks the values instead of 'knowing' that
it is supposed to balance the pendulum. It is designed to show how to track marker and kinematic data.

Note that the final node is not tracked.
"""

import platform

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
    PhaseDynamics,
    SolutionMerge,
)
from casadi import MX, SX, vertcat, sin, Function, DM, reshape
import numpy as np
import numpy.testing as npt
import pytest

from ..utils import TestUtils


def time_dynamic(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    algebraic_states: MX | SX,
    numerical_timeseries: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, a, d)

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
    algebraic_states: MX | SX
        The Algebraic states variables of the system
    numerical_timeseries: MX | SX
        The numerical timeseries of the system
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
    ddq = nlp.model.forward_dynamics(with_contact=False)(q, qdot, tau, [], [])

    return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)


def custom_configure(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries: dict[str, np.ndarray] = None
):
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
    phase_dynamics: PhaseDynamics = PhaseDynamics.ONE_PER_NODE,
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
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node

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
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=i, quadratic=True)
        if minimize_time:
            target = 1 if i == 0 else 2
            objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_TIME, target=target, weight=20000, phase=i, quadratic=True
            )

    # Dynamics
    dynamics = DynamicsList()
    expand = not isinstance(ode_solver, OdeSolver.IRK)
    for i in range(len(bio_model)):
        dynamics.add(
            custom_configure,
            dynamic_function=time_dynamic,
            phase=i,
            ode_solver=ode_solver,
            expand_dynamics=expand,
            phase_dynamics=phase_dynamics,
        )

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
        control_type=control_type,
        use_sx=use_sx,
    )


def integrate(time_vector, states, controls, dyn_fun, n_shooting=30, n_steps=5):
    n_q = 2
    tf = time_vector[-1]
    dt = tf / n_shooting
    h = dt / n_steps
    x_integrated = DM.zeros((n_q * 2, n_shooting + 1))
    x_integrated[:, 0] = states[:, 0]
    for i_shooting in range(n_shooting):
        x_this_time = x_integrated[:, i_shooting]
        u_this_time = controls[:, i_shooting]
        current_time = dt * i_shooting
        for i_step in range(n_steps):
            k1 = dyn_fun(current_time, x_this_time, u_this_time)
            k2 = dyn_fun(current_time + h / 2, x_this_time + h / 2 * k1, u_this_time)
            k3 = dyn_fun(current_time + h / 2, x_this_time + h / 2 * k2, u_this_time)
            k4 = dyn_fun(current_time + h, x_this_time + h * k3, u_this_time)
            x_this_time += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            current_time += h
        x_integrated[:, i_shooting + 1] = x_this_time
    return x_integrated


@pytest.mark.parametrize("n_phase", [1, 2])
@pytest.mark.parametrize("integrator", [OdeSolver.IRK, OdeSolver.RK4, OdeSolver.COLLOCATION, OdeSolver.TRAPEZOIDAL])
@pytest.mark.parametrize("control_type", [ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS])
@pytest.mark.parametrize("minimize_time", [True, False])
@pytest.mark.parametrize("use_sx", [False, True])
def test_time_dependent_problem(n_phase, integrator, control_type, minimize_time, use_sx):
    """
    Firstly, it solves the getting_started/pendulum.py example.
    It then creates and solves this ocp and show the results.
    """
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = TestUtils.module_folder(ocp_module)

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
        return

    if integrator == OdeSolver.TRAPEZOIDAL and control_type == ControlType.CONSTANT:
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
        return

    # --- Solve the program --- #
    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_phase=n_phase,
        ode_solver=integrator(),
        control_type=control_type,
        minimize_time=minimize_time,
        use_sx=use_sx,
    )
    # Check the values which will be sent to the solver
    np.random.seed(42)
    match integrator:
        case OdeSolver.RK4:
            if control_type == ControlType.CONSTANT:
                v_len = 185 * n_phase
                expected = (
                    [18.5, 80006.0 if minimize_time else 6.0, 0.8715987034298607]
                    if n_phase == 1
                    else [37.0, 400012.00000000006 if minimize_time else 12.0, 6.033764148108874]
                )
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                v_len = 187 * n_phase
                expected = (
                    [18.699999999999996, 80006.0 if minimize_time else 6.0, 0.8715987034298607]
                    if n_phase == 1
                    else [37.39999999999999, 400012.00000000006 if minimize_time else 12.0, 6.033764148108878]
                )
            else:
                raise ValueError("Test not implemented")

        case OdeSolver.COLLOCATION:
            if control_type == ControlType.CONSTANT:
                v_len = 665 * n_phase
                expected = (
                    [66.5, 80006.0 if minimize_time else 6.0, 6.035552847184389]
                    if n_phase == 1
                    else [133.0, 400012.00000000006 if minimize_time else 12.0, 28.618666282170977]
                )
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                v_len = 667 * n_phase
                if minimize_time:
                    if n_phase == 1:
                        expected = [66.69999999999999, 80006.0, 6.035552847184389]
                    else:
                        expected = [133.39999999999998, 400012.00000000006, 28.61866628217097]
                else:
                    if n_phase == 1:
                        expected = [66.69999999999999, 6.000000000000002, 6.035552847184389]
                    else:
                        expected = [133.39999999999998, 12.000000000000009, 28.61866628217097]

        case OdeSolver.IRK:
            if control_type == ControlType.CONSTANT:
                v_len = 305 * n_phase
                expected = (
                    [30.5, 320010.0 if minimize_time else 10.0, 4.283653839663469]
                    if n_phase == 1
                    else [61.0, 1600019.9999999998 if minimize_time else 20.0, 8.125629434161866]
                )
            elif control_type == ControlType.LINEAR_CONTINUOUS:
                v_len = 307 * n_phase
                if minimize_time:
                    if n_phase == 1:
                        expected = [30.699999999999996, 320010.0, 4.283653839663469]
                    else:
                        expected = [61.39999999999999, 1600019.9999999998, 8.125629434161866]
                else:
                    if n_phase == 1:
                        expected = [30.699999999999996, 10.000000000000005, 4.283653839663469]
                    else:
                        expected = [61.39999999999999, 20.000000000000014, 8.12562943416187]

        case OdeSolver.TRAPEZOIDAL:
            v_len = 187 * n_phase
            expected = (
                [18.699999999999996, 80006.0 if minimize_time else 6.0, 1.5103810164979388]
                if n_phase == 1
                else [37.39999999999999, 400012.00000000006 if minimize_time else 12.0, 7.154696449039014]
            )

        case _:
            raise ValueError("Test not implemented")

    TestUtils.compare_ocp_to_solve(
        ocp,
        v=np.ones((v_len, 1)) / 10,  # Random creates nan in the g vector
        expected_v_f_g=expected,
        decimal=6,
    )
    if platform.system() == "Windows":
        return

    return
    sol = ocp.solve()

    if integrator is OdeSolver.IRK:
        if minimize_time:
            if control_type is ControlType.CONSTANT:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[219.90675564]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.18884500361053447,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        1.2938273882793678,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.03894288570447333,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.02238, decimal=5)
                else:
                    return
        else:
            if control_type is ControlType.CONSTANT:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[231.35087767]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.2089236570383936,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        1.3301078174187717,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.030406254549304543,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1)
                else:
                    return

    elif integrator is OdeSolver.RK4:
        if minimize_time:
            if control_type is ControlType.CONSTANT:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[208.39179206]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.5153208032266005,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        0.3607299730920017,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.5482178516886668,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.01985, decimal=5)

                    time_sym = ocp.nlp[0].time_cx
                    states_sym = ocp.nlp[0].states.cx
                    controls_sym = ocp.nlp[0].controls.cx

                    dyn_fun = Function(
                        "dynamics",
                        [time_sym, states_sym, controls_sym],
                        [
                            time_dynamic(
                                time_sym,
                                states_sym,
                                controls_sym,
                                [],
                                [],
                                [],
                                ocp.nlp[0],
                            ).dxdt
                        ],
                    )
                    sol_time_vector = sol.decision_time()
                    sol_states = vertcat(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"],
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["qdot"],
                    )
                    sol_controls = sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"]
                    x_integrated = integrate(
                        sol_time_vector, sol_states, sol_controls, dyn_fun, n_shooting=30, n_steps=5
                    )

                    npt.assert_almost_equal(
                        np.array(reshape(x_integrated, 4 * 31, 1)), np.array(reshape(sol_states, 4 * 31, 1))
                    )
                else:
                    return
            elif control_type is ControlType.LINEAR_CONTINUOUS:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[186.81107206]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.5135348067780378,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        1.2321603659895992,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.6102933497526636,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.01535, decimal=5)
                else:
                    return
        else:
            if control_type is ControlType.CONSTANT:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[439.46711618]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.5399146804724992,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        0.3181014748510472,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.8458229444008145,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.0)
                else:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[263.27075989]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.5163043019429967,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        0.36302465583173776,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.5619591749485379,
                    )
                    npt.assert_almost_equal(sol.decision_time()[0][-1], 1.0)
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][1][10],
                        -0.4914893459283523,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][1][10], 0.0
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][1][20], 0.0
                    )
                    npt.assert_almost_equal(sol.decision_time()[1][-1], 2.0)
            elif control_type is ControlType.LINEAR_CONTINUOUS:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[192.03219177]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.5161382753215996,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        1.2921455698655684,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.6099846721957292,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.0)
                else:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[276.52833014]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.5325968728935088,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        1.3326929308535087,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.47419291694176013,
                    )
                    npt.assert_almost_equal(sol.decision_time()[0][-1], 1.0)
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][1][10],
                        -0.5005869650796351,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][1][10], 0.0
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][1][20], 0.0
                    )
                    npt.assert_almost_equal(sol.decision_time()[1][-1], 2.0)

    elif integrator is OdeSolver.COLLOCATION:
        if minimize_time:
            if control_type is ControlType.CONSTANT:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[338.20966265]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.03150376452725097,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        0.6012990041197794,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.32918646060774515,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.0396, decimal=5)
                else:
                    return
        else:
            if control_type is ControlType.CONSTANT:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[322.05408485]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.029122530316589967,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        0.5375069618928111,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.35409975042643815,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.0)
                else:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[365.2257133]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.029312343780174756,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        0.4773666763760592,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.40384124674585303,
                    )
                    npt.assert_almost_equal(sol.decision_time()[0][-1], 1.0)
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][1][10],
                        -0.03076974549056092,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][1][10], 0.0
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][1][20], 0.0
                    )
                    npt.assert_almost_equal(sol.decision_time()[1][-1], 2.0)

    elif integrator is OdeSolver.TRAPEZOIDAL:
        if minimize_time:
            if control_type is ControlType.LINEAR_CONTINUOUS:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[123.2679704]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.5893077706761598,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        0.9882866205489946,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.7305987779476445,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.01854, decimal=5)
                else:
                    return
        else:
            if control_type is ControlType.LINEAR_CONTINUOUS:
                if n_phase == 1:
                    npt.assert_almost_equal(np.array(sol.cost), np.array([[132.03064443]]))
                    npt.assert_almost_equal(
                        sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["q"][0][10],
                        0.5845805647849496,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][10],
                        0.8966300293800638,
                    )
                    npt.assert_almost_equal(
                        sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"][0][20],
                        0.7528919448851801,
                    )
                    npt.assert_almost_equal(sol.decision_time()[-1], 1.0)
                else:
                    return
