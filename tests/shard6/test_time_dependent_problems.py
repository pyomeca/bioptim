"""
This example uses the data from the balanced pendulum example to generate the data to track.
When it optimizes the program, contrary to the vanilla pendulum, it tracks the values instead of 'knowing' that
it is supposed to balance the pendulum. It is designed to show how to track marker and kinematic data.

Note that the final node is not tracked.
"""
import os
import pytest
import platform

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
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[172.8834048]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.2827123610541368)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 1.0740355622653601)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -0.19302027184867174)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0165215559480536)
                    else:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[253.86966024]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.2878593914994324)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 1.0696023177041738)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -0.1984735983859142)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0099613938981316)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], -0.07332566736521029)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -0.06242161890699912)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 0.2929151968697335)
                        np.testing.assert_almost_equal(sol.time[1][-1], 3.004624977588497)
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[178.92793056]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.2947267283269733)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 1.0627378962634675)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -0.20852896653142494)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[219.56328194]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.29472672830496854)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 1.0627378962736056)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -0.20852896657892694)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], -0.8806191138987077)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.6123414728154551)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 0.31457777214014526)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.RK4:
            if minimize_time:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[394.66600426]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.5511483100068987)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.3104608832105364)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.6964645358818702)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0410501449351302)
                    else:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[249.02062056]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.515465729973584)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.36085663521048567)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.5491001342038282)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0181503582573952)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], -0.7360658513679542)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.31965099349592874)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.03411819095493515)
                        np.testing.assert_almost_equal(sol.time[1][-1], 3.0173286199672997)
                elif control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[192.8215926]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.49725527306572986)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.6933785288605654)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.09656390900731922)
                        np.testing.assert_almost_equal(sol.time[-1], 1.016504038138323)
                    else:
                        if platform.system() == "Windows":
                            return
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[1638.27930348]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.5107221153599056)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.7166824738415234)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.07145353235356225)
                        np.testing.assert_almost_equal(sol.time[0][-1], 0.9288827513626345)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], 1.1315705465545554)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -0.5182190005284218)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], 1.0804115102547298)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.841120112778656)
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[439.46711618]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.5399146804724992)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.3181014748510472)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.8458229444008145)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[263.27075989]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.5163043019429964)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.36302465583174054)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.5619591749485324)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], -0.9922794818013333)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.3933344616906924)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.18690024664611413)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)
                elif control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[192.03219177]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.5161382753215992)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.6099846721957277)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.26790239968843726)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[235.04680652]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.516138275321599)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.6099846721957268)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.26790239968843604)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], -1.0000000090290213)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.40211868315342186)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.28370338722588506)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.COLLOCATION:
            if minimize_time:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[186.61842748]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.019120660739247904)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.2922771720859445)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.2840926049009388)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0196384456953451)
                    else:
                        # if platform.system() == "Linux":
                        return
                        # np.testing.assert_almost_equal(
                        #     np.array(sol.cost), np.array([[609.7634933]]) if use_sx else np.array([[615.51412616]])
                        # )
                        # np.testing.assert_almost_equal(
                        #     sol.states[0]["q"][0][10], 0.08784141530212256 if use_sx else 0.09235037702709002
                        # )
                        # np.testing.assert_almost_equal(
                        #     sol.controls[0]["tau"][0][10], -0.5052767479051853 if use_sx else -0.7893836755444296
                        # )
                        # np.testing.assert_almost_equal(
                        #     sol.controls[0]["tau"][0][20], 0.2809180374342609 if use_sx else 0.2815936910217755
                        # )
                        # np.testing.assert_almost_equal(
                        #     sol.time[0][-1], 0.9769595610422147 if use_sx else 0.9773408698970485
                        # )
                        # np.testing.assert_almost_equal(
                        #     sol.states[1]["q"][0][10], -0.18360049737736453 if use_sx else -0.1837617745360787
                        # )
                        # np.testing.assert_almost_equal(
                        #     sol.controls[1]["tau"][0][10], -0.18375233077943384 if use_sx else -0.1840170883295579
                        # )
                        # np.testing.assert_almost_equal(
                        #     sol.controls[1]["tau"][0][20], 0.24334657164303342 if use_sx else 0.24354313552685838
                        # )
                        # np.testing.assert_almost_equal(
                        #     sol.time[1][-1], 2.951825464285147 if use_sx else 2.9521189420277105
                        # )
            else:
                if control_type is ControlType.CONSTANT:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[253.91175755]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.01837629161465852)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.2621255903085372)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], 0.3940840842831783)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        # if platform.system() == "Linux":
                        #     return
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[359.26094374]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.02243344477603267)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.1611312749047648)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.40945960507427026)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], -0.503878161059821)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.32689719054740884)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.2131135407233949)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)

        elif integrator is OdeSolver.TRAPEZOIDAL:
            if minimize_time:
                if control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[133.93264169]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.5690273997201437)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.7792794037533405)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -1.4769186440398776)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0217388521900082)
                    else:
                        if platform.system() == "Linux":
                            return
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[245.65193273]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.5632172417569231)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.7384287794069126)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], 0.45380684243834224)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0568042922465213)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], -0.691691238183481)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], 0.15904268963103146)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.0739183851785058)
                        np.testing.assert_almost_equal(sol.time[1][-1], 3.057239461784291)
            else:
                if control_type is ControlType.LINEAR_CONTINUOUS:
                    if n_phase == 1:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[147.72809392]]))
                        np.testing.assert_almost_equal(sol.states["q"][0][10], 0.5677646798955323)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][10], 0.8003065430295088)
                        np.testing.assert_almost_equal(sol.controls["tau"][0][20], -1.5700440196300944)
                        np.testing.assert_almost_equal(sol.time[-1], 1.0)
                    else:
                        np.testing.assert_almost_equal(np.array(sol.cost), np.array([[174.85917125]]))
                        np.testing.assert_almost_equal(sol.states[0]["q"][0][10], 0.5677646798955326)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][10], 0.8003065430295143)
                        np.testing.assert_almost_equal(sol.controls[0]["tau"][0][20], -1.570044019630101)
                        np.testing.assert_almost_equal(sol.time[0][-1], 1.0)
                        np.testing.assert_almost_equal(sol.states[1]["q"][0][10], -0.9987158065510906)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][10], -0.007854806227095378)
                        np.testing.assert_almost_equal(sol.controls[1]["tau"][0][20], -0.11333730470915207)
                        np.testing.assert_almost_equal(sol.time[1][-1], 2.0)
