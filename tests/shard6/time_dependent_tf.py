"""
This example uses the data from the balanced pendulum example to generate the data to track.
When it optimizes the program, contrary to the vanilla pendulum, it tracks the values instead of 'knowing' that
it is supposed to balance the pendulum. It is designed to show how to track marker and kinematic data.

Note that the final node is not tracked.
"""

from casadi import MX, SX, vertcat, sin, Function, DM, reshape
import os
import numpy as np

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
    ConstraintList,
    Node, Solver,
)


def custom_configure_as_time(
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
    ConfigureProblem.configure_new_variable(
        "t",
        "0",
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=False,
    )
    ConfigureProblem.configure_new_variable(
        "final_time",
        "0",
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=False,
    )

    ConfigureProblem.configure_dynamics_function(ocp, nlp, dynamic_as_time)


def dynamic_as_time(
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
    t = DynamicsFunctions.get(nlp.states["t"], states)
    # tf_minus_t = DynamicsFunctions.get(nlp.states["final_time"], states)
    # tau = DynamicsFunctions.get(nlp.controls["tau"], controls) * (sin(tf_minus_t) * time.ones(nlp.model.nb_tau) * 10)
    tf = nlp.tf_mx
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls) * (sin(tf - t) * time.ones(nlp.model.nb_tau) * 10)

    # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = nlp.model.forward_dynamics(q, qdot, tau)

    return DynamicsEvaluation(dxdt=vertcat(dq, ddq, 1, -1), defects=None)

def final_time_equals_tf_init(controller):
    tf = controller.states["final_time"].cx
    real_final_time = controller.tf.cx
    return tf - real_final_time


def prepare_ocp_state_as_time(
    biorbd_model_path: str,
    n_phase: int,
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
    final_time = 1
    n_shooting = 30

    # Add objective functions
    objective_functions = ObjectiveList()
    for i in range(len(bio_model)):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=i, quadratic=True)
        if minimize_time:
            target = 1 if i == 0 else 2
            objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_TIME, target=target, weight=20000, phase=i, quadratic=True
            )

    # Add constraints
    constraints = ConstraintList()
    constraints.add(final_time_equals_tf_init, node=Node.START)

    # Dynamics
    dynamics = DynamicsList()
    for i in range(len(bio_model)):
        dynamics.add(
            custom_configure_as_time,
            dynamic_function=dynamic_as_time,
            phase=i,
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
        )

    # Define states path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model[0].bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][-1, -1] = 3.14
    x_bounds["qdot"] = bio_model[0].bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0
    x_bounds.add("t", min_bound=np.array([[0, 0, 0]]), max_bound=np.array([[0, 10, 10]]))
    x_bounds.add("final_time", min_bound=np.array([[0, 0, 0]]), max_bound=np.array([[10, 10, 10]]))


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
        constraints=constraints,
        control_type=control_type,
        use_sx=use_sx,
        ode_solver=OdeSolver.RK4(n_integration_steps=1),
    )



def integrate_RK4(time_vector, dt, states, controls, dyn_fun, n_shooting=30, n_steps=5):
    n_q = 2
    tf = time_vector[-1]
    h = dt / n_steps
    x_integrated = DM.zeros((n_q*2, n_shooting + 1))
    x_integrated[:, 0] = states[:, 0]
    for i_shooting in range(n_shooting):
        x_this_time = x_integrated[:, i_shooting]
        u_this_time = controls[:, i_shooting]
        current_time = dt * i_shooting
        for i_step in range(n_steps):
            k1 = dyn_fun(current_time, tf, x_this_time, u_this_time)
            k2 = dyn_fun(current_time + h / 2, tf, x_this_time + h / 2 * k1, u_this_time)
            k3 = dyn_fun(current_time + h / 2, tf, x_this_time + h / 2 * k2, u_this_time)
            k4 = dyn_fun(current_time + h, tf, x_this_time + h * k3, u_this_time)
            x_this_time += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            current_time += h
        x_integrated[:, i_shooting + 1] = x_this_time
    return x_integrated



def integrate_RK1(time_vector, dt, states, controls, dyn_fun, n_shooting=30, n_steps=5):
    n_q = 2
    tf = time_vector[-1]
    h = dt / n_steps
    x_integrated = np.zeros((n_q*2, n_shooting + 1))
    x_integrated[:, 0] = np.array(states[:, 0]).reshape(4, )
    for i_shooting in range(n_shooting):
        x_this_time = np.zeros((n_q*2, n_steps+1))
        x_this_time[:, 0] = np.reshape(x_integrated[:, i_shooting], (4, ))
        u_this_time = controls[:, i_shooting]
        current_time = dt * i_shooting
        for i_step in range(1, n_steps+1):
            current_time += h
            x_this_time[:, i_step] = np.reshape(h * dyn_fun(current_time, tf, x_this_time[:, i_step-1], u_this_time), (4, ))
        x_integrated[:, i_shooting + 1] = x_this_time[:, -1]
    return x_integrated


# --- Solve the program --- #
ocp = prepare_ocp_state_as_time(
    biorbd_model_path="/home/mickaelbegon/Documents/Eve/BiorbdOptim/bioptim/examples/getting_started/models/pendulum.bioMod",
    n_phase=1,
    control_type=ControlType.CONSTANT,
    minimize_time=True,
    use_sx=False,
)
sol = ocp.solve()
# sol.graphs()

time_sym = MX.sym('T', 1, 1)
tf_sym = MX.sym('Tf', 1, 1)
states_sym = MX.sym('Q_Qdot', 4, 1)
controls_sym = MX.sym('Tau', 2, 1)

tau_dyn = controls_sym * (sin(tf_sym - time_sym) * MX.ones(ocp.nlp[0].model.nb_tau) * 10)
out_dyn = vertcat(states_sym[ocp.nlp[0].model.nb_q:],
                  ocp.nlp[0].model.forward_dynamics(states_sym[:ocp.nlp[0].model.nb_q],
                                                    states_sym[ocp.nlp[0].model.nb_q:],
                                                    tau_dyn))

dyn_fun = Function('dynamics',
                   [time_sym, tf_sym, states_sym, controls_sym],
                   [out_dyn],
                   )
sol_time_vector = sol.decision_time()
sol_states = vertcat(sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])['q'],
                     sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])['qdot'])
sol_controls = sol.decision_controls(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])["tau"]
x_integrated = integrate_RK4(sol_time_vector, sol_time_vector[1], sol_states, sol_controls, dyn_fun, n_shooting=30, n_steps=1)
# x_integrated = integrate_RK1(sol_time_vector, sol_time_vector[1], sol_states, sol_controls, dyn_fun, n_shooting=30, n_steps=1)

# Time behaves as expected
print(np.array(sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])['t']).T - np.array(sol_time_vector)[:, :, 0])
print(np.array(sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])['final_time']).T - (sol_time_vector[-1] - np.array(sol_time_vector)[:, :, 0]))

# RK4 n_step=1 is the same in my version and in Bioptim
print(x_integrated[:4, :] - sol_states)


node_idx = 1
step_idx = 1
t_span = DM.zeros(2)
t_span[0] = sol_time_vector[node_idx] + step_idx * sol_time_vector[1]
t_span[1] = sol_time_vector[1]
x = DM.zeros(6)
x[:4] = sol_states[:, node_idx]
x[4] = t_span[0]
x[-1] = sol_time_vector[-1] - t_span[0]
u = sol_controls[:, node_idx]

# The dynamics is the same
print(ocp.nlp[0].dynamics_func(t_span, x, u, [], [], []))
print(dyn_fun(t_span[0], sol_time_vector[-1], x[:-2], u))


# This does not give the same result
x_end, x_all = ocp.nlp[0].dynamics[0](t_span[0], x, u, [], [], [])
x_end_moi = integrate_RK4(np.array([t_span[0], t_span[0]+t_span[1]]), t_span[1], reshape(x[:-2], 4, 1), reshape(u, 2, 1), dyn_fun, n_shooting=1, n_steps=1)[:, 1]
print(x_end)
print(x_end_moi)



