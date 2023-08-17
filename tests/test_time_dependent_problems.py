"""
This example uses the data from the balanced pendulum example to generate the data to track.
When it optimizes the program, contrary to the vanilla pendulum, it tracks the values instead of 'knowing' that
it is supposed to balance the pendulum. It is designed to show how to track marker and kinematic data.

Note that the final node is not tracked.
"""
import os
import pytest

from casadi import MX, SX, vertcat, sin
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    DynamicsEvaluation,
    InitialGuessList,
    ControlType,
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

    bio_model = BiorbdModel(biorbd_model_path)
    bio_model = [bio_model] * n_phase
    final_time = [1] * n_phase
    n_shooting = [30] * n_phase

    # Add objective functions
    objective_functions = ObjectiveList()
    for i in range(len(bio_model)):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=i)
        if minimize_time:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=100)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    for i in range(len(bio_model)):
        dynamics.add(custom_configure, dynamic_function=time_dynamic, phase=i, expand=expand)

    # Define states path constraint
    x_bounds = BoundsList()
    x_bounds_q = bio_model[0].bounds_from_ranges("q")
    x_bounds_q[:, [0, -1]] = 0  # Start and end at 0...
    x_bounds_q[1, -1] = 3.14  # ...but end with pendulum 180 degrees rotated
    x_bounds_qdot = bio_model[0].bounds_from_ranges("qdot")
    x_bounds_qdot[:, [0, -1]] = 0  # Start and end without any velocity
    for i in range(len(bio_model)):
        x_bounds.add("q", bounds=x_bounds_q, phase=i)
        x_bounds.add("qdot", bounds=x_bounds_qdot, phase=i)

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
@pytest.mark.parametrize("integrator, control_type, minimize_time, use_sx", [
    (OdeSolver.RK4(), ControlType.CONSTANT, False, False),
    (OdeSolver.RK4(), ControlType.CONSTANT, False, True),
    (OdeSolver.RK4(), ControlType.CONSTANT, True, False),
    (OdeSolver.RK4(), ControlType.CONSTANT, True, True),
    (OdeSolver.RK4(), ControlType.LINEAR_CONTINUOUS, False, False),
    (OdeSolver.RK4(), ControlType.LINEAR_CONTINUOUS, False, True),
    (OdeSolver.RK4(), ControlType.LINEAR_CONTINUOUS, True, True),
    (OdeSolver.IRK(), ControlType.CONSTANT, False, False),
    (OdeSolver.IRK(), ControlType.LINEAR_CONTINUOUS, False, False),
    (OdeSolver.COLLOCATION(), ControlType.CONSTANT, False, False),
    (OdeSolver.COLLOCATION(), ControlType.CONSTANT, False, True),
    (OdeSolver.COLLOCATION(), ControlType.CONSTANT, True, True),
    (OdeSolver.TRAPEZOIDAL(), ControlType.LINEAR_CONTINUOUS, False, False),
    (OdeSolver.TRAPEZOIDAL(), ControlType.LINEAR_CONTINUOUS, False, True),
    (OdeSolver.TRAPEZOIDAL(), ControlType.LINEAR_CONTINUOUS, True, True),
])
def test_time_dependent_problem(n_phase, integrator, control_type, minimize_time, use_sx):
    """
    Firstly, it solves the getting_started/pendulum.py example.
    It then creates and solves this ocp and show the results.
    """
    from bioptim.examples.torque_driven_ocp import example_multi_biorbd_model as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    ocp = prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        n_phase=n_phase,
        ode_solver=integrator,
        control_type=control_type,
        minimize_time=minimize_time,
        use_sx=use_sx,
    )

    # --- Solve the program --- #
    sol = ocp.solve()
    # sol.graphs(show_bounds=True)
