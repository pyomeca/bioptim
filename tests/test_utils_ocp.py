"""
This file implements the 'old' custom_dynamics.
It is used in many tests, so it has been displaced here.
"""

from casadi import MX, SX, vertcat

from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsOptionsList,
    DynamicsOptions,
    DynamicsFunctions,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    PhaseDynamics,
    States,
    Controls,
)


class CustomBiorbdModel(BiorbdModel):
    def __init__(self, biorbd_model_path: str, **kwargs):
        BiorbdModel.__init__(self, biorbd_model_path)

        # Define the variables to configure here
        self.state_type = [States.Q, States.QDOT]
        self.control_type = [Controls.TAU]
        self.algebraic_type = []
        self.functions = []
        self.extra_dynamics = None
        self.extra_parameters = kwargs

    def dynamics(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_timeseries: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

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
            The algebraic states of the system
        nlp: NonLinearProgram
            A reference to the phase
        my_additional_factor: int
            An example of an extra parameter sent by the user

        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """
        my_additional_factor = self.extra_parameters["my_additional_factor"]

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot) * my_additional_factor
        ddq = nlp.model.forward_dynamics()(q, qdot, tau, [], nlp.parameters.cx)
        dxdt = vertcat(dq, ddq)

        # the user has to choose if want to return the explicit dynamics dx/dt = f(x,u,p)
        # as the first argument of DynamicsEvaluation or
        # the implicit dynamics f(x,u,p,xdot)=0 as the second argument
        # which may be useful for IRK or COLLOCATION integrators
        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            # Implicit dynamics
            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slopes = vertcat(slope_q, slope_qdot)
            defects = slopes * nlp.dt - dxdt * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)


def prepare_ocp(
    biorbd_model_path: str,
    problem_type_custom: bool = True,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    problem_type_custom: bool
        If the preparation should be done using the user-defined dynamic function or the normal TORQUE_DRIVEN.
        They should return the same results
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = CustomBiorbdModel(biorbd_model_path, my_additional_factor=1)

    # Problem parameters
    n_shooting = 30
    final_time = 2

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, multi_thread=False)

    # DynamicsOptions
    dynamics = DynamicsOptionsList()
    if problem_type_custom:
        dynamics.add(
            DynamicsOptions(
                ode_solver=ode_solver,
                expand_dynamics=expand_dynamics,
                phase_dynamics=phase_dynamics,
            )
        )
    else:
        dynamics.add(
            DynamicsOptions(
                ode_solver=ode_solver,
                expand_dynamics=expand_dynamics,
                phase_dynamics=phase_dynamics,
            )
        )

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bio_model.bounds_from_ranges("qdot"))
    x_bounds["q"][1:, [0, -1]] = 0
    x_bounds["q"][2, -1] = 1.57
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model.nb_tau, max_bound=[tau_max] * bio_model.nb_tau)

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        use_sx=use_sx,
    )
