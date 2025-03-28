from casadi import vertcat

from bioptim import (
    PenaltyController,
    DynamicsEvaluation,
    DynamicsFunctions,
    NonLinearProgram,
    OptimalControlProgram,
    ConfigureProblem,
    OdeSolver,
)


def constraint_holonomic_end(
    controllers: PenaltyController,
):
    """
    Minimize the distance between two markers
    By default this function is quadratic, meaning that it minimizes distance between them.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    """

    q_u = controllers.states["q_u"]
    q_u_complete = q_u.mapping.to_second.map(q_u.cx)

    q_v = controllers.algebraic_states["q_v"]
    q_v_complete = q_v.mapping.to_second.map(q_v.cx)

    q = controllers.model.state_from_partition(q_u_complete, q_v_complete)

    return controllers.model.holonomic_constraints(q)


def constraint_holonomic(
    controllers: PenaltyController,
):
    """
    Minimize the distance between two markers
    By default this function is quadratic, meaning that it minimizes distance between them.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    """

    q_u = controllers.states["q_u"]
    q_u_complete = q_u.mapping.to_second.map(q_u.cx)

    q_v = controllers.algebraic_states["q_v"]
    q_v_complete = q_v.mapping.to_second.map(q_v.cx)

    q = controllers.model.state_from_partition(q_u_complete, q_v_complete)

    holonomic_constraints = controllers.model.holonomic_constraints(q)

    for q_u_cx, q_v_cx in zip(q_u.cx_intermediates_list, q_v.cx_intermediates_list):
        q_u_complete = q_u.mapping.to_second.map(q_u_cx)
        q_v_complete = q_v.mapping.to_second.map(q_v_cx)
        q = controllers.model.state_from_partition(q_u_complete, q_v_complete)
        holonomic_constraints = vertcat(holonomic_constraints, controllers.model.holonomic_constraints(q))

    return holonomic_constraints


def holonomic_torque_driven_with_qv(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    numerical_timeseries,
    nlp,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, a, d)

    Parameters
    ----------
    time: MX.sym | SX.sym
        The time of the system
    states: MX.sym | SX.sym
        The state of the system
    controls: MX.sym | SX.sym
        The controls of the system
    parameters: MX.sym | SX.sym
        The parameters acting on the system
    algebraic_states: MX.sym | SX.sym
        The algebraic states of the system
    numerical_timeseries: MX.sym | SX.sym
        The numerical timeseries of the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
    qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    q_v = DynamicsFunctions.get(nlp.algebraic_states["q_v"], algebraic_states)
    qddot_u = nlp.model.partitioned_forward_dynamics_with_qv()(q_u, q_v, qdot_u, tau)

    dxdt = vertcat(qdot_u, qddot_u)

    defects = None
    if isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
        slope_q_u = DynamicsFunctions.get(nlp.states_dot["q_u"], nlp.states_dot.scaled.cx)
        slope_qdot_u = DynamicsFunctions.get(nlp.states_dot["qdot_u"], nlp.states_dot.scaled.cx)
        defects = vertcat(slope_q_u, slope_qdot_u) * nlp.dt - dxdt * nlp.dt

    return DynamicsEvaluation(dxdt=dxdt, defects=defects)


def configure_holonomic_torque_driven(
    ocp: OptimalControlProgram,
    nlp: NonLinearProgram,
    numerical_data_timeseries=None,
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

    name = "q_u"
    names_u = [nlp.model.name_dof[i] for i in nlp.model.independent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_u,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=True,
    )

    name = "q_v"
    names_v = [nlp.model.name_dof[i] for i in nlp.model.dependent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_v,
        ocp,
        nlp,
        as_states=False,
        as_controls=False,
        as_algebraic_states=True,
    )

    name = "qdot_u"
    names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
    names_udot = [names_qdot[i] for i in nlp.model.independent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_udot,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=True,
        # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
        # see _set_kinematic_phase_mapping method
        # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
    )

    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    # extra plots
    ConfigureProblem.configure_qv(ocp, nlp, nlp.model.compute_q_v)
    ConfigureProblem.configure_qdotv(ocp, nlp, nlp.model._compute_qdot_v)
    ConfigureProblem.configure_lagrange_multipliers_function(ocp, nlp, nlp.model.compute_the_lagrangian_multipliers)

    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        holonomic_torque_driven_with_qv,
    )
