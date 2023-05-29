"""
Variational integrator.
"""
from casadi import MX, SX, jacobian, transpose, Function, vertcat

import biorbd_casadi

from bioptim import (
    OptimalControlProgram,
    MultinodeConstraintList,
    PenaltyController,
    NonLinearProgram,
    DynamicsEvaluation,
    ConfigureProblem,
    DynamicsFunctions,
)

from enums import QuadratureRule, ControlType


# --- Variational integrator functions --- #
def lagrangian(biorbd_model: biorbd_casadi.Model, q: MX | SX, qdot: MX | SX) -> MX | SX:
    """
    Compute the Lagrangian of a biorbd model.

    Parameters
    ----------
    biorbd_model: biorbd_casadi.Model
        The biorbd model.
    q: MX | SX
        The generalized coordinates.
    qdot: MX | SX
        The generalized velocities.

    Returns
    -------
    The Lagrangian.
    """

    return biorbd_model.KineticEnergy(q, qdot).to_mx() - biorbd_model.PotentialEnergy(q).to_mx()


def discrete_lagrangian(
        biorbd_model: biorbd_casadi.Model,
        q1: MX | SX,
        q2: MX | SX,
        time_step: float,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL
) -> MX | SX:
    """
    Compute the discrete Lagrangian of a biorbd model.

    Parameters
    ----------
    biorbd_model: biorbd_casadi.Model
        The biorbd model.
    q1: MX | SX
        The generalized coordinates at the first time step.
    q2: MX | SX
        The generalized coordinates at the second time step.
    time_step: float
        The time step.
    discrete_approximation: QuadratureRule
        The quadrature rule to use for the discrete Lagrangian.

    Returns
    -------
    The discrete Lagrangian.
    """
    if discrete_approximation == QuadratureRule.MIDPOINT:
        q_discrete = (q1 + q2) / 2
        qdot_discrete = (q2 - q1) / time_step
        return time_step * lagrangian(biorbd_model, q_discrete, qdot_discrete)
    elif discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
        q_discrete = q1
        qdot_discrete = (q2 - q1) / time_step
        return time_step * lagrangian(biorbd_model, q_discrete, qdot_discrete)
    elif discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
        q_discrete = q2
        qdot_discrete = (q2 - q1) / time_step
        return time_step * lagrangian(biorbd_model, q_discrete, qdot_discrete)
    elif discrete_approximation == QuadratureRule.TRAPEZOIDAL:
        # from : M. West, “Variational integrators,” Ph.D. dissertation, California Inst.
        # Technol., Pasadena, CA, 2004. p 13
        qdot_discrete = (q2 - q1) / time_step
        return time_step / 2 * (
                lagrangian(biorbd_model, q1, qdot_discrete) + lagrangian(biorbd_model, q2, qdot_discrete)
        )
    else:
        raise NotImplementedError(f"Discrete Lagrangian {discrete_approximation} is not implemented")


def control_approximation(
        control_minus: MX | SX,
        control_plus: MX | SX,
        time_step: float,
        control_type: ControlType = ControlType.PIECEWISE_CONSTANT,
        discrete_approximation: QuadratureRule = QuadratureRule.MIDPOINT,
):
    """
    Compute the term associated to the discrete forcing. The term associated to the controls in the Lagrangian
    equations is homogeneous to a force or a torque multiplied by a time.

    Parameters
    ----------
    control_minus: MX | SX
        Control at t_k (or t{k-1})
    control_plus: MX | SX
        Control at t_{k+1} (or tk)
    time_step: float
        The time step.
    control_type: ControlType
        The type of control.
    discrete_approximation: QuadratureRule
        The quadrature rule to use for the discrete Lagrangian.

    Returns
    ----------
    The term associated to the controls in the Lagrangian equations.
    Johnson, E. R., & Murphey, T. D. (2009).
    Scalable Variational Integrators for Constrained Mechanical Systems in Generalized Coordinates.
    IEEE Transactions on Robotics, 25(6), 1249–1261. doi:10.1109/tro.2009.2032955
    """
    if control_type == ControlType.PIECEWISE_CONSTANT:
        return 1 / 2 * control_minus * time_step

    elif control_type == ControlType.PIECEWISE_LINEAR:
        if discrete_approximation == QuadratureRule.MIDPOINT:
            return 1 / 2 * (control_minus + control_plus) / 2 * time_step
        elif discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
            return 1 / 2 * control_minus * time_step
        elif discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
            return 1 / 2 * control_plus * time_step
        elif discrete_approximation == QuadratureRule.TRAPEZOIDAL:
            raise NotImplementedError(
                f"Discrete {discrete_approximation} is not implemented for {control_type}"
            )


def discrete_euler_lagrange_equations(
        biorbd_model: biorbd_casadi.Model,
        time_step: float,
        q_prev: MX | SX,
        q_cur: MX | SX,
        q_next: MX | SX,
        control_prev: MX | SX,
        control_cur: MX | SX,
        control_next: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas: MX | SX = None,
) -> MX | SX:
    """
    Compute the discrete Euler-Lagrange equations of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd_casadi.Model
        The biorbd model.
    time_step: float
        The time step.
    q_prev: MX | SX
        The generalized coordinates at the first time step.
    q_cur: MX | SX
        The generalized coordinates at the second time step.
    q_next: MX | SX
        The generalized coordinates at the third time step.
    control_prev: MX | SX
        The generalized forces at the first time step.
    control_cur: MX | SX
        The generalized forces at the second time step.
    control_next: MX | SX
        The generalized forces at the third time step.
    constraints: Function
        The constraints.
    jac: Function
        The jacobian of the constraints.
    lambdas: MX | SX
        The Lagrange multipliers.
    """
    p_current = transpose(jacobian(discrete_lagrangian(biorbd_model, q_prev, q_cur, time_step), q_cur))

    D1_Ld_qcur_qnext = transpose(jacobian(discrete_lagrangian(biorbd_model, q_cur, q_next, time_step), q_cur))
    constraint_term = (
        transpose(jac(q_cur)) @ lambdas if constraints is not None else MX.zeros(p_current.shape)
    )

    residual = (
            p_current
            + D1_Ld_qcur_qnext
            - constraint_term
            + control_approximation(control_prev, control_cur, time_step)
            + control_approximation(control_cur, control_next, time_step)
    )

    if constraints is not None:
        return vertcat(residual, constraints(q_next))
    else:
        return residual


def compute_initial_states(
        biorbd_model: biorbd_casadi.Model,
        time_step: MX | SX,
        q0: MX | SX,
        q0_dot: MX | SX,
        q1: MX | SX,
        control0: MX | SX,
        control1: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdas0: MX | SX = None,
):
    """
    Compute the initial states of the system from the initial position and velocity.
    """
    # The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
    # constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
    # indications given just before the equation (18) for p0 and pN.
    D2_L_q0_q0dot = transpose(jacobian(lagrangian(biorbd_model, q0, q0_dot), q0_dot))
    D1_Ld_q0_q1 = transpose(jacobian(discrete_lagrangian(biorbd_model, q0, q1, time_step), q0))
    f0_minus = control_approximation(control0, control1, time_step)
    constraint_term = (
        1 / 2 * transpose(jac(q0)) @ lambdas0
        if constraints is not None
        else MX.zeros(biorbd_model.nbQ(), 1)
    )
    residual = D2_L_q0_q0dot + D1_Ld_q0_q1 + f0_minus - constraint_term

    if constraints is not None:
        return vertcat(residual, constraints(q0), constraints(q1))  # constraints(0) is never evaluated if not here
    else:
        return residual


def compute_final_states(
        biorbd_model: biorbd_casadi.Model,
        time_step: MX | SX,
        qN_minus_1: MX | SX,
        qN: MX | SX,
        qN_dot: MX | SX,
        controlN_minus_1: MX | SX,
        controlN: MX | SX,
        constraints: Function = None,
        jac: Function = None,
        lambdasN: MX | SX = None,
        ):
    """
    Compute the initial states of the system from the initial position and velocity.
    """
    # The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
    # constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
    # indications given just before the equation (18) for p0 and pN.
    D2_L_qN_qN_dot = transpose(jacobian(lagrangian(biorbd_model, qN, qN_dot), qN_dot))
    D2_Ld_qN_minus_1_qN = transpose(jacobian(discrete_lagrangian(biorbd_model, qN_minus_1, qN, time_step), qN))
    fd_plus = control_approximation(controlN_minus_1, controlN, time_step)
    constraint_term = (
        1 / 2 * transpose(jac(qN)) @ lambdasN
        if constraints is not None
        else MX.zeros(biorbd_model.nbQ(), 1)
    )

    residual = -D2_L_qN_qN_dot + D2_Ld_qN_minus_1_qN + fd_plus - constraint_term
    # constraints(qN) has already been evaluated in the last constraint calling discrete_euler_lagrange_equations, thus
    # it is not necessary to evaluate it again here.
    return residual


# --- Parameters for the initial and final velocity --- #
def qdot_function(model, value):
    """
    It is currently mandatory to provide a function to the method add of ParameterList.
    Parameters
    ----------
    model
    value
    """
    pass


# --- Dynamics --- #
def custom_dynamics_function(
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        bio_model,
        constraints: Function = None,
        jac: Function = None,
        expand: bool = True):
    """
    Configure the dynamics of the system

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    bio_model: BiorbdModel
        The biorbd model
    constraints: Function
        The constraint function
    jac: Function
        The jacobian of the constraints
    expand: bool
        If the dynamics should be expanded with casadi
    """

    nlp.parameters = ocp.v.parameters_in_list
    DynamicsFunctions.apply_parameters(nlp.parameters.mx, nlp)

    dynamics_eval = DynamicsEvaluation(MX(0), MX(0))
    dynamics_dxdt = dynamics_eval.dxdt
    if isinstance(dynamics_dxdt, (list, tuple)):
        dynamics_dxdt = vertcat(*dynamics_dxdt)

    # Note: useless but needed to run bioptim as it need to test the size of xdot
    nlp.dynamics_func = Function(
        "ForwardDyn",
        [nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced, nlp.parameters.mx],
        [dynamics_dxdt],
        ["x", "u", "p"],
        ["xdot"],
    )

    time_step = nlp.tf / nlp.ns
    q_prev = MX.sym("q_prev", nlp.model.nb_q, 1)
    q_cur = MX.sym("q_cur", nlp.model.nb_q, 1)
    q_next = MX.sym("q_next", nlp.model.nb_q, 1)
    control_prev = MX.sym("control_prev", nlp.model.nb_q, 1)
    control_cur = MX.sym("control_cur", nlp.model.nb_q, 1)
    control_next = MX.sym("control_next", nlp.model.nb_q, 1)
    q0 = MX.sym("q0", nlp.model.nb_q, 1)
    q0_dot = MX.sym("q0_dot", nlp.model.nb_q, 1)
    q1 = MX.sym("q1", nlp.model.nb_q, 1)
    control0 = MX.sym("control0", nlp.model.nb_q, 1)
    control1 = MX.sym("control1", nlp.model.nb_q, 1)
    qN = MX.sym("qN", nlp.model.nb_q, 1)
    qN_dot = MX.sym("qN_dot", nlp.model.nb_q, 1)
    qN_minus_1 = MX.sym("qN_minus_1", nlp.model.nb_q, 1)
    controlN_minus_1 = MX.sym("controlN_minus_1", nlp.model.nb_q, 1)
    controlN = MX.sym("controlN", nlp.model.nb_q, 1)
    if constraints is not None:
        lambdas = MX.sym("lambda", constraints.nnz_out(), 1)
        nlp.implicit_dynamics_func = Function(
            "ThreeNodesIntegration",
            [q_prev, q_cur, q_next, control_prev, control_cur, control_next, lambdas],
            [discrete_euler_lagrange_equations(
                bio_model.model,
                time_step,
                q_prev,
                q_cur,
                q_next,
                control_prev,
                control_cur,
                control_next,
                constraints,
                jac,
                lambdas,
            )]
        )

        nlp.implicit_dynamics_func_first_node = Function(
            "TwoFirstNodesIntegration",
            [q0, q0_dot, q1, control0, control1, lambdas],
            [compute_initial_states(
                bio_model.model, time_step, q0, q0_dot, q1, control0, control1, constraints, jac, lambdas)]
        )

        nlp.implicit_dynamics_func_last_node = Function(
            "TwoLastNodesIntegration",
            [qN_minus_1, qN, qN_dot, controlN_minus_1, controlN, lambdas],
            [compute_final_states(
                bio_model.model, time_step, qN_minus_1, qN, qN_dot, controlN_minus_1, controlN, constraints, jac,
                lambdas)]
        )

    else:
        nlp.implicit_dynamics_func = Function(
            "ThreeNodesIntegration",
            [q_prev, q_cur, q_next, control_prev, control_cur, control_next],
            [discrete_euler_lagrange_equations(
                bio_model.model,
                time_step,
                q_prev,
                q_cur,
                q_next,
                control_prev,
                control_cur,
                control_next,
            )]
        )

        nlp.implicit_dynamics_func_first_node = Function(
            "TwoFirstNodesIntegration",
            [q0, q0_dot, q1, control0, control1],
            [compute_initial_states(bio_model.model, time_step, q0, q0_dot, q1, control0, control1)]
        )

        nlp.implicit_dynamics_func_last_node = Function(
            "TwoLastNodesIntegration",
            [qN_minus_1, qN, qN_dot, controlN_minus_1, controlN],
            [compute_final_states(bio_model.model, time_step, qN_minus_1, qN, qN_dot, controlN_minus_1, controlN)]
        )

    if expand:
        nlp.dynamics_func = nlp.dynamics_func.expand()
        nlp.implicit_dynamics_func = nlp.implicit_dynamics_func.expand()
        nlp.implicit_dynamics_func_first_node = nlp.implicit_dynamics_func_first_node.expand()
        nlp.implicit_dynamics_func_last_node = nlp.implicit_dynamics_func_last_node.expand()


def custom_configure_unconstrained(ocp: OptimalControlProgram, nlp: NonLinearProgram, bio_model, expand: bool = True):
    """
    If the problem is not constrained, use this custom configuration.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp.
    nlp: NonLinearProgram
        A reference to the phase.
    bio_model: BiorbdModel
        The biorbd model.
    expand: bool
        If the dynamics should be expanded with casadi.
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    custom_dynamics_function(ocp, nlp, bio_model, expand=expand)


# --- Constraints continuity --- #
def variational_integrator_three_nodes(
        controllers: list[PenaltyController, PenaltyController, PenaltyController],
        use_constraints: bool = False,
):
    """
    The discrete Euler Lagrange equations for the main integration.

    Parameters
    ----------
    controllers
    use_constraints

    Returns
    -------

    """
    if use_constraints:
        return controllers[0].get_nlp.implicit_dynamics_func(
            controllers[0].states["q"].cx,
            controllers[1].states["q"].cx,
            controllers[2].states["q"].cx,
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
            controllers[2].controls["tau"].cx,
            controllers[1].states["lambdas"].cx,
        )
    else:
        return controllers[0].get_nlp.implicit_dynamics_func(
            controllers[0].states["q"].cx,
            controllers[1].states["q"].cx,
            controllers[2].states["q"].cx,
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
            controllers[2].controls["tau"].cx,
        )


def variational_integrator_initial(
        controllers: list[PenaltyController, PenaltyController],
        use_constraints: bool = False,
):
    """
    The initial continuity constraint for the integration.

    Parameters
    ----------
    controllers
    use_constraints

    Returns
    -------

    """
    if use_constraints:
        return controllers[0].get_nlp.implicit_dynamics_func_first_node(
            controllers[0].states["q"].cx,
            controllers[0].parameters.cx[0],
            controllers[1].states["q"].cx,
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
            controllers[0].states["lambdas"].cx,
        )
    else:
        return controllers[0].get_nlp.implicit_dynamics_func_first_node(
            controllers[0].states["q"].cx,
            controllers[0].parameters.cx[0],
            controllers[1].states["q"].cx,
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
        )


def variational_integrator_final(
        controllers: list[PenaltyController, PenaltyController],
        use_constraints: bool = False,
):
    """
    The final continuity constraint for the integration.

    Parameters
    ----------
    use_constraints
    controllers

    Returns
    -------

    """
    if use_constraints:
        return controllers[0].get_nlp.implicit_dynamics_func_last_node(
            controllers[0].states["q"].cx,
            controllers[1].states["q"].cx,
            controllers[0].parameters.cx[1],
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
            controllers[1].states["lambdas"].cx,
        )
    else:
        return controllers[0].get_nlp.implicit_dynamics_func_last_node(
            controllers[0].states["q"].cx,
            controllers[1].states["q"].cx,
            controllers[0].parameters.cx[1],
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
        )


def variational_continuity(n_shooting, use_constraints: bool = False) -> MultinodeConstraintList:
    """
    The continuity constraint for the integration.

    Parameters
    ----------
    n_shooting
    use_constraints

    Returns
    -------
    The list of continuity constraints for the integration.
    """
    multinode_constraints = MultinodeConstraintList()
    for i in range(n_shooting - 1):
        multinode_constraints.add(
            variational_integrator_three_nodes,
            nodes_phase=(0, 0, 0),
            nodes=(i, i + 1, i + 2),
            use_constraints=use_constraints,
        )
    # add initial and final constraints
    multinode_constraints.add(
        variational_integrator_initial,
        nodes_phase=(0, 0),
        nodes=(0, 1),
        use_constraints=use_constraints,
    )

    multinode_constraints.add(
        variational_integrator_final,
        nodes_phase=(0, 0),
        nodes=(n_shooting - 1, n_shooting),
        use_constraints=use_constraints,
    )
    return multinode_constraints
