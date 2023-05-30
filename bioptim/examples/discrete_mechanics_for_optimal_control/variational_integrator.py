"""
Variational integrator.
"""
from casadi import MX, Function, vertcat

from bioptim import (
    OptimalControlProgram,
    MultinodeConstraintList,
    PenaltyController,
    NonLinearProgram,
    DynamicsEvaluation,
    ConfigureProblem,
    DynamicsFunctions,
)

from biorbd_model_holonomic import BiorbdModelCustomHolonomic


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
    bio_model: BiorbdModelCustomHolonomic,
    constraints: Function = None,
    jac: Function = None,
    expand: bool = True,
):
    """
    Configure the dynamics of the system

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    bio_model: BiorbdModelCustomHolonomic
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

    ts = MX.sym("ts")
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
            [ts, q_prev, q_cur, q_next, control_prev, control_cur, control_next, lambdas],
            [
                bio_model.discrete_euler_lagrange_equations(
                    ts,
                    q_prev,
                    q_cur,
                    q_next,
                    control_prev,
                    control_cur,
                    control_next,
                    constraints,
                    jac,
                    lambdas,
                )
            ],
        )

        nlp.implicit_dynamics_func_first_node = Function(
            "TwoFirstNodesIntegration",
            [ts, q0, q0_dot, q1, control0, control1, lambdas],
            [bio_model.compute_initial_states(ts, q0, q0_dot, q1, control0, control1, constraints, jac, lambdas)],
        )

        nlp.implicit_dynamics_func_last_node = Function(
            "TwoLastNodesIntegration",
            [ts, qN_minus_1, qN, qN_dot, controlN_minus_1, controlN, lambdas],
            [
                bio_model.compute_final_states(
                    ts,
                    qN_minus_1,
                    qN,
                    qN_dot,
                    controlN_minus_1,
                    controlN,
                    constraints,
                    jac,
                    lambdas,
                )
            ],
        )

    else:
        nlp.implicit_dynamics_func = Function(
            "ThreeNodesIntegration",
            [ts, q_prev, q_cur, q_next, control_prev, control_cur, control_next],
            [
                bio_model.discrete_euler_lagrange_equations(
                    ts,
                    q_prev,
                    q_cur,
                    q_next,
                    control_prev,
                    control_cur,
                    control_next,
                )
            ],
        )

        nlp.implicit_dynamics_func_first_node = Function(
            "TwoFirstNodesIntegration",
            [ts, q0, q0_dot, q1, control0, control1],
            [bio_model.compute_initial_states(ts, q0, q0_dot, q1, control0, control1)],
        )

        nlp.implicit_dynamics_func_last_node = Function(
            "TwoLastNodesIntegration",
            [ts, qN_minus_1, qN, qN_dot, controlN_minus_1, controlN],
            [bio_model.compute_final_states(ts, qN_minus_1, qN, qN_dot, controlN_minus_1, controlN)],
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
            controllers[0].get_nlp.dt,
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
            controllers[0].get_nlp.dt,
            controllers[0].states["q"].cx,
            controllers[1].states["q"].cx,
            controllers[2].states["q"].cx,
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
            controllers[2].controls["tau"].cx,
        )


def variational_integrator_initial(
    controllers: list[PenaltyController, PenaltyController],
    n_qdot: int,
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
            controllers[0].get_nlp.dt,
            controllers[0].states["q"].cx,
            controllers[0].parameters.cx[:n_qdot],
            controllers[1].states["q"].cx,
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
            controllers[0].states["lambdas"].cx,
        )
    else:
        return controllers[0].get_nlp.implicit_dynamics_func_first_node(
            controllers[0].get_nlp.dt,
            controllers[0].states["q"].cx,
            controllers[0].parameters.cx[:n_qdot],
            controllers[1].states["q"].cx,
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
        )


def variational_integrator_final(
    controllers: list[PenaltyController, PenaltyController],
    n_qdot: int,
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
            controllers[0].get_nlp.dt,
            controllers[0].states["q"].cx,
            controllers[1].states["q"].cx,
            controllers[0].parameters.cx[n_qdot : 2 * n_qdot],
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
            controllers[1].states["lambdas"].cx,
        )
    else:
        return controllers[0].get_nlp.implicit_dynamics_func_last_node(
            controllers[0].get_nlp.dt,
            controllers[0].states["q"].cx,
            controllers[1].states["q"].cx,
            controllers[0].parameters.cx[n_qdot : 2 * n_qdot],
            controllers[0].controls["tau"].cx,
            controllers[1].controls["tau"].cx,
        )


def variational_continuity(n_shooting, n_qdot, use_constraints: bool = False) -> MultinodeConstraintList:
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
        n_qdot=n_qdot,
    )

    multinode_constraints.add(
        variational_integrator_final,
        nodes_phase=(0, 0),
        nodes=(n_shooting - 1, n_shooting),
        use_constraints=use_constraints,
        n_qdot=n_qdot,
    )
    return multinode_constraints
