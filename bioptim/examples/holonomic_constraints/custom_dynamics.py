from casadi import vertcat

from bioptim import (
    PenaltyController,
    DynamicsEvaluation,
    DynamicsFunctions,
    NonLinearProgram,
    OptimalControlProgram,
    ConfigureProblem,
    OdeSolver,
    HolonomicTorqueBiorbdModel,
    ConfigureVariables,
    HolonomicConstraintsList,
)


def constraint_holonomic_end(
    controllers: PenaltyController,
):
    """
    The custom constraint function that provides the holonomic constraints at the end node
    This function is used to compute the holonomic constraints at the end node of the phase because the last interval does not have cx_intermediates_list variables.

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
    Applies the holonomic constraints on each collocation node into the constraint set of solver.
    The holonomic constraints are not any more embedded in the equations of motion,
    but rather in the constraint set of the solver.

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


def configure_qv(ocp, nlp, as_states, as_controls, as_algebraic_states):
    name = "q_v"
    names_v = [nlp.model.name_dof[i] for i in nlp.model.dependent_joint_index]
    ConfigureVariables.configure_new_variable(
        name,
        names_v,
        ocp,
        nlp,
        as_states=False,
        as_controls=False,
        as_algebraic_states=True,
    )


class ModifiedHolonomicTorqueBiorbdModel(HolonomicTorqueBiorbdModel):
    def __init__(
        self,
        bio_model_path: str,
        holonomic_constraints: HolonomicConstraintsList = None,
        independent_joint_index: tuple[int] = None,
        dependent_joint_index: tuple[int] = None,
    ):
        super().__init__(
            bio_model_path,
            holonomic_constraints=holonomic_constraints,
            independent_joint_index=independent_joint_index,
            dependent_joint_index=dependent_joint_index,
        )
        self.algebraic_configuration += [configure_qv]

    @staticmethod
    def dynamics(
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
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q_u = DynamicsFunctions.get(nlp.states_dot["q_u"], nlp.states_dot.scaled.cx)
            slope_qdot_u = DynamicsFunctions.get(nlp.states_dot["qdot_u"], nlp.states_dot.scaled.cx)
            defects = vertcat(slope_q_u, slope_qdot_u) * nlp.dt - dxdt * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)
