from typing import Callable

import numpy as np
from casadi import vertcat, DM

import biorbd
from bioptim import (
    ConfigureVariables,
    Controls,
    DynamicsEvaluation,
    DynamicsFunctions,
    HolonomicBiorbdModel,
    HolonomicConstraintsList,
    HolonomicTorqueBiorbdModel,
    HolonomicTorqueDynamics,
    OdeSolver,
    ParameterList,
    PenaltyController,
    States,
    DefectType,
    TorqueDynamics,
    MusclesDynamics,
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
    The custom constraint function that provides the holonomic constraints at each collocation node.
    This function is used to add the holonomic constraints to the solver's constraint set.
    Please note that the holonomic constraints are NOT embedded in the equations of motion,
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
            defects = vertcat(slope_q_u, slope_qdot_u) - dxdt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)


class HolonomicMusclesDynamics(HolonomicTorqueDynamics):

    def __init__(self):
        super().__init__()
        self.state_configuration = [States.Q_U, States.QDOT_U]
        self.control_configuration = [Controls.TAU, Controls.MUSCLE_EXCITATION]
        self.algebraic_configuration = []
        self.functions = [
            ConfigureVariables.configure_qv,
            ConfigureVariables.configure_qdotv,
            ConfigureVariables.configure_lagrange_multipliers_function,
        ]
        self.with_residual_torque = True

    def get_basic_variables(
        self,
        nlp,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
    ):

        # Get variables from the right place
        # q = DynamicsFunctions.get(nlp.states["q"], states)
        # qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
        qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
        q_v_init = DM.zeros(nlp.model.nb_dependent_joints)
        mus_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
        fatigue_states, mus_activations = DynamicsFunctions.get_fatigue_states(states, nlp, mus_activations)

        q = nlp.model.compute_q()(q_u, q_v_init)
        qdot = nlp.model.compute_qdot()(q, qdot_u)

        # Compute the torques due to muscles
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)

        # Add additional torques
        if self.with_residual_torque:
            muscles_tau += DynamicsFunctions.get_fatigable_tau(nlp, states, controls)
        muscles_tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        # Get external forces
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q_u, qdot_u, muscles_tau, external_forces, mus_activations

    def dynamics(
        self,
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ):

        # q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
        # qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
        # tau = DynamicsFunctions.get(nlp.controls["tau"], controls) # Get torques from muscles + residual torques
        q_u, qdot_u, tau, _, _ = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )
        q_v_init = DM.zeros(nlp.model.nb_dependent_joints)

        qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)
        dxdt = vertcat(qdot_u, qddot_u)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q_u = DynamicsFunctions.get(nlp.states_dot["q_u"], nlp.states_dot.scaled.cx)
            slope_qdot_u = DynamicsFunctions.get(nlp.states_dot["qdot_u"], nlp.states_dot.scaled.cx)
            defects = vertcat(slope_q_u, slope_qdot_u) - dxdt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    def get_rigid_contact_forces(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        return

    @property
    def extra_dynamics(self):
        return None


class HolonomicMusclesBiorbdModel(HolonomicBiorbdModel, HolonomicMusclesDynamics):
    def __init__(
        self,
        bio_model: str | biorbd.Model,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        holonomic_constraints: HolonomicConstraintsList | None = None,
        dependent_joint_index: list[int] | tuple[int, ...] = None,
        independent_joint_index: list[int] | tuple[int, ...] = None,
    ):
        HolonomicBiorbdModel.__init__(self, bio_model, friction_coefficients, parameters)
        if holonomic_constraints is not None:
            self.set_holonomic_configuration(holonomic_constraints, dependent_joint_index, independent_joint_index)
        HolonomicMusclesDynamics.__init__(self)

    def serialize(self) -> tuple[Callable, dict]:
        return HolonomicMusclesBiorbdModel, dict(bio_model=self.path, friction_coefficients=self.friction_coefficients)


class AlgebraicHolonomicMusclesBiorbdModel(HolonomicMusclesBiorbdModel):
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

    def get_basic_variables(
        self,
        nlp,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
    ):

        q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
        qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
        q_v = DynamicsFunctions.get(nlp.algebraic_states["q_v"], algebraic_states)
        mus_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
        fatigue_states, mus_activations = DynamicsFunctions.get_fatigue_states(states, nlp, mus_activations)

        q = nlp.model.compute_q()(q_u, q_v)
        qdot = nlp.model.compute_qdot()(q, qdot_u)

        # Compute the torques due to muscles
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)

        # Add additional torques
        if self.with_residual_torque:
            muscles_tau += DynamicsFunctions.get_fatigable_tau(nlp, states, controls)
        muscles_tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        # Get external forces
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q_u, qdot_u, muscles_tau, external_forces, mus_activations
