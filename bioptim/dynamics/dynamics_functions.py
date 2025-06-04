from casadi import horzcat, vertcat, MX, SX, DM

from .dynamics_evaluation import DynamicsEvaluation
from .ode_solvers import OdeSolver
from .fatigue.fatigue_dynamics import FatigueList
from ..limits.holonomic_constraints import HolonomicConstraintsFcn
from ..misc.enums import DefectType, ContactType
from ..misc.mapping import BiMapping
from ..optimization.optimization_variable import OptimizationVariable
from ..misc.parameters_types import (
    Bool,
    AnyListOptional,
    CX,
    CXOptional,
)


class DynamicsFunctions:
    """
    Implementation of all the dynamic functions

    Methods
    -------
    custom -> MX
        Interface to custom dynamic function provided by the user
    torque_driven
        Forward dynamics driven by joint torques
    torque_activations_driven -> MX:
        Forward dynamics driven by joint torques activations.
    torque_derivative_driven-> MX:
        Forward dynamics driven by joint torques derivatives
    forces_from_torque_driven -> MX:
        Contact forces of a forward dynamics driven by joint torques with contact constraints.
    muscles_driven -> MX:
        Forward dynamics driven by muscle.
    forces_from_muscle_driven -> MX:
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.
    get:
        Main accessor to a variable in states or controls (cx)
    reshape_qdot:
        Easy accessor to derivative of q
    forward_dynamics:
        Easy accessor to derivative of qdot
    compute_muscle_dot:
        Easy accessor to derivative of muscle activations
    compute_tau_from_muscle:
        Easy accessor to tau computed from muscles
    """

    @staticmethod
    def custom(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ) -> DynamicsEvaluation:
        """
        Interface to custom dynamic function provided by the user.

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic_states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym | SX.sym
            The derivative of the states
        MX.sym | SX.sym
            The defects of the implicit dynamics
        """

        return nlp.dynamics_type.dynamic_function(
            time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp
        )

    @staticmethod
    def torque_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        fatigue: FatigueList,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system
        fatigue : FatigueList
            A list of fatigue elements

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        tau = DynamicsFunctions.get_fatigable_tau(nlp, states, controls, fatigue)
        if nlp.model.nb_passive_joint_torques > 0:
            tau += nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.nb_ligaments > 0:
            tau += nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.friction_coefficients is not None:
            tau -= nlp.model.friction_coefficients @ qdot

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )

        forward_dynamics_contact_types = ContactType.get_equivalent_explicit_contacts(nlp.model.contact_types)
        ddq_fd = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, forward_dynamics_contact_types, external_forces)

        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[nlp.states["q"].index, 0] = dq
        dxdt[nlp.states["qdot"].index, 0] = ddq_fd

        if fatigue is not None and "tau" in fatigue:
            dxdt = fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            for key in nlp.states.keys():
                if nlp.variable_mappings[key].actually_does_a_mapping:
                    raise NotImplementedError(
                        f"COLLOCATION transcription is not compatible with mapping for states."
                        "Please note that concept of states mapping in already sketchy on it's own, but is particularly not appropriate for COLLOCATION transcriptions."
                    )

            # Do not use DynamicsFunctions.get to get the slopes because we do not want them mapped
            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx
            defects = nlp.cx(nlp.states.shape, 1)

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:

                dxdt_defects = nlp.cx(nlp.states.shape, 1)
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, nlp.model.contact_types, external_forces)
                dxdt_defects[nlp.states["q"].index, 0] = qdot
                dxdt_defects[nlp.states["qdot"].index, 0] = ddq

                slopes = nlp.cx(nlp.states.shape, 1)
                slopes[nlp.states["q"].index, 0] = slope_q
                slopes[nlp.states["qdot"].index, 0] = slope_qdot

                if fatigue is not None and "tau" in fatigue:
                    dxdt_defects = fatigue["tau"].dynamics(dxdt_defects, nlp, states, controls)
                    state_keys = nlp.states.keys()
                    if state_keys[0] != "q" or state_keys[1] != "qdot":
                        raise NotImplementedError(
                            "The accession of tau fatigue states is not implemented generically yet."
                        )

                    slopes_fatigue = nlp.cx()
                    fatigue_indices = []
                    for key in state_keys[2:]:
                        if not key.startswith("tau_"):
                            raise NotImplementedError(
                                "The accession of muscles tau states is not implemented generically yet."
                            )
                        slopes_fatigue = vertcat(slopes_fatigue, nlp.states_dot[key].cx)
                        fatigue_indices += list(nlp.states[key].index)

                    slopes[fatigue_indices, 0] = slopes_fatigue

                defects = slopes * nlp.dt - dxdt_defects * nlp.dt

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                if fatigue is not None:
                    raise NotImplementedError("Fatigue is not implemented yet with inverse dynamics defects.")

                defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - dq * nlp.dt

                tau_id = DynamicsFunctions.inverse_dynamics(
                    nlp,
                    q=q,
                    qdot=qdot,
                    qddot=slope_qdot,
                    contact_types=nlp.model.contact_types,
                    external_forces=external_forces,
                )
                tau_defects = tau - tau_id
                defects[nlp.states["qdot"].index, 0] = tau_defects
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for torque driven dynamics."
                )

            # We append the defects with the algebraic states implicit constraints
            if ContactType.RIGID_IMPLICIT in nlp.model.contact_types:
                rigid_contact_defect = (
                    nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)
                    - nlp.algebraic_states["rigid_contact_forces"].cx
                )
                _, _, acceleration_constraint_func = HolonomicConstraintsFcn.rigid_contacts(nlp.model)
                contact_acceleration_defect = acceleration_constraint_func(q, qdot, slope_qdot, nlp.parameters.cx)
                defects = vertcat(defects, rigid_contact_defect, contact_acceleration_defect)

            if ContactType.SOFT_IMPLICIT in nlp.model.contact_types:
                soft_contact_defect = (
                    nlp.model.soft_contact_forces().expand()(q, qdot, nlp.parameters.cx)
                    - nlp.algebraic_states["soft_contact_forces"].cx
                )
                defects = vertcat(defects, soft_contact_defect)

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def torque_driven_free_floating_base(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques without actuation of the free floating base

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
        q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
        qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
        qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
        tau_joints = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)

        q_full = vertcat(q_roots, q_joints)
        qdot_full = vertcat(qdot_roots, qdot_joints)
        dq = DynamicsFunctions.compute_qdot(nlp, q_full, qdot_full)

        if nlp.model.nb_passive_joint_torques > 0:
            tau_joints += nlp.model.passive_joint_torque()(q_full, qdot_full, nlp.parameters.cx)
        if nlp.model.nb_ligaments > 0:
            tau_joints += nlp.model.ligament_joint_torque()(q_full, qdot_full)
        if nlp.model.friction_coefficients is not None:
            tau_joints -= nlp.model.friction_coefficients @ qdot_joints

        tau_full = vertcat(nlp.cx.zeros(nlp.model.nb_root), tau_joints)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )

        forward_dynamics_contact_typess = ContactType.get_equivalent_explicit_contacts(nlp.model.contact_types)
        ddq_fd = DynamicsFunctions.forward_dynamics(
            nlp,
            q_full,
            qdot_full,
            tau_full,
            contact_types=forward_dynamics_contact_typess,
            external_forces=external_forces,
        )
        q_index = list(nlp.states["q_roots"].index) + list(nlp.states["q_joints"].index)
        qdot_index = list(nlp.states["qdot_roots"].index) + list(nlp.states["qdot_joints"].index)
        dxdt = nlp.cx(nlp.states.shape, ddq_fd.shape[1])
        dxdt[q_index, :] = dq
        dxdt[qdot_index, :] = ddq_fd

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx
            defects = nlp.cx(nlp.states.shape)

            # qdot = polynomial slope
            defects[nlp.states["q"].index] = slope_q * nlp.dt - dq * nlp.dt

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                ddq = DynamicsFunctions.forward_dynamics(
                    nlp, q_full, qdot_full, tau_full, nlp.model.contact_types, external_forces
                )
                defects[nlp.states["qdot"].index] = slope_qdot * nlp.dt - ddq * nlp.dt

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                tau_id = DynamicsFunctions.inverse_dynamics(
                    nlp,
                    q=q_full,
                    qdot=qdot_full,
                    qddot=slope_qdot,
                    contact_types=nlp.model.contact_types,
                    external_forces=external_forces,
                )
                tau_defects = tau_full - tau_id
                defects[nlp.states["qdot"].index] = tau_defects
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for torque driven dynamics."
                )

            # We append the defects with the algebraic states implicit constraints
            if ContactType.RIGID_IMPLICIT in nlp.model.contact_types:
                rigid_contact_defect = (
                    nlp.model.rigid_contact_forces()(q_full, qdot_full, tau_full, external_forces, nlp.parameters.cx)
                    - nlp.algebraic_states["rigid_contact_forces"].cx
                )
                _, _, acceleration_constraint_func = HolonomicConstraintsFcn.rigid_contacts(nlp.model)
                contact_acceleration_defect = acceleration_constraint_func(
                    q_full, qdot_full, slope_qdot, nlp.parameters.cx
                )
                defects = vertcat(defects, rigid_contact_defect, contact_acceleration_defect)

            if ContactType.SOFT_IMPLICIT in nlp.model.contact_types:
                soft_contact_defect = (
                    nlp.model.soft_contact_forces().expand()(q_full, qdot_full, nlp.parameters.cx)
                    - nlp.algebraic_states["soft_contact_forces"].cx
                )
                defects = vertcat(defects, soft_contact_defect)

        return DynamicsEvaluation(dxdt, defects=defects)

    @staticmethod
    def stochastic_torque_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics subject to motor and sensory noise driven by torques

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states variables of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        if (
            ContactType.SOFT_EXPLICIT in nlp.model.contact_types
            or ContactType.SOFT_IMPLICIT in nlp.model.contact_types
            or ContactType.RIGID_IMPLICIT in nlp.model.contact_types
        ):
            raise NotImplementedError(
                "soft contacts and implicit contacts not implemented yet with stochastic torque driven dynamics."
            )

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        if external_forces.shape != (0, 1):
            raise NotImplementedError("External forces are not implemented yet with stochastic torque driven dynamics.")

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
        motor_noise = DynamicsFunctions.get(nlp.parameters["motor_noise"], parameters)
        sensory_noise = DynamicsFunctions.get(nlp.parameters["sensory_noise"], parameters)

        tau += nlp.model.compute_torques_from_noise_and_feedback(
            nlp=nlp,
            time=time,
            states=states,
            controls=controls,
            parameters=parameters,
            algebraic_states=algebraic_states,
            numerical_timeseries=numerical_timeseries,
            sensory_noise=sensory_noise,
            motor_noise=motor_noise,
        )

        if nlp.model.friction_coefficients is not None:
            tau -= nlp.model.friction_coefficients @ qdot

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(
            nlp, q, qdot, tau, contact_types=nlp.model.contact_types, external_forces=None
        )
        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[nlp.states["q"].index, 0] = dq
        dxdt[nlp.states["qdot"].index, 0] = ddq

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx
            defects = nlp.cx(nlp.states.shape, 1)

            # qdot = polynomial slope
            defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - dq * nlp.dt

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                defects[nlp.states["qdot"].index, 0] = slope_qdot * nlp.dt - ddq * nlp.dt
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for stochastic torque driven dynamics."
                )
        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def stochastic_torque_driven_free_floating_base(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics subject to motor and sensory noise driven by joint torques

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        if (
            ContactType.SOFT_EXPLICIT in nlp.model.contact_types
            or ContactType.SOFT_IMPLICIT in nlp.model.contact_types
            or ContactType.RIGID_IMPLICIT in nlp.model.contact_types
        ):
            raise NotImplementedError(
                "soft contacts and implicit contacts not implemented yet with stochastic torque driven dynamics."
            )

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        if external_forces.shape != (0, 1):
            raise NotImplementedError("External forces are not implemented yet with stochastic torque driven dynamics.")

        q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
        q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
        qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
        qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
        tau_joints = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)
        motor_noise = DynamicsFunctions.get(nlp.parameters["motor_noise"], parameters)
        sensory_noise = DynamicsFunctions.get(nlp.parameters["sensory_noise"], parameters)

        q_full = vertcat(q_roots, q_joints)
        qdot_full = vertcat(qdot_roots, qdot_joints)
        dq = DynamicsFunctions.compute_qdot(nlp, q_full, qdot_full)

        tau_joints += nlp.model.compute_torques_from_noise_and_feedback(
            nlp=nlp,
            time=time,
            states=states,
            controls=controls,
            parameters=parameters,
            algebraic_states=algebraic_states,
            motor_noise=motor_noise,
            sensory_noise=sensory_noise,
        )
        if nlp.model.friction_coefficients is not None:
            tau_joints -= nlp.model.friction_coefficients @ qdot_joints

        tau_full = vertcat(nlp.cx.zeros(nlp.model.nb_root), tau_joints)

        # Free floating base is by definition without contacts
        ddq = DynamicsFunctions.forward_dynamics(
            nlp, q_full, qdot_full, tau_full, contact_types=[], external_forces=None
        )

        q_index = list(nlp.states["q_roots"].index) + list(nlp.states["q_joints"].index)
        qdot_index = list(nlp.states["qdot_roots"].index) + list(nlp.states["qdot_joints"].index)
        dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
        dxdt[q_index, :] = dq
        dxdt[qdot_index, :] = ddq

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx
            defects = nlp.cx(nlp.states.shape)

            # qdot = polynomial slope
            defects[nlp.states["q"].index] = slope_q * nlp.dt - dq * nlp.dt

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                defects[nlp.states["qdot"].index] = slope_qdot * nlp.dt - ddq * nlp.dt
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for stochastic torque driven free floating base dynamics."
                )

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def get_fatigable_tau(nlp, states: CX, controls: CX, fatigue: FatigueList) -> CX:
        """
        Apply the forward dynamics including (or not) the torque fatigue

        Parameters
        ----------
        nlp: NonLinearProgram
            The current phase
        states: MX | SX
            The states variable that may contains the tau and the tau fatigue variables
        controls: MX | SX
            The controls variable that may contains the tau
        fatigue: FatigueList
            The dynamics for the torque fatigue

        Returns
        -------
        The generalized accelerations
        """
        tau_var, tau_cx = (nlp.controls, controls) if "tau" in nlp.controls else (nlp.states, states)
        tau = nlp.get_var_from_states_or_controls("tau", states, controls)
        if fatigue is not None and "tau" in fatigue:
            tau_fatigue = fatigue["tau"]
            tau_suffix = fatigue["tau"].suffix

            # Only homogeneous state_only is implemented yet
            n_state_only = sum([t.models.state_only for t in tau_fatigue])
            if 0 < n_state_only < len(fatigue["tau"]):
                raise NotImplementedError("fatigue list without homogeneous state_only flag is not supported yet")
            apply_to_joint_dynamics = sum([t.models.apply_to_joint_dynamics for t in tau_fatigue])
            if 0 < n_state_only < len(fatigue["tau"]):
                raise NotImplementedError(
                    "fatigue list without homogeneous apply_to_joint_dynamics flag is not supported yet"
                )
            if apply_to_joint_dynamics != 0:
                raise NotImplementedError("apply_to_joint_dynamics is not implemented for joint torque")

            if not tau_fatigue[0].models.split_controls and "tau" in nlp.controls:
                pass
            elif tau_fatigue[0].models.state_only:
                tau = sum([DynamicsFunctions.get(tau_var[f"tau_{suffix}"], tau_cx) for suffix in tau_suffix])
            else:
                tau = nlp.cx()
                for i, t in enumerate(tau_fatigue):
                    tau_tp = nlp.cx(1, 1)
                    for suffix in tau_suffix:
                        model = t.models.models[suffix]
                        tau_tp += (
                            DynamicsFunctions.get(nlp.states[f"tau_{suffix}_{model.dynamics_suffix()}"], states)[i]
                            * model.scaling
                        )
                    tau = vertcat(tau, tau_tp)
        return tau

    @staticmethod
    def get_fatigue_states(
        states,
        nlp,
        fatigue,
        mus_activations,
    ):

        fatigue_states = None
        if fatigue is not None and "muscles" in fatigue:
            mus_fatigue = fatigue["muscles"]
            fatigue_name = mus_fatigue.suffix[0]

            # Sanity check
            n_state_only = sum([m.models.state_only for m in mus_fatigue])
            if 0 < n_state_only < len(fatigue["muscles"]):
                raise NotImplementedError(
                    f"{fatigue_name} list without homogeneous state_only flag is not supported yet"
                )
            apply_to_joint_dynamics = sum([m.models.apply_to_joint_dynamics for m in mus_fatigue])
            if 0 < apply_to_joint_dynamics < len(fatigue["muscles"]):
                raise NotImplementedError(
                    f"{fatigue_name} list without homogeneous apply_to_joint_dynamics flag is not supported yet"
                )

            dyn_suffix = mus_fatigue[0].models.models[fatigue_name].dynamics_suffix()
            fatigue_suffix = mus_fatigue[0].models.models[fatigue_name].fatigue_suffix()
            for m in mus_fatigue:
                for key in m.models.models:
                    if (
                        m.models.models[key].dynamics_suffix() != dyn_suffix
                        or m.models.models[key].fatigue_suffix() != fatigue_suffix
                    ):
                        raise ValueError(f"{fatigue_name} must be of all same types")

            if n_state_only == 0:
                mus_activations = DynamicsFunctions.get(nlp.states[f"muscles_{dyn_suffix}"], states)

            if apply_to_joint_dynamics > 0:
                fatigue_states = DynamicsFunctions.get(nlp.states[f"muscles_{fatigue_suffix}"], states)

        return fatigue_states, mus_activations

    @staticmethod
    def torque_activations_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_residual_torque: Bool,
    ):
        """
        Forward dynamics driven by joint torques activations.

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system
        with_residual_torque: bool
            If the dynamic should be added with residual torques

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        if (
            ContactType.SOFT_EXPLICIT in nlp.model.contact_types
            or ContactType.SOFT_IMPLICIT in nlp.model.contact_types
            or ContactType.RIGID_IMPLICIT in nlp.model.contact_types
        ):
            raise NotImplementedError(
                "soft contacts and implicit contacts not implemented yet with stochastic torque driven dynamics."
            )

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau_activation = DynamicsFunctions.get(nlp.controls["tau"], controls)

        tau = nlp.model.torque()(tau_activation, q, qdot, nlp.parameters.cx)
        if with_residual_torque:
            tau += DynamicsFunctions.get(nlp.controls["residual_tau"], controls)

        if nlp.model.nb_passive_joint_torques > 0:
            tau += nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.nb_ligaments > 0:
            tau += nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.friction_coefficients is not None:
            tau -= nlp.model.friction_coefficients @ qdot

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        forward_dynamics_contact_types = ContactType.get_equivalent_explicit_contacts(nlp.model.contact_types)
        ddq_fd = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, forward_dynamics_contact_types, external_forces)

        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[nlp.states["q"].index, 0] = dq
        dxdt[nlp.states["qdot"].index, 0] = ddq_fd

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            defects = nlp.cx(nlp.states.shape, 1)

            # qdot = polynomial slope
            defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - dq * nlp.dt

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, nlp.model.contact_types, external_forces)
                defects[nlp.states["qdot"].index, 0] = slope_qdot * nlp.dt - ddq * nlp.dt
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for torque activations driven dynamics."
                )

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def torque_derivative_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques derivatives

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.states["tau"], states)
        taudot = DynamicsFunctions.get(nlp.controls["taudot"], controls)

        if nlp.model.nb_passive_joint_torques > 0:
            tau += nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.nb_ligaments > 0:
            tau += nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.friction_coefficients is not None:
            tau -= nlp.model.friction_coefficients @ qdot

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )

        forward_dynamics_contact_types = ContactType.get_equivalent_explicit_contacts(nlp.model.contact_types)
        ddq_fd = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, forward_dynamics_contact_types, external_forces)

        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[nlp.states["q"].index, 0] = dq
        dxdt[nlp.states["qdot"].index, 0] = ddq_fd
        dxdt[nlp.states["tau"].index, 0] = taudot

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_tau = DynamicsFunctions.get(nlp.states_dot["tau"], nlp.states_dot.scaled.cx)
            defects = nlp.cx(nlp.states.shape, 1)

            # qdot = polynomial slope
            defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - dq * nlp.dt
            defects[nlp.states["tau"].index, 0] = slope_tau * nlp.dt - taudot * nlp.dt

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, nlp.model.contact_types, external_forces)
                defects[nlp.states["qdot"].index, 0] = slope_qdot * nlp.dt - ddq * nlp.dt

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:

                tau_id = DynamicsFunctions.inverse_dynamics(
                    nlp,
                    q=q,
                    qdot=qdot,
                    qddot=slope_qdot,
                    contact_types=nlp.model.contact_types,
                    external_forces=external_forces,
                )

                tau_defects = tau - tau_id
                defects[nlp.states["qdot"].index, 0] = tau_defects
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for torque derivative driven dynamics."
                )

            if ContactType.RIGID_IMPLICIT in nlp.model.contact_types:
                rigid_contact_defect = (
                    nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)
                    - nlp.algebraic_states["rigid_contact_forces"].cx
                )
                _, _, acceleration_constraint_func = HolonomicConstraintsFcn.rigid_contacts(nlp.model)
                contact_acceleration_defect = acceleration_constraint_func(q, qdot, slope_qdot, nlp.parameters.cx)
                defects = vertcat(defects, rigid_contact_defect, contact_acceleration_defect)

            if ContactType.SOFT_IMPLICIT in nlp.model.contact_types:
                soft_contact_defect = (
                    nlp.model.soft_contact_forces().expand()(q, qdot, nlp.parameters.cx)
                    - nlp.algebraic_states["soft_contact_forces"].cx
                )
                defects = vertcat(defects, soft_contact_defect)

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def forces_from_torque_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries: MX.sym,
        nlp,
    ) -> CX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym | SX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        tau = nlp.get_var_from_states_or_controls("tau", states, controls)
        if nlp.model.nb_passive_joint_torques > 0:
            tau += nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.nb_ligaments > 0:
            tau += nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )

        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

    @staticmethod
    def forces_from_torque_activation_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ) -> CX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym | SX.sym
            The contact forces that ensure no acceleration at these contact points
        """
        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        tau_activations = nlp.get_var_from_states_or_controls("tau", states, controls)
        tau = nlp.model.torque()(tau_activations, q, qdot, nlp.parameters.cx)
        if nlp.model.nb_passive_joint_torques > 0:
            tau += nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.nb_ligaments > 0:
            tau += nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

    @staticmethod
    def muscles_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_residual_torque: Bool = False,
        fatigue=None,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by muscle.

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system
        fatigue: FatigueDynamicsList
            To define fatigue elements
        with_residual_torque: bool
            If the dynamic should be added with residual torques

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        residual_tau = (
            DynamicsFunctions.get_fatigable_tau(nlp, states, controls, fatigue) if with_residual_torque else None
        )
        mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)
        fatigue_states, mus_activations = DynamicsFunctions.get_fatigue_states(states, nlp, fatigue, mus_activations)

        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        if nlp.model.nb_passive_joint_torques > 0:
            tau += nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.nb_ligaments > 0:
            tau += nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.friction_coefficients is not None:
            tau -= nlp.model.friction_coefficients @ qdot

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )

        forward_dynamics_contact_types = ContactType.get_equivalent_explicit_contacts(nlp.model.contact_types)
        ddq_fd = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, forward_dynamics_contact_types, external_forces)

        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[nlp.states["q"].index, 0] = dq
        dxdt[nlp.states["qdot"].index, 0] = ddq_fd

        has_excitation = True if "muscles" in nlp.states else False
        if has_excitation:
            mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
            dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
            dxdt[nlp.states["muscles"].index, 0] = dmus

        if fatigue is not None and "muscles" in fatigue:
            dxdt = fatigue["muscles"].dynamics(dxdt, nlp, states, controls)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):

            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx
            defects = nlp.cx(nlp.states.shape, 1)

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:

                dxdt_defects = nlp.cx(nlp.states.shape, 1)
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, nlp.model.contact_types, external_forces)
                dxdt_defects[nlp.states["q"].index, 0] = qdot
                dxdt_defects[nlp.states["qdot"].index, 0] = ddq

                slopes = nlp.cx(nlp.states.shape, 1)
                slopes[nlp.states["q"].index, 0] = slope_q
                slopes[nlp.states["qdot"].index, 0] = slope_qdot

                has_excitation = True if "muscles" in nlp.states else False
                if has_excitation:
                    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                    dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
                    dxdt_defects[nlp.states["muscles"].index, 0] = dmus
                    slope_mus = nlp.states_dot["muscles"].cx
                    slopes[nlp.states["qdot"].index, 0] = slope_mus

                if fatigue is not None and "muscles" in fatigue:
                    dxdt_defects = fatigue["muscles"].dynamics(dxdt_defects, nlp, states, controls)
                    state_keys = nlp.states.keys()
                    if state_keys[0] != "q" or state_keys[1] != "qdot":
                        raise NotImplementedError(
                            "The accession of muscles fatigue states is not implemented generically yet."
                        )

                    slopes_fatigue = nlp.cx()
                    fatigue_indices = []
                    for key in state_keys[2:]:
                        if not key.startswith("muscles_"):
                            raise NotImplementedError(
                                "The accession of muscles fatigue states is not implemented generically yet."
                            )
                        slopes_fatigue = vertcat(slopes_fatigue, nlp.states_dot[key].cx)
                        fatigue_indices += list(nlp.states[key].index)

                    slopes[fatigue_indices, 0] = slopes_fatigue

                defects = slopes * nlp.dt - dxdt_defects * nlp.dt

            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                if fatigue is not None:
                    raise NotImplementedError("Fatigue is not implemented yet with inverse dynamics defects.")

                defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - dq * nlp.dt

                tau_id = DynamicsFunctions.inverse_dynamics(
                    nlp,
                    q=q,
                    qdot=qdot,
                    qddot=slope_qdot,
                    contact_types=nlp.model.contact_types,
                    external_forces=external_forces,
                )
                tau_defects = tau - tau_id
                defects[nlp.states["qdot"].index, 0] = tau_defects

                if has_excitation:
                    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                    dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
                    slope_mus = nlp.states_dot["muscles"].cx
                    defects[nlp.states["muscles"].index, 0] = slope_mus * nlp.dt - dmus * nlp.dt

            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for muscles driven dynamics."
                )

            # We append the defects with the algebraic states implicit constraints
            if ContactType.RIGID_IMPLICIT in nlp.model.contact_types:
                rigid_contact_defect = (
                    nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)
                    - nlp.algebraic_states["rigid_contact_forces"].cx
                )
                _, _, acceleration_constraint_func = HolonomicConstraintsFcn.rigid_contacts(nlp.model)
                contact_acceleration_defect = acceleration_constraint_func(q, qdot, slope_qdot, nlp.parameters.cx)
                defects = vertcat(defects, rigid_contact_defect, contact_acceleration_defect)

            if ContactType.SOFT_IMPLICIT in nlp.model.contact_types:
                soft_contact_defect = (
                    nlp.model.soft_contact_forces().expand()(q, qdot, nlp.parameters.cx)
                    - nlp.algebraic_states["soft_contact_forces"].cx
                )
                defects = vertcat(defects, soft_contact_defect)

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def forces_from_muscle_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ) -> CX:
        """
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym | SX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        residual_tau = nlp.get_var_from_states_or_controls("tau", states, controls) if "tau" in nlp.controls else None
        mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        if nlp.model.nb_passive_joint_torques > 0:
            tau += nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        if nlp.model.nb_ligaments > 0:
            tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

    @staticmethod
    def joints_acceleration_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joints accelerations of a free floating body.

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym | SX.sym
            The derivative of states
        """

        if len(nlp.model.contact_types) > 0:
            raise RuntimeError("Joints acceleration driven dynamics cannot be used with contacts by definition.")

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        if external_forces.shape != (0, 1):
            raise RuntimeError("Joints acceleration driven dynamics cannot be used with external forces by definition.")

        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        qddot_joints = nlp.get_var_from_states_or_controls("qddot", states, controls)

        qddot_root = nlp.model.forward_dynamics_free_floating_base()(q, qdot, qddot_joints, nlp.parameters.cx)
        qddot_reordered = nlp.model.reorder_qddot_root_joints(qddot_root, qddot_joints)

        qdot_mapped = nlp.variable_mappings["qdot"].to_first.map(qdot)
        qddot_mapped = nlp.variable_mappings["qdot"].to_first.map(qddot_reordered)

        dxdt = nlp.cx(nlp.states.shape, 1)
        dxdt[nlp.states["q"].index, 0] = qdot_mapped
        dxdt[nlp.states["qdot"].index, 0] = qddot_mapped

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = nlp.states_dot["q"].cx
            slope_qdot = nlp.states_dot["qdot"].cx
            defects = nlp.cx(nlp.states.shape, 1)

            # qdot = polynomial slope
            defects[nlp.states["q"].index, 0] = slope_q * nlp.dt - qdot_mapped * nlp.dt

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                defects[nlp.states["qdot"].index, 0] = slope_qdot * nlp.dt - qddot_mapped * nlp.dt

            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for joints acceleration driven dynamics."
                )

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def get(var: OptimizationVariable, cx: CX):
        """
        Main accessor to a variable in states or controls (cx)

        Parameters
        ----------
        var: OptimizationVariable
            The variable from nlp.states["name"] or nlp.controls["name"]
        cx: MX | SX
            The actual SX or MX variables

        Returns
        -------
        The sliced values
        """

        return var.mapping.to_second.map(cx[var.index, :])

    @staticmethod
    def compute_qdot(nlp, q: CX, qdot: CX):
        """
        Easy accessor to derivative of q

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"

        Returns
        -------
        The derivative of q
        """

        if "q" in nlp.states:
            mapping = nlp.states["q"].mapping
        elif "q_roots" and "q_joints" in nlp.states:
            mapping = BiMapping(
                to_first=list(nlp.states["q_roots"].mapping.to_first.map_idx)
                + [i + nlp.model.nb_root for i in nlp.states["q_joints"].mapping.to_first.map_idx],
                to_second=list(nlp.states["q_roots"].mapping.to_second.map_idx)
                + [i + nlp.model.nb_root for i in nlp.states["q_joints"].mapping.to_second.map_idx],
            )
        elif q in nlp.controls:
            mapping = nlp.controls["q"].mapping
        else:
            raise RuntimeError("Your q key combination was not found in states or controls")
        return mapping.to_first.map(nlp.model.reshape_qdot()(q, qdot, nlp.parameters.cx))

    @staticmethod
    def get_external_forces_from_contacts(nlp, q, qdot, contact_types, external_forces: MX | SX):

        external_forces = nlp.cx() if external_forces is None else external_forces
        if ContactType.RIGID_IMPLICIT in contact_types:
            if external_forces.shape[0] != 0:
                raise NotImplementedError("ContactType.RIGID_IMPLICIT cannot be used with external forces yet")
            if "rigid_contact_forces" in nlp.states:
                contact_forces = nlp.states["rigid_contact_forces"].cx
            elif "rigid_contact_forces" in nlp.algebraic_states:
                contact_forces = nlp.algebraic_states["rigid_contact_forces"].cx
            else:
                raise RuntimeError("The key 'rigid_contact_forces' was not found in states or algebraic_states")
            external_forces = vertcat(
                external_forces,
                nlp.model.map_rigid_contact_forces_to_global_forces(contact_forces, q, nlp.parameters.cx),
            )

        if ContactType.SOFT_EXPLICIT in contact_types:
            contact_forces = nlp.model.soft_contact_forces().expand()(q, qdot, nlp.parameters.cx)
            external_forces = vertcat(
                external_forces, nlp.model.map_soft_contact_forces_to_global_forces(contact_forces)
            )

        if ContactType.SOFT_IMPLICIT in contact_types:
            contact_forces = nlp.algebraic_states["soft_contact_forces"].cx
            external_forces = vertcat(
                external_forces, nlp.model.map_soft_contact_forces_to_global_forces(contact_forces)
            )

        external_forces = [] if external_forces.shape == (0, 1) else external_forces

        return external_forces

    @staticmethod
    def forward_dynamics(
        nlp,
        q: CX,
        qdot: CX,
        tau: CX,
        contact_types: list[ContactType] | tuple[ContactType],
        external_forces: AnyListOptional = None,
    ):
        """
        Easy accessor to derivative of qdot

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        tau: MX | SX
            The value of tau from "get"
        contact_types: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        external_forces: MX | SX
            The external forces
        Returns
        -------
        The derivative of qdot
        """
        # Get the mapping of the output
        if "qdot" in nlp.states:
            qdot_var_mapping = nlp.states["qdot"].mapping.to_first
        elif "qdot" in nlp.controls:
            qdot_var_mapping = nlp.controls["qdot"].mapping.to_first
        else:
            qdot_var_mapping = BiMapping([i for i in range(qdot.shape[0])], [i for i in range(qdot.shape[0])]).to_first

        external_forces = DynamicsFunctions.get_external_forces_from_contacts(
            nlp, q, qdot, contact_types, external_forces
        )
        with_contact = ContactType.RIGID_EXPLICIT in contact_types

        qddot = nlp.model.forward_dynamics(with_contact=with_contact)(
            q,
            qdot,
            tau,
            external_forces,
            nlp.parameters.cx,
        )
        return qdot_var_mapping.map(qddot)

    @staticmethod
    def inverse_dynamics(
        nlp,
        q: CX,
        qdot: CX,
        qddot: CX,
        contact_types: list[ContactType] | tuple[ContactType],
        external_forces: MX = None,
    ):
        """
        Easy accessor to torques from inverse dynamics

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        qddot: MX | SX
            The value of qddot from "get"
        contact_types: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        external_forces: MX
            The external forces

        Returns
        -------
        Torques in tau
        """
        # TODO: Charbie -> Check if the mapping can be applied or not.
        # Old comment: we ignore on purpose the mapping to keep zeros in the defects of the dynamic.

        # Get the mapping of the output
        if "tau" in nlp.states:
            tau_var_mapping = nlp.states["tau"].mapping.to_first
        elif "tau" in nlp.controls:
            tau_var_mapping = nlp.controls["tau"].mapping.to_first
        else:
            raise RuntimeError("The key 'tau' was not found in states or controls")

        if ContactType.RIGID_EXPLICIT in contact_types:
            raise NotImplementedError("Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet")

        external_forces = DynamicsFunctions.get_external_forces_from_contacts(
            nlp, q, qdot, contact_types, external_forces
        )

        tau = nlp.model.inverse_dynamics(with_contact=False)(q, qdot, qddot, external_forces, nlp.parameters.cx)

        return tau_var_mapping.map(tau)

    @staticmethod
    def compute_muscle_dot(nlp, muscle_excitations: CX, muscle_activations: CX):
        """
        Easy accessor to derivative of muscle activations

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        muscle_excitations: MX | SX
            The value of muscle_excitations from "get"
        muscle_activations: MX | SX
            The value of muscle_activations from "get"

        Returns
        -------
        The derivative of muscle activations
        """

        return nlp.model.muscle_activation_dot()(muscle_excitations, muscle_activations, nlp.parameters.cx)

    @staticmethod
    def compute_tau_from_muscle(
        nlp,
        q: CX,
        qdot: CX,
        muscle_activations: CX,
        fatigue_states: CXOptional = None,
    ):
        """
        Easy accessor to tau computed from muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        muscle_activations: MX | SX
            The value of muscle_activations from "get"
        fatigue_states: MX | SX
            The states of fatigue

        Returns
        -------
        The generalized forces computed from the muscles
        """

        activations = type(q)()
        for k in range(len(nlp.controls["muscles"])):
            if fatigue_states is not None:
                activations = vertcat(activations, muscle_activations[k] * (1 - fatigue_states[k]))
            else:
                activations = vertcat(activations, muscle_activations[k])
        return nlp.model.muscle_joint_torque()(activations, q, qdot, nlp.parameters.cx)

    @staticmethod
    def holonomic_torque_driven(
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
        q_v_init = DM.zeros(nlp.model.nb_dependent_joints)

        qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)
        dxdt = vertcat(qdot_u, qddot_u)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot_u"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot_u"], nlp.states_dot.scaled.cx)
            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)
                derivative = vertcat(qdot_u, qddot_u)
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt
            else:
                raise NotImplementedError(
                    f"The defect type {nlp.dynamics_type.ode_solver.defects_type} is not implemented yet for holonomic torque driven dynamics."
                )

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)
