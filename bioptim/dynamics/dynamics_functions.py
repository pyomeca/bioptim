from casadi import horzcat, vertcat, MX, SX

from ..misc.enums import RigidBodyDynamics, DefectType
from .fatigue.fatigue_dynamics import FatigueList
from ..optimization.optimization_variable import OptimizationVariable
from ..optimization.non_linear_program import NonLinearProgram
from .dynamics_evaluation import DynamicsEvaluation
from ..models.protocols.stochastic_biomodel import StochasticBioModel
from ..misc.mapping import BiMapping


class DynamicsFunctions:
    """
    Implementation of all the dynamic functions

    Methods
    -------
    custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Interface to custom dynamic function provided by the user
    torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool)
        Forward dynamics driven by joint torques, optional external forces can be declared.
    torque_activations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact) -> MX:
        Forward dynamics driven by joint torques activations.
    torque_derivative_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        Forward dynamics driven by joint torques, optional external forces can be declared.
    forces_from_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        Contact forces of a forward dynamics driven by joint torques with contact constraints.
    muscles_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        Forward dynamics driven by muscle.
    forces_from_muscle_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.
    get(var: OptimizationVariable, cx: MX | SX):
        Main accessor to a variable in states or controls (cx)
    apply_parameters(parameters: MX.sym, nlp: NonLinearProgram)
        Apply the parameter variables to the model. This should be called before calling the dynamics
    reshape_qdot(nlp: NonLinearProgram, q: MX | SX, qdot: MX | SX):
        Easy accessor to derivative of q
    forward_dynamics(nlp: NonLinearProgram, q: MX | SX, qdot: MX | SX, tau: MX | SX, with_contact: bool):
        Easy accessor to derivative of qdot
    compute_muscle_dot(nlp: NonLinearProgram, muscle_excitations: MX | SX):
        Easy accessor to derivative of muscle activations
    compute_tau_from_muscle(nlp: NonLinearProgram, q: MX | SX, qdot: MX | SX, muscle_activations: MX | SX):
        Easy accessor to tau computed from muscles
    """

    @staticmethod
    def custom(
        time: MX.sym, states: MX.sym, controls: MX.sym, parameters: MX.sym, stochastic_variables: MX.sym, nlp
    ) -> DynamicsEvaluation:
        """
        Interface to custom dynamic function provided by the user.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        MX.sym
            The defects of the implicit dynamics
        """

        return nlp.dynamics_type.dynamic_function(time, states, controls, parameters, stochastic_variables, nlp)

    @staticmethod
    def torque_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_contact: bool,
        with_passive_torque: bool,
        with_ligament: bool,
        with_friction: bool,
        rigidbody_dynamics: RigidBodyDynamics,
        fatigue: FatigueList,
        external_forces: list = None,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques, optional external forces can be declared.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with friction should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        fatigue : FatigueList
            A list of fatigue elements
        external_forces: list[Any]
            The external forces

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        tau = DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue)
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau
        tau = tau + nlp.model.friction_coefficients @ qdot if with_friction else tau

        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
        ):
            dxdt = MX(nlp.states.shape, 1)
            dxdt[nlp.states["q"].index, :] = dq
            dxdt[nlp.states["qdot"].index, :] = DynamicsFunctions.get(nlp.controls["qddot"], controls)
        elif (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
        ):
            dxdt = MX(nlp.states.shape, 1)
            dxdt[nlp.states["q"].index, :] = dq
            qddot = DynamicsFunctions.get(nlp.states["qddot"], states)
            dxdt[nlp.states["qdot"].index, :] = qddot
            dxdt[nlp.states["qddot"].index, :] = DynamicsFunctions.get(nlp.controls["qdddot"], controls)
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
            dxdt = MX(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot"].index, :] = ddq

        if fatigue is not None and "tau" in fatigue:
            dxdt = fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        defects = None
        # TODO: contacts and fatigue to be handled with implicit dynamics
        if rigidbody_dynamics is not RigidBodyDynamics.ODE or (
            rigidbody_dynamics is RigidBodyDynamics.ODE and nlp.ode_solver.defects_type == DefectType.IMPLICIT
        ):
            if not with_contact and fatigue is None:
                qddot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.mx_reduced)
                tau_id = DynamicsFunctions.inverse_dynamics(nlp, q, qdot, qddot, with_contact)
                defects = MX(dq.shape[0] + tau_id.shape[0], tau_id.shape[1])

                dq_defects = []
                for _ in range(tau_id.shape[1]):
                    dq_defects.append(
                        dq
                        - DynamicsFunctions.compute_qdot(
                            nlp,
                            q,
                            DynamicsFunctions.get(nlp.states_dot.scaled["qdot"], nlp.states_dot.scaled.mx_reduced),
                        )
                    )
                defects[: dq.shape[0], :] = horzcat(*dq_defects)
                # We modified on purpose the size of the tau to keep the zero in the defects in order to respect the dynamics
                defects[dq.shape[0] :, :] = tau - tau_id

        return DynamicsEvaluation(dxdt, defects)

    @staticmethod
    def torque_driven_free_floating_base(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_contact: bool,
        with_passive_torque: bool,
        with_ligament: bool,
        with_friction: bool,
        external_forces: list = None,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques without actuation of the free floating base, optional external forces can be declared.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with friction should be used
        external_forces: list[Any]
            The external forces

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
        n_q = q_full.shape[0]

        tau_joints = (
            tau_joints + nlp.model.passive_joint_torque(q_full, qdot_full) if with_passive_torque else tau_joints
        )
        tau_joints = tau_joints + nlp.model.ligament_joint_torque(q_full, qdot_full) if with_ligament else tau_joints
        tau_joints = tau_joints + nlp.model.friction_coefficients @ qdot_full if with_friction else tau_joints

        tau_full = vertcat(MX.zeros(nlp.model.nb_root), tau_joints)

        ddq = DynamicsFunctions.forward_dynamics(nlp, q_full, qdot_full, tau_full, with_contact, external_forces)
        dxdt = MX(nlp.states.shape, ddq.shape[1])
        dxdt[:n_q, :] = horzcat(*[qdot_full for _ in range(ddq.shape[1])])
        dxdt[n_q:, :] = ddq

        defects = None

        return DynamicsEvaluation(dxdt, defects)

    @staticmethod
    def stochastic_torque_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_contact: bool,
        with_friction: bool,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics subject to motor and sensory noise driven by joint torques, optional external forces can be declared.

        Parameters
        ----------
        time: MX.sym
            The time
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic variables of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used
        with_friction: bool
            If the dynamic with friction should be used

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        ref = DynamicsFunctions.get(nlp.stochastic_variables["ref"], stochastic_variables)
        k = DynamicsFunctions.get(nlp.stochastic_variables["k"], stochastic_variables)
        k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)

        sensory_input = nlp.model.sensory_reference(states, controls, parameters, stochastic_variables, nlp)

        mapped_motor_noise = nlp.model.motor_noise_sym
        mapped_sensory_feedback_torque = nlp.model.compute_torques_from_noise_and_feedback(k_matrix, sensory_input, ref)
        if "tau" in nlp.model.motor_noise_mapping.keys():
            mapped_motor_noise = nlp.model.motor_noise_mapping["tau"].to_second.map(nlp.model.motor_noise_sym)
            mapped_sensory_feedback_torque = nlp.model.motor_noise_mapping["tau"].to_second.map(
                mapped_sensory_feedback_torque
            )
        tau += mapped_motor_noise + mapped_sensory_feedback_torque
        tau = tau + nlp.model.friction_coefficients @ qdot if with_friction else tau

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)
        dxdt = MX(nlp.states.shape, ddq.shape[1])
        dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
        dxdt[nlp.states["qdot"].index, :] = ddq

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def __get_fatigable_tau(nlp: NonLinearProgram, states: MX, controls: MX, fatigue: FatigueList) -> MX:
        """
        Apply the forward dynamics including (or not) the torque fatigue

        Parameters
        ----------
        nlp: NonLinearProgram
            The current phase
        states: MX
            The states variable that may contains the tau and the tau fatigue variables
        controls: MX
            The controls variable that may contains the tau
        fatigue: FatigueList
            The dynamics for the torque fatigue

        Returns
        -------
        The generalized accelerations
        """

        tau_var, tau_mx = (nlp.controls, controls) if "tau" in nlp.controls else (nlp.states, states)
        tau = DynamicsFunctions.get(tau_var["tau"], tau_mx)
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
                tau = sum([DynamicsFunctions.get(tau_var[f"tau_{suffix}"], tau_mx) for suffix in tau_suffix])
            else:
                tau = MX()
                for i, t in enumerate(tau_fatigue):
                    tau_tp = MX(1, 1)
                    for suffix in tau_suffix:
                        model = t.models.models[suffix]
                        tau_tp += (
                            DynamicsFunctions.get(nlp.states[f"tau_{suffix}_{model.dynamics_suffix()}"], states)[i]
                            * model.scaling
                        )
                    tau = vertcat(tau, tau_tp)
        return tau

    @staticmethod
    def torque_activations_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_contact: bool,
        with_passive_torque: bool,
        with_residual_torque: bool,
        with_ligament: bool,
        external_forces: list = None,
    ):
        """
        Forward dynamics driven by joint torques activations.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_residual_torque: bool
            If the dynamic should be added with residual torques
        with_ligament: bool
            If the dynamic with ligament should be used
        external_forces: list[Any]
            The external forces

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau_activation = DynamicsFunctions.get(nlp.controls["tau"], controls)

        tau = nlp.model.torque(tau_activation, q, qdot)
        if with_passive_torque:
            tau += nlp.model.passive_joint_torque(q, qdot)
        if with_residual_torque:
            tau += DynamicsFunctions.get(nlp.controls["residual_tau"], controls)
        if with_ligament:
            tau += nlp.model.ligament_joint_torque(q, qdot)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)

        dq = horzcat(*[dq for _ in range(ddq.shape[1])])

        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)

    @staticmethod
    def torque_derivative_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        rigidbody_dynamics: RigidBodyDynamics,
        with_contact: bool,
        with_passive_torque: bool,
        with_ligament: bool,
        external_forces: list = None,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques, optional external forces can be declared.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        external_forces: list[Any]
            The external forces

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)

        tau = DynamicsFunctions.get(nlp.states["tau"], states)
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dtau = DynamicsFunctions.get(nlp.controls["taudot"], controls)

        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
        ):
            ddq = DynamicsFunctions.get(nlp.states["qddot"], states)
            dddq = DynamicsFunctions.get(nlp.controls["qdddot"], controls)

            dxdt = MX(nlp.states.shape, 1)
            dxdt[nlp.states["q"].index, :] = dq
            dxdt[nlp.states["qdot"].index, :] = ddq
            dxdt[nlp.states["qddot"].index, :] = dddq
            dxdt[nlp.states["tau"].index, :] = dtau
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
            dxdt = MX(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot"].index, :] = ddq
            dxdt[nlp.states["tau"].index, :] = horzcat(*[dtau for _ in range(ddq.shape[1])])

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def forces_from_torque_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        external_forces: list = None,
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic variables of the system
        nlp: NonLinearProgram
            The definition of the system
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        external_forces: list[Any]
            The external forces

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        def get_var_from_states_or_controls(key: str, nlp: NonLinearProgram, states: MX.sym, controls: MX.sym):
            if key in nlp.states:
                out_nlp, out_var = (nlp.states[key], states)
                out = DynamicsFunctions.get(out_nlp, out_var)
            elif key in nlp.controls:
                out_nlp, out_var = (nlp.controls[key], controls)
                out = DynamicsFunctions.get(out_nlp, out_var)
            elif f"{key}_roots" in nlp.states and f"{key}_joints" in nlp.states:
                out_roots_nlp, out_roots_var = (nlp.states[f"{key}_roots"], states)
                out_roots = DynamicsFunctions.get(out_roots_nlp, out_roots_var)
                out_joints_nlp, out_joints_var = (nlp.states[f"{key}_joints"], states)
                out_joints = DynamicsFunctions.get(out_joints_nlp, out_joints_var)
                out = vertcat(out_roots, out_joints)
            elif f"{key}_joints" in nlp.controls:
                out_joints_nlp, out_joints_var = (nlp.controls[f"{key}_joints"], controls)
                out = DynamicsFunctions.get(out_joints_nlp, out_joints_var)
            else:
                raise RuntimeError(f"{key} not found in states or controls")
            return out

        q = get_var_from_states_or_controls("q", nlp, states, controls)
        qdot = get_var_from_states_or_controls("qdot", nlp, states, controls)
        tau = get_var_from_states_or_controls("tau", nlp, states, controls)
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        return nlp.model.contact_forces(q, qdot, tau, external_forces)

    @staticmethod
    def forces_from_torque_activation_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        external_forces: list = None,
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        external_forces: list[Any]
            The external forces

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q_nlp, q_var = (nlp.states["q"], states) if "q" in nlp.states else (nlp.controls["q"], controls)
        qdot_nlp, qdot_var = (nlp.states["qdot"], states) if "qdot" in nlp.states else (nlp.controls["qdot"], controls)
        tau_nlp, tau_var = (nlp.states["tau"], states) if "tau" in nlp.states else (nlp.controls["tau"], controls)
        q = DynamicsFunctions.get(q_nlp, q_var)
        qdot = DynamicsFunctions.get(qdot_nlp, qdot_var)
        tau_activations = DynamicsFunctions.get(tau_nlp, tau_var)
        tau = nlp.model.torque(tau_activations, q, qdot)
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        return nlp.model.contact_forces(q, qdot, tau, external_forces)

    @staticmethod
    def muscles_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_contact: bool,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        with_residual_torque: bool = False,
        fatigue=None,
        external_forces: list = None,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by muscle.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        fatigue: FatigueDynamicsList
            To define fatigue elements
        with_residual_torque: bool
            If the dynamic should be added with residual torques
        external_forces: list[Any]
            The external forces

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        residual_tau = (
            DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue) if with_residual_torque else None
        )

        mus_act_nlp, mus_act = (nlp.states, states) if "muscles" in nlp.states else (nlp.controls, controls)
        mus_activations = DynamicsFunctions.get(mus_act_nlp["muscles"], mus_act)
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
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            ddq = DynamicsFunctions.get(nlp.controls["qddot"], controls)
            dxdt = MX(nlp.states.shape, 1)
            dxdt[nlp.states["q"].index, :] = dq
            dxdt[nlp.states["qdot"].index, :] = DynamicsFunctions.get(nlp.controls["qddot"], controls)
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
            dxdt = MX(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot"].index, :] = ddq

        has_excitation = True if "muscles" in nlp.states else False
        if has_excitation:
            mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
            dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations)
            dxdt[nlp.states["muscles"].index, :] = horzcat(*[dmus for _ in range(ddq.shape[1])])

        if fatigue is not None and "muscles" in fatigue:
            dxdt = fatigue["muscles"].dynamics(dxdt, nlp, states, controls)

        defects = None
        # TODO: contacts and fatigue to be handled with implicit dynamics
        if rigidbody_dynamics is not RigidBodyDynamics.ODE or (
            rigidbody_dynamics is RigidBodyDynamics.ODE and nlp.ode_solver.defects_type == DefectType.IMPLICIT
        ):
            if not with_contact and fatigue is None:
                qddot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.mx_reduced)
                tau_id = DynamicsFunctions.inverse_dynamics(nlp, q, qdot, qddot, with_contact, external_forces)
                defects = MX(dq.shape[0] + tau_id.shape[0], tau_id.shape[1])

                dq_defects = []
                for _ in range(tau_id.shape[1]):
                    dq_defects.append(
                        dq
                        - DynamicsFunctions.compute_qdot(
                            nlp,
                            q,
                            DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.mx_reduced),
                        )
                    )
                defects[: dq.shape[0], :] = horzcat(*dq_defects)
                defects[dq.shape[0] :, :] = tau - tau_id

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def forces_from_muscle_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        external_forces: list = None,
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        external_forces: list[Any]
            The external forces
        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if "tau" in nlp.controls else None

        mus_act_nlp, mus_act = (nlp.states, states) if "muscles" in nlp.states else (nlp.controls, controls)
        mus_activations = DynamicsFunctions.get(mus_act_nlp["muscles"], mus_act)
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        return nlp.model.contact_forces(q, qdot, tau, external_forces)

    @staticmethod
    def joints_acceleration_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joints accelerations of a free floating body.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        stochastic_variables: MX.sym
            The stochastic_variables of the system
        nlp: NonLinearProgram
            The definition of the system
        rigidbody_dynamics: RigidBodyDynamics
            which rigid body dynamics to use

        Returns
        ----------
        MX.sym
            The derivative of states
        """
        if rigidbody_dynamics != RigidBodyDynamics.ODE:
            raise NotImplementedError("Implicit dynamics not implemented yet.")

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        qddot_joints = DynamicsFunctions.get(nlp.controls["qddot_joints"], controls)

        qddot_root = nlp.model.forward_dynamics_free_floating_base(q, qdot, qddot_joints)
        qddot_reordered = nlp.model.reorder_qddot_root_joints(qddot_root, qddot_joints)

        qdot_mapped = nlp.variable_mappings["qdot"].to_first.map(qdot)
        qddot_mapped = nlp.variable_mappings["qdot"].to_first.map(qddot_reordered)
        qddot_root_mapped = nlp.variable_mappings["qddot_roots"].to_first.map(qddot_root)
        qddot_joints_mapped = nlp.variable_mappings["qddot_joints"].to_first.map(qddot_joints)

        # defects
        defects = None
        if rigidbody_dynamics is not RigidBodyDynamics.ODE or (
            rigidbody_dynamics is RigidBodyDynamics.ODE and nlp.ode_solver.defects_type == DefectType.IMPLICIT
        ):
            qddot_root_defects = DynamicsFunctions.get(nlp.states_dot["qddot_roots"], nlp.states_dot.mx_reduced)
            qddot_defects_reordered = nlp.model.reorder_qddot_root_joints(qddot_root_defects, qddot_joints)

            floating_base_constraint = nlp.model.inverse_dynamics(q, qdot, qddot_defects_reordered)[: nlp.model.nb_root]

            defects = MX(qdot_mapped.shape[0] + qddot_root_mapped.shape[0] + qddot_joints_mapped.shape[0], 1)

            defects[: qdot_mapped.shape[0], :] = qdot_mapped - nlp.variable_mappings["qdot"].to_first.map(
                DynamicsFunctions.compute_qdot(
                    nlp, q, DynamicsFunctions.get((nlp.states_dot["qdot"]), nlp.states_dot.mx_reduced)
                )
            )

            defects[
                qdot_mapped.shape[0] : (qdot_mapped.shape[0] + qddot_root_mapped.shape[0]), :
            ] = floating_base_constraint
            defects[
                (qdot_mapped.shape[0] + qddot_root_mapped.shape[0]) :, :
            ] = qddot_joints_mapped - nlp.variable_mappings["qddot_joints"].to_first.map(
                DynamicsFunctions.get(nlp.states_dot["qddot_joints"], nlp.states_dot.mx_reduced)
            )

        return DynamicsEvaluation(dxdt=vertcat(qdot_mapped, qddot_mapped), defects=defects)

    @staticmethod
    def get(var: OptimizationVariable, cx: MX | SX):
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
    def apply_parameters(parameters: MX.sym, nlp):
        """
        Apply the parameter variables to the model. This should be called before calling the dynamics

        Parameters
        ----------
        parameters: MX.sym
            The state of the system
        nlp: NonLinearProgram
            The definition of the system
        """

        for param in nlp.parameters:
            # Call the pre dynamics function
            if param.function[0]:
                param.function[0](nlp.model, parameters[param.index], **param.params)

    @staticmethod
    def compute_qdot(nlp: NonLinearProgram, q: MX | SX, qdot: MX | SX):
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

        q_nlp = nlp.states["q"] if "q" in nlp.states else nlp.controls["q"]
        return q_nlp.mapping.to_first.map(nlp.model.reshape_qdot(q, qdot))

    @staticmethod
    def forward_dynamics(
        nlp: NonLinearProgram,
        q: MX | SX,
        qdot: MX | SX,
        tau: MX | SX,
        with_contact: bool,
        external_forces: list = None,
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
        with_contact: bool
            If the dynamics with contact should be used
        external_forces: list[]
            The external forces
        Returns
        -------
        The derivative of qdot
        """
        if "qdot" in nlp.states:
            qdot_var_mapping = nlp.states["qdot"].mapping.to_first
        elif "qdot" in nlp.controls:
            qdot_var_mapping = nlp.controls["qdot"].mapping.to_first
        else:
            qdot_var_mapping = BiMapping([i for i in range(qdot.shape[0])], [i for i in range(qdot.shape[0])]).to_first

        if external_forces is None:
            if with_contact:
                qddot = nlp.model.constrained_forward_dynamics(q, qdot, tau)
            else:
                qddot = nlp.model.forward_dynamics(q, qdot, tau)

            return qdot_var_mapping.map(qddot)
        else:
            dxdt = MX(len(qdot_var_mapping), nlp.ns)
            # Todo: Should be added to pass f_ext in controls (as a symoblic value)
            #  this would avoid to create multiple equations of motions per node
            for i, f_ext in enumerate(external_forces):
                if with_contact:
                    qddot = nlp.model.constrained_forward_dynamics(q, qdot, tau, f_ext)
                else:
                    qddot = nlp.model.forward_dynamics(q, qdot, tau, f_ext)
                dxdt[:, i] = qdot_var_mapping.map(qddot)
            return dxdt

    @staticmethod
    def inverse_dynamics(
        nlp: NonLinearProgram,
        q: MX | SX,
        qdot: MX | SX,
        qddot: MX | SX,
        with_contact: bool,
        external_forces: list = None,
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
        with_contact: bool
            If the dynamics with contact should be used
        external_forces: list[]
            The external forces

        Returns
        -------
        Torques in tau
        """

        if nlp.external_forces is None:
            tau = nlp.model.inverse_dynamics(q, qdot, qddot)
        else:
            if "tau" in nlp.states:
                tau_shape = nlp.states["tau"].mx.shape[0]
            elif "tau" in nlp.controls:
                tau_shape = nlp.controls["tau"].mx.shape[0]
            else:
                tau_shape = nlp.model.nb_tau
            tau = MX(tau_shape, nlp.ns)
            # Todo: Should be added to pass f_ext in controls (as a symoblic value)
            #  this would avoid to create multiple equations of motions per node
            for i, f_ext in enumerate(nlp.external_forces):
                tau[:, i] = nlp.model.inverse_dynamics(q, qdot, qddot, f_ext)
        return tau  # We ignore on purpose the mapping to keep zeros in the defects of the dynamic.

    @staticmethod
    def compute_muscle_dot(nlp: NonLinearProgram, muscle_excitations: MX | SX):
        """
        Easy accessor to derivative of muscle activations

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        muscle_excitations: MX | SX
            The value of muscle_excitations from "get"

        Returns
        -------
        The derivative of muscle activations
        """

        return nlp.model.muscle_activation_dot(muscle_excitations)

    @staticmethod
    def compute_tau_from_muscle(
        nlp: NonLinearProgram,
        q: MX | SX,
        qdot: MX | SX,
        muscle_activations: MX | SX,
        fatigue_states: MX | SX = None,
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

        activations = []
        for k in range(len(nlp.controls["muscles"])):
            if fatigue_states is not None:
                activations.append(muscle_activations[k] * (1 - fatigue_states[k]))
            else:
                activations.append(muscle_activations[k])
        return nlp.model.muscle_joint_torque(activations, q, qdot)

    @staticmethod
    def holonomic_torque_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp: NonLinearProgram,
        external_forces: list = None,
    ) -> DynamicsEvaluation:
        """
        The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, s)

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters acting on the system
        stochastic_variables: MX.sym
            The stochastic variables of the system
        nlp: NonLinearProgram
            A reference to the phase
        external_forces: list[Any]
            The external forces

        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
        qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
        qddot_u = nlp.model.partitioned_forward_dynamics(q_u, qdot_u, tau, external_forces=external_forces)

        return DynamicsEvaluation(dxdt=vertcat(qdot_u, qddot_u), defects=None)
