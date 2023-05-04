from casadi import horzcat, vertcat, MX, SX, Function

from ..misc.enums import RigidBodyDynamics
from .fatigue.fatigue_dynamics import FatigueList
from ..optimization.optimization_variable import OptimizationVariable
from ..optimization.non_linear_program import NonLinearProgram
from .dynamics_evaluation import DynamicsEvaluation


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
    def custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> DynamicsEvaluation:
        """
        Interface to custom dynamic function provided by the user.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        MX.sym
            The defects of the implicit dynamics
        """

        return nlp.dynamics_type.dynamic_function(states, controls, parameters, nlp)

    @staticmethod
    def torque_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        with_contact: bool,
        with_passive_torque: bool,
        with_ligament: bool,
        rigidbody_dynamics: RigidBodyDynamics,
        fatigue: FatigueList,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques, optional external forces can be declared.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
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
        fatigue : FatigueList
            A list of fatigue elements

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states[0]["q"], states)  # TODO: [0] to [node_index]
        qdot = DynamicsFunctions.get(nlp.states[0]["qdot"], states)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        tau = DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue)
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
        ):
            dxdt = MX(nlp.states[0].shape, 1)  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["q"].index, :] = dq  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["qdot"].index, :] = DynamicsFunctions.get(
                nlp.controls[0]["qddot"], controls
            )  # TODO: [0] to [node_index]
        elif (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
        ):
            dxdt = MX(nlp.states[0].shape, 1)  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["q"].index, :] = dq  # TODO: [0] to [node_index]
            qddot = DynamicsFunctions.get(nlp.states[0]["qddot"], states)  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["qdot"].index, :] = qddot  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["qddot"].index, :] = DynamicsFunctions.get(
                nlp.controls[0]["qdddot"], controls
            )  # TODO: [0] to [node_index]
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)
            dxdt = MX(nlp.states[0].shape, ddq.shape[1])
            dxdt[nlp.states[0]["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states[0]["qdot"].index, :] = ddq  # TODO: [0] to [node_index]

        if fatigue is not None and "tau" in fatigue:
            dxdt = fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        defects = None
        # TODO: contacts and fatigue to be handled with implicit dynamics
        if not with_contact and fatigue is None:
            qddot = DynamicsFunctions.get(
                nlp.states_dot[0]["qddot"], nlp.states_dot[0]["scaled"].mx_reduced
            )  # TODO: [0] to [node_index]
            tau_id = DynamicsFunctions.inverse_dynamics(nlp, q, qdot, qddot, with_contact)
            defects = MX(dq.shape[0] + tau_id.shape[0], tau_id.shape[1])

            dq_defects = []
            for _ in range(tau_id.shape[1]):
                dq_defects.append(
                    dq
                    - DynamicsFunctions.compute_qdot(
                        nlp,
                        q,
                        DynamicsFunctions.get(
                            nlp.states_dot[0]["scaled"]["qdot"], nlp.states_dot[0]["scaled"].mx_reduced
                        ),  # TODO: [0] to [node_index]
                    )
                )
            defects[: dq.shape[0], :] = horzcat(*dq_defects)
            # We modified on purpose the size of the tau to keep the zero in the defects in order to respect the dynamics
            defects[dq.shape[0] :, :] = tau - tau_id

        return DynamicsEvaluation(dxdt, defects)

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

        tau_var, tau_mx = (
            (nlp.controls[0], controls) if "tau" in nlp.controls[0] else (nlp.states[0], states)
        )  # TODO: [0] to [node_index]
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
                            DynamicsFunctions.get(nlp.states[0][f"tau_{suffix}_{model.dynamics_suffix()}"], states)[
                                i
                            ]  # TODO: [0] to [node_index]
                            * model.scaling
                        )
                    tau = vertcat(tau, tau_tp)
        return tau

    @staticmethod
    def torque_activations_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        with_contact: bool,
        with_passive_torque: bool,
        with_residual_torque: bool,
        with_ligament: bool,
    ):
        """
        Forward dynamics driven by joint torques activations.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
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

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states[0]["q"], states)  # TODO: [0] to [node_index]
        qdot = DynamicsFunctions.get(nlp.states[0]["qdot"], states)  # TODO: [0] to [node_index]
        tau_activation = DynamicsFunctions.get(nlp.controls[0]["tau"], controls)  # TODO: [0] to [node_index]

        tau = nlp.model.torque(tau_activation, q, qdot)
        if with_passive_torque:
            tau += nlp.model.passive_joint_torque(q, qdot)
        if with_residual_torque:
            tau += DynamicsFunctions.get(nlp.controls[0]["residual_tau"], controls)  # TODO: [0] to [node_index]
        if with_ligament:
            tau += nlp.model.ligament_joint_torque(q, qdot)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)

        dq = horzcat(*[dq for _ in range(ddq.shape[1])])

        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)

    @staticmethod
    def torque_derivative_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        rigidbody_dynamics: RigidBodyDynamics,
        with_contact: bool,
        with_passive_torque: bool,
        with_ligament: bool,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joint torques, optional external forces can be declared.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
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

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states[0]["q"], states)  # TODO: [0] to [node_index]
        qdot = DynamicsFunctions.get(nlp.states[0]["qdot"], states)  # TODO: [0] to [node_index]

        tau = DynamicsFunctions.get(nlp.states[0]["tau"], states)  # TODO: [0] to [node_index]
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dtau = DynamicsFunctions.get(nlp.controls[0]["taudot"], controls)  # TODO: [0] to [node_index]

        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
        ):
            ddq = DynamicsFunctions.get(nlp.states[0]["qddot"], states)  # TODO: [0] to [node_index]
            dddq = DynamicsFunctions.get(nlp.controls[0]["qdddot"], controls)  # TODO: [0] to [node_index]

            dxdt = MX(nlp.states[0].shape, 1)  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["q"].index, :] = dq  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["qdot"].index, :] = ddq  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["qddot"].index, :] = dddq  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["tau"].index, :] = dtau  # TODO: [0] to [node_index]
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)
            dxdt = MX(nlp.states[0].shape, ddq.shape[1])  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["qdot"].index, :] = ddq  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["tau"].index, :] = horzcat(
                *[dtau for _ in range(ddq.shape[1])]
            )  # TODO: [0] to [node_index]

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def forces_from_torque_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q_nlp, q_var = (
            (nlp.states[0]["q"], states) if "q" in nlp.states[0] else (nlp.controls[0]["q"], controls)
        )  # TODO: [0] to [node_index]
        qdot_nlp, qdot_var = (
            (nlp.states[0]["qdot"], states) if "qdot" in nlp.states[0] else (nlp.controls[0]["qdot"], controls)
        )  # TODO: [0] to [node_index]
        tau_nlp, tau_var = (
            (nlp.states[0]["tau"], states) if "tau" in nlp.states[0] else (nlp.controls[0]["tau"], controls)
        )  # TODO: [0] to [node_index]

        q = DynamicsFunctions.get(q_nlp, q_var)
        qdot = DynamicsFunctions.get(qdot_nlp, qdot_var)
        tau = DynamicsFunctions.get(tau_nlp, tau_var)
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        return nlp.model.contact_forces(q, qdot, tau, nlp.external_forces)

    @staticmethod
    def forces_from_torque_activation_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q_nlp, q_var = (
            (nlp.states[0]["q"], states) if "q" in nlp.states[0] else (nlp.controls[0]["q"], controls)
        )  # TODO: [0] to [node_index]
        qdot_nlp, qdot_var = (
            (nlp.states[0]["qdot"], states) if "qdot" in nlp.states[0] else (nlp.controls[0]["qdot"], controls)
        )  # TODO: [0] to [node_index]
        tau_nlp, tau_var = (
            (nlp.states[0]["tau"], states) if "tau" in nlp.states[0] else (nlp.controls[0]["tau"], controls)
        )  # TODO: [0] to [node_index]
        q = DynamicsFunctions.get(q_nlp, q_var)
        qdot = DynamicsFunctions.get(qdot_nlp, qdot_var)
        tau_activations = DynamicsFunctions.get(tau_nlp, tau_var)
        tau = nlp.model.torque(tau_activations, q, qdot)
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        return nlp.model.contact_forces(q, qdot, tau, nlp.external_forces)

    @staticmethod
    def muscles_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        with_contact: bool,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        with_residual_torque: bool = False,
        fatigue=None,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by muscle.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
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

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states[0]["q"], states)  # TODO: [0] to [node_index]
        qdot = DynamicsFunctions.get(nlp.states[0]["qdot"], states)  # TODO: [0] to [node_index]
        residual_tau = (
            DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue) if with_residual_torque else None
        )

        mus_act_nlp, mus_act = (
            (nlp.states[0], states) if "muscles" in nlp.states[0] else (nlp.controls[0], controls)
        )  # TODO: [0] to [node_index]
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
                mus_activations = DynamicsFunctions.get(
                    nlp.states[0][f"muscles_{dyn_suffix}"], states
                )  # TODO: [0] to [node_index]

            if apply_to_joint_dynamics > 0:
                fatigue_states = DynamicsFunctions.get(
                    nlp.states[0][f"muscles_{fatigue_suffix}"], states
                )  # TODO: [0] to [node_index]
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            ddq = DynamicsFunctions.get(nlp.controls[0]["qddot"], controls)  # TODO: [0] to [node_index]
            dxdt = MX(nlp.states[0].shape, 1)  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["q"].index, :] = dq  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["qdot"].index, :] = DynamicsFunctions.get(
                nlp.controls[0]["qddot"], controls
            )  # TODO: [0] to [node_index]
        else:
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)
            dxdt = MX(nlp.states[0].shape, ddq.shape[1])  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])  # TODO: [0] to [node_index]
            dxdt[nlp.states[0]["qdot"].index, :] = ddq  # TODO: [0] to [node_index]

        has_excitation = True if "muscles" in nlp.states[0] else False  # TODO: [0] to [node_index]
        if has_excitation:
            mus_excitations = DynamicsFunctions.get(nlp.controls[0]["muscles"], controls)  # TODO: [0] to [node_index]
            dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations)
            dxdt[nlp.states[0]["muscles"].index, :] = horzcat(
                *[dmus for _ in range(ddq.shape[1])]
            )  # TODO: [0] to [node_index]

        if fatigue is not None and "muscles" in fatigue:
            dxdt = fatigue["muscles"].dynamics(dxdt, nlp, states, controls)

        defects = None
        # TODO: contacts and fatigue to be handled with implicit dynamics
        if not with_contact and fatigue is None:  # TODO: [0] to [node_index]
            qddot = DynamicsFunctions.get(
                nlp.states_dot[0]["qddot"], nlp.states_dot[0].mx_reduced
            )  # TODO: [0] to [node_index]
            tau_id = DynamicsFunctions.inverse_dynamics(nlp, q, qdot, qddot, with_contact)
            defects = MX(dq.shape[0] + tau_id.shape[0], tau_id.shape[1])

            dq_defects = []
            for _ in range(tau_id.shape[1]):  # TODO: [0] to [node_index]
                dq_defects.append(
                    dq
                    - DynamicsFunctions.compute_qdot(
                        nlp,
                        q,
                        DynamicsFunctions.get(
                            nlp.states_dot[0]["qdot"], nlp.states_dot[0].mx_reduced
                        ),  # TODO: [0] to [node_index]
                    )
                )
            defects[: dq.shape[0], :] = horzcat(*dq_defects)
            defects[dq.shape[0] :, :] = tau - tau_id

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def forces_from_muscle_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q = DynamicsFunctions.get(nlp.states[0]["q"], states)  # TODO: [0] to [node_index]
        qdot = DynamicsFunctions.get(nlp.states[0]["qdot"], states)  # TODO: [0] to [node_index]
        residual_tau = (
            DynamicsFunctions.get(nlp.controls[0]["tau"], controls) if "tau" in nlp.controls[0] else None
        )  # TODO: [0] to [node_index]

        mus_act_nlp, mus_act = (
            (nlp.states[0], states) if "muscles" in nlp.states[0] else (nlp.controls[0], controls)
        )  # TODO: [0] to [node_index]
        mus_activations = DynamicsFunctions.get(mus_act_nlp["muscles"], mus_act)
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau

        return nlp.model.contact_forces(q, qdot, tau, nlp.external_forces)

    @staticmethod
    def joints_acceleration_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        nlp,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by joints accelerations of a free floating body.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
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

        q = DynamicsFunctions.get(nlp.states[0]["q"], states)  # TODO: [0] to [node_index]
        qdot = DynamicsFunctions.get(nlp.states[0]["qdot"], states)  # TODO: [0] to [node_index]
        qddot_joints = DynamicsFunctions.get(nlp.controls[0]["qddot_joints"], controls)  # TODO: [0] to [node_index]

        qddot_root, qddot_reordered = DynamicsFunctions.forward_dynamics_free_floating(nlp, q, qdot, qddot_joints)

        # defects
        qddot_root_defects = DynamicsFunctions.get(nlp.states_dot[0]["qddot_roots"], nlp.states_dot[0].mx_reduced)
        qddot_defects_reordered = nlp.model.reorder_qddot_root_joints(qddot_root_defects, qddot_joints)

        floating_base_constraint = nlp.model.inverse_dynamics(q, qdot, qddot_defects_reordered)[: nlp.model.nb_root]

        qdot_mapped = nlp.variable_mappings["qdot"].to_first.map(qdot)
        qddot_mapped = nlp.variable_mappings["qdot"].to_first.map(qddot_reordered)
        qddot_root_mapped = nlp.variable_mappings["qddot_roots"].to_first.map(qddot_root)
        qddot_joints_mapped = nlp.variable_mappings["qddot_joints"].to_first.map(qddot_joints)

        defects = MX(qdot_mapped.shape[0] + qddot_root_mapped.shape[0] + qddot_joints_mapped.shape[0], 1)

        defects[: qdot_mapped.shape[0], :] = qdot_mapped - nlp.variable_mappings["qdot"].to_first.map(
            DynamicsFunctions.compute_qdot(
                nlp, q, DynamicsFunctions.get((nlp.states_dot[0]["qdot"]), nlp.states_dot[0].mx_reduced)
            )
        )

        defects[
            qdot_mapped.shape[0] : (qdot_mapped.shape[0] + qddot_root_mapped.shape[0]), :
        ] = floating_base_constraint
        defects[(qdot_mapped.shape[0] + qddot_root_mapped.shape[0]) :, :] = qddot_joints_mapped - nlp.variable_mappings[
            "qddot_joints"
        ].to_first.map(DynamicsFunctions.get(nlp.states_dot[0]["qddot_joints"], nlp.states_dot[0].mx_reduced))

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

        offset = 0
        for param in nlp.parameters:
            # Call the pre dynamics function
            if param.function:
                param.function(nlp.model, parameters[offset : offset + param.size], **param.params)
                offset += param.size

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

        q_nlp = nlp.states[0]["q"] if "q" in nlp.states[0] else nlp.controls[0]["q"]  # TODO: [0] to [node_index]
        return q_nlp.mapping.to_first.map(nlp.model.reshape_qdot(q, qdot))

    @staticmethod
    def forward_dynamics(
        nlp: NonLinearProgram,
        q: MX | SX,
        qdot: MX | SX,
        tau: MX | SX,
        with_contact: bool,
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

        Returns
        -------
        The derivative of qdot
        """
        qdot_var = (
            nlp.states[0]["qdot"] if "qdot" in nlp.states[0] else nlp.controls[0]["qdot"]
        )  # TODO: [0] to [node_index]

        if nlp.external_forces:
            dxdt = MX(len(qdot_var.mapping.to_first), nlp.ns)
            for i, f_ext in enumerate(nlp.external_forces):
                if with_contact:
                    qddot = nlp.model.constrained_forward_dynamics(q, qdot, tau, f_ext)
                else:
                    qddot = nlp.model.forward_dynamics(q, qdot, tau, f_ext)
                dxdt[:, i] = qdot_var.mapping.to_first.map(qddot)
            return dxdt
        else:
            if with_contact:
                qddot = nlp.model.constrained_forward_dynamics(q, qdot, tau)
            else:
                qddot = nlp.model.forward_dynamics(q, qdot, tau)

            return qdot_var.mapping.to_first.map(qddot)

    @staticmethod
    def forward_dynamics_free_floating(nlp, q, qdot, qddot_joints) -> list[MX, MX]:
        """
        Easy accessor to derivative of free floating base dynamics

        Returns
        -------
        The derivative of qdot concatenated such that qddot = qddot_root, qddot_joints
        """

        q_temporary = MX.sym("Q", nlp.model.nb_q)
        qdot_temporary = MX.sym("Qdot",  nlp.model.nb_qdot)
        qddot_joints_temporary = MX.sym("Qddot_joints", nlp.model.nb_qddot - nlp.model.nb_root)

        qddot_root_temporary = nlp.model.forward_dynamics_free_floating_base(q_temporary, qdot_temporary, qddot_joints_temporary)
        q_root_func = Function(
            "qddot_root_func", [q_temporary, qdot_temporary, qddot_joints_temporary], [qddot_root_temporary]
        ).expand()
        qddot_root = q_root_func(q, qdot, qddot_joints)
        qddot_reordered = nlp.model.reorder_qddot_root_joints(qddot_root, qddot_joints)

        return qddot_root, qddot_reordered

    @staticmethod
    def inverse_dynamics(nlp: NonLinearProgram, q: MX | SX, qdot: MX | SX, qddot: MX | SX, with_contact: bool):
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

        Returns
        -------
        Torques in tau
        """

        if len(nlp.external_forces) != 0:
            if "tau" in nlp.states[0]:  # TODO: [0] to [node_index]
                tau_shape = nlp.states[0]["tau"].mx.shape[0]  # TODO: [0] to [node_index]
            elif "tau" in nlp.controls[0]:  # TODO: [0] to [node_index]
                tau_shape = nlp.controls[0]["tau"].mx.shape[0]  # TODO: [0] to [node_index]
            else:
                tau_shape = nlp.model.nb_tau
            tau = MX(tau_shape, nlp.ns)
            for i, f_ext in enumerate(nlp.external_forces):
                tau[:, i] = nlp.model.inverse_dynamics(q, qdot, qddot, f_ext)
        else:
            tau = nlp.model.inverse_dynamics(q, qdot, qddot)
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
        for k in range(len(nlp.controls[0]["muscles"])):  # TODO: [0] to [node_index]
            if fatigue_states is not None:
                activations.append(muscle_activations[k] * (1 - fatigue_states[k]))
            else:
                activations.append(muscle_activations[k])
        return nlp.model.muscle_joint_torque(activations, q, qdot)
