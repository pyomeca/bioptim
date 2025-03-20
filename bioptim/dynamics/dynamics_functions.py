from casadi import horzcat, vertcat, MX, SX, DM

from .dynamics_evaluation import DynamicsEvaluation
from .fatigue.fatigue_dynamics import FatigueList
from .ode_solvers import OdeSolver
from ..misc.enums import DefectType, ContactType
from ..misc.mapping import BiMapping
from ..optimization.optimization_variable import OptimizationVariable


class DynamicsFunctions:
    """
    Implementation of all the dynamic functions

    Methods
    -------
    custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Interface to custom dynamic function provided by the user
    torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool)
        Forward dynamics driven by joint torques
    torque_activations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact) -> MX:
        Forward dynamics driven by joint torques activations.
    torque_derivative_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        Forward dynamics driven by joint torques derivatives
    forces_from_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        Contact forces of a forward dynamics driven by joint torques with contact constraints.
    muscles_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        Forward dynamics driven by muscle.
    forces_from_muscle_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.
    get(var: OptimizationVariable, cx: MX | SX):
        Main accessor to a variable in states or controls (cx)
    reshape_qdot(nlp: NonLinearProgram, q: MX | SX, qdot: MX | SX):
        Easy accessor to derivative of q
    forward_dynamics(nlp: NonLinearProgram, q: MX | SX, qdot: MX | SX, tau: MX | SX, with_contact: bool):
        Easy accessor to derivative of qdot
    compute_muscle_dot(nlp: NonLinearProgram, muscle_excitations: MX | SX, muscle_activations: MX | SX):
        Easy accessor to derivative of muscle activations
    compute_tau_from_muscle(nlp: NonLinearProgram, q: MX | SX, qdot: MX | SX, muscle_activations: MX | SX):
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
        with_contact: bool,
        with_passive_torque: bool,
        with_ligament: bool,
        with_friction: bool,
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
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with friction should be used
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

        tau = DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue)
        tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau
        tau = tau - nlp.model.friction_coefficients @ qdot if with_friction else tau

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)

        dxdt, defects = None, None
        if not isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):  # not COLLOCATION or IRK
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
            dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot"].index, :] = ddq

            if fatigue is not None and "tau" in fatigue:
                dxdt = fatigue["tau"].dynamics(dxdt, nlp, states, controls)

        else:
            # TODO: contacts and fatigue to be handled with implicit dynamics
            # TODO: Charbie -> change the defect type
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:
                if not with_contact and fatigue is None:
                    tau_id = DynamicsFunctions.inverse_dynamics(
                        nlp, q, slope_q, slope_qdot, with_contact, external_forces
                    )
                    defects = nlp.cx(dq.shape[0] + tau_id.shape[0], tau_id.shape[1])

                    dq_defects = []
                    for _ in range(tau_id.shape[1]):
                        dq_defects.append(
                            dq
                            - DynamicsFunctions.compute_qdot(
                                nlp,
                                q,
                                slope_q,
                            )
                        )
                    defects[: dq.shape[0], :] = horzcat(*dq_defects)
                    # We modified on purpose the size of the tau to keep the zero in the defects in order to respect the dynamics
                    defects[dq.shape[0] :, :] = tau - tau_id

            else:  # TODO: Charbie -> DefectType.Forward dynamics
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
                derivative = nlp.cx(nlp.states.shape, ddq.shape[1])
                derivative[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
                derivative[nlp.states["qdot"].index, :] = ddq

                if fatigue is not None and "tau" in fatigue:
                    derivative = fatigue["tau"].dynamics(derivative, nlp, states, controls)
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt

        return DynamicsEvaluation(dxdt, defects)

    @staticmethod
    def torque_driven_free_floating_base(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_passive_torque: bool,
        with_ligament: bool,
        with_friction: bool,
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
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with friction should be used

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
        n_q = nlp.model.nb_q
        n_qdot = nlp.model.nb_qdot

        tau_joints = (
            tau_joints + nlp.model.passive_joint_torque()(q_full, qdot_full, nlp.parameters.cx)
            if with_passive_torque
            else tau_joints
        )
        tau_joints = tau_joints + nlp.model.ligament_joint_torque()(q_full, qdot_full) if with_ligament else tau_joints
        tau_joints = tau_joints - nlp.model.friction_coefficients @ qdot_joints if with_friction else tau_joints

        tau_full = vertcat(nlp.cx.zeros(nlp.model.nb_root), tau_joints)

        dxdt, defects = None, None
        if not isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
            ddq = DynamicsFunctions.forward_dynamics(
                nlp, q_full, qdot_full, tau_full, with_contact=False, external_forces=None
            )
            dxdt = nlp.cx(n_q + n_qdot, ddq.shape[1])
            dxdt[:n_q, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[n_q:, :] = ddq
        else:
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:  # TODO
                raise NotImplementedError("Implicit dynamics with free floating base is not implemented yet")
            else:
                ddq = DynamicsFunctions.forward_dynamics(
                    nlp, q_full, qdot_full, tau_full, with_contact=False, external_forces=None
                )
                derivative = nlp.cx(n_q + n_qdot, ddq.shape[1])
                derivative[:n_q, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
                derivative[n_q:, :] = ddq
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt

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
        contact_type: ContactType,
        with_friction: bool,
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
        with_contact: bool
            If the dynamic with contact should be used
        with_friction: bool
            If the dynamic with friction should be used

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """
        if contact_type == ContactType.SOFT:
            raise NotImplementedError("soft contacts not implemented yet with stochastic torque driven dynamics.")
        with_contact = contact_type == ContactType.RIGID

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
        tau = tau - nlp.model.friction_coefficients @ qdot if with_friction else tau

        dxdt, defects = None, None
        if not isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
            dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces=None)
            dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot"].index, :] = ddq
        else:
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:
                raise NotImplementedError(
                    "Implicit dynamics with stochastic torque driven dynamics is not implemented yet"
                )
            else:
                dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces=None)
                derivative = nlp.cx(nlp.states.shape, ddq.shape[1])
                derivative[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
                derivative[nlp.states["qdot"].index, :] = ddq
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt

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
        with_friction: bool,
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
        with_friction: bool
            If the dynamic with friction should be used

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
        motor_noise = DynamicsFunctions.get(nlp.parameters["motor_noise"], parameters)
        sensory_noise = DynamicsFunctions.get(nlp.parameters["sensory_noise"], parameters)

        q_full = vertcat(q_roots, q_joints)
        qdot_full = vertcat(qdot_roots, qdot_joints)
        n_q = q_full.shape[0]

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
        tau_joints = (tau_joints - nlp.model.friction_coefficients @ qdot_joints) if with_friction else tau_joints

        tau_full = vertcat(nlp.cx.zeros(nlp.model.nb_root), tau_joints)

        dxdt, defects = None, None
        if not isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
            dq = DynamicsFunctions.compute_qdot(nlp, q_full, qdot_full)
            ddq = DynamicsFunctions.forward_dynamics(
                nlp, q_full, qdot_full, tau_full, with_contact=False, external_forces=None
            )
            dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
            dxdt[:n_q, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[n_q:, :] = ddq
        else:
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:
                raise NotImplementedError(
                    "Implicit dynamics with stochastic torque driven dynamics is not implemented yet"
                )
            else:
                dq = DynamicsFunctions.compute_qdot(nlp, q_full, qdot_full)
                ddq = DynamicsFunctions.forward_dynamics(
                    nlp, q_full, qdot_full, tau_full, with_contact=False, external_forces=None
                )
                derivative = nlp.cx(nlp.states.shape, ddq.shape[1])
                derivative[:n_q, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
                derivative[n_q:, :] = ddq
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    @staticmethod
    def __get_fatigable_tau(nlp, states: MX | SX, controls: MX | SX, fatigue: FatigueList) -> MX | SX:
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
    def torque_activations_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
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

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau_activation = DynamicsFunctions.get(nlp.controls["tau"], controls)

        tau = nlp.model.torque()(tau_activation, q, qdot, nlp.parameters.cx)
        if with_passive_torque:
            tau += nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        if with_residual_torque:
            tau += DynamicsFunctions.get(nlp.controls["residual_tau"], controls)
        if with_ligament:
            tau += nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx)

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)

        dxdt, defects = None, None
        if not isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
            dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
            dq = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt = vertcat(dq, ddq)
        else:
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:
                raise NotImplementedError(
                    "Implicit dynamics with torque activations driven dynamics is not implemented yet"
                )
            else:
                dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
                dq = horzcat(*[dq for _ in range(ddq.shape[1])])
                derivative = vertcat(dq, ddq)
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt

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
        with_contact: bool,
        with_passive_torque: bool,
        with_ligament: bool,
        with_friction: bool,
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
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with friction should be used

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)

        tau = DynamicsFunctions.get(nlp.states["tau"], states)
        tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau
        tau = tau - nlp.model.friction_coefficients @ qdot if with_friction else tau

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        dtau = DynamicsFunctions.get(nlp.controls["taudot"], controls)

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)

        dxdt, defects = None, None
        if not isinstance(nlp.dynamics.ode_solver, OdeSolver.COLLOCATION):
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
            dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot"].index, :] = ddq
            dxdt[nlp.states["tau"].index, :] = horzcat(*[dtau for _ in range(ddq.shape[1])])
        else:
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:
                raise NotImplementedError(
                    "Implicit dynamics with torque derivative driven dynamics is not implemented yet"
                )
            else:
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
                derivative = nlp.cx(nlp.states.shape, ddq.shape[1])
                derivative[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
                derivative[nlp.states["qdot"].index, :] = ddq
                derivative[nlp.states["tau"].index, :] = horzcat(*[dtau for _ in range(ddq.shape[1])])
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt

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
        with_passive_torque: bool = False,
        with_ligament: bool = False,
    ) -> MX | SX:
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
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used

        Returns
        ----------
        MX.sym | SX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        tau = nlp.get_var_from_states_or_controls("tau", states, controls)
        tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)

        return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

    @staticmethod
    def forces_from_torque_activation_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
    ) -> MX | SX:
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
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used

        Returns
        ----------
        MX.sym | SX.sym
            The contact forces that ensure no acceleration at these contact points
        """
        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        tau_activations = nlp.get_var_from_states_or_controls("tau", states, controls)
        tau = nlp.model.torque()(tau_activations, q, qdot, nlp.parameters.cx)
        tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)
        return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

    @staticmethod
    def muscles_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_contact: bool,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        with_friction: bool = False,
        with_residual_torque: bool = False,
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
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with friction should be used
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
            DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue) if with_residual_torque else None
        )
        mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)
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
        tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau
        tau = tau - nlp.model.friction_coefficients @ qdot if with_friction else tau

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)

        dxdt, defects = None, None
        if not isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
            ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
            dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
            dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
            dxdt[nlp.states["qdot"].index, :] = ddq

            has_excitation = True if "muscles" in nlp.states else False
            if has_excitation:
                mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
                dxdt[nlp.states["muscles"].index, :] = horzcat(*[dmus for _ in range(ddq.shape[1])])

            if fatigue is not None and "muscles" in fatigue:
                dxdt = fatigue["muscles"].dynamics(dxdt, nlp, states, controls)
        else:
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
            slopes = vertcat(slope_q, slope_qdot)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:
                raise NotImplementedError("Implicit dynamics with muscles driven dynamics is not implemented yet")
            else:
                ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
                derivative = nlp.cx(nlp.states.shape, ddq.shape[1])
                derivative[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
                derivative[nlp.states["qdot"].index, :] = ddq

                has_excitation = True if "muscles" in nlp.states else False
                if has_excitation:
                    mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                    dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
                    derivative[nlp.states["muscles"].index, :] = horzcat(*[dmus for _ in range(ddq.shape[1])])
                    slope_mus = DynamicsFunctions.get(nlp.states_dot["muscles"], nlp.states_dot.scaled.cx)
                    slopes = vertcat(slopes, slope_mus)
                if fatigue is not None and "muscles" in fatigue:
                    derivative = fatigue["muscles"].dynamics(derivative, nlp, states, controls)
                defects = slopes * nlp.dt - derivative * nlp.dt

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
        with_passive_torque: bool = False,
        with_ligament: bool = False,
    ) -> MX | SX:
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
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used

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
        tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)
        return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

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
        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        qddot_joints = nlp.get_var_from_states_or_controls("qddot", states, controls)

        dxdt, defects = None, None
        if not isinstance(nlp.ode_solver, OdeSolver.COLLOCATION):
            qddot_root = nlp.model.forward_dynamics_free_floating_base()(q, qdot, qddot_joints, nlp.parameters.cx)
            qddot_reordered = nlp.model.reorder_qddot_root_joints(qddot_root, qddot_joints)

            qdot_mapped = nlp.variable_mappings["qdot"].to_first.map(qdot)
            qddot_mapped = nlp.variable_mappings["qdot"].to_first.map(qddot_reordered)

            dxdt = vertcat(qdot_mapped, qddot_mapped)
        else:
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:
                raise NotImplementedError(
                    "Implicit dynamics with joints acceleration driven dynamics is not implemented yet"
                )
            else:
                qddot_root = nlp.model.forward_dynamics_free_floating_base()(q, qdot, qddot_joints, nlp.parameters.cx)
                qddot_reordered = nlp.model.reorder_qddot_root_joints(qddot_root, qddot_joints)

                qdot_mapped = nlp.variable_mappings["qdot"].to_first.map(qdot)
                qddot_mapped = nlp.variable_mappings["qdot"].to_first.map(qddot_reordered)

                derivative = vertcat(qdot_mapped, qddot_mapped)
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

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
    def compute_qdot(nlp, q: MX | SX, qdot: MX | SX):
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
    def forward_dynamics(
        nlp,
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
        # Get the mapping of the output
        if "qdot" in nlp.states:
            qdot_var_mapping = nlp.states["qdot"].mapping.to_first
        elif "qdot" in nlp.controls:
            qdot_var_mapping = nlp.controls["qdot"].mapping.to_first
        else:
            qdot_var_mapping = BiMapping([i for i in range(qdot.shape[0])], [i for i in range(qdot.shape[0])]).to_first

        external_forces = [] if external_forces is None else external_forces
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
        q: MX | SX,
        qdot: MX | SX,
        qddot: MX | SX,
        with_contact: bool,
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
        with_contact: bool
            If the dynamics with contact should be used
        external_forces: MX
            The external forces

        Returns
        -------
        Torques in tau
        """
        if external_forces is None:
            tau = nlp.model.inverse_dynamics(with_contact=with_contact)(q, qdot, qddot, [], nlp.parameters.cx)
        else:
            tau = nlp.model.inverse_dynamics(with_contact=with_contact)(
                q, qdot, qddot, external_forces, nlp.parameters.cx
            )
        return tau  # We ignore on purpose the mapping to keep zeros in the defects of the dynamic.

    @staticmethod
    def compute_muscle_dot(nlp, muscle_excitations: MX | SX, muscle_activations: MX | SX):
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

        dxdt, defects = None, None
        if not isinstance(nlp.dynamics.ode_solver, OdeSolver.COLLOCATION):
            qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)
            dxdt = vertcat(qdot_u, qddot_u)
        else:
            slope_q = DynamicsFunctions.get(nlp.states_dot["qdot_u"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot_u"], nlp.states_dot.scaled.cx)
            if nlp.ode_solver.defects_type == DefectType.IMPLICIT:
                raise NotImplementedError(
                    "Implicit dynamics with holonomic torque driven dynamics is not implemented yet"
                )
            else:
                qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)
                derivative = vertcat(qdot_u, qddot_u)
                defects = vertcat(slope_q, slope_qdot) * nlp.dt - derivative * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)
