from casadi import vertcat, MX
import biorbd


class DynamicsFunctions:
    """
    Different dynamics types
    """

    @staticmethod
    def custom(states, controls, parameters, nlp):
        qdot, qddot = nlp["problem_type"]["dynamic"](states, controls, parameters, nlp)
        return vertcat(qdot, qddot)

    @staticmethod
    def forward_dynamics_torque_driven(states, controls, parameters, nlp):
        """
        Forward dynamics (q, qdot, qddot -> tau) with external forces driven by joint torques (controls).
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :param parameters: The MX associated to the parameters
        :return: Vertcat of derived states. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)

        if "external_forces" in nlp:
            dxdt = MX(nlp["nx"], nlp["ns"])
            for i, f_ext in enumerate(nlp["external_forces"]):
                qddot = nlp["model"].ForwardDynamics(q, qdot, tau, f_ext).to_mx()
                qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
                dxdt[:, i] = vertcat(qdot_reduced, qddot_reduced)
        else:
            qddot = nlp["model"].ForwardDynamics(q, qdot, tau).to_mx()
            qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
            dxdt = vertcat(qdot_reduced, qddot_reduced)

        return dxdt

    @staticmethod
    def forward_dynamics_torque_driven_with_contact(states, controls, parameters, nlp):
        """
        Forward dynamics (q, qdot, qddot -> tau) with contact force driven by joint torques (controls).
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Vertcat of derived states. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forces_from_forward_dynamics_with_contact(states, controls, parameters, nlp):
        """
        Returns contact forces computed from forward dynamics with contact force
        (forward_dynamics_torque_driven_with_contact)
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Contact forces. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)

        return cs.getForce().to_mx()

    @staticmethod
    def forward_dynamics_torque_activations_driven(states, controls, parameters, nlp):
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, torque_act = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        tau = nlp["model"].torque(torque_act, q, qdot).to_mx()
        qddot = nlp["model"].ForwardDynamics(q, qdot, tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_activations_driven_with_contact(states, controls, parameters, nlp):
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, torque_act = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        tau = nlp["model"].torque(torque_act, q, qdot).to_mx()
        qddot = nlp["model"].ForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_muscle_driven(states, controls, parameters, nlp):
        """
        Forward dynamics (q, qdot, qddot -> tau) without external forces driven by joint torques and muscles (controls).
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Vertcat of derived states. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        muscles_activations = controls[nlp["nbTau"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_muscle_activations_and_torque_driven_with_contact(states, controls, parameters, nlp):
        """
        Forward dynamics (q, qdot, qddot -> tau) with contact force driven by joint torques and muscles (controls).
        :param states: Sates. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Vertcat of derived states. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        muscles_activations = controls[nlp["nbTau"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forces_from_forward_dynamics_muscle_activations_and_torque_driven_with_contact(
        states, controls, parameters, nlp
    ):
        """
        Returns contact forces computed from forward dynamics with contact force
        (forward_dynamics_torque_muscle_driven_with_contact)
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Contact forces. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        muscles_activations = controls[nlp["nbTau"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)

        return cs.getForce().to_mx()

    @staticmethod
    def forward_dynamics_muscle_activations_driven(states, controls, parameters, nlp):
        DynamicsFunctions.apply_parameters(parameters, nlp)

        nq = nlp["q_mapping"].reduce.len
        q = nlp["q_mapping"].expand.map(states[:nq])
        qdot = nlp["q_dot_mapping"].expand.map(states[nq:])

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        muscles_activations = controls

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, muscles_tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_muscle_excitations_driven(states, controls, parameters, nlp):
        """
        Forward dynamics (q, qdot, qddot -> tau) without external forces driven by muscle excitation (controls).
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Vertcat of derived states. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)

        nq = nlp["q_mapping"].reduce.len
        q = nlp["q_mapping"].expand.map(states[:nq])
        qdot = nlp["q_dot_mapping"].expand.map(states[nq:])

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        muscles_excitation = controls
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp["model"].activationDot(muscles_states).to_mx()

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, muscles_tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forward_dynamics_muscle_excitations_and_torque_driven(states, controls, parameters, nlp):
        """
        Forward dynamics (q, qdot, qddot -> tau) without external forces driven by muscle excitation
        and joint torques (controls).
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Vertcat of derived states. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        muscles_excitation = controls[nlp["nbTau"] :]
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp["model"].activationDot(muscles_states).to_mx()

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forward_dynamics_muscle_excitations_and_torque_driven_with_contact(states, controls, parameters, nlp):
        """
        Forward dynamics (q, qdot, qddot -> tau) with contact force driven by muscle excitation and
        joint torques (controls).
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Vertcat of derived states. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        muscles_excitation = controls[nlp["nbTau"] :]
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp["model"].activationDot(muscles_states).to_mx()

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        q_dot = nlp["model"].computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp["q_mapping"].reduce.map(q_dot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact(
        states, controls, parameters, nlp
    ):
        """
        Returns contact forces computed from forward dynamics with contact force
        (forward_dynamics_muscle_excitations_and_torque_driven_with_contact)
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: Contact forces. (MX.sym from CasADi)
        """
        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        muscles_excitation = controls[nlp["nbTau"] :]
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)
        return cs.getForce().to_mx()

    @staticmethod
    def dispatch_q_qdot_tau_data(states, controls, nlp):
        """
        Returns q, qdot, tau (unreduced by a potential symmetry) and qdot_reduced
        from states, controls and mapping through nlp to condense this code.
        :param states: States. (MX.sym from CasADi)
        :param controls: Controls. (MX.sym from CasADi)
        :param nlp: An OptimalControlProgram class.
        :return: q -> Generalized coordinates positions. (MX.sym from CasADi),
        qdot -> Generalized coordinates velocities. (MX.sym from CasADi) and
        tau -> Joint torques. (MX.sym from CasADi)
        """
        nq = nlp["q_mapping"].reduce.len
        q = nlp["q_mapping"].expand.map(states[:nq])
        qdot = nlp["q_dot_mapping"].expand.map(states[nq:])
        tau = nlp["tau_mapping"].expand.map(controls[: nlp["nbTau"]])

        return q, qdot, tau

    @staticmethod
    def apply_parameters(mx, nlp):
        for key in nlp["parameters_to_optimize"]:
            param = nlp["parameters_to_optimize"][key]

            # Call the pre dynamics function
            if param["func"]:
                param["func"](nlp["model"], mx, **param["extra_params"])
