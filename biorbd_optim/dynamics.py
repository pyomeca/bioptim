from casadi import vertcat, MX
import biorbd


class Dynamics:
    @staticmethod
    def forward_dynamics_torque_driven(states, controls, nlp):
        """
        :param states: MX.sym from CasADi.
        :param controls: MX.sym from CasADi.
        :param nlp: An OptimalControlProgram class
        :return: Vertcat of derived states.
        """
        q, qdot, tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        if "external_forces" in nlp:
            dxdt = MX(nlp["nx"], nlp["ns"])
            for i, f_ext in enumerate(nlp["external_forces"]):
                qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau, f_ext).to_mx()
                qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
                dxdt[:, i] = vertcat(qdot_reduced, qddot_reduced)
        else:
            qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()
            qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
            dxdt = vertcat(qdot_reduced, qddot_reduced)

        return dxdt

    @staticmethod
    def forward_dynamics_torque_driven_with_contact(states, controls, nlp):
        """
        :param states: MX.sym from CasADi.
        :param controls: MX.sym from CasADi.
        :param nlp: An OptimalControlProgram class
        :return: Vertcat of derived states.
        """
        q, qdot, tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forces_from_forward_dynamics_with_contact(states, controls, nlp):
        q, qdot, tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)

        return cs.getForce().to_mx()

    @staticmethod
    def forward_dynamics_torque_activations_driven(states, controls, nlp):
        q, qdot, torque_act = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        tau = nlp["model"].torque(torque_act, q, qdot).to_mx()
        qddot = nlp["model"].ForwardDynamics(q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_activations_driven_with_contact(states, controls, nlp):
        q, qdot, torque_act = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        tau = nlp["model"].torque(torque_act, q, qdot).to_mx()
        qddot = nlp["model"].ForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_muscle_driven(states, controls, nlp):
        q, qdot, residual_tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_activations = controls[nlp["nbTau"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_muscle_driven_with_contact(states, controls, nlp):
        q, qdot, residual_tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_activations = controls[nlp["nbTau"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forces_from_forward_dynamics_torque_muscle_driven_with_contact(states, controls, nlp):
        q, qdot, residual_tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_activations = controls[nlp["nbTau"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)

        return cs.getForce().to_mx()

    @staticmethod
    def forward_dynamics_muscle_excitations_driven(states, controls, nlp):
        nq = nlp["q_mapping"].reduce.len
        q = nlp["q_mapping"].expand.map(states[:nq])
        qdot = nlp["q_dot_mapping"].expand.map(states[nq:])

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_excitation = controls
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp["model"].activationDot(muscles_states).to_mx()

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, muscles_tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forward_dynamics_muscle_excitations_and_torque_driven(states, controls, nlp):
        q, qdot, residual_tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_excitation = controls[nlp["nbTau"] :]
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp["model"].activationDot(muscles_states).to_mx()

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forward_dynamics_muscle_excitations_and_torque_driven_with_contact(states, controls, nlp):
        q, qdot, residual_tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_excitation = controls[nlp["nbTau"] :]
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp["model"].activationDot(muscles_states).to_mx()

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact(states, controls, nlp):
        q, qdot, residual_tau = Dynamics.__dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_excitation = controls[nlp["nbTau"] :]
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp["model"].activationDot(muscles_states).to_mx()

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)
        return cs.getForce().to_mx()

    @staticmethod
    def __dispatch_q_qdot_tau_data(states, controls, nlp):
        """
        Returns q, qdot, tau (unreduced by a potential symmetry) and qdot_reduced
        from states, controls and mapping through nlp to condense this code.
        """
        nq = nlp["q_mapping"].reduce.len
        q = nlp["q_mapping"].expand.map(states[:nq])
        qdot = nlp["q_dot_mapping"].expand.map(states[nq:])
        tau = nlp["tau_mapping"].expand.map(controls[: nlp["nbTau"]])

        return q, qdot, tau
