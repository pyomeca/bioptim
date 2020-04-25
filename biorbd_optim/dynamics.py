from casadi import vertcat
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
        q, qdot, tau = Dynamics.__dispatch_data(states, controls, nlp)

        qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_driven_with_contact(states, controls, nlp):
        """
        :param states: MX.sym from CasADi.
        :param controls: MX.sym from CasADi.
        :param nlp: An OptimalControlProgram class
        :return: Vertcat of derived states.
        """
        q, qdot, tau = Dynamics.__dispatch_data(states, controls, nlp)

        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()

        qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
        qddot_reduced = nlp["q_dot_mapping"].reduce.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forces_from_forward_dynamics_with_contact(states, controls, nlp):
        q, qdot, tau = Dynamics.__dispatch_data(states, controls, nlp)

        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)

        return cs.getForce().to_mx()

    @staticmethod
    def forward_dynamics_torque_muscle_driven(states, controls, nlp):
        q, qdot, residual_tau = Dynamics.__dispatch_data(states, controls, nlp)

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
        q, qdot, residual_tau = Dynamics.__dispatch_data(states, controls, nlp)

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
    def forward_dynamics_excitation_driven(states, controls, nlp):
        q, qdot, qdot_reduced, residual_tau = Dynamics.__dispatch_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_excitation = controls[nlp["nbTau"]:]
        muscles_activations = states[nlp["nbQ"] + nlp["nbQdot"]:]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].Excitation(muscles_excitation[k])
            muscles_activations[k] = muscles_states[k].activation()

        # muscles_activations =

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()
        qddot_reduced = nlp["qdot_mapping"].reduce.map(qddot)

        return vertcat(muscles_activations, qdot_reduced, qddot_reduced)

    @staticmethod
    def __dispatch_data(states, controls, nlp):
        """
        Returns q, qdot, tau (unreduced by a potential symmetry) and qdot_reduced
        from states, controls and mapping through nlp to condense this code.
        """
        nq = nlp["q_mapping"].reduce.len
        q = nlp["q_mapping"].expand.map(states[:nq])
        qdot = nlp["q_dot_mapping"].expand.map(states[nq:])
        tau = nlp["tau_mapping"].expand.map(controls[: nlp["nbTau"]])

        return q, qdot, tau
