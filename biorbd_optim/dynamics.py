from casadi import vertcat
import biorbd


class Dynamics:

    @staticmethod
    def __dispatch_data(states, controls, nlp):
        """
        Returns q, qdot, tau (unreduced by a potential symmetry) and qdot_reduced
        from states, controls and mapping through nlp to condense this code.
        """
        nq = nlp["dof_mapping"].nb_reduced
        q = nlp["dof_mapping"].expand(states[:nq])
        qdot_reduced = states[nq:]
        qdot = nlp["dof_mapping"].expand(qdot_reduced)
        tau = nlp["dof_mapping"].expand(controls[: nlp["nbTau"]])

        return q, qdot, qdot_reduced, tau

    @staticmethod
    def forward_dynamics_torque_driven_no_contact(states, controls, nlp):
        """
        :param states: MX.sym from CasADi.
        :param controls: MX.sym from CasADi.
        :param nlp: An OptimalControlProgram class
        :return: Vertcat of derived states.
        """
        q, qdot, qdot_reduced, tau = Dynamics.__dispatch_data(states, controls, nlp)

        qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()
        qddot_reduced = nlp["dof_mapping"].reduce(qddot)

        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_driven_with_contact(states, controls, nlp):
        """
        :param states: MX.sym from CasADi.
        :param controls: MX.sym from CasADi.
        :param nlp: An OptimalControlProgram class
        :return: Vertcat of derived states.
        """
        q, qdot, qdot_reduced, tau = Dynamics.__dispatch_data(states, controls, nlp)

        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()
        qddot_reduced = nlp["dof_mapping"].reduce(qddot)

        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_muscle_driven(states, controls, nlp):
        q, qdot, qdot_reduced, residual_tau = Dynamics.__dispatch_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
        muscles_activations = controls[nlp["nbTau"] :]

        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()
        qddot_reduced = nlp["dof_mapping"].reduce(qddot)

        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def get_forces_from_contact(states, controls, nlp):
        q, qdot, qdot_reduced, tau = Dynamics.__dispatch_data(states, controls, nlp)

        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)

        return cs.getForce().to_mx()
