from casadi import vertcat
import biorbd


class Dynamics:
    @staticmethod
    def forward_dynamics_torque_driven(states, controls, nlp):
        """
        :param states: MX.sym from CasADi.
        :param controls: MX.sym from CasADi.
        :param nlp: Instance of an OptimalControlProgram class
        :return: Vertcat of derived states.
        """
        # TODO: Paul = move the dispatch in a private method
        nq = nlp["dof_mapping"].nb_reduced
        q = nlp["dof_mapping"].expand(states[:nq])
        qdot_reduced = states[nq:]
        qdot = nlp["dof_mapping"].expand(qdot_reduced)
        tau = nlp["dof_mapping"].expand(controls)

        qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()
        qddot_reduced = nlp["dof_mapping"].reduce(qddot)

        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_muscle_driven(states, controls, nlp):
        nq = nlp["dof_mapping"].nb_reduced
        q = nlp["dof_mapping"].expand(states[:nq])
        qdot_reduced = states[nq:]
        qdot = nlp["dof_mapping"].expand(qdot_reduced)
        residual_tau = nlp["dof_mapping"].expand(controls[: nlp["nbTau"]])

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
        nq = nlp["dof_mapping"].nb_reduced
        q = nlp["dof_mapping"].expand(states[:nq])
        qdot_reduced = states[nq:]
        qdot = nlp["dof_mapping"].expand(qdot_reduced)
        tau = nlp["dof_mapping"].expand(controls[: nlp["nbTau"]])

        cs = nlp["model"].getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau, cs)

        # TODO move to a proper forward_dynamics
        # qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp["model"], q, qdot, tau).to_mx()
        # qddot_reduced = nlp["dof_mapping"].reduce(qddot)
        return cs.getForce().to_mx()
