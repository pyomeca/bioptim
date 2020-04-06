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
        nq = nlp.dof_mapping.nb_reduced
        q = nlp.dof_mapping.expand(states[:nq])
        qdot_reduced = states[nq:]
        qdot = nlp.dof_mapping.expand(qdot_reduced)
        tau = nlp.dof_mapping.expand(controls)

        qddot = biorbd.Model.ForwardDynamics(nlp.model, q, qdot, tau).to_mx()
        qddot_reduced = nlp.dof_mapping.reduce(qddot)

        return vertcat(qdot_reduced, qddot_reduced)

        # def forward_dynamics_torque_muscle_driven(states, controls, model):
        #     muscles_states = []
        #
        #     for k in range (model.nbMuscleTotal()):
        #         muscles_states.append(StateDynamics())
        #
        #
        #     q = states[:model.nbQ()]
        #     qdot = states[model.nbQ():]
        #
        #     if model.nbMuscleTotal() > 0:
        #         for k in range (model.nbMuscleTotal()):
        #             muscles_states[k].setActivation(controls[k])
        #         Tau = model.muscularJointTorque(muscles_states, true, Q, QDot);
        #
        #     else:
        #         Tau.setZero();
        #
        #     Tau += controls
        #
        #
        #     qddot = biorbd.Model.ForwardDynamics(model, Q, QDot, Tau).to_mx()
        #     return vertcat(qdot, qddot)
