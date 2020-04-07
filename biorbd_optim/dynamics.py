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

    @staticmethod
    def forward_dynamics_torque_muscle_driven(states, controls, model):
        q = states[:model.nbQ()]
        qdot = states[model.nbQ():]
        muscles_states = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
        muscles_activations = controls[:model.nbMuscleTotal()]
        residual_tau = controls[model.nbMuscleTotal():]

        for k in range(model.nbMuscleTotal()):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = model.muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamics(model, q, qdot, tau).to_mx()
        return vertcat(qdot, qddot)
