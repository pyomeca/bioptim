from casadi import vertcat
import biorbd


class Dynamics:
    @staticmethod
    def forward_dynamics_torque_driven(states, controls, model):
        """
        :param states: MX.sym from CasADi.
        :param controls: MX.sym from CasADi.
        :param model: Biorbd model loaded from the biorbd.Model() function.
        :return: Vertcat of derived states.
        """
        q = states[: model.nbQ()]
        qdot = states[model.nbQ() :]
        tau = controls

        qddot = biorbd.Model.ForwardDynamics(model, q, qdot, tau).to_mx()
        return vertcat(qdot, qddot)

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
