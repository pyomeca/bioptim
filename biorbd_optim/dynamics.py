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

    def forward_dynamics_torque_muscle_driven(states, controls, model):
        muscles_states = []

        for k in range (model.nbMuscleTotal()):
            muscles_states.append(biorbd.StateDynamics)

        q = states[:model.nbQ()]
        qdot = states[model.nbQ():]

        if model.nbMuscleTotal() > 0:
            for k in range (model.nbMuscleTotal()):
                muscles_states[k].setActivation(biorbd.StateDynamics, controls[k])
            Tau = model.muscularJointTorque(muscles_states, true, Q, QDot)
        else:
            Tau.setZero()

        Tau += controls

        qddot = biorbd.Model.ForwardDynamics(model, Q, QDot, Tau).to_mx()
        return vertcat(qdot, qddot)
