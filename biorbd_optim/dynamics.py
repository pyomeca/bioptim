from casadi import vertcat
import biorbd


class Dynamics:
    @staticmethod
    def forward_dynamics_torque_driven(states, controls, model):
        """
        :param states: MX.sym form CasaDI.
        :param controls: MX.sym form CasaDI.
        :param model: Biorbd model loaded from the biorbd.Model() function.
        :return: Vertcat of derived states.
        """
        q = states[:model.nbQ()]
        qdot = states[model.nbQ():]
        tau = controls

        qddot = biorbd.Model.ForwardDynamics(model, q, qdot, tau).to_mx()
        return vertcat(qdot, qddot)
