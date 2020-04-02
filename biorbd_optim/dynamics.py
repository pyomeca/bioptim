from casadi import vertcat
import biorbd


class Dynamics:
    @staticmethod
    def forward_dynamics_torque_driven(states, controls, model):
        q = states[:model.nbQ()]
        qdot = states[model.nbQ():]
        tau = controls

        qddot = biorbd.Model.ForwardDynamics(model, q, qdot, tau).to_mx()
        return vertcat(qdot, qddot)
