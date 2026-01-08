from casadi import vertcat

from ..configure_variables import States, Controls
from ..dynamics_functions import DynamicsFunctions
from .torque_dynamics import TorqueDynamics


class TorqueFreeFloatingBaseDynamics(TorqueDynamics):
    """
    This class is used to create a model actuated through joint torques with a free floating base.

    x = [q_roots, q_joints, qdot_roots, qdot_joints]
    u = [tau_joints]
    """

    def __init__(self, **kwargs):
        super().__init__(fatigue=None, **kwargs)

    @property
    def state_configuration_functions(self):
        return [States.Q_ROOTS, States.Q_JOINTS, States.QDOT_ROOTS, States.QDOT_JOINTS]

    @property
    def control_configuration_functions(self):
        return [Controls.TAU_JOINTS]

    @staticmethod
    def get_q_qdot_indices(nlp):
        """
        Get the indices of the states and controls in the free floating base dynamics
        """
        return list(nlp.states["q_roots"].index) + list(nlp.states["q_joints"].index), list(
            nlp.states["qdot_roots"].index
        ) + list(nlp.states["qdot_joints"].index)

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):

        # Get variables from the right place
        q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
        q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
        qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
        qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
        tau_joints = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)

        q_full = vertcat(q_roots, q_joints)
        qdot_full = vertcat(qdot_roots, qdot_joints)
        tau_full = vertcat(nlp.cx(nlp.model.nb_root, 1), tau_joints)

        # Add additional torques
        tau_full += DynamicsFunctions.collect_tau(nlp, q_full, qdot_full, parameters)

        # Get external forces
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q_full, qdot_full, tau_full, external_forces

    def get_basic_slopes(self, nlp):
        slope_q = vertcat(nlp.states_dot["q_roots"].cx, nlp.states_dot["q_joints"].cx)
        slope_qdot = vertcat(nlp.states_dot["qdot_roots"].cx, nlp.states_dot["qdot_joints"].cx)
        return slope_q, slope_qdot
