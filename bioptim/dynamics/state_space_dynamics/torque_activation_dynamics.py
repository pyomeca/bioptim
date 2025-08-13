from ..configure_variables import Controls
from ..dynamics_functions import DynamicsFunctions
from ..fatigue.fatigue_dynamics import FatigueList
from ...misc.parameters_types import Bool
from .torque_dynamics import TorqueDynamics


class TorqueActivationDynamics(TorqueDynamics):
    def __init__(self, with_residual_torque: Bool, fatigue: FatigueList):
        super().__init__(fatigue=fatigue)

        if with_residual_torque:
            self.control_configuration += [Controls.RESIDUAL_TAU]
        self.with_residual_torque = with_residual_torque

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):
        if nlp.model.fatigue is not None:
            raise NotImplementedError("Fatigue is not implemented yet for torque activation dynamics.")

        # Get variables from the right place
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau_activation = DynamicsFunctions.get(nlp.controls["tau"], controls)

        # Convert tau activations to joint torque
        tau = nlp.model.torque()(tau_activation, q, qdot, nlp.parameters.cx)
        if self.with_residual_torque:
            tau += DynamicsFunctions.get(nlp.controls["residual_tau"], controls)

        # Add additional torques
        tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        # Get external forces
        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q, qdot, tau, external_forces
