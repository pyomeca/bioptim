from ..configure_variables import States, Controls
from ..dynamics_evaluation import DynamicsEvaluation
from ..configure_variables import ConfigureVariables
from .abstract_dynamics import StateDynamics


class VariationalTorqueDynamics(StateDynamics):

    def __init__(self):
        super().__init__()
        # TODO: QDOT are declared as parameters in VariationalOptimalControlProgram, these parameters should instead be
        #   declared here as self.parameter_configuration = [Parameter.QDOT_INIT, Parameter.QDOT_END]
        # If the model has no holonomic constraint, there will be no lambdas defined
        self.state_configuration = [States.Q, Controls.LAMBDA]
        self.control_configuration = [Controls.TAU]
        self.algebraic_configuration = []
        self.functions = [
            ConfigureVariables.configure_variational_functions,
        ]

    def dynamics(
        self,
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ):
        # For variational integrator, the traditional dynamics is skipped, but its declaration is mandatory in bioptim
        return DynamicsEvaluation(
            dxdt=nlp.cx(
                nlp.states.shape,
            ),
            defects=nlp.cx(0),
        )

    @property
    def extra_dynamics(self):
        return None
