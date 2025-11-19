"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""

from bioptim import NonLinearProgram, DynamicsEvaluation, StateDynamics, States, ConfigureVariables, Controls
from casadi import sin, MX, Function, vertcat


class MyModel(StateDynamics):
    """
    This is a custom model that inherits from bioptim.StateDynamics which is the base class for all dynamic models in bioptim.
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self):
        super().__init__()
        # The next two lines tells bioptim what are the states and controls of the model.
        # One could use [States.Q, States.QDOT], but for sake of showcasing how to define custom variables,
        # we define "State.Q" manually (but it could obviously be any kind of variable).
        #   From that point on, "nlp.states" will have the variable (here "q"), and "len(name_elements)"
        #   (here "name_elements=self.name_dofs") will determine the length of that variable.
        self.state_configuration = [
            lambda **kwargs: ConfigureVariables.configure_new_variable("q", self.name_dofs, as_states=True, **kwargs),
            States.QDOT,
        ]
        self.control_configuration = [Controls.TAU]

        # custom values for the example
        self.com = [-0.0005, 0.0688, -0.9542]
        self.inertia = 0.0391
        self.mass = 1.0

        # CasADi symbols for the custom forward dynamics, these are not strictly necessary but helps prevent cases of
        # free variables in the CasADi functions
        self.q = MX.sym("q", 1)
        self.qdot = MX.sym("qdot", 1)
        self.tau = MX.sym("tau", 1)

    # ---- Mandatory methods ---- #
    @property
    def name(self):
        return "MyModel"

    @property
    def name_dofs(self):
        return ["rotx"]

    def dynamics(
        self,
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        Parameters
        ----------
        states: MX | SX
            The state of the system
        controls: MX | SX
            The controls of the system
        parameters: MX | SX
            The parameters acting on the system
        nlp: NonLinearProgram
            A reference to the phase
        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        return DynamicsEvaluation(
            dxdt=vertcat(states[1], self.forward_dynamics()(states[0], states[1], controls[0], [])),
            defects=None,
        )

    def forward_dynamics(self) -> Function:
        # This is where you can implement your own forward dynamics
        # with casadi it your are dealing with mechanical systems
        d = 0  # damping
        L = self.com[2]
        I = self.inertia
        m = self.mass
        g = 9.81
        casadi_return = 1 / (I + m * L**2) * (-self.qdot * d - g * m * L * sin(self.q) + self.tau)
        casadi_fun = Function("forward_dynamics", [self.q, self.qdot, self.tau, MX()], [casadi_return])
        return casadi_fun
