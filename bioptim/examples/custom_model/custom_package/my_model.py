"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""

from typing import Callable

import numpy as np
from casadi import sin, MX, Function, vertcat
from typing import Callable
from bioptim import NonLinearProgram, DynamicsEvaluation, TorqueDynamics


class MyModel(TorqueDynamics):
    """
    This is a custom model that inherits from bioptim.CustomModel
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self):
        super().__init__()
        # custom values for the example
        self.com = MX(np.array([-0.0005, 0.0688, -0.9542]))
        self.inertia = MX(0.0391)
        self.q = MX.sym("q", 1)
        self.qdot = MX.sym("qdot", 1)
        self.tau = MX.sym("tau", 1)
        self.contact_types = ()

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return MyModel, dict(com=self.com, inertia=self.inertia)

    # ---- Needed for the example ---- #

    @property
    def name(self) -> str:
        return "MyModel"

    @property
    def nb_tau(self):
        return 1

    @property
    def nb_q(self):
        return 1

    @property
    def nb_qdot(self):
        return 1

    @property
    def nb_qddot(self):
        return 1

    @property
    def mass(self):
        return 1

    @property
    def name_dof(self):
        return ["rotx"]

    def forward_dynamics(self, with_contact=False) -> Function:
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
            dxdt=vertcat(states[1], self.forward_dynamics(with_contact=False)(states[0], states[1], controls[0], [])),
            defects=None,
        )
