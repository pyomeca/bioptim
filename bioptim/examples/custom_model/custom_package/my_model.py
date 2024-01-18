"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""
from typing import Callable

import numpy as np
from casadi import sin, MX


class MyModel:
    """
    This is a custom model that inherits from bioptim.CustomModel
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self):
        # custom values for the example
        self.com = MX(np.array([-0.0005, 0.0688, -0.9542]))
        self.inertia = MX(0.0391)

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

    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        # This is where you can implement your own forward dynamics
        # with casadi it your are dealing with mechanical systems
        d = 0  # damping
        L = self.com[2]
        I = self.inertia
        m = self.mass
        g = 9.81
        return 1 / (I + m * L**2) * (-qdot[0] * d - g * m * L * sin(q[0]) + tau[0])

    # def system_dynamics(self, *args):
    # This is where you can implement your system dynamics with casadi if you are dealing with other systems
