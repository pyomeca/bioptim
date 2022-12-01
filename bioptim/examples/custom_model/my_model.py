"""
This script implements a custom model to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd. This is an example of how to use bioptim with a custom model.
"""
from abc import ABCMeta

import numpy as np
from casadi import sin, MX

from bioptim import (
    CustomModel,
)


class MyModel(CustomModel, metaclass=ABCMeta):
    """
    This is a custom model that inherits from bioptim.CustomModel
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self):
        # custom values for the example
        self.com = MX(np.array([-0.0005, 0.0688, -0.9542]))
        self.inertia = MX(0.0391)

    # ---- absolutely needed to be implemented ---- #
    def nb_quat(self):
        """Number of quaternion in the model"""
        return 0

    # ---- Needed for the example ---- #
    def nb_tau(self):
        return 1

    def nb_q(self):
        return 1

    def nb_qdot(self):
        return 1

    def nb_qddot(self):
        return 1

    def mass(self):
        return 1

    def name_dof(self):
        return ["rotx"]

    def path(self):
        # note: can we do something with this?
        return None

    def forward_dynamics(self, q, qdot, tau, fext=None, f_contacts=None):
        # This is where you can implement your own forward dynamics
        # with casadi it your are dealing with mechanical systems
        d = 0  # damping
        L = self.com[2]
        I = self.inertia
        m = self.mass()
        g = 9.81
        return 1 / (I + m * L**2) * (-qdot[0] * d - g * m * L * sin(q[0]) + tau[0])

    # def system_dynamics(self, *args):
    # This is where you can implement your system dynamics with casadi if you are dealing with other systems
