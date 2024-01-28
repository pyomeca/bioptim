"""
This file contains the model used in the example of Lyapunov matrix in Rockit.
"""

from typing import Callable
from casadi import vertcat, MX, DM, sqrt
import numpy as np

from bioptim import DynamicsEvaluation, DynamicsFunctions, SocpType


class RockitModel:
    """
    This allows to generate the same model as in Rockit's example.
    """

    def __init__(
        self,
        motor_noise_magnitude: np.ndarray | DM = None,
        polynomial_degree: int = 1,
        socp_type=None,
    ):
        self.socp_type = socp_type

        self.motor_noise_magnitude = motor_noise_magnitude

        self.sensory_noise_magnitude = (
            []
        )  # This is necessary to have the right shapes in bioptim's internal constraints
        self.sensory_noise_magnitude = np.ndarray((0, 1))
        self.sensory_reference = None

        n_noised_states = 2
        self.polynomial_degree = polynomial_degree
        self.matrix_shape_cov = (n_noised_states, n_noised_states)
        self.matrix_shape_m = (n_noised_states, n_noised_states * (polynomial_degree + 1))

    @property
    def nb_q(self):
        return 1

    @property
    def nb_qdot(self):
        return 1

    @property
    def nb_tau(self):
        return 1

    @property
    def nb_root(self):
        return 0

    @property
    def name_dof(self):
        return ["x"]

    @property
    def name_u(self):
        return ["U"]

    def dynamics(self, states, controls, parameters, algebraic_states, nlp, with_noise=False):
        """
        The dynamics from equation (line 42).
        """
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        u = DynamicsFunctions.get(nlp.controls["u"], controls)

        motor_noise = 0
        if with_noise:
            motor_noise = nlp.parameters["motor_noise"].mx

        qddot = -0.1 * (1 - q**2) * qdot - q + u + motor_noise

        return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=None)

    def dynamics_numerical(self, states, controls, motor_noise=0):
        q = states[: self.nb_q]
        qdot = states[self.nb_q:]
        u = controls

        qddot = -0.1 * (1 - q**2) * qdot - q + u + motor_noise

        return vertcat(qdot, qddot)

    def serialize(self):
        return RockitModel
