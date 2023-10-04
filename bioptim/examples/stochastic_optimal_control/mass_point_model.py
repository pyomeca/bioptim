"""
This file contains the model used in the article.
"""

from typing import Callable
from casadi import vertcat, MX, DM, sqrt
import numpy as np

from bioptim import DynamicsEvaluation, DynamicsFunctions, SocpType


class MassPointModel:
    """
    This allows to generate the same model as in the paper.
    """

    def __init__(
        self,
        motor_noise_magnitude: np.ndarray | DM = None,
        polynomial_degree: int = 1,
        socp_type: SocpType = None,
    ):
        self.socp_type = socp_type

        self.motor_noise_sym = MX()
        if motor_noise_magnitude is not None:
            print("Add motor noise")
            self.motor_noise_magnitude = motor_noise_magnitude
            self.motor_noise_sym = MX.sym("motor_noise", motor_noise_magnitude.shape[0])

        self.sensory_noise_magnitude = (
            []
        )  # This is necessary to have the right shapes in bioptim's internal constraints
        self.sensory_noise_sym = MX()
        self.sensory_reference = None

        n_noised_states = 4
        self.polynomial_degree = polynomial_degree
        self.matrix_shape_cov = (n_noised_states, n_noised_states)
        self.matrix_shape_m = (n_noised_states, n_noised_states * (polynomial_degree + 1))

        self.kapa = 10
        self.c = 1
        self.beta = 1

        self.super_ellipse_center_x = [0, 1]
        self.super_ellipse_center_y = [0, 0.5]
        self.super_ellipse_a = [1, 0.5]
        self.super_ellipse_b = [1, 2]
        self.super_ellipse_n = [4, 4]

    def serialize(self) -> tuple[Callable, dict]:
        return MassPointModel, dict(
            super_ellipse_center_x=self.super_ellipse_center_x,
            super_ellipse_center_y=self.super_ellipse_center_y,
            super_ellipse_a=self.super_ellipse_a,
            super_ellipse_b=self.super_ellipse_b,
            super_ellipse_n=self.super_ellipse_n,
        )

    @property
    def nb_q(self):
        return 2

    @property
    def nb_qdot(self):
        return 2

    @property
    def nb_u(self):
        return 2

    @property
    def nb_root(self):
        return 0

    @property
    def name_dof(self):
        return ["Px", "Py"]

    @property
    def name_u(self):
        return ["Ux", "Uy"]

    def dynamics(self, states, controls, parameters, stochastic_variables, nlp, with_noise=False):
        """
        The dynamics from equation (22).
        """
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        u = DynamicsFunctions.get(nlp.controls["u"], controls)
        nu = u.shape[0]
        motor_noise = 0
        if with_noise:
            motor_noise = self.motor_noise_sym
            # motor_noise = self.motor_noise_sym
        # qddot = -self.kapa * (q - u) - self.beta * qdot * sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise
        qddot = (
            -self.kapa * (q - u) - self.beta * qdot * sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise * 5
        )

        return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=None)

    def dynamics_numerical(self, states, controls, motor_noise=0):
        """
        The dynamics from equation (22).
        """
        q = states[: self.nb_q]
        qdot = states[self.nb_q :]
        u = controls

        qddot = (
            -self.kapa * (q - u) - self.beta * qdot * sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise * 5
        )

        return vertcat(qdot, qddot)
