"""
This file contains the model used in the article.
"""

from typing import Callable
from casadi import vertcat, DM, sqrt
import numpy as np

from bioptim import (
    DynamicsEvaluation,
    DynamicsFunctions,
    OdeSolver,
    States,
    Controls,
    AlgebraicStates,
    ConfigureVariables,
    SocpType,
    AbstractStateSpaceDynamics,
)


class MassPointModel(AbstractStateSpaceDynamics):
    """
    This allows to generate the same model as in the paper.
    """

    def __init__(self, motor_noise_magnitude: np.ndarray | DM = None, polynomial_degree: int = 1, problem_type=None):
        self.problem_type = problem_type

        self.motor_noise_magnitude = motor_noise_magnitude

        # This is necessary to have the right shapes in bioptim's internal constraints
        self.sensory_noise_magnitude = np.ndarray((0, 1))

        self.sensory_reference = None
        self.contact_types = ()

        n_noised_states = 4
        self.polynomial_degree = polynomial_degree
        self.matrix_shape_cov = (n_noised_states, n_noised_states)

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
    def nb_tau(self):
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

    def configure_u(self, ocp, nlp, as_states, as_controls, as_algebraic_states):
        return ConfigureVariables.configure_new_variable(
            "u", self.name_u, ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False
        )


class MassPointDynamicsModel(MassPointModel):
    def __init__(
        self, problem_type: SocpType, motor_noise_magnitude: np.ndarray | DM = None, polynomial_degree: int = 1
    ):
        super().__init__(
            problem_type=problem_type, motor_noise_magnitude=motor_noise_magnitude, polynomial_degree=polynomial_degree
        )
        self.fatigue = None
        self.state_configuration = [States.Q, States.QDOT]
        self.control_configuration = [self.configure_u]
        self.algebraic_configuration = []
        self.functions = []

    def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        """
        The dynamics from equation (22).
        """
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        u = DynamicsFunctions.get(nlp.controls["u"], controls)
        motor_noise = 0

        qddot = -self.kapa * (q - u) - self.beta * qdot * sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise

        dxdt = vertcat(qdot, qddot)
        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            # Defects
            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            defects = vertcat(slope_q, slope_qdot) * nlp.dt - vertcat(qdot, qddot) * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    def dynamics_numerical(self, states, controls, motor_noise=0):
        """
        The dynamics from equation (22).
        """
        # to avoid dimension pb with solve_ivp
        if states.ndim == 2:
            states = states.reshape((-1,))

        q = states[: self.nb_q]
        qdot = states[self.nb_q :]
        u = controls
        qddot = -self.kapa * (q - u) - self.beta * qdot * sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise

        return vertcat(qdot, qddot)


class StochasticMassPointDynamicsModel(MassPointModel):
    def __init__(
        self, problem_type: SocpType, motor_noise_magnitude: np.ndarray | DM = None, polynomial_degree: int = 1
    ):
        super().__init__(
            problem_type=problem_type, motor_noise_magnitude=motor_noise_magnitude, polynomial_degree=polynomial_degree
        )
        self.state_configuration = [States.Q, States.QDOT]
        self.control_configuration = [
            self.configure_u,
            lambda ocp, nlp, as_states, as_controls, as_algebraic_states: Controls.COV(
                ocp, nlp, as_states, as_controls, as_algebraic_states, n_noised_states=4
            ),
        ]
        self.algebraic_configuration = [
            lambda ocp, nlp, as_states, as_controls, as_algebraic_states: AlgebraicStates.M(
                ocp, nlp, as_states, as_controls, as_algebraic_states, n_noised_states=4
            )
        ]
        self.functions = []
        self.fatigue = None

    def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        """
        The dynamics from equation (22).
        """
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        u = DynamicsFunctions.get(nlp.controls["u"], controls)
        motor_noise = 0

        qddot = -self.kapa * (q - u) - self.beta * qdot * sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise

        dxdt = vertcat(qdot, qddot)
        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            # Defects
            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            defects = vertcat(slope_q, slope_qdot) * nlp.dt - vertcat(qdot, qddot) * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    def dynamics_numerical(self, states, controls, motor_noise=0):
        """
        The dynamics from equation (22).
        """
        # to avoid dimension pb with solve_ivp
        if states.ndim == 2:
            states = states.reshape((-1,))

        q = states[: self.nb_q]
        qdot = states[self.nb_q :]
        u = controls
        qddot = -self.kapa * (q - u) - self.beta * qdot * sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise

        return vertcat(qdot, qddot)

    def extra_dynamics(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        """
        The dynamics from equation (22).
        """
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        u = DynamicsFunctions.get(nlp.controls["u"], controls)
        motor_noise = DynamicsFunctions.get(nlp.parameters["motor_noise"], parameters)

        qddot = -self.kapa * (q - u) - self.beta * qdot * sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise

        dxdt = vertcat(qdot, qddot)
        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            # Defects
            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            defects = vertcat(slope_q, slope_qdot) * nlp.dt - vertcat(qdot, qddot) * nlp.dt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)
