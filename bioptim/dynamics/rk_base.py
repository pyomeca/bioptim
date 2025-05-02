from casadi import horzcat

from .ode_solver_base import OdeSolverBase
from ..misc.enums import ControlType, DefectType
from ..misc.parameters_types import (
    Bool,
    Int,
    Str,
)


class RK(OdeSolverBase):
    """
    The base class for Runge-Kutta
    """

    def __init__(self, n_integration_steps: Int = 5, **kwargs):
        """
        Parameters
        ----------
        n_integration_steps: int
            The number of steps for the integration
        """

        super(RK, self).__init__(**kwargs)
        self.n_integration_steps = n_integration_steps

    @property
    def is_direct_collocation(self) -> Bool:
        return False

    @property
    def is_direct_shooting(self) -> Bool:
        return True

    @property
    def n_required_cx(self) -> Int:
        return 1

    @property
    def defects_type(self) -> DefectType:
        return DefectType.NOT_APPLICABLE

    def initialize_integrator(self, *args, **kwargs):
        return super(RK, self).initialize_integrator(
            *args, **kwargs, number_of_finite_elements=self.n_integration_steps
        )

    def x_ode(self, nlp):
        return nlp.states.scaled.cx_start

    def p_ode(self, nlp):
        if nlp.control_type in (
            ControlType.CONSTANT,
            ControlType.CONSTANT_WITH_LAST_NODE,
        ):
            return nlp.controls.scaled.cx_start
        elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
            return horzcat(nlp.controls.scaled.cx_start, nlp.controls.scaled.cx_end)
        elif nlp.control_type == ControlType.NONE:
            return nlp.cx()
        else:
            raise NotImplementedError(f"The control_type {nlp.control_type} is not implemented.")

    def a_ode(self, nlp):
        return nlp.algebraic_states.scaled.cx_start

    def d_ode(self, nlp):
        return nlp.numerical_timeseries.cx_start

    def __str__(self) -> Str:
        ode_solver_string = f"{self.integrator.__name__} {self.n_integration_steps} step"
        if self.n_integration_steps > 1:
            ode_solver_string += "s"

        return ode_solver_string
