from abc import abstractmethod
from typing import Any

from .fatigue_dynamics import FatigueModel, MultiFatigueInterface
from ..dynamics_functions import DynamicsFunctions
from ...misc.enums import VariableType
from ...misc.parameters_types import (
    Bool,
    Int,
    Float,
    Str,
    CX,
)


class MuscleFatigue(FatigueModel):
    """
    A placeholder for fatigue dynamics.
    """

    @abstractmethod
    def apply_dynamics(self, target_load: Float | CX, *states: Any):
        """
        Apply the dynamics to the system (return dx/dt)

        Parameters
        ----------
        target_load: float | MX | SX
            The target load the muscle must accomplish
        states: Any
            The list of states the dynamics should compute

        Returns
        -------
        The derivative of all states
        """

    @staticmethod
    def type() -> Str:
        return "muscles"

    @property
    def multi_type(self):
        return MultiFatigueInterfaceMuscle

    def _get_target_load(self, nlp, controls: CX, index: Int) -> CX:
        if self.type() not in nlp.controls:
            raise NotImplementedError(f"Fatigue dynamics without {self.type()} controls is not implemented yet")

        return DynamicsFunctions.get(nlp.controls[self.type()], controls)[index, :]

    def dynamics(self, dxdt: CX, nlp, index: Int, states: CX, controls: CX) -> CX:
        target_load = self._get_target_load(nlp, controls, index)
        fatigue = [
            DynamicsFunctions.get(nlp.states[f"{self.type()}_{s}"], states)[index, :]
            for s in self.suffix(VariableType.STATES)
        ]
        current_dxdt = self.apply_dynamics(target_load, *fatigue)

        for i, s in enumerate(self.suffix(variable_type=VariableType.STATES)):
            dxdt[nlp.states[f"{self.type()}_{s}"].index[index], :] = current_dxdt[i]

        return dxdt

    def default_state_only(self) -> Bool:
        return False

    def default_apply_to_joint_dynamics(self) -> Bool:
        return False


class MultiFatigueInterfaceMuscle(MultiFatigueInterface):
    @staticmethod
    def model_type() -> Str:
        """
        The type of Fatigue
        """
        return "muscles"

    def default_state_only(self) -> Bool:
        return False

    def default_apply_to_joint_dynamics(self) -> Bool:
        return False
