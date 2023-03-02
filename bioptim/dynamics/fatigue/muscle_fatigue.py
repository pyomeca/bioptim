from abc import abstractmethod
from typing import Any

from casadi import SX, MX

from .fatigue_dynamics import FatigueModel, MultiFatigueInterface
from ..dynamics_functions import DynamicsFunctions
from ...misc.enums import VariableType


class MuscleFatigue(FatigueModel):
    """
    A placeholder for fatigue dynamics.
    """

    @abstractmethod
    def apply_dynamics(self, target_load: float | MX | SX, *states: Any):
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
    def type() -> str:
        return "muscles"

    @property
    def multi_type(self):
        return MultiFatigueInterfaceMuscle

    def _get_target_load(self, nlp, controls, index):
        if self.type() not in nlp.controls:
            raise NotImplementedError(f"Fatigue dynamics without {self.type()} controls is not implemented yet")

        return DynamicsFunctions.get(nlp.controls[self.type()], controls)[index, :]

    def dynamics(self, dxdt, nlp, index, states, controls):
        target_load = self._get_target_load(nlp, controls, index)
        fatigue = [
            DynamicsFunctions.get(nlp.states[f"{self.type()}_{s}"], states)[index, :]
            for s in self.suffix(VariableType.STATES)
        ]
        current_dxdt = self.apply_dynamics(target_load, *fatigue)

        for i, s in enumerate(self.suffix(variable_type=VariableType.STATES)):
            dxdt[nlp.states[f"{self.type()}_{s}"].index[index], :] = current_dxdt[i]

        return dxdt

    def default_state_only(self) -> bool:
        return False

    def default_apply_to_joint_dynamics(self) -> bool:
        return False


class MultiFatigueInterfaceMuscle(MultiFatigueInterface):
    @staticmethod
    def model_type() -> str:
        """
        The type of Fatigue
        """
        return "muscles"

    def default_state_only(self) -> bool:
        return False

    def default_apply_to_joint_dynamics(self) -> bool:
        return False
