from typing import List, Any, Optional, Callable
from abc import ABC, abstractmethod

from ...dynamics.dynamics_functions import DynamicsEvaluation
from ...misc.parameters_types import CXOptional
from ...dynamics.configure_variables import States, Controls, AlgebraicStates


class AbstractModel(ABC):
    def __init__(self):
        self.state_configuration: List[States] = []
        self.control_configuration: List[Controls] = []
        self.algebraic_configuration: List[AlgebraicStates] = []
        self.functions: List[Callable] = []
        self.fatigue = None

    @abstractmethod
    def dynamics(self) -> DynamicsEvaluation:
        raise NotImplementedError("You must implement the dynamics method in your model dynamics class")

    def get_rigid_contact_forces(
        self,
        time: Any,
        states: Any,
        controls: Any,
        parameters: Any,
        algebraic_states: Any,
        numerical_timeseries: Any,
        nlp: Any,
    ) -> CXOptional:
        return None

    @property
    def extra_dynamics(self) -> Optional[DynamicsEvaluation]:

        return None
