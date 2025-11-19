from typing import List, Any, Optional, Callable
from abc import ABC, abstractmethod

from ..dynamics_evaluation import DynamicsEvaluation
from ...misc.enums import ContactType
from ...misc.parameters_types import CXOptional
from ..configure_variables import States, Controls, AlgebraicStates


class StateDynamics(ABC):
    def __init__(self):
        self.state_configuration: List[States] = []
        self.control_configuration: List[Controls] = []
        self.algebraic_configuration: List[AlgebraicStates] = []
        self.functions: List[Callable] = []
        self.fatigue = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the dynamics model
        """

    @property
    @abstractmethod
    def name_dofs(self) -> List[str]:
        """
        Get the name of the degrees of freedom of the model. For historical reasons, this property must exists in the
        protocol, however, it only make sense to provide values for this if "q" is part of the state variables (then
        this should return the names of the "q_names").
        """

    @abstractmethod
    def dynamics(self) -> DynamicsEvaluation:
        """
        When inheriting from this class, the dynamics method must be implemented to provide the system dynamics.
        """

    @property
    def extra_dynamics(self) -> Optional[DynamicsEvaluation]:
        """
        When inheriting from this class, the extra_dynamics @property can be overridden to provide additional
        dynamics using the same arguments as the dynamics function.
        """
        return None


class StateDynamicsWithContacts(StateDynamics):
    def get_rigid_contact_forces(
        self,
        time: CXOptional,
        states: CXOptional,
        controls: CXOptional,
        parameters: CXOptional,
        algebraic_states: CXOptional,
        numerical_timeseries: Any,
        nlp: Any,
    ) -> Any:
        """
        When inheriting from this class, the get_rigid_contact_forces method must be implemented to provide rigid contact forces.
        """
        raise NotImplementedError(
            "You must implement the get_rigid_contact_forces method if your model has rigid contacts."
        )

    @property
    @abstractmethod
    def contact_types(self) -> List[ContactType]:
        """
        Get the contact types of the model
        """
