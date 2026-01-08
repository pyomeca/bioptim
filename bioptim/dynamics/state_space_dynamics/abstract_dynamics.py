from typing import List, Any, Optional, Callable
from abc import ABC, abstractmethod

from ..dynamics_evaluation import DynamicsEvaluation
from ...misc.enums import ContactType
from ...misc.parameters_types import CXOptional
from ..configure_variables import States, Controls, AlgebraicStates


class StateDynamics(ABC):
    def __init__(self, *args, fatigue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fatigue = fatigue

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the dynamics model
        """

    @property
    @abstractmethod
    def state_configuration_functions(self) -> List[States | Callable]:
        """
        Provide a list of state configuration functions of the model. These functions should configure the states of the
        model when called by themselves calling `ConfigureVariables.configure_new_variable` with the `as_state` parameter
        set to True.
        """

    @property
    @abstractmethod
    def control_configuration_functions(self) -> List[Controls | Callable]:
        """
        Provide a list of control configuration functions of the model. These functions should configure the controls of
        the model when called by themselves calling `ConfigureVariables.configure_new_variable` with the `as_control` parameter
        set to True.
        """

    @property
    @abstractmethod
    def algebraic_configuration_functions(self) -> List[AlgebraicStates | Callable]:
        """
        Provide a list of algebraic state configuration functions of the model. These functions should configure the algebraic
        states of the model when called by themselves calling `ConfigureVariables.configure_new_variable` with the `as_algebraic` parameter
        set to True.
        """

    @property
    @abstractmethod
    def extra_configuration_functions(self) -> List[Callable]:
        """
        Provide any extra configuration functions of the model. Please note, when technically all the configuration
        functions provided by state_configuration_functions, control_configuration_functions, and algebraic_configuration_functions
        could be put in this property, it is kept separate for clarity and easier understanding of the model structure.

        If for any reason, a function declared in this property needs to be passed around, the `nlp` object can be used
        as it is availble everywhere where dynamics and variables are used (constraints, objectives, dynamics, etc.).
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
