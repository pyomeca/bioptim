from typing import Callable, Any, Union

from enum import Enum
from .problem import Problem
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric


class DynamicsFcn(Enum):
    """
    Selection of valid dynamics functions
    """

    TORQUE_DRIVEN = (Problem.torque_driven,)
    TORQUE_DRIVEN_WITH_CONTACT = (Problem.torque_driven_with_contact,)
    TORQUE_ACTIVATIONS_DRIVEN = (Problem.torque_activations_driven,)
    TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT = (Problem.torque_activations_driven_with_contact,)

    MUSCLE_ACTIVATIONS_DRIVEN = (Problem.muscle_activations_driven,)
    MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN = (Problem.muscle_activations_and_torque_driven,)
    MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT = (Problem.muscle_activations_and_torque_driven_with_contact,)

    MUSCLE_EXCITATIONS_DRIVEN = (Problem.muscle_excitations_driven,)
    MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN = (Problem.muscle_excitations_and_torque_driven,)
    MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT = (Problem.muscle_excitations_and_torque_driven_with_contact,)

    CUSTOM = (Problem.custom,)


class Dynamics(OptionGeneric):
    """
    A placeholder for the chosen dynamics by the user

    Attributes
    ----------
    dynamic_function: function
        The custom dynamic function provided by the user
    configure: function
        The configuration function provided by the user that declares the NLP (states and controls),
        usually only necessary when defining custom functions

    """

    def __init__(
        self,
        dynamics_type: Union[Callable, DynamicsFcn],
        configure: Callable = None,
        dynamic_function: Callable = None,
        **params
    ):
        """
        Parameters
        ----------
        dynamics_type: Union[Callable, DynamicsFcn]
            The chosen dynamic functions
        configure: function
            The configuration function provided by the user that declares the NLP (states and controls),
            usually only necessary when defining custom functions
        dynamic_function: function
            The custom dynamic function provided by the user
        params: dict
            Any parameters to pass to the dynamic and configure functions
        """

        if not isinstance(dynamics_type, DynamicsFcn):
            configure = dynamics_type
            dynamics_type = DynamicsFcn.CUSTOM

        super(Dynamics, self).__init__(type=dynamics_type, **params)
        self.dynamic_function = dynamic_function
        self.configure = configure


class DynamicsList(UniquePerPhaseOptionList):
    """
    A list of Dynamics if more than one is required, typically when more than one phases are declared

    Methods
    -------
    add(dynamics: DynamicsFcn, **extra_parameters)
        Add a new Dynamics to the list
    print(self)
        Print the DynamicsList to the console
    """

    def add(self, dynamics_type: Union[Callable, Dynamics, DynamicsFcn], **extra_parameters: Any):
        """
        Add a new Dynamics to the list

        Parameters
        ----------
        dynamics_type: Union[Callable, Dynamics, DynamicsFcn]
            The chosen dynamic functions
        extra_parameters: dict
            Any parameters to pass to Dynamics
        """

        if isinstance(dynamics_type, Dynamics):
            self.copy(dynamics_type)

        else:
            super(DynamicsList, self)._add(dynamics_type=dynamics_type, option_type=Dynamics, **extra_parameters)

    def print(self):
        """
        Print the DynamicsList to the console
        """
        raise NotImplementedError("Printing of DynamicsList is not ready yet")
