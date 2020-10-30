from enum import Enum

from .problem import Problem
from ..misc.options_lists import UniquePerPhaseOptionList, OptionGeneric


class DynamicsTypeOption(OptionGeneric):
    def __init__(self, dynamics_type, dynamics=None, configure=None, dynamic_function=None, **params):
        params["dynamic_function"] = dynamic_function
        if not isinstance(dynamics_type, DynamicsType):
            configure = dynamics_type
            dynamics_type = DynamicsType.CUSTOM

        super(DynamicsTypeOption, self).__init__(type=dynamics_type, **params)
        self.dynamics = dynamics
        self.configure = configure


class DynamicsTypeList(UniquePerPhaseOptionList):
    def add(self, dynamics_type, **extra_parameters):
        if isinstance(dynamics_type, DynamicsTypeOption):
            self.copy(dynamics_type)

        else:
            super(DynamicsTypeList, self)._add(
                dynamics_type=dynamics_type, option_type=DynamicsTypeOption, **extra_parameters
            )


class DynamicsType(Enum):
    MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN = (Problem.muscle_excitations_and_torque_driven,)
    MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN = (Problem.muscle_activations_and_torque_driven,)
    MUSCLE_ACTIVATIONS_DRIVEN = (Problem.muscle_activations_driven,)
    MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT = (Problem.muscle_excitations_and_torque_driven_with_contact,)
    MUSCLE_EXCITATIONS_DRIVEN = (Problem.muscle_excitations_driven,)
    MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT = (Problem.muscle_activations_and_torque_driven_with_contact,)

    TORQUE_DRIVEN = (Problem.torque_driven,)
    TORQUE_ACTIVATIONS_DRIVEN = (Problem.torque_activations_driven,)
    TORQUE_ACTIVATIONS_DRIVEN_WITH_CONTACT = (Problem.torque_activations_driven_with_contact,)
    TORQUE_DRIVEN_WITH_CONTACT = (Problem.torque_driven_with_contact,)

    CUSTOM = (Problem.custom,)
