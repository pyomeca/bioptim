from enum import Enum

from .problem import Problem
from ..misc.options_lists import UniquePerPhaseOptionList, OptionGeneric


class DynamicsOption(OptionGeneric):
    def __init__(self, dynamics=None, configure=None, **params):
        super(DynamicsOption, self).__init__(**params)
        self.dynamics = dynamics
        self.configure = configure


class DynamicsTypeList(UniquePerPhaseOptionList):
    def add(self, type, dynamic_function=None, phase=-1):
        extra_arguments = {}
        if not isinstance(type, DynamicsType):
            extra_arguments["configure"] = type
            type = DynamicsType.CUSTOM

        super(DynamicsTypeList, self)._add(
            option_type=DynamicsOption, type=type, phase=phase, dynamic_function=dynamic_function, **extra_arguments
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
