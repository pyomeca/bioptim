from enum import Enum

from .problem import Problem


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
