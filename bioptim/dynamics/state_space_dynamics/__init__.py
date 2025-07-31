from typing import TypeAlias

from .abstract_dynamics import StateSpaceDynamics
from .joint_acceleration_dynamics import JointAccelerationDynamics
from .torque_dynamics_holonomic import HolonomicTorqueDynamics
from .muscle_dynamics import MusclesDynamics
from .stochastic_dynamics import StochasticTorqueDynamics
from .stochastic_torque_dynamics_free_floating import StochasticTorqueFreeFloatingBaseDynamics
from .torque_activation_dynamics import TorqueActivationDynamics
from .torque_derivative_dynamics import TorqueDerivativeDynamics
from .torque_dynamics_free_floating_base import TorqueFreeFloatingBaseDynamics
from .torque_dynamics import TorqueDynamics
from .torque_dynamics_variational import VariationalTorqueDynamics

StateSpaceDynamics: TypeAlias = (
    TorqueDynamics
    | StochasticTorqueDynamics
    | TorqueFreeFloatingBaseDynamics
    | StochasticTorqueFreeFloatingBaseDynamics
    | MusclesDynamics
    | TorqueActivationDynamics
    | TorqueDerivativeDynamics
    | JointAccelerationDynamics
    | HolonomicTorqueDynamics
    | VariationalTorqueDynamics
)
