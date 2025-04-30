from casadi import if_else

from .muscle_fatigue import MuscleFatigue
from .tau_fatigue import TauFatigue
from ...misc.enums import VariableType
from ...misc.parameters_types import (
    Bool,
    Float,
    Str,
    IntTuple,
    StrTuple,
    AnyTuple,
)


class EffortPerception(MuscleFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, effort_threshold: Float, effort_factor: Float, scaling: Float = 1):
        """
        Parameters
        ----------
        effort_threshold: float
            The activation level at which the structure starts to build/relax the effort perception
        effort_factor: float
            Effort perception build up rate
        scaling: float
            The scaling factor to the max value of the target load
        """

        super(EffortPerception, self).__init__(scaling=scaling)
        self.effort_threshold = effort_threshold / self.scaling
        self.effort_factor = effort_factor

    @staticmethod
    def suffix(variable_type: VariableType) -> StrTuple:
        if variable_type == VariableType.STATES:
            return ("mf",)
        else:
            return ("",)

    @staticmethod
    def color() -> StrTuple:
        return ("tab:brown",)

    def default_state_only(self) -> Bool:
        return True

    def default_initial_guess(self) -> IntTuple:
        return (0,)

    def default_bounds(self, variable_type: VariableType) -> AnyTuple:
        return (0,), (1,)

    def apply_dynamics(self, target_load, *states):
        effort = states[0]

        effort_perception_dot = (
            self.effort_factor
            * (target_load - self.effort_threshold)
            * if_else(
                target_load > self.effort_threshold,
                1 / (1 - self.effort_threshold) * (1 - effort),
                1 / self.effort_threshold * effort,
            )
        )
        return effort_perception_dot

    @staticmethod
    def dynamics_suffix() -> Str:
        return ""

    @staticmethod
    def fatigue_suffix() -> Str:
        return "mf"


class TauEffortPerception(TauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, minus: EffortPerception, plus: EffortPerception, **kwargs):
        """
        Parameters
        ----------
        minus: EffortPerception
            The Michaud model for the negative tau
        plus: EffortPerception
            The Michaud model for the positive tau
        """

        super(TauEffortPerception, self).__init__(
            minus=minus, plus=plus, state_only=True, apply_to_joint_dynamics=False, **kwargs
        )

    @staticmethod
    def default_state_only() -> Bool:
        return True

    @staticmethod
    def dynamics_suffix() -> Str:
        return "effort"

    @staticmethod
    def fatigue_suffix() -> Str:
        return "fatigue"
