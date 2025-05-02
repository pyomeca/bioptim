from casadi import vertcat, lt, gt, if_else

from .muscle_fatigue import MuscleFatigue, MultiFatigueInterfaceMuscle
from .tau_fatigue import TauFatigue
from ...misc.enums import VariableType
from ...misc.parameters_types import (
    Bool,
    Float,
    Str,
    StrTuple,
    FloatTuple,
    CX,
)


class MichaudFatigue(MuscleFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    @staticmethod
    def dynamics_suffix() -> Str:
        return "ma"

    @staticmethod
    def fatigue_suffix() -> Str:
        return "mf"

    def __init__(
        self,
        LD: Float,
        LR: Float,
        F: Float,
        R: Float,
        effort_threshold: Float,
        effort_factor: Float,
        stabilization_factor: Float = 1,
        **kwargs
    ):
        """
        Parameters
        ----------
        LD: float
            Development coefficient
        LR: float
            Relaxation coefficient
        F: float
            Recovery rate
        R: float
            Relaxation rate
        effort_threshold: float
            The activation level at which the structure starts to build/relax the effort perception
        effort_factor: float
            Effort perception build up rate
        stabilization_factor: float
            Stabilization factor so: ma + mr + mf => 1
        scaling: float
            The scaling factor to convert so input / scale => TL
        """

        super(MichaudFatigue, self).__init__(**kwargs)
        # Xia parameters
        self.LD = LD
        self.LR = LR
        self.F = F
        self.R = R

        # Stabilisation factor and effort factor
        self.stabilization_factor = stabilization_factor
        self.effort_factor = effort_factor
        self.effort_threshold = effort_threshold

    @staticmethod
    def suffix(variable_type: VariableType) -> StrTuple:
        if variable_type == VariableType.STATES:
            return "ma", "mr", "mf_xia", "mf"
        else:
            return ("",)

    @staticmethod
    def color() -> StrTuple:
        return "tab:green", "tab:orange", "tab:red", "tab:brown"

    def default_initial_guess(self) -> FloatTuple:
        return 0, 1, 0, 0

    def default_bounds(self, variable_type: VariableType) -> tuple[FloatTuple]:
        return (0, 0, 0, 0), (1, 1, 1, 1)

    def apply_dynamics(self, target_load: CX, *states) -> CX:
        # Implementation of modified Xia dynamics
        ma, mr, mf, effort = states

        c = if_else(
            lt(ma, target_load),
            if_else(gt(mr, target_load - ma), self.LD * (target_load - ma), self.LD * mr),
            self.LR * (target_load - ma),
        )

        effort_perception_dot = (
            self.effort_factor
            * (target_load - self.effort_threshold)
            * if_else(
                target_load > self.effort_threshold,
                1 / (1 - self.effort_threshold) * (1 - effort),
                1 / self.effort_threshold * effort,
            )
        )

        ma_dot = c - self.F * ma - if_else(target_load > self.effort_threshold, effort_perception_dot, 0)
        mr_dot = -c + self.R * mf - if_else(target_load < self.effort_threshold, effort_perception_dot, 0)
        mf_dot = self.F * ma - self.R * mf
        effort_perception_dot += self.stabilization_factor * (1 - ma - mr - mf - effort)

        return vertcat(ma_dot, mr_dot, mf_dot, effort_perception_dot)

    @property
    def multi_type(self):
        return MultiFatigueInterfaceMuscle


class MichaudTauFatigue(TauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    @staticmethod
    def dynamics_suffix() -> Str:
        return "ma"

    @staticmethod
    def fatigue_suffix() -> Str:
        return "mf"

    def __init__(
        self,
        minus: MichaudFatigue,
        plus: MichaudFatigue,
        state_only: Bool = False,
        apply_to_joint_dynamics: Bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        minus: MichaudFatigue
            The Michaud model for the negative tau
        plus: MichaudFatigue
            The Michaud model for the positive tau
        state_only: bool
            If the dynamics should be passed to tau or only computed
        apply_to_joint_dynamics: bool
            If the fatigue should be applied to joint directly
        """

        super(MichaudTauFatigue, self).__init__(
            minus, plus, state_only=state_only, apply_to_joint_dynamics=apply_to_joint_dynamics, **kwargs
        )
