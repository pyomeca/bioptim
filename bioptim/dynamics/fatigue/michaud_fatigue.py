from casadi import vertcat, lt, gt, if_else

from .muscle_fatigue import MuscleFatigue, MultiFatigueInterfaceMuscle
from .tau_fatigue import TauFatigue
from ...misc.enums import VariableType


class MichaudFatigue(MuscleFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    @staticmethod
    def dynamics_suffix() -> str:
        return "ma"

    def __init__(
        self,
        LD: float,
        LR: float,
        F: float,
        R: float,
        effort_threshold: float,
        effort_factor: float,
        stabilization_factor: float = 1,
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
    def suffix(variable_type: VariableType) -> tuple:
        if variable_type == VariableType.STATES:
            return "ma", "mr", "mf_xia", "mf"
        else:
            return ("",)

    @staticmethod
    def color() -> tuple:
        return "tab:green", "tab:orange", "tab:red", "tab:brown"

    def default_initial_guess(self) -> tuple:
        return 0, 1, 0, 0

    def default_bounds(self, variable_type: VariableType) -> tuple:
        return (0, 0, 0, 0), (1, 1, 1, 1)

    def apply_dynamics(self, target_load, *states):
        # Implementation of modified Xia dynamics
        ma, mr, mf_xia, mf_long = states

        c = if_else(
            lt(ma, target_load),
            if_else(gt(mr, target_load - ma), self.LD * (target_load - ma), self.LD * mr),
            self.LR * (target_load - ma),
        )

        fatigue_load = target_load - self.effort_threshold
        fatigue_dyn = self.effort_factor * if_else(gt(fatigue_load, 0), 1 - mf_long, -mf_long)

        ma_dot = c - self.F * ma - if_else(gt(fatigue_load, 0), fatigue_dyn, 0)
        mr_dot = -c + self.R * mf_xia - if_else(lt(fatigue_load, 0), fatigue_dyn, 0)
        mf_dot = self.F * ma - self.R * mf_xia
        mf_long_dot = fatigue_dyn + self.stabilization_factor * (1 - ma - mr - mf_xia - mf_long)

        return vertcat(ma_dot, mr_dot, mf_dot, mf_long_dot)

    @property
    def multi_type(self):
        return MultiFatigueInterfaceMuscle


class MichaudTauFatigue(TauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    @staticmethod
    def dynamics_suffix() -> str:
        return "ma"

    def __init__(self, minus: MichaudFatigue, plus: MichaudFatigue, state_only: bool = False, **kwargs):
        """
        Parameters
        ----------
        minus: MichaudFatigue
            The Michaud model for the negative tau
        plus: MichaudFatigue
            The Michaud model for the positive tau
        """

        super(MichaudTauFatigue, self).__init__(minus, plus, state_only=state_only, **kwargs)
