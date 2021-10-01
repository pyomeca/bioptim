from casadi import vertcat, lt, gt, if_else

from .xia_fatigue import XiaFatigue, XiaTauFatigue, MultiFatigueInterfaceMuscle
from ...misc.enums import VariableType


class MichaudFatigue(XiaFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(
        self,
        LD: float,
        LR: float,
        F: float,
        R: float,
        fatigue_threshold: float,
        L: float,
        S: float = 1,
        scale: float = 1,
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
        fatigue_threshold: float
            The activation level at which the structure starts to build long-term fatigue
        L: float
            Long-term fatigable rate
        S: float
            Stabilization factor so: ma + mr + mf => 1
        scale: float
            The scaling factor to convert so input / scale => TL
        """

        super(MichaudFatigue, self).__init__(LD, LR, F, R, scale)
        self.S = S
        self.L = L
        self.fatigue_threshold = fatigue_threshold

    @staticmethod
    def suffix(variable_type: VariableType) -> tuple:
        if variable_type == VariableType.STATES:
            return "ma", "mr", "mf_xia", "mf"
        else:
            return "",

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

        fatigue_load = target_load - self.fatigue_threshold
        fatigue_dyn = self.L * if_else(gt(fatigue_load, 0), 1 - mf_long, -mf_long)

        ma_dot = c - self.F * ma - if_else(gt(fatigue_load, 0), fatigue_dyn, 0)
        mr_dot = -c + self.R * mf_xia - if_else(lt(fatigue_load, 0), fatigue_dyn, 0)
        mf_dot = self.F * ma - self.R * mf_xia
        mf_long_dot = fatigue_dyn + self.S * (1 - ma - mr - mf_xia - mf_long)

        return vertcat(ma_dot, mr_dot, mf_dot, mf_long_dot)

    @property
    def multi_type(self):
        return MultiFatigueInterfaceMuscle


class MichaudTauFatigue(XiaTauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

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


class MichaudFatigueSimple(MichaudFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(
            self,
            fatigue_threshold: float,
            L: float,
            scale: float = 1,
    ):
        """
        Parameters
        ----------
        fatigue_threshold: float
            The activation level at which the structure starts to build long-term fatigue
        L: float
            Long-term fatigable rate
        scale: float
            The scaling factor to convert so input / scale => TL
        """

        super(MichaudFatigueSimple, self).__init__(LD=0, LR=0, F=0, R=0, fatigue_threshold=fatigue_threshold, L=L, S=0,
                                                   scale=scale)

    @staticmethod
    def suffix(variable_type: VariableType) -> tuple:
        if variable_type == VariableType.STATES:
            return "mf",
        else:
            return "",

    @staticmethod
    def color() -> tuple:
        return "tab:brown",

    def default_initial_guess(self) -> tuple:
        return 0,

    def default_bounds(self, variable_type: VariableType) -> tuple:
        if variable_type == VariableType.STATES:
            return (0,), (1,)
        elif variable_type == VariableType.CONTROLS:
            raise RuntimeError("default_bounds for CONTROLS cannot be called for this model")

    def apply_dynamics(self, target_load, *states):
        # Implementation of modified Xia dynamics
        mf_long = states[0]
        fatigue_load = target_load - self.fatigue_threshold
        mf_long_dot = self.L * if_else(gt(fatigue_load, 0), 1 - mf_long, -mf_long)
        return mf_long_dot

    @staticmethod
    def dynamics_suffix() -> str:
        raise RuntimeError("MichaudSimple cannot be used with state_only=False")

    @property
    def multi_type(self):
        return MichaudTauFatigueSimple


class MichaudTauFatigueSimple(MichaudTauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, minus: MichaudFatigueSimple, plus: MichaudFatigueSimple, state_only: bool = True, **kwargs):
        """
        Parameters
        ----------
        minus: MichaudFatigueSimple
            The Michaud model for the negative tau
        plus: MichaudFatigueSimple
            The Michaud model for the positive tau
        """

        super(MichaudTauFatigueSimple, self).__init__(minus, plus, state_only=state_only, **kwargs)
