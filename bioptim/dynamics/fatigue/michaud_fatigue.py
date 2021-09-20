from casadi import vertcat, lt, gt, if_else

from .xia_fatigue import XiaFatigue, XiaTauFatigue


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
    def suffix() -> list:
        return ["ma", "mr", "mf", "mf_long"]

    @staticmethod
    def default_initial_guess() -> list:
        return [0, 0.5, 0, 0.5]

    @staticmethod
    def default_bounds() -> list:
        return [[0, 1], [0, 1], [0, 1], [0, 1]]

    def apply_dynamics(self, target_load, *states):
        # Implementation of modified Xia dynamics
        ma, mr, mf, mf_long = states

        c = if_else(
            lt(ma, target_load),
            if_else(gt(mr, target_load - ma), self.LD * (target_load - ma), self.LD * mr),
            self.LR * (target_load - ma),
        )

        fatigue_load = target_load - self.fatigue_threshold
        fatigue_dyn = self.L * if_else(gt(fatigue_load, 0), 1 - mf_long, -mf_long)

        # The fatigue_load should be separated between ma/mr (if >0 or <0),
        # but since LD >> L, we can save 2 if_else by putting everything on ma
        ma_dot = c - self.F * ma - fatigue_dyn  # - if_else(gt(fatigue_load, 0), fatigue_dyn, 0)
        mr_dot = -c + self.R * mf               # - if_else(lt(fatigue_load, 0), fatigue_dyn, 0)
        mf_dot = self.F * ma - self.R * mf
        mf_long_dot = fatigue_dyn + self.S * (1 - ma - mr - mf - mf_long)

        return vertcat(ma_dot, mr_dot, mf_dot, mf_long_dot)


class MichaudTauFatigue(XiaTauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, minus: MichaudFatigue, plus: MichaudFatigue):
        """
        Parameters
        ----------
        minus: XiaFatigue
            The Xia model for the negative tau
        plus: XiaFatigue
            The Xia model for the positive tau
        """

        super(MichaudTauFatigue, self).__init__(minus, plus)
