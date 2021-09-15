from casadi import vertcat, lt, gt, if_else, logic_and

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

    def apply_dynamics(self, target_load, ma, mr, mf):
        # Implementation of Xia dynamics
        c = if_else(
            lt(ma, target_load),
            if_else(gt(mr, target_load - ma), self.LD * (target_load - ma), self.LD * mr),
            self.LR * (target_load - ma),
        )

        fatigue_level = target_load - self.fatigue_threshold
        target_load_fatigue = if_else(logic_and(gt(mf, 0), lt(mf, 1)), self.L * fatigue_level, 0)

        ma_dot = c - self.F * ma - target_load_fatigue  # * (1 - mf)
        mr_dot = -c + self.R * mf
        mf_dot = self.F * ma + target_load_fatigue - self.R * mf + self.S * (1 - (ma + mr + mf))
        return vertcat(ma_dot, mr_dot, mf_dot)


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
