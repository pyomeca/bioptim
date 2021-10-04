from casadi import vertcat, lt, gt, if_else

from .muscle_fatigue import MuscleFatigue
from .tau_fatigue import TauFatigue
from ...misc.enums import VariableType


class XiaFatigue(MuscleFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, LD: float, LR: float, F: float, R: float, **kwargs):
        """
        Parameters
        ----------
        LD: float
            Joint development coefficient
        LR: float
            Joint relaxation coefficient
        F: float
            Joint fibers recovery rate
        R: float
            Joint fibers relaxation rate
        """

        super(XiaFatigue, self).__init__(**kwargs)
        self.LR = LR
        self.LD = LD
        self.F = F
        self.R = R

    @staticmethod
    def suffix(variable_type: VariableType) -> tuple:
        if variable_type == VariableType.STATES:
            return "ma", "mr", "mf"
        else:
            return "",

    @staticmethod
    def color() -> tuple:
        return "tab:green", "tab:orange", "tab:red"

    def default_initial_guess(self) -> tuple:
        return 0, 1, 0

    def default_bounds(self, variable_type: VariableType) -> tuple:
        return (0, 0, 0), (1, 1, 1)

    @staticmethod
    def dynamics_suffix() -> str:
        return "ma"

    def apply_dynamics(self, target_load, *states):
        ma, mr, mf = states
        # Implementation of Xia dynamics
        c = if_else(
            lt(ma, target_load),
            if_else(gt(mr, target_load - ma), self.LD * (target_load - ma), self.LD * mr),
            self.LR * (target_load - ma),
        )
        ma_dot = c - self.F * ma
        mr_dot = -c + self.R * mf
        mf_dot = self.F * ma - self.R * mf
        return vertcat(ma_dot, mr_dot, mf_dot)


class XiaTauFatigue(TauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    @staticmethod
    def dynamics_suffix() -> str:
        return "ma"
