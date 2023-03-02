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
            Muscle development coefficient
        LR: float
            Muscle relaxation coefficient
        F: float
            Muscle fiber recovery rate
        R: float
            Muscle fiber relaxation rate
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
            return ("",)

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

    @staticmethod
    def fatigue_suffix() -> str:
        return "mf"

    def apply_dynamics(self, target_load, *states):
        """
        The dynamics of the fatigue model that returns the derivatives of the states.
        with Xia's model.

        according to the paper: "A theoretical approach for modeling peripheral muscle fatigue and recovery"
        by Xia, et al. 2008, doi: 10.1016/j.biomech.2008.07.013

        Parameters
        ----------
        target_load: float | MX | SX
            The target load the actuator must accomplish
        states: float | MX | SX
            The values of the states used to compute the dynamics
        Returns
        -------
        float | MX | SX
        The derivative of the states: (ma_dot, mr_dot, mf_dot)

        """
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


class XiaFatigueStabilized(XiaFatigue):
    def __init__(self, LD: float, LR: float, F: float, R: float, stabilization_factor: float, **kwargs):
        """
        stabilization_factor: float
            Stabilization factor so: ma + mr + mf => 1
        """
        super(XiaFatigueStabilized, self).__init__(LD, LR, F, R, **kwargs)
        self.stabilization_factor = stabilization_factor

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
        mf_dot = self.F * ma - self.R * mf + self.stabilization_factor * (1 - ma - mr - mf)
        return vertcat(ma_dot, mr_dot, mf_dot)


class XiaTauFatigue(TauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    @staticmethod
    def dynamics_suffix() -> str:
        return "ma"

    @staticmethod
    def fatigue_suffix() -> str:
        return "mf"
