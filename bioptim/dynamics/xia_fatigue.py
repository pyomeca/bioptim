from casadi import vertcat, lt, gt, if_else

from .dynamics_functions import DynamicsFunctions
from .fatigue_dynamics import FatigueModel, TauFatigue


class XiaFatigue(FatigueModel):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, LD: float, LR: float, F: float, R: float, scale: float = 1):
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
        scale: float
            The scaling factor to convert so input / scale => TL
        """

        super(XiaFatigue, self).__init__()
        self.LR = LR
        self.LD = LD
        self.F = F
        self.R = R
        self.scale = scale

    @staticmethod
    def type() -> str:
        return "muscles"

    @staticmethod
    def suffix() -> list:
        return ["ma", "mr", "mf"]

    @staticmethod
    def default_initial_guess() -> list:
        return [0, 1, 0]

    @staticmethod
    def default_bounds() -> list:
        return [[0, 1], [0, 1], [0, 1]]

    @staticmethod
    def dynamics_suffix() -> str:
        return "ma"

    def apply_dynamics(self, target_load, ma, mr, mf):
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

    def _get_target_load(self, nlp, controls, index):
        if "muscles" not in nlp.controls:
            raise NotImplementedError("Fatigue dynamics without muscle controls is not implemented yet")

        return DynamicsFunctions.get(nlp.controls["muscles"], controls)[index, :]

    def dynamics(self, dxdt, nlp, index, states, controls):
        target_load = self._get_target_load(nlp, controls, index)
        fatigue = [DynamicsFunctions.get(nlp.states[f"muscles_{s}"], states)[index, :] for s in self.suffix()]
        current_dxdt = self.apply_dynamics(target_load, *fatigue)

        for i, s in enumerate(self.suffix()):
            dxdt[nlp.states[f"muscles_{s}"].index[index], :] = current_dxdt[i]

        return dxdt


class XiaTauFatigue(TauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, minus: XiaFatigue, plus: XiaFatigue):
        """
        Parameters
        ----------
        minus: XiaFatigue
            The Xia model for the negative tau
        plus: XiaFatigue
            The Xia model for the positive tau
        """

        super(XiaTauFatigue, self).__init__(minus, plus)

    @staticmethod
    def suffix() -> list:
        return ["minus", "plus"]

    @staticmethod
    def default_initial_guess() -> list:
        raise RuntimeError("default_initial_guess cannot by called at XiaTauFatigue level")

    @staticmethod
    def default_bounds() -> list:
        raise RuntimeError("default_bounds cannot by called at XiaTauFatigue level")

    @staticmethod
    def dynamics_suffix() -> str:
        return "ma"

    def _dynamics_per_suffix(self, dxdt, suffix, nlp, index, states, controls):
        var = getattr(self, suffix)
        target_load = self._get_target_load(var, suffix, nlp, controls, index)
        fatigue = [
            DynamicsFunctions.get(nlp.states[f"tau_{suffix}_{dyn_suffix}"], states)[index, :]
            for dyn_suffix in var.suffix()
        ]
        current_dxdt = var.apply_dynamics(target_load, *fatigue)

        for i, dyn_suffix in enumerate(var.suffix()):
            dxdt[nlp.states[f"tau_{suffix}_{dyn_suffix}"].index[index], :] = current_dxdt[i]

        return dxdt

    def _get_target_load(self, var: XiaFatigue, suffix: str, nlp, controls, index: int):
        if "tau" not in nlp.controls:
            raise NotImplementedError("Fatigue dynamics without tau controls is not implemented yet")

        return DynamicsFunctions.get(nlp.controls[f"tau_{suffix}"], controls)[index, :] / var.scale
