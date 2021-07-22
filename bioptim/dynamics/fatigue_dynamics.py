from typing import Any, Union

from casadi import vertcat, if_else, lt, gt, MX

from .dynamics_functions import DynamicsFunctions
from ..misc.enums import Fatigue
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric, OptionList


class XiaTorqueFatigue:
    """
    A placeholder for torque fatigue parameters.
    """

    def __init__(
        self,
        LD: float,
        LR: float,
        F: float,
        R: float,
        tau_max: float,
    ):
        """
        Parameters
        ----------
        LD: int
            Joint development coefficient
        LR: int
            Joint relaxation coefficient
        F: int
            Joint fibers recovery rate
        R: int
            Joint fibers relaxation rate
        tau_max:
            The maximum positive or negative torque
        """

        self.LR = LR
        self.LD = LD
        self.F = F
        self.R = R
        self.tau_max = tau_max


class FatigueDynamics(OptionGeneric):
    def __init__(self, list_index, phase):
        super(FatigueDynamics, self).__init__(phase=phase, list_index=list_index)

    def dynamics(self, dxdt, nlp, index, states, controls):
        raise NotImplementedError("FatigueDynamics is abstract")

    @property
    def shape(self):
        raise NotImplementedError("FatigueDynamics is abstract")


class XiaMuscleFatigueDynamics(FatigueDynamics):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(
        self,
        LD: float,
        LR: float,
        F: float,
        R: float,
        list_index: int,
        phase: int = 0,
    ):
        """
        Parameters
        ----------
        LD: int
            Joint development coefficient
        LR: int
            Joint relaxation coefficient
        F: int
            Joint fibers recovery rate
        R: int
            Joint fibers relaxation rate
        list_index: int
            The index of the structure on which the current dynamics are applied
        phase: int
            At which phase this objective function must be applied
        """

        super(XiaMuscleFatigueDynamics, self).__init__(list_index=list_index, phase=phase)
        self.LR = LR
        self.LD = LD
        self.F = F
        self.R = R

    def apply_dynamics(self, TL, ma, mr, mf):
        # Implementation of Xia dynamics
        c = if_else(lt(ma, TL), if_else(gt(mr, TL - ma), self.LD * (TL - ma), self.LD * mr), self.LR * (TL - ma))
        madot = c - self.F * ma
        mrdot = -c + self.R * mf
        mfdot = self.F * ma - self.R * mf + 100 * (1 - (ma + mr + mf))
        return vertcat(madot, mrdot, mfdot)

    def dynamics(self, dxdt, nlp, index, states, controls):
        if "muscles" not in nlp.controls:
            raise NotImplementedError("Fatigue dynamics without muscle controls is not implemented yet")
        TL = DynamicsFunctions.get(nlp.controls["muscles"], controls)[index, :]
        ma = DynamicsFunctions.get(nlp.states["muscles_ma"], states)[index, :]
        mr = DynamicsFunctions.get(nlp.states["muscles_mr"], states)[index, :]
        mf = DynamicsFunctions.get(nlp.states["muscles_mf"], states)[index, :]
        current_dxdt = self.apply_dynamics(TL, ma, mr, mf)

        dxdt[nlp.states["muscles_ma"].index[index], :] = current_dxdt[0]
        dxdt[nlp.states["muscles_mr"].index[index], :] = current_dxdt[1]
        dxdt[nlp.states["muscles_mf"].index[index], :] = current_dxdt[2]

        return dxdt

    @property
    def shape(self):
        return 3


class XiaTorqueFatigueDynamics(XiaMuscleFatigueDynamics):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(
        self,
        LD: float,
        LR: float,
        F: float,
        R: float,
        tau_max: float,
        list_index: int,
        direction: int,
        phase: int = 0,
    ):
        """
        Parameters
        ----------
        tau_min: int
            The minimal negative torque
        tau_max:
            The maximum positive torque
        LD: int
            Joint development coefficient
        LR: int
            Joint relaxation coefficient
        F: int
            Joint fibers recovery rate
        R: int
            Joint fibers relaxation rate
        list_index: int
            The index of the structure on which the current dynamics are applied
        direction: int
            Positive or negative pool
        phase: int
            At which phase this objective function must be applied
        """

        super(XiaTorqueFatigueDynamics, self).__init__(
            LD=LD,
            LR=LR,
            F=F,
            R=R,
            list_index=list_index,
            phase=phase,
        )
        self.tau_max = tau_max
        self.direction = direction

    def dynamics(self, dxdt, nlp, index, states, controls):
        def apply_direction(direction):
            tau_name = f"tau_{direction}"
            ma_name = f"tau_ma_{direction}"
            mr_name = f"tau_mr_{direction}"
            mf_name = f"tau_mf_{direction}"

            tau_nlp, tau_mx = (nlp.controls, controls) if tau_name in nlp.controls else (nlp.states, states)
            tau_max = self.tau_max
            TL = DynamicsFunctions.get(tau_nlp[tau_name], tau_mx)[index, :] / tau_max
            ma = DynamicsFunctions.get(nlp.states[ma_name], states)[index, :]
            mr = DynamicsFunctions.get(nlp.states[mr_name], states)[index, :]
            mf = DynamicsFunctions.get(nlp.states[mf_name], states)[index, :]
            current_dxdt = self.apply_dynamics(TL, ma, mr, mf)

            dxdt[nlp.states[ma_name].index[index], :] = current_dxdt[0]
            dxdt[nlp.states[mr_name].index[index], :] = current_dxdt[1]
            dxdt[nlp.states[mf_name].index[index], :] = current_dxdt[2]
            return dxdt

        dxdt = apply_direction(self.direction)

        return dxdt

    @property
    def shape(self):
        return 6


class FatigueDynamicsList(OptionList):
    """
    Abstract class for fatigue dynamics
    """

    def __init__(self):
        super(FatigueDynamicsList, self).__init__()
        self.n_torque_fatigued = 0
        self.n_muscle_fatigued = 0

    def add_muscle(self, phase, index):
        raise NotImplementedError("FatigueDynamicsList is abstract")

    def add_torque(self, phase, index):
        raise NotImplementedError("FatigueDynamicsList is abstract")


class FatigueUniqueList(UniquePerPhaseOptionList):
    def __init__(self):
        super(FatigueUniqueList, self).__init__()

    def __next__(self) -> Any:
        """
        Get the next option of the list

        Returns
        -------
        The next option of the list
        """
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.options[self._iter_idx - 1][0] if self.options[self._iter_idx - 1] else None

    def dynamics(self, dxdt, nlp, states, controls):
        for i, elt in enumerate(self):
            dxdt = elt.dynamics(dxdt, nlp, i, states, controls)
        return dxdt


class XiaFatigueDynamicsList(FatigueDynamicsList):
    """
    A list of FatigueDynamics if more than one is required

    Methods
    -------
    add(self, fatigue_dynamics: FatigueDynamics, **extra_arguments)
        Add a new FatigueDynamics to the list
    print(self):
        Print the FatigueDynamicsList to the console
    """

    def __init__(self):
        super(XiaFatigueDynamicsList, self).__init__()
        # First element is muscle fatigue, second is tau fatigue
        self.options = [[FatigueUniqueList(), [FatigueUniqueList(), FatigueUniqueList()]]]

    def add_muscle(self, LD: float, LR: float, F: float, R: float, phase: int = 0, index: int = -1):
        self.options[phase][Fatigue.MUSCLES]._add(
            option_type=XiaMuscleFatigueDynamics, phase=index, LD=LD, LR=LR, F=F, R=R
        )
        self.n_muscle_fatigued = self.n_muscle_fatigued + 3

    def add_torque(
        self,
        torque_minus: XiaTorqueFatigue,
        torque_plus: XiaTorqueFatigue,
        index: int = -1,
        phase: int = 0,
    ):
        """
        Add a new FatigueDynamics to the list
        """

        if torque_minus.tau_max > 0:
            raise RuntimeError("tau_max is supposed to be negative for the negative pool")
        if torque_plus.tau_max < 0:
            raise RuntimeError("tau_max is supposed to be positive for the positive pool")

        negative_direction = 0
        positive_direction = 1
        self.options[phase][Fatigue.TAU][negative_direction]._add(
            option_type=XiaTorqueFatigueDynamics,
            phase=index,
            LD=torque_minus.LD,
            LR=torque_minus.LR,
            F=torque_minus.F,
            R=torque_minus.R,
            tau_max=torque_minus.tau_max,
            direction="minus",
        )
        self.options[phase][Fatigue.TAU][positive_direction]._add(
            option_type=XiaTorqueFatigueDynamics,
            phase=index,
            LD=torque_plus.LD,
            LR=torque_plus.LR,
            F=torque_plus.F,
            R=torque_plus.R,
            tau_max=torque_plus.tau_max,
            direction="plus",
        )
