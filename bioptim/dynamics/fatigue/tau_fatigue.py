from abc import abstractmethod

from casadi import if_else, lt, gt, MX

from .fatigue_dynamics import MultiFatigueModel, FatigueModel
from ..dynamics_functions import DynamicsFunctions
from ...misc.enums import VariableType
from ...misc.parameters_types import (
    Bool,
    Int,
    Str,
    StrTuple,
    DoubleIntTuple,
    AnyTuple,
)


class TauFatigue(MultiFatigueModel):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(
        self,
        minus: FatigueModel,
        plus: FatigueModel,
        state_only: Bool = True,
        apply_to_joint_dynamics: Bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        minus: XiaFatigue
            The Xia model for the negative tau
        plus: XiaFatigue
            The Xia model for the positive tau
        state_only: bool
            If the dynamics should be passed to tau or only computed
        apply_to_joint_dynamics: bool
            If the fatigue should be applied to joint directly
        """

        super(TauFatigue, self).__init__(
            [minus, plus], state_only=state_only, apply_to_joint_dynamics=apply_to_joint_dynamics, **kwargs
        )

    def suffix(self) -> StrTuple:
        return "minus", "plus"

    @staticmethod
    def model_type() -> Str:
        return "tau"

    @staticmethod
    def color() -> StrTuple:
        return "tab:orange", "tab:green"

    @staticmethod
    def plot_factor() -> DoubleIntTuple:
        return -1, 1

    @staticmethod
    @abstractmethod
    def dynamics_suffix() -> Str:
        """
        Returns
        -------
        The name of the dynamic variable in the suffix variables
        """

    @staticmethod
    @abstractmethod
    def fatigue_suffix() -> Str:
        """
        The suffix that is appended to the variable name that describes the fatigue
        """

    def _dynamics_per_suffix(self, dxdt: MX, suffix, nlp, index: Int, states, controls) -> MX:
        var = self.models[suffix]
        target_load = self._get_target_load(var, suffix, nlp, controls, index)
        fatigue = [
            DynamicsFunctions.get(nlp.states[f"{self.model_type()}_{suffix}_{dyn_suffix}"], states)[index, :]
            for dyn_suffix in var.suffix(variable_type=VariableType.STATES)
        ]
        current_dxdt = var.apply_dynamics(target_load, *fatigue)

        for i, dyn_suffix in enumerate(var.suffix(variable_type=VariableType.STATES)):
            dxdt[nlp.states[f"{self.model_type()}_{suffix}_{dyn_suffix}"].index[index], :] = current_dxdt[i]

        return dxdt

    def _get_target_load(self, var: FatigueModel, suffix: Str, nlp, controls, index: Int):
        if self.model_type() not in nlp.controls:
            raise NotImplementedError(f"Fatigue dynamics without {self.model_type()} controls is not implemented yet")

        val = DynamicsFunctions.get(nlp.controls[f"{self.model_type()}_{suffix}"], controls)[index, :]
        if not self.split_controls:
            if var.scaling < 0:
                val = if_else(lt(val, 0), val, 0)
            else:
                val = if_else(gt(val, 0), val, 0)
        return val / var.scaling

    @staticmethod
    def default_state_only() -> Bool:
        return False

    @staticmethod
    def default_apply_to_joint_dynamics() -> Bool:
        return False

    def default_bounds(self, index: int, variable_type: VariableType) -> AnyTuple:
        key = self._convert_to_models_key(index)

        if variable_type == VariableType.STATES:
            return self.models[key].default_bounds(variable_type)
        else:
            if self.split_controls:
                scaling = self.models[key].scaling
                return ((scaling if index == 0 else 0),), ((scaling if index == 1 else 0),)
            else:
                return tuple((self.models[s].scaling,) for s in self.suffix())

    def default_initial_guess(self, index: Int, variable_type: VariableType):
        key = self._convert_to_models_key(index)
        return self.models[key].default_initial_guess()
