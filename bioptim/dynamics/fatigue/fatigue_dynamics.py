from typing import Any, Union
from abc import ABC, abstractmethod

from casadi import MX

from ...misc.options import UniquePerPhaseOptionList, OptionDict, OptionGeneric


class FatigueModel(ABC):
    @staticmethod
    @abstractmethod
    def type() -> str:
        """
        The type of Fatigue
        """
        pass

    @staticmethod
    @abstractmethod
    def suffix() -> list:
        """
        The type of Fatigue
        """
        pass

    @staticmethod
    def color() -> list:
        """
        The coloring when drawn
        """
        pass

    @staticmethod
    def default_initial_guess() -> list:
        """
        The initial guess the fatigue parameters are expected to have
        """
        pass

    @staticmethod
    def default_bounds() -> list:
        """
        The bounds the fatigue parameters are expected to have
        """
        pass

    @staticmethod
    @abstractmethod
    def dynamics_suffix() -> str:
        """
        The type of Fatigue
        """
        pass

    @abstractmethod
    def dynamics(self, dxdt, nlp, index, states, controls) -> MX:
        """
        Augment the dxdt vector with the derivative of the fatigue states

        Parameters
        ----------
        dxdt: MX
            The MX vector to augment
        nlp: NonLinearProgram
            The current phase
        index: int
            The index of the current fatigue element
        states: OptionVariable
            The state variable
        controls: OptionVariable
            The control variable

        Returns
        -------
        dxdt augmented
        """
        pass


class TauFatigue(FatigueModel, ABC):
    def __init__(self, minus: FatigueModel, plus: FatigueModel):
        """
        Parameters
        ----------
        minus: FatigueModel
            The model for the negative tau
        plus: FatigueModel
            The model for the positive tau
        """
        super(TauFatigue, self).__init__()
        self.minus = minus
        self.plus = plus

    @staticmethod
    def type() -> str:
        return "tau"

    @staticmethod
    def color() -> list:
        """
        The color to be draw
        """
        return ["tab:orange", "tab:green"]

    def dynamics(self, dxdt, nlp, index, states, controls):
        for suffix in self.suffix():
            dxdt = self._dynamics_per_suffix(dxdt, suffix, nlp, index, states, controls)

        return dxdt

    @abstractmethod
    def _dynamics_per_suffix(self, dxdt, suffix, nlp, index, states, controls):
        """

        Parameters
        ----------
        dxdt: MX
            The MX vector to augment
        suffix: str
            The str for each suffix
        nlp: NonLinearProgram
            The current phase
        index: int
            The index of the current fatigue element
        states: OptionVariable
            The state variable
        controls: OptionVariable
            The control variable

        Returns
        -------

        """
        pass


class FatigueOption(OptionGeneric):
    def __init__(self, model: FatigueModel, state_only: bool, **params):
        """
        model: FatigueModel
            The actual fatigue model
        state_only: bool
            If the added fatigue should be used in the dynamics or only computed
        params: Any
            Any other parameters to pass to OptionGeneric
        """

        super(FatigueOption, self).__init__(**params)
        self.model = model
        self.state_only = state_only


class FatigueUniqueList(UniquePerPhaseOptionList):
    def add(self, **extra_arguments: Any):
        self._add(option_type=FatigueOption, **extra_arguments)

    def __init__(self, suffix: list):
        """
        Parameters
        ----------
        suffix: list
            The type of Fatigue
        """

        super(FatigueUniqueList, self).__init__()
        self.suffix = suffix

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
            dxdt = elt.model.dynamics(dxdt, nlp, i, states, controls)
        return dxdt


class FatigueList(OptionDict):
    def add(self, model: FatigueModel, index: int = -1, state_only: bool = False):
        """
        Add a muscle to the dynamic list

        Parameters
        ----------
        model: FatigueModel
            The dynamics to add, if more than one dynamics is required, a list can be sent
        index: int
            The index of the muscle, referring to the muscles order in the bioMod
        state_only: bool
            If the added fatigue should be used in the dynamics or only computed
        """

        if model.type() not in self.options[0]:
            self.options[0][model.type()] = FatigueUniqueList(model.suffix())

        self.options[0][model.type()].add(model=model, phase=index, state_only=state_only)

    def dynamics(self, dxdt, nlp, index, states, controls):
        raise NotImplementedError("FatigueDynamics is abstract")

    def __contains__(self, item):
        return item in self.options[0]

    def __getitem__(self, item: Union[int, str, list, tuple]) -> FatigueUniqueList:
        return super(FatigueList, self).__getitem__(item)
