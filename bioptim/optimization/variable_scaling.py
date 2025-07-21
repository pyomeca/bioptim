import numpy as np
from typing import Any

from ..misc.options import OptionGeneric, OptionDict
from ..misc.parameters_types import (
    Bool,
    Int,
    Str,
    StrOptional,
    AnyDict,
    FloatList,
    NpArrayOptional,
    DoubleIntTuple,
)


class VariableScaling(OptionGeneric):
    def __init__(self, key: Str, scaling: NpArrayOptional | FloatList = None, **kwargs):
        """
        Parameters
        ----------
        scaling: np.ndarray | list
            The scaling of the variables
        """

        super(VariableScaling, self).__init__(**kwargs)

        if isinstance(scaling, list):
            scaling = np.array(scaling)
        elif not (isinstance(scaling, np.ndarray) or isinstance(scaling, VariableScaling)):
            raise RuntimeError(f"Scaling must be a list or a numpy array, not {type(scaling)}")

        if len(scaling.shape) == 1:
            scaling = scaling[:, np.newaxis]
        elif len(scaling.shape) > 2:
            raise ValueError(f"Scaling must be a 1- or 2- dimensional numpy array")

        if (scaling < 0).any():
            raise ValueError(f"Scaling factors must be strictly greater than zero.")

        self.key = key
        self.scaling = scaling

    @property
    def value(self) -> NpArrayOptional:
        return self.scaling

    @property
    def shape(self) -> DoubleIntTuple:
        return self.scaling.shape

    def to_vector(self, repeats: Int) -> NpArrayOptional:
        """
        Repeat the scaling to match the variables vector format
        """

        return self.scaling.repeat(repeats, axis=0).reshape((-1, 1))

    def to_array(self, repeats: Int = 1) -> NpArrayOptional:
        """
        Repeat the scaling to match the variables array format
        """

        return np.repeat(self.scaling, repeats=repeats, axis=1)


class VariableScalingList(OptionDict):
    """
    A list of VariableScaling if more than one is required

    Methods
    -------
    add(self, scaling: np.ndarray | list = None)
        Add a new variable scaling to the list
    __getitem__(self, item) -> Bounds
        Get the ith option of the list
    print(self)
        Print the VariableScalingList to the console
    """

    def __init__(self) -> None:
        super(VariableScalingList, self).__init__(sub_type=VariableScaling)

    def add(
        self,
        key: StrOptional = None,
        scaling: NpArrayOptional | list | VariableScaling = None,
        phase: int = -1,
    ) -> None:
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        key: str
            The name of the variable to apply the scaling to
        scaling
            The value of the scaling for the variable
        phase
            The phase to apply the scaling to
        """

        if isinstance(scaling, VariableScaling):
            self.add(key=scaling.key, scaling=scaling.scaling, phase=phase)
        elif isinstance(scaling, list) or isinstance(scaling, np.ndarray):
            for i, elt in enumerate(scaling):
                if elt <= 0:
                    raise RuntimeError(
                        f"Scaling factors must be strictly greater than zero. {i}th element {elt} is not > 0."
                    )
            super(VariableScalingList, self)._add(key=key, phase=phase, scaling=scaling)
        else:
            if scaling is None:
                raise ValueError("Scaling cannot be None")
            else:
                raise ValueError(f"Scaling must be a VariableScaling, a list or a numpy array, not {type(scaling)}")

    def copy(self) -> "VariableScalingList":
        out = VariableScalingList()
        for key in self.keys():
            out.add(key, self[key])
        return out

    @property
    def param_when_copying(self) -> AnyDict:
        return {}

    def __contains__(self, item: Any) -> Bool:
        return item in self.options[0]

    def print(self) -> None:
        """
        Print the VariableScalingList to the console
        """

        raise NotImplementedError("Printing of VariableScalingList is not ready yet")
