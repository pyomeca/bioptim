from typing import Any

import numpy as np

from ..misc.options import OptionGeneric, OptionDict


class VariableScaling(OptionGeneric):
    def __init__(self, key: str, scaling: np.ndarray | list = None, **kwargs):
        """
        Parameters
        ----------
        scaling: np.ndarray | list
            The scaling of the variables
        """

        super(VariableScaling, self).__init__()

        if isinstance(scaling, list):
            scaling = np.array(scaling)
        elif not (isinstance(scaling, np.ndarray) or isinstance(scaling, VariableScaling)):
            raise RuntimeError(f"Scaling must be a list or a numpy array, not {type(scaling)}")

        self.scaling = scaling
        self.key = key

    @property
    def shape(self):
        return self.scaling.shape

    def to_vector(self, repeats: int):
        """
        Repeat the scaling to match the variables vector format
        """

        return self.scaling[np.newaxis, :].repeat(repeats, axis=0).reshape((-1, 1))

    def to_array(self, repeats: int = 1):
        """
        Repeat the scaling to match the variables array format
        """

        return np.repeat(self.scaling[:, np.newaxis], repeats=repeats, axis=1)


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

    def __init__(self):
        super(VariableScalingList, self).__init__()
        self._all: tuple[str, ...] | None = None

    def add(
        self,
        key: str = None,
        scaling: np.ndarray | list | VariableScaling = None,
        phase: int = -1,
    ):
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
        else:
            if scaling is None:
                raise ValueError("Scaling cannot be None")

            for i, elt in enumerate(scaling):
                if elt <= 0:
                    raise RuntimeError(
                        f"Scaling factors must be strictly greater than zero. {i}th element {elt} is not > 0."
                    )
            super(VariableScalingList, self)._add(key=key, phase=phase, scaling=scaling, option_type=VariableScaling)

    def define_all(self, all: tuple[str, ...]):
        """
        Define what "all" means (basically the order of the all. This should not be set by the user as it is
        automatically set

        Parameters
        ----------
        all
            The tuple of the all
        """
        self._all = all

    def __getitem__(self, item: int | tuple[str, ...] | str) -> VariableScaling | Any:
        """
        Get the ith option of the list

        Parameters
        ----------
        item: int
            The index of the option to get

        Returns
        -------
        The ith option of the list (Any being a VariableScalingList
        """

        if isinstance(item, str) and item == "all":
            if self._all is None:
                raise RuntimeError("The scaling with 'all' was called prior to be defined. Please call the "
                                   "'define_all' method before trying to get the 'all' index")
            return self[self._all]

        if isinstance(item, (list, tuple)):
            out = np.ndarray((0, 1))
            for i in item:
                out = np.append(out, self[i].scaling)
            return VariableScaling("not named", out)

        if isinstance(item, int):
            # Request VariableScalingList for a particular phase
            out = VariableScalingList()
            for key in self.keys():
                out.add(key, self.options[item][key])
            return out

        if isinstance(item, str):
            if len(self.options) != 1:
                raise ValueError("Indexing VariableScalingList with 'str' with more than one dimension is a mistake."
                                 "Call the function index first with the index of the phase you want to fetch")
            return self.options[0][item]

        raise ValueError("Wrong type in getting scaling")

    def copy(self):
        out = VariableScalingList()
        for key in self.keys():
            out.add(key, self[key])
        return out

    def __contains__(self, item):
        return item in self.options[0]

    def print(self):
        """
        Print the VariableScalingList to the console
        """

        raise NotImplementedError("Printing of VariableScalingList is not ready yet")
