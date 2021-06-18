from typing import Union

import numpy as np
from casadi import MX, SX, vertcat

from ..misc.mapping import BiMapping


class OptimizationVariable:
    """
    An optimization variable and the indices to find this variable in its state or control vector

    Attributes
    ----------
    name: str
        The name of the variable
    mx: MX
        The MX variable associated with this variable
    index: range
        The indices to find this variable
    """

    def __init__(self, name: str, mx: MX, index: [range, list], mapping: BiMapping = None):
        """
        Parameters
        ----------
        name: str
            The name of the variable
        mx: MX
            The MX variable associated with this variable
        index: [range, list]
            The indices to find this variable
        """
        self.name: str = name
        self.mx: MX = mx
        self.index: [range, list] = index
        self.mapping: BiMapping = mapping

    def __len__(self):
        """
        Returns
        -------
        The number of element (correspond to the nrows of the MX)
        """
        return len(self.index)


class OptimizationVariableList:
    """
    A list of OptimizationVariable

    Attributes
    ----------
    elements: list
        Each of the variable separated
    cx: Union[MX, SX]
        The symbolic MX or SX of the list
    """

    def __init__(self):
        self.elements: list = []
        self.cx: Union[MX, SX, np.ndarray] = np.array([])

    def __getitem__(self, item: Union[int, str]):
        """
        Get a specific variable in the list, whether by name or by index

        Parameters
        ----------
        item: Union[int, str]
            The index or name of the element to return

        Returns
        -------
        The specific variable in the list
        """

        if isinstance(item, int):
            return self.elements[item]
        elif isinstance(item, str):
            if item == "all":
                index = []
                for elt in self.elements:
                    index.extend(list(elt.index))
                return OptimizationVariable("all", self.mx, index)

            for elt in self.elements:
                if item == elt.name:
                    return elt
            raise KeyError(f"{item} is not in the list")
        elif isinstance(item, (list, tuple)):
            mx = [elt.mx for elt in self.elements if elt.name in item]
            index = []
            for elt in self.elements:
                if elt.name in item:
                    index.extend(list(elt.index))
            return OptimizationVariable("some", mx, index)
        else:
            raise ValueError("OptimizationVariableList can be sliced with int or str only")

    def append(self, name: str, cx: Union[MX, SX], mx: MX, bimapping: BiMapping):
        """
        Add a new variable to the list

        Parameters
        ----------
        name: str
            The name of the variable
        cx: Union[MX, SX]
            The SX or MX variable associated with this variable
        mx: MX
            The MX variable associated with this variable
        bimapping: BiMapping
            The Mapping of the MX against CX
        """

        index = range(self.cx.shape[0], self.cx.shape[0] + cx.shape[0])
        self.cx = vertcat(self.cx, cx)
        self.elements.append(OptimizationVariable(name, mx, index, bimapping))

    @property
    def mx(self):
        """
        Returns
        -------
        The MX of all variable concatenated together
        """

        return vertcat(*[elt.mx for elt in self.elements])

    def __contains__(self, item: str):
        """
        Parameters
        ----------
        item: str
            The name of the item
        Returns
        -------
        If the item of name item is in the list
        """

        for elt in self.elements:
            if item == elt.name:
                return True
        else:
            return False

    def keys(self):
        """
        Returns
        -------
        All the names of the elements in the list
        """

        return [elt for elt in self]

    @property
    def shape(self):
        """
        Returns
        -------
        The size of the CX
        """

        return self.cx.shape[0]

    def __len__(self):
        """
        Returns
        -------
        The number of variables in the list
        """
        return len(self.elements)

    def __iter__(self):
        """
        Allow for the list to be used in a for loop

        Returns
        -------
        A reference to self
        """

        self._iter_idx = 0
        return self

    def __next__(self):
        """
        Get the next phase of the option list

        Returns
        -------
        The next phase of the option list
        """

        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self[self._iter_idx - 1].name
