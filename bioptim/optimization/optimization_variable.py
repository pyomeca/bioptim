from typing import Union

from casadi import MX, SX, vertcat


class OptimizationVariable:
    name: str
    mx: MX
    index: range

    def __init__(self, name: str, mx: MX, index: range):
        self.name = name
        self.mx = mx
        self.index = index

    def __len__(self):
        return len(self.index)


class OptimizationVariableList:
    cx: Union[MX, SX]  # The symbolic variable

    def __init__(self):
        self.elements = list()

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.elements[item]
        elif isinstance(item, str):
            for elt in self.elements:
                if item == elt.name:
                    return elt
            raise KeyError(f"{item} is not in the list")
        else:
            raise ValueError("OptimizationVariableList can be sliced with int or str only")

    def append(self, name: str, cx: Union[MX, SX], mx: MX):
        index = range(self.cx.shape[0], self.cx.shape[0] + cx.shape[0])
        self.cx = vertcat(self.cx, cx)
        self.elements.append(OptimizationVariable(name, mx, index))

    @property
    def mx(self):
        return vertcat(*[elt.mx for elt in self.elements])

    def __contains__(self, item):
        for elt in self.elements:
            if item == elt.name:
                return True
        else:
            return False

    def keys(self):
        return [elt.name for elt in self]

    @property
    def shape(self):
        return self.cx.shape[0]

    def __len__(self):
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
