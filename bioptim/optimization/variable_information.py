from typing import Union


class VariableInformation:
    def __init__(self, name: str, n: int, index: Union[list, tuple]):
        self.name = name
        self.n = n
        self.index = index


class VariableInformationList:
    def __init__(self):
        self.elements = list()

    def __getitem__(self, item):
        for elt in self.elements:
            if item == elt.name:
                return elt
        raise KeyError(f"{item} is not in the list")

    def append(self, name: str, n: int, index: Union[list, tuple]):
        self.elements.append(VariableInformation(name, n, index))

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
