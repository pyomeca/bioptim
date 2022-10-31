from typing import Union, Any

import numpy as np
from casadi import MX, SX, vertcat, horzcat

from ..misc.mapping import BiMapping
from ..misc.options import OptionGeneric, UniquePerPhaseOptionList


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
    mapping: BiMapping
        The mapping of the MX
    parent_list: OptimizationVariableList
        The parent that added this entry

    Methods
    -------
    __len__(self)
        The len of the MX reduced
    cx(self)
        The CX of the variable (starting point)
    cx_end(self)
        The CX of the variable (ending point)
    """

    def __init__(self, name: str, mx: MX, index: [range, list], mapping: BiMapping = None, parent_list=None):
        """
        Parameters
        ----------
        name: str
            The name of the variable
        mx: MX
            The MX variable associated with this variable
        index: [range, list]
            The indices to find this variable
        parent_list: OptimizationVariableList
            The list the OptimizationVariable is in
        """
        self.name: str = name
        self.mx: MX = mx
        self.index: [range, list] = index
        self.mapping: BiMapping = mapping
        self.parent_list: OptimizationVariableList = parent_list

    def __len__(self):
        """
        The len of the MX reduced

        Returns
        -------
        The number of element (correspond to the nrows of the MX)
        """
        return len(self.index)

    @property
    def cx(self):
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "OptimizationVariable must have been created by OptimizationVariableList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx[self.index, :]

    @property
    def cx_end(self):
        if self.parent_list is None:
            raise RuntimeError(
                "OptimizationVariable must have been created by OptimizationVariableList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_end[self.index, :]


class OptimizationVariableList:
    """
    A list of OptimizationVariable

    Attributes
    ----------
    elements: list
        Each of the variable separated
    _cx: Union[MX, SX]
        The symbolic MX or SX of the list (starting point)
    _cx_end: Union[MX, SX]
        The symbolic MX or SX of the list (ending point)
    mx_reduced: MX
        The reduced MX to the size of _cx

    Methods
    -------
    __getitem__(self, item: Union[int, str])
        Get a specific variable in the list, whether by name or by index
    append(self, name: str, cx: list, mx: MX, bimapping: BiMapping)
        Add a new variable to the list
    cx(self)
        The the cx of all elements together (starting point)
    cx_end(self)
        The the cx of all elements together (ending point)
    mx(self)
        The MX of all variable concatenated together
    shape(self)
        The size of the CX
    __len__(self)
        The number of variables in the list
    """

    def __init__(self):
        self.elements: list = []
        self.fake_elements: list = []
        self._cx: Union[MX, SX, np.ndarray] = np.array([])
        self._cx_end: Union[MX, SX, np.ndarray] = np.array([])
        self._cx_intermediates: list = []
        self.mx_reduced: MX = MX.sym("var", 0, 0)

    def __getitem__(self, item: Union[int, str, list, range]):
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
            for elt in self.fake_elements:
                if item == elt.name:
                    return elt
            raise KeyError(f"{item} is not in the list")
        elif isinstance(item, (list, tuple)) or isinstance(item, range):
            mx = vertcat([elt.mx for elt in self.elements if elt.name in item])
            index = []
            for elt in self.elements:
                if elt.name in item:
                    index.extend(list(elt.index))
            return OptimizationVariable("some", mx, index)
        else:
            raise ValueError("OptimizationVariableList can be sliced with int, list, range or str only")

    def append_fake(self, name: str, index: Union[MX, SX, list], mx: MX, bimapping: BiMapping):
        """
        Add a new variable to the fake list which add something without changing the size of the normal elements

        Parameters
        ----------
        name: str
            The name of the variable
        index: Union[MX, SX]
            The SX or MX variable associated with this variable. Is interpreted as index if is_fake is true
        mx: MX
            The MX variable associated with this variable
        bimapping: BiMapping
            The Mapping of the MX against CX
        """

        self.fake_elements.append(OptimizationVariable(name, mx, index, bimapping, self))

    def append(self, name: str, cx: list, mx: MX, bimapping: BiMapping):
        """
        Add a new variable to the list

        Parameters
        ----------
        name: str
            The name of the variable
        cx: list
            The list of SX or MX variable associated with this variable
        mx: MX
            The MX variable associated with this variable
        bimapping: BiMapping
            The Mapping of the MX against CX
        """

        index = range(self._cx.shape[0], self._cx.shape[0] + cx[0].shape[0])
        self._cx = vertcat(self._cx, cx[0])
        self._cx_end = vertcat(self._cx_end, cx[-1])
        for i, c in enumerate(cx[1:-1]):
            if i >= len(self._cx_intermediates):
                self._cx_intermediates.append(c)
            else:
                self._cx_intermediates[i] = vertcat(self._cx_intermediates[i], c)
        self.mx_reduced = vertcat(self.mx_reduced, MX.sym("var", cx[0].shape))

        self.elements.append(OptimizationVariable(name, mx, index, bimapping, self))

    @property
    def cx(self):
        """
        The cx of all elements together (starting point)
        """

        return self._cx[:, 0]

    @property
    def cx_end(self):
        """
        The cx of all elements together (ending point)
        """

        return self._cx_end[:, 0]

    @property
    def cx_intermediates_list(self):
        """
        The cx of all elements together (starting point)
        """

        return self._cx_intermediates

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
        If the item of name item is in the list
        """

        for elt in self.elements:
            if item == elt.name:
                return True
        for elt in self.fake_elements:
            if item == elt.name:
                return True
        else:
            return False

    def keys(self):
        """
        All the names of the elements in the list
        """

        return [elt for elt in self]

    @property
    def shape(self):
        """
        The size of the CX
        """

        return self._cx.shape[0]

    def __len__(self):
        """
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


class VariableScaling(OptionGeneric):
    def __init__(
        self,
        scaling: Union[np.ndarray, list] = None,
    ):
        """
        Parameters
        ----------
        scaling: Union[np.ndarray, list]
            The scaling of the variables
        """
        if isinstance(scaling, list):
            scaling = np.array(scaling)
        self.scaling = scaling
        super(VariableScaling, self).__init__()

    def __bool__(self) -> bool:
        """
        Get if the VariableScaling is empty

        Returns
        -------
        If the VariableScaling is empty
        """

        return len(self.scaling) > 0

    @property
    def shape(self) -> int:
        """
        Get the size of the InitialGuess

        Returns
        -------
        The size of the InitialGuess
        """

        return self.scaling.shape

    def __setitem__(self, _slice: Union[slice, list, tuple], value: Union[np.ndarray, list, float]):
        """
        Allows to set from square brackets

        Parameters
        ----------
        _slice: Union[slice, list, tuple]
            The slice where to put the data
        value: Union[np.ndarray, float]
            The value to set
        """

        self.scaling[_slice] = value

    def to_vector(self, n_elements: int, n_shooting: int):
        """
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay

        """

        if self.scaling.shape[0] != n_elements:
            raise RuntimeError(
                f"The number of elements in the scaling ({self.scaling.shape[0]}) is not the same as the number of elements ({n_elements})"
            )

        scaling_vector = np.zeros((n_elements * n_shooting, 1))
        for i in range(n_shooting):
            scaling_vector[i * n_elements : (i + 1) * n_elements] = np.reshape(self.scaling, (n_elements, 1))

        return scaling_vector


class VariableScalingList(UniquePerPhaseOptionList):
    """
    A list of VariableScaling if more than one is required

    Methods
    -------
    add(self, scaling: Union[np.ndarray, list] = None)
        Add a new variable scaling to the list
    __getitem__(self, item) -> Bounds
        Get the ith option of the list
    print(self)
        Print the VariableScalingList to the console
    """

    def add(
        self,
        scaling: Union[np.ndarray, list, VariableScaling] = None,
    ):
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        min_bound: Union[PathCondition, np.ndarray, list, tuple]
            The minimum path condition. If min_bound if defined, then max_bound must be so and bound should be None
        max_bound: [PathCondition, np.ndarray, list, tuple]
            The maximum path condition. If max_bound if defined, then min_bound must be so and bound should be None
        bounds: Bounds
            Copy a Bounds. If bounds is defined, min_bound and max_bound should be None
        extra_arguments: dict
            Any parameters to pass to the Bounds
        """

        if isinstance(scaling, VariableScaling):
            if scaling.phase == -1:
                scaling.phase = len(self.options) if self.options[0] else 0
            self.copy(scaling)
        else:
            super(VariableScalingList, self)._add(scaling=scaling, option_type=VariableScaling)

    def add(
        self,
        scaling: Union[np.ndarray, list, VariableScaling] = None,
    ):
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        min_bound: Union[PathCondition, np.ndarray, list, tuple]
            The minimum path condition. If min_bound if defined, then max_bound must be so and bound should be None
        max_bound: [PathCondition, np.ndarray, list, tuple]
            The maximum path condition. If max_bound if defined, then min_bound must be so and bound should be None
        bounds: Bounds
            Copy a Bounds. If bounds is defined, min_bound and max_bound should be None
        extra_arguments: dict
            Any parameters to pass to the Bounds
        """

        if isinstance(scaling, VariableScaling):
            if scaling.phase == -1:
                scaling.phase = len(self.options) if self.options[0] else 0
            self.copy(scaling)
        else:
            super(VariableScalingList, self)._add(scaling=scaling, option_type=VariableScaling)

    def __getitem__(self, item) -> VariableScaling:
        """
        Get the ith option of the list

        Parameters
        ----------
        item: int
            The index of the option to get

        Returns
        -------
        The ith option of the list
        """

        return super(VariableScalingList, self).__getitem__(item)

    def print(self):
        """
        Print the VariableScalingList to the console
        """

        raise NotImplementedError("Printing of VariableScalingList is not ready yet")
