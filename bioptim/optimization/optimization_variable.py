from typing import Callable

import numpy as np
from casadi import MX, SX, vertcat

from ..misc.enums import PhaseDynamics
from ..misc.mapping import BiMapping
from ..misc.parameters_types import (
    Int,
    Str,
    Range,
    Bool,
    IntList,
    StrList,
    AnyList,
    AnyListOptional,
    AnyIterable,
    Indexer,
    CX,
    CXList,
)


class OptimizationVariable:
    """
    An optimization variable and the indices to find this variable in its state or control vector

    Attributes
    ----------
    name: str
        The name of the variable
    cx_start: MX | SX
        The symbolic variable associated with this variable
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
    cx_start(self)
        The CX of the variable (starting point)
    cx_end(self)
        The CX of the variable (ending point)
    """

    def __init__(
        self,
        name: Str,
        cx_start: AnyListOptional,
        index: Range | IntList,
        mapping: BiMapping | None = None,
        parent_list: "OptimizationVariableList" = None,
    ) -> None:
        """
        Parameters
        ----------
        name: str
            The name of the variable
        cx_start: MX | SX
            The symbolic variable associated with this variable
        index: range | list
            The indices to find this variable
        parent_list: OptimizationVariableList
            The list the OptimizationVariable is in
        """
        self.name: Str = name
        self.original_cx: AnyListOptional = cx_start
        self.index: Range | IntList = index
        self.mapping: BiMapping = mapping
        self.parent_list: OptimizationVariableList = parent_list

    def __len__(self) -> Int:
        """
        The len of the MX reduced

        Returns
        -------
        The number of element (correspond to the nrows of the MX)
        """
        return len(self.index)

    @property
    def shape(self) -> Int:
        """
        The len of the MX reduced

        Returns
        -------
        The number of element (correspond to the nrows of the MX)
        """
        return len(self)

    @property
    def cx(self) -> CX:
        if self.parent_list is not None:
            if self.parent_list.current_cx_to_get == 0:
                return self.cx_start
            elif self.parent_list.current_cx_to_get == 1:
                return self.cx_mid
            elif self.parent_list.current_cx_to_get == 2:
                return self.cx_end
            else:
                raise NotImplementedError("This cx is not implemented. Please contact a programmer")
        else:
            return self.cx_start

    @property
    def cx_start(self) -> CX:
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "OptimizationVariable must have been created by OptimizationVariableList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_start[self.index, :]

    @property
    def cx_mid(self) -> CX:
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "OptimizationVariable must have been created by OptimizationVariableList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_mid[self.index, :]

    @property
    def cx_end(self) -> CX:
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "OptimizationVariable must have been created by OptimizationVariableList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_end[self.index, :]

    @property
    def cx_intermediates_list(self) -> CXList:
        """
        The cx of all elements together (starting point)
        """

        return [collocation_point[self.index, :] for collocation_point in self.parent_list.cx_intermediates_list]


class OptimizationVariableList:
    """
    A list of OptimizationVariable

    Attributes
    ----------
    elements: list
        Each of the variable separated
    _cx_start: MX | SX
        The symbolic MX or SX of the list (starting point)
    _cx_mid: MX | SX
        The symbolic MX or SX of the list (mid point)
    _cx_end: MX | SX
        The symbolic MX or SX of the list (ending point)
    cx_constructor: Callable
        The casadi symbolic type used for the optimization (MX or SX)

    Methods
    -------
    __getitem__(self, item: int | str)
        Get a specific variable in the list, whether by name or by index
    append(self, name: str, cx: list, bimapping: BiMapping)
        Add a new variable to the list
    cx(self)
        The cx of all elements together (starting point)
    cx_end(self)
        The cx of all elements together (ending point)
    shape(self)
        The size of the CX
    __len__(self)
        The number of variables in the list
    """

    def __init__(self, cx_constructor: Callable, phase_dynamics: PhaseDynamics) -> None:
        self.elements: list = []
        self.fake_elements: list = []
        self._cx_start: CX | np.ndarray = np.array([])
        self._cx_mid: CX | np.ndarray = np.array([])
        self._cx_end: CX | np.ndarray = np.array([])
        self._cx_intermediates: list = [cx_constructor([])]
        self.cx_constructor = cx_constructor
        self._current_cx_to_get = 0
        self.phase_dynamics = phase_dynamics

    def __getitem__(self, item: Int | Str | Indexer) -> "OptimizationVariable":
        """
        Get a specific variable in the list, whether by name or by index

        Parameters
        ----------
        item: int | str
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
                return OptimizationVariable("all", self.cx_start, index, None, self)

            for elt in self.elements:
                if item == elt.name:
                    return elt
            for elt in self.fake_elements:
                if item == elt.name:
                    return elt
            raise KeyError(f"{item} is not in the list")
        elif isinstance(item, (list, tuple)) or isinstance(item, range):
            index = []
            for elt in self.elements:
                if elt.name in item:
                    index.extend(list(elt.index))
            return OptimizationVariable("some", None, index)
        else:
            raise ValueError("OptimizationVariableList can be sliced with int, list, range or str only")

    def __setitem__(self, key: Int | Str, value: OptimizationVariable) -> None:
        self.elements.append(value)

    @property
    def current_cx_to_get(self) -> Int:
        return self._current_cx_to_get

    @current_cx_to_get.setter
    def current_cx_to_get(self, index: Int) -> None:
        """
        Set the value of current_cx_to_get to corresponding index (cx_start for 0, cx_mid for 1, cx_end for 2) if
        phase_dynamics == PhaseDynamics.SHARED_DURING_PHASE. Otherwise, it is always cx_start

        Parameters
        ----------
        index
            The index to set the current cx to
        """

        if index < -1 or index > 2:
            raise ValueError(
                "Valid values for setting the cx is 0, 1 or 2. If you reach this error message, you probably tried to "
                "add more penalties than available in a multinode constraint. You can try to split the constraints "
                "into more penalties or use phase_dynamics=PhaseDynamics.ONE_PER_NODE"
            )

        else:
            self._current_cx_to_get = index if index != -1 else 2

    def append_fake(self, name: Str, index: CX | AnyList, bimapping: BiMapping) -> None:
        """
        Add a new variable to the fake list which add something without changing the size of the normal elements

        Parameters
        ----------
        name: str
            The name of the variable
        index: MX | SX
            The SX or MX variable associated with this variable. Is interpreted as index if is_fake is true
        bimapping: BiMapping
            The Mapping of the MX against CX
        """

        self.fake_elements.append(OptimizationVariable(name, None, index, bimapping, self))

    def append(self, name: Str, cx: CXList, bimapping: BiMapping) -> None:
        """
        Add a new variable to the list

        Parameters
        ----------
        name: str
            The name of the variable
        cx: list
            The list of SX or MX variable associated with this variable
        """

        if len(cx) < 2:
            raise NotImplementedError("cx should be of dimension 2 (start, [mid], end)")

        index = range(self._cx_start.shape[0], self._cx_start.shape[0] + cx[0].shape[0])
        self._cx_start = vertcat(self._cx_start, cx[0])
        if len(cx) > 2:
            self._cx_mid = vertcat(self._cx_mid, cx[(len(cx) - 1) // 2])
        self._cx_end = vertcat(self._cx_end, cx[-1])

        for i, c in enumerate(cx[1:-1]):
            self._cx_intermediates[i] = vertcat(self._cx_intermediates[i], c)

        self.elements.append(OptimizationVariable(name, cx, index, bimapping, parent_list=self))

    def append_from_scaled(
        self,
        name: Str,
        cx: CXList,
        scaled_optimization_variable: "OptimizationVariableList",
    ) -> None:
        """
        Add a new variable to the list

        Parameters
        ----------
        name: str
            The name of the variable
        cx: list
            The list of SX or MX variable associated with this variable
        scaled_optimization_variable: OptimizationVariable
            The scaled optimization variable associated with this variable
        """

        if len(cx) < 2:
            raise NotImplementedError("cx should be of dimension 2 (start, [mid], end)")

        self._cx_start = vertcat(self._cx_start, cx[0])
        if len(cx) > 2:
            self._cx_mid = vertcat(self._cx_mid, cx[(len(cx) - 1) // 2])
        self._cx_end = vertcat(self._cx_end, cx[-1])

        for i, c in enumerate(cx[1:-1]):
            if i >= len(self._cx_intermediates):
                self._cx_intermediates.append(c)
            else:
                self._cx_intermediates[i] = vertcat(self._cx_intermediates[i], c)

        var = scaled_optimization_variable[name]
        self.elements.append(OptimizationVariable(name, cx, var.index, var.mapping, self))

    @property
    def cx(self):
        """
        Return the current cx to get. This is to get the user an easy accessor to the proper cx to get. Indeed, in the
        backend, we may need to redirect the user toward cx_start, cx_mid or cx_end, but they may not know which to use.
        CX is expected to return the proper cx to the user.

        Returns
        -------
        Tne cx related to the current state
        """

        if self._current_cx_to_get == 0:
            return self.cx_start
        elif self._current_cx_to_get == 1:
            return self.cx_mid
        elif self._current_cx_to_get == 2:
            return self.cx_end
        else:
            NotImplementedError("This should not happen, please contact a programmer!")

    @property
    def cx_start(self) -> CX:
        """
        The cx of all elements together
        """

        # Recast in CX since if it happens to be empty it is transformed into a DM behind the scene
        is_empty = self.shape == 0 or np.prod(self._cx_start.shape) == 0
        return self.cx_constructor([] if is_empty else self._cx_start[:, 0])

    @property
    def cx_mid(self) -> CX:
        """
        The cx of all elements together
        """

        # Recast in CX since if it happens to be empty it is transformed into a DM behind the scene
        is_empty = self.shape == 0 or np.prod(self._cx_mid.shape) == 0
        return self.cx_constructor([] if is_empty else self._cx_mid[:, 0])

    @property
    def cx_end(self) -> CX:
        """
        The cx of all elements together
        """

        # Recast in CX since if it happens to be empty it is transformed into a DM behind the scene
        is_empty = self.shape == 0 or np.prod(self._cx_end.shape) == 0
        return self.cx_constructor([] if is_empty else self._cx_end[:, 0])

    @property
    def cx_intermediates_list(self) -> CXList:
        """
        The cx of all elements together (starting point)
        """

        return self._cx_intermediates

    def __contains__(self, item: Str) -> Bool:
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

    def keys(self) -> StrList:
        """
        All the names of the elements in the list
        """

        return [elt for elt in self]

    @property
    def shape(self) -> Int:
        """
        The size of the CX
        """

        return self._cx_start.shape[0]

    def __len__(self) -> Int:
        """
        The number of variables in the list
        """

        return len(self.elements)

    def __iter__(self) -> "OptimizationVariableList":
        """
        Allow for the list to be used in a for loop

        Returns
        -------
        A reference to self
        """

        self._iter_idx = 0
        return self

    def __next__(self) -> Str:
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


class OptimizationVariableContainer:
    def __init__(self, phase_dynamics: PhaseDynamics):
        """
        This is merely a declaration function, it is mandatory to call initialize_from_shooting to get valid structures

        Parameters
        ----------
        phase_dynamics: PhaseDynamics
            If the dynamics is the same for all the phase (effectively always setting _node_index to 0 even though the
            user sets it to something else)
        """
        self.cx_constructor = None
        self._unscaled: list[OptimizationVariableList] = []
        self._scaled: list[OptimizationVariableList] = []
        self._node_index = 0  # TODO: [0] to [node_index]
        self.phase_dynamics = phase_dynamics

    @property
    def node_index(self) -> Int:
        return self._node_index

    @node_index.setter
    def node_index(self, value: Int) -> None:
        if self.phase_dynamics == PhaseDynamics.ONE_PER_NODE:
            self._node_index = value

    def initialize_from_shooting(self, n_shooting: Int, cx: Callable) -> None:
        """
        Initialize the Container so the dimensions are up to number of shooting points of the program

        Parameters
        ----------
        n_shooting
            The number of shooting points
        cx
            The type (MX or SX) of the variable
        """

        if not isinstance(cx, Callable):
            raise ValueError("This entry should be either casadi.MX or casadi.SX")

        for node_index in range(n_shooting):
            self.cx_constructor = cx
            self._scaled.append(OptimizationVariableList(cx, self.phase_dynamics))
            self._unscaled.append(OptimizationVariableList(cx, self.phase_dynamics))

    def initialize_intermediate_cx(self, n_shooting: Int, n_cx: Int) -> None:
        """
        Initialize the containers so the dimensions are up to the required intermediate points,
        especially important in collocations

        Parameters
        ----------
        n_shooting: int
            The number of shooting points
        n_cx: int
            The number of intermediate points
        """

        for node_index in range(n_shooting):
            self._scaled[node_index]._cx_intermediates = [self.cx_constructor([]) for _ in range(n_cx)]
            self._unscaled[node_index]._cx_intermediates = [self.cx_constructor([]) for _ in range(n_cx)]

    def __getitem__(self, item: Int | Str) -> OptimizationVariable:
        if isinstance(item, int):
            raise ValueError("To get a specific node, please set the node_index property then call the desired method.")
        elif isinstance(item, str):
            return self.unscaled[item]
        else:
            raise ValueError("item should be a node index or the name of the variable")

    @property
    def unscaled(self) -> OptimizationVariableList:
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._unscaled[self.node_index]

    @property
    def scaled(self) -> OptimizationVariableList:
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._scaled[self.node_index]

    def keys(self) -> StrList:
        return self._unscaled[0].keys()

    def key_index(self, key: Str) -> Range | IntList:
        return self._unscaled[0][key].index

    @property
    def shape(self) -> Int:
        if isinstance(self._unscaled, list) and len(self._unscaled) == 0:
            raise RuntimeError("The optimization variable container is empty. Please initialize it first.")
        else:
            return self._unscaled[0].shape

    @property
    def cx(self) -> CX:
        return self.unscaled.cx

    @property
    def cx_start(self) -> CX:
        return self.unscaled.cx_start

    @property
    def cx_intermediates_list(self) -> CXList:
        return self.unscaled.cx_intermediates_list

    @property
    def cx_end(self) -> CX:
        return self.unscaled.cx_end

    def append(
        self,
        name: Str,
        cx: CXList,
        cx_scaled: CXList,
        mapping: BiMapping,
        node_index: Int,
    ) -> None:
        """
        Add a new variable to the list

        Parameters
        ----------
        name: str
            The name of the variable
        cx: list
            The list of unscaled SX or MX variable associated with this variable
        cx_scaled: list
            The list of scaled SX or MX variable associated with this variable
        mapping
            The mapping to apply to the unscaled variable
        node_index
            The index of the node for the scaled variable
        """
        self._scaled[node_index].append(name, cx_scaled, mapping)
        self._unscaled[node_index].append_from_scaled(name, cx, self._scaled[node_index])

    def __contains__(self, item: Str) -> Bool:
        """
        If the item of name item is in the list
        """
        if item == "scaled" or item == "unscaled":
            return True

        return item in self.unscaled

    def __len__(self) -> Int:
        """
        The number of variables in the list
        """

        return len(self.unscaled)

    def __iter__(self) -> "OptimizationVariableContainer":
        """
        Allow for the list to be used in a for loop

        Returns
        -------
        A reference to self
        """

        self._iter_idx = 0
        return self

    def __next__(self) -> Str:
        """
        Get the next phase of the option list

        Returns
        -------
        The next phase of the option list
        """

        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self.unscaled[self._iter_idx - 1].name