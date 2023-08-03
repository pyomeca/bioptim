from typing import Callable

import numpy as np
from casadi import MX, SX, vertcat

from ..misc.mapping import BiMapping
from ..misc.enums import Node


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
    cx_start(self)
        The CX of the variable (starting point)
    cx_end(self)
        The CX of the variable (ending point)
    """

    def __init__(
        self,
        name: str,
        mx: MX,
        cx_start: list | None,
        index: [range, list],
        mapping: BiMapping = None,
        parent_list=None,
    ):
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
        self.original_cx: list = cx_start
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
    def shape(self):
        """
        The len of the MX reduced

        Returns
        -------
        The number of element (correspond to the nrows of the MX)
        """
        return len(self)

    @property
    def cx(self):
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
            return self.cx_start()

    @property
    def cx_start(self):
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
    def cx_mid(self):
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
    def cx_end(self):
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "OptimizationVariable must have been created by OptimizationVariableList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_end[self.index, :]

    @staticmethod
    def reshape_to_vector(matrix):
        """
        Restore the vector form of the matrix
        """
        shape_0 = matrix.shape[0]
        shape_1 = matrix.shape[1]
        vector = MX.zeros(shape_0 * shape_1)
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                vector[shape_0 * s0 + s1] = matrix[s0, s1]
        return vector

    @staticmethod
    def reshape_to_matrix(variable, shape_0, shape_1, node: Node, key: str):
        """
        Restore the matrix form of the variables
        """
        matrix = MX(shape_0, shape_1)
        i = 0
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                if node == Node.START:
                    matrix[s0, s1] = variable[key].cx_start[i]
                elif node == Node.MID:
                    matrix[s0, s1] = variable[key].cx_mid[i]
                elif node == Node.END:
                    matrix[s0, s1] = variable[key].cx_end[i]
                else:
                    raise RuntimeError(
                        "Node must be a Node.START for cx_start, Node.MID for cx_mid, or Node.END for cx_end"
                    )
                i += 1
        return matrix

    @staticmethod
    def reshape_to_cholesky_matrix(variable, shape_0, node: Node, key: str):
        """
        Restore the lower diagonal matrix form of the variables vector
        """
        matrix = MX.zeros(shape_0, shape_0)
        i = 0
        for s0 in range(shape_0):
            for s1 in range(s0 + 1):
                if node == Node.START:
                    matrix[s0, s1] = variable[key].cx_start[i]
                elif node == Node.MID:
                    matrix[s0, s1] = variable[key].cx_mid[i]
                elif node == Node.END:
                    matrix[s0, s1] = variable[key].cx_end[i]
                else:
                    raise RuntimeError(
                        "Node must be a Node.START for cx_start, Node.MID for cx_mid, or Node.END for cx_end"
                    )
                i += 1
        return matrix


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
    mx_reduced: MX
        The reduced MX to the size of _cx
    cx_constructor: Callable
        The casadi symbolic type used for the optimization (MX or SX)

    Methods
    -------
    __getitem__(self, item: int | str)
        Get a specific variable in the list, whether by name or by index
    append(self, name: str, cx: list, mx: MX, bimapping: BiMapping)
        Add a new variable to the list
    cx(self)
        The cx of all elements together (starting point)
    cx_end(self)
        The cx of all elements together (ending point)
    mx(self)
        The MX of all variable concatenated together
    shape(self)
        The size of the CX
    __len__(self)
        The number of variables in the list
    """

    def __init__(self, cx_constructor, assume_phase_dynamics):
        self.elements: list = []
        self.fake_elements: list = []
        self._cx_start: MX | SX | np.ndarray = np.array([])
        self._cx_mid: MX | SX | np.ndarray = np.array([])
        self._cx_end: MX | SX | np.ndarray = np.array([])
        self._cx_intermediates: list = []
        self.mx_reduced: MX = MX.sym("var", 0, 0)
        self.cx_constructor = cx_constructor
        self._current_cx_to_get = 0
        self.assume_phase_dynamics = assume_phase_dynamics

    def __getitem__(self, item: int | str | list | range):
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
                # TODO Benjamin
                index = []
                for elt in self.elements:
                    index.extend(list(elt.index))
                return OptimizationVariable("all", self.mx, self.cx_start, index, None, self)

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
            return OptimizationVariable("some", mx, None, index)
        else:
            raise ValueError("OptimizationVariableList can be sliced with int, list, range or str only")

    def __setitem__(self, key, value: OptimizationVariable):
        self.elements.append(value)

    @property
    def current_cx_to_get(self):
        return self._current_cx_to_get

    @current_cx_to_get.setter
    def current_cx_to_get(self, index: int):
        """
        Set the value of current_cx_to_get to corresponding index (cx_start for 0, cx_mid for 1, cx_end for 2) if
        ocp.assume_phase_dynamics. Otherwise, it is always cx_start

        Parameters
        ----------
        index
            The index to set the current cx to
        """

        if not self.assume_phase_dynamics:
            self._current_cx_to_get = 0
            return

        if index < 0 or index > 2:
            raise ValueError(
                "Valid values for setting the cx is 0, 1 or 2. If you reach this error message, you probably tried to "
                "add more penalties than available in a multinode constraint. You can try to split the constraints "
                "into more penalties or use assume_phase_dynamics=False."
            )

        else:
            self._current_cx_to_get = index

    def append_fake(self, name: str, index: MX | SX | list, mx: MX, bimapping: BiMapping):
        """
        Add a new variable to the fake list which add something without changing the size of the normal elements

        Parameters
        ----------
        name: str
            The name of the variable
        index: MX | SX
            The SX or MX variable associated with this variable. Is interpreted as index if is_fake is true
        mx: MX
            The MX variable associated with this variable
        bimapping: BiMapping
            The Mapping of the MX against CX
        """

        self.fake_elements.append(OptimizationVariable(name, mx, None, index, bimapping, self))

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

        if len(cx) < 3:
            raise NotImplementedError("cx should be of dimension 3 (start, mid, end)")

        index = range(self._cx_start.shape[0], self._cx_start.shape[0] + cx[0].shape[0])
        self._cx_start = vertcat(self._cx_start, cx[0])
        self._cx_mid = vertcat(self._cx_mid, cx[(len(cx) - 1) // 2])
        self._cx_end = vertcat(self._cx_end, cx[-1])

        for i, c in enumerate(cx[1:-1]):
            if i >= len(self._cx_intermediates):
                self._cx_intermediates.append(c)
            else:
                self._cx_intermediates[i] = vertcat(self._cx_intermediates[i], c)

        self.mx_reduced = vertcat(self.mx_reduced, MX.sym("var", cx[0].shape[0]))
        self.elements.append(OptimizationVariable(name, mx, cx, index, bimapping, self))

    def append_from_scaled(
        self,
        name: str,
        cx: list,
        scaled_optimization_variable: "OptimizationVariableList",
    ):
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

        if len(cx) < 3:
            raise NotImplementedError("cx should be of dimension 3 (start, mid, end)")

        self._cx_start = vertcat(self._cx_start, cx[0])
        self._cx_mid = vertcat(self._cx_mid, cx[(len(cx) - 1) // 2])
        self._cx_end = vertcat(self._cx_end, cx[-1])

        for i, c in enumerate(cx[1:-1]):
            if i >= len(self._cx_intermediates):
                self._cx_intermediates.append(c)
            else:
                self._cx_intermediates[i] = vertcat(self._cx_intermediates[i], c)

        self.mx_reduced = scaled_optimization_variable.mx_reduced
        var = scaled_optimization_variable[name]
        self.elements.append(OptimizationVariable(name, var.mx, cx, var.index, var.mapping, self))

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
    def cx_start(self):
        """
        The cx of all elements together
        """

        # Recast in CX since if it happens to be empty it is transformed into a DM behind the scene
        return self.cx_constructor([] if self.shape == 0 else self._cx_start[:, 0])

    @property
    def cx_mid(self):
        """
        The cx of all elements together
        """

        # Recast in CX since if it happens to be empty it is transformed into a DM behind the scene
        return self.cx_constructor([] if self.shape == 0 else self._cx_mid[:, 0])

    @property
    def cx_end(self):
        """
        The cx of all elements together
        """

        # Recast in CX since if it happens to be empty it is transformed into a DM behind the scene
        return self.cx_constructor([] if self.shape == 0 else self._cx_end[:, 0])

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

        return self._cx_start.shape[0]

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


class OptimizationVariableContainer:
    def __init__(self, assume_phase_dynamics: bool):
        """
        This is merely a declaration function, it is mandatory to call initialize_from_shooting to get valid structures

        Parameters
        ----------
        assume_phase_dynamics
            If the dynamics is the same for all the phase (effectively always setting _node_index to 0 even though the
            user sets it to something else)
        """
        self.cx_constructor = None
        self._unscaled: list[OptimizationVariableList, ...] = []
        self._scaled: list[OptimizationVariableList, ...] = []
        self._node_index = 0  # TODO: [0] to [node_index]
        self.assume_phase_dynamic = assume_phase_dynamics

    @property
    def node_index(self):
        return self._node_index

    @node_index.setter
    def node_index(self, value):
        if not self.assume_phase_dynamic:
            self._node_index = value

    def initialize_from_shooting(self, n_shooting: int, cx: Callable):
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
            self._scaled.append(OptimizationVariableList(cx, self.assume_phase_dynamic))
            self._unscaled.append(OptimizationVariableList(cx, self.assume_phase_dynamic))

    def __getitem__(self, item: int | str):
        if isinstance(item, int):
            raise ValueError("To get a specific node, please set the node_index property then call the desired method.")
        elif isinstance(item, str):
            return self.unscaled[item]
        else:
            raise ValueError("item should be a node index or the name of the variable")

    @property
    def unscaled(self):
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._unscaled[self.node_index]

    @property
    def scaled(self):
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._scaled[self.node_index]

    def keys(self):
        return self.unscaled.keys()

    @property
    def shape(self):
        return self.unscaled.shape

    @property
    def mx(self):
        return self.unscaled.mx

    @property
    def mx_reduced(self):
        return self.unscaled.mx_reduced

    @property
    def cx_start(self):
        return self.unscaled.cx_start

    @property
    def cx_intermediates_list(self):
        return self.unscaled.cx_intermediates_list

    @property
    def cx_end(self):
        return self.unscaled.cx_end

    def append(
        self,
        name: str,
        cx: list,
        cx_scaled: list,
        mx: MX,
        mapping: BiMapping,
        node_index: int,
    ):
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
        mx: MX
            The symbolic variable associated to this variable
        mapping
            The mapping to apply to the unscaled variable
        node_index
            The index of the node for the scaled variable
        """
        self._scaled[node_index].append(name, cx_scaled, mx, mapping)
        self._unscaled[node_index].append_from_scaled(name, cx, self._scaled[node_index])

    def __contains__(self, item: str):
        """
        If the item of name item is in the list
        """
        if item == "scaled" or item == "unscaled":
            return True

        return item in self.unscaled

    def __len__(self):
        """
        The number of variables in the list
        """

        return len(self.unscaled)

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
        return self.unscaled[self._iter_idx - 1].name

    def reshape_to_vector(self, matrix):
        """
        Restore the vector form of the matrix
        """
        shape_0 = matrix.shape[0]
        shape_1 = matrix.shape[1]
        vector = MX.zeros(shape_0 * shape_1)
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                vector[shape_0 * s0 + s1] = matrix[s0, s1]
        return vector

    def reshape_to_matrix(self, variable, shape_0, shape_1, node: Node, key: str):
        """
        Restore the matrix form of the variables
        """
        matrix = MX(shape_0, shape_1)
        i = 0
        for s0 in range(shape_0):
            for s1 in range(shape_1):
                if node == Node.START:
                    matrix[s0, s1] = variable[key].cx_start[i]
                elif node == Node.MID:
                    matrix[s0, s1] = variable[key].cx_mid[i]
                elif node == Node.END:
                    matrix[s0, s1] = variable[key].cx_end[i]
                else:
                    raise RuntimeError(
                        "Node must be a Node.START for cx_start, Node.MID for cx_mid, or Node.END for cx_end"
                    )
                i += 1
        return matrix
