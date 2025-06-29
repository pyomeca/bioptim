from typing import Callable, Any

from casadi import MX, SX, vertcat
import numpy as np

from ..misc.mapping import BiMapping
from ..misc.parameters_types import (
    Str,
    Int,
    IntList,
    Range,
    Bool,
    CX,
    CXList,
    StrList,
)
from ..models.protocols.biomodel import BioModel
from ..optimization.variable_scaling import VariableScaling, VariableScalingList
from ..optimization.optimization_variable import (
    OptimizationVariable,
    OptimizationVariableList,
    OptimizationVariableContainer,
)
from ..misc.enums import PhaseDynamics


class Parameter(OptimizationVariable):
    """
    A Parameter.
    """

    def __init__(
        self,
        name: Str,
        mx: MX,
        cx_start: CXList | None,
        index: Range | IntList,
        mapping: BiMapping = None,
        parent_list: "ParameterList" = None,
        function: Callable | None = None,
        size: Int | None = None,
        cx_type: Callable | CX | None = None,
        scaling: VariableScaling | None = None,
        **kwargs: Any,
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
        parent_list: ParameterList
            The list the OptimizationVariable is in
        """
        super(Parameter, self).__init__(name, cx_start, index, mapping, parent_list)
        self.function = function
        self.size = size
        self.cx_type = cx_type

        if scaling is None:
            scaling = VariableScaling(self.name, np.ones((self.size, 1)))
        if not isinstance(scaling, VariableScaling):
            raise ValueError("Parameter scaling must be a VariableScaling")

        if scaling.shape[0] != self.size:
            raise ValueError(f"Parameter scaling must be of size {self.size}, not {scaling.shape[0]}.")
        if scaling.shape[1] != 1:
            raise ValueError(f"Parameter scaling must have exactly one column, not {scaling.shape[1]}.")

        self.scaling = scaling
        self.index = index
        self.kwargs = kwargs
        self._mx = mx

    @property
    def mx(self) -> MX:
        # TODO: this should removed and placed in the BiorbdModel
        return self._mx

    @property
    def cx(self) -> CX:
        if self.parent_list is not None:
            return self.cx_start
        else:
            return self.cx_start()

    @property
    def cx_start(self) -> CX:
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "Parameter must have been created by ParameterList to have a cx. " "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_start[self.index]

    def cx_mid(self) -> None:
        raise RuntimeError("cx_mid is not available for parameters, only cx_start is accepted.")

    def cx_end(self) -> None:
        raise RuntimeError("cx_end is not available for parameters, only cx_start is accepted.")

    def cx_intermediates_list(self) -> None:
        raise RuntimeError("cx_intermediates_list is not available for parameters, only cx_start is accepted.")

    def apply_parameter(self, model: "BioModel") -> None:
        """
        Apply the parameter variables to the model. This should be called during the creation of the biomodel
        """
        if self.function:
            param_scaling = self.scaling.scaling
            param_reduced = self.mx  # because this function will be used directly in the biorbd model
            self.function(model, param_reduced * param_scaling, **self.kwargs)


class ParameterList(OptimizationVariableList):
    """
    A list of Parameters.
    """

    def __init__(self, use_sx: Bool):
        cx_constructor = SX if use_sx else MX
        super(ParameterList, self).__init__(cx_constructor, phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)
        self.cx_type = cx_constructor
        self._cx_mid = None
        self._cx_end = None
        self._cx_intermediates = None
        self.function = []
        self.scaling = VariableScalingList()

    @property
    def current_cx_to_get(self) -> Int:
        return self._current_cx_to_get

    @current_cx_to_get.setter
    def current_cx_to_get(self, index: Int):
        raise RuntimeError(
            "current_cx_to_get is not changable for parameters, only cx_start is accepted (current_cx_to_get is always 0)."
        )

    def add(
        self,
        name: Str,
        function: Callable,
        size: Int,
        scaling: VariableScaling,
        mapping: BiMapping = None,
        allow_reserved_name: Bool = False,
        **kwargs: Any,
    ):
        """
        Add a new Parameter to the list
        This function should be called by the user. It will create the parameter and add it to the list

        Parameters
        ----------
        name: str
            The name of the parameter
        function: callable
            The user defined function that modify the model
        size: int
            The number of variables this parameter has
        scaling: VariableScaling
            The scaling of the parameter
        kwargs: dict
            Any argument that should be passed to the user defined function
        """

        if not allow_reserved_name and name == "dt":
            raise KeyError("It is not possible to declare a parameter with the key 'dt' as it is a reserved name.")

        if "phase" in kwargs:
            raise ValueError(
                "Parameters are declared for all phases at once. You must therefore not use "
                "'phase' but 'list_index' instead."
            )

        if not isinstance(scaling, VariableScaling):
            raise ValueError("Scaling must be a VariableScaling, not " + str(type(scaling)))
        else:
            if len(scaling.scaling.shape) != 1:
                if scaling.scaling.shape[1] != 1:
                    raise ValueError("Parameter scaling must have exactly one column")
            elif scaling.scaling.shape[0] != size:
                raise ValueError(f"Parameter scaling must be of size {size}, not {scaling.shape[0]}.")
        cx = [self.cx_constructor.sym(name, size)]

        if len(cx) != 1:
            raise NotImplementedError("cx should be of dimension 1 for parameters (there is no mid or end)")

        self.scaling.add(key=name, scaling=scaling)
        index = range(self._cx_start.shape[0], self._cx_start.shape[0] + cx[0].shape[0])
        self._cx_start = vertcat(self._cx_start, cx[0])
        mx = MX.sym(name, size)
        self.elements.append(
            Parameter(
                name=name,
                mx=mx,
                cx_start=cx,
                index=index,
                mapping=mapping,
                size=size,
                parent_list=self,
                function=function,
                cx_type=self.cx_type,
                scaling=scaling,
                **kwargs,
            )
        )

    def copy(self) -> "ParameterList":
        """
        Copy a parameters list to new one
        """

        parameters_copy = ParameterList(use_sx=(True if self.cx_type == SX else False))
        for element in self.elements:
            parameters_copy.elements.append(
                Parameter(
                    name=element.name,
                    mx=element.mx,
                    cx_start=element.cx_start,
                    index=element.index,
                    mapping=element.mapping,
                    parent_list=element.parent_list,
                    function=element.function,
                    size=element.size,
                    cx_type=element.cx_type,
                    scaling=VariableScaling(key=element.name, scaling=element.scaling.scaling),
                    **element.kwargs,
                )
            )
            parameters_copy._cx_start = vertcat(parameters_copy._cx_start, element.cx_start)

        return parameters_copy

    def cx_mid(self) -> None:
        raise RuntimeError("cx_mid is not available for parameters, only cx_start is accepted.")

    def cx_end(self) -> None:
        raise RuntimeError("cx_end is not available for parameters, only cx_start is accepted.")

    def cx_intermediates_list(self) -> None:
        raise RuntimeError("cx_intermediates_list is not available for parameters, only cx_start is accepted.")

    def add_a_copied_element(self, element_to_copy_index: Int) -> None:
        self.elements.append(self.elements[element_to_copy_index])

    @property
    def mx(self) -> MX:
        """
        Returns
        -------
        The MX of all variable concatenated together
        """
        out = MX()
        for elt in self.elements:
            if type(elt.mx) == MX:
                out = vertcat(out, elt.mx)
            else:
                out = vertcat(out, MX())
        return out


class ParameterContainer(OptimizationVariableContainer):
    """
    A parameter container (i.e., the list of scaled parameters and a list of unscaled parameters).
    """

    def __init__(self, use_sx: Bool):
        super(ParameterContainer, self).__init__(phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)
        self._parameters: ParameterList = ParameterList(use_sx=use_sx)

    @property
    def node_index(self) -> Int:
        return self._node_index

    @node_index.setter
    def node_index(self, value: Int):
        raise RuntimeError("node_index is not changable for parameters since it is the same for every node.")

    def initialize(self, parameters: ParameterList) -> None:
        """
        Initialize the Container so the dimensions are correct.

        Since the parameters are used in the model and outside of the model, we need to have a symbolic version of the CX
        for both of the scaled and the unscaled parameters. Therefore, one must explicitly scale or unscale the parameters
        when using .cx (contrary to the states and controls, which are done automatically).

        Parameters
        ----------

        """
        self._parameters = parameters.copy()
        return

    def initialize_from_shooting(self, n_shooting: Int, cx: Callable) -> None:
        raise RuntimeError("initialize_from_shooting is not available for parameters, only initialize is accepted.")

    @property
    def parameters(self) -> ParameterList:
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._parameters

    @property
    def unscaled(self) -> ParameterList:
        """
        This method allows to intercept the scaled item and return the current node_index

        Notes
        -----
        The return parameters is actually the scaled version of it.
        But for internal reasons, it is expected.
        This can be confusing for the user who expects an unscaled version though.
        Therefore, `.unscaled` method should only be call by the programmer who KNOWS what there are doing.
        """
        return self._parameters

    @property
    def scaled(self) -> ParameterList:
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._parameters

    def keys(self) -> StrList:
        return self._parameters.keys()

    def key_index(self, key: Str) -> Range | IntList:
        return self._parameters[key].index

    @property
    def shape(self) -> Int:
        return self._parameters.shape

    @property
    def cx_intermediates_list(self) -> None:
        raise RuntimeError("cx_intermediates_list is not available for parameters, only cx_start is accepted.")

    @property
    def cx_mid(self) -> None:
        raise RuntimeError("cx_mid is not available for parameters, only cx_start is accepted.")

    @property
    def cx_end(self) -> None:
        raise RuntimeError("cx_end is not available for parameters, only cx_start is accepted.")

    @property
    def mx(self) -> MX:
        return self.parameters.mx