from typing import Callable, Any

from casadi import MX, SX, DM, vertcat, Function
import numpy as np

from ..limits.penalty_controller import PenaltyController
from ..limits.penalty import PenaltyOption
from ..misc.options import UniquePerProblemOptionList
from ..misc.mapping import BiMapping
from ..optimization.variable_scaling import VariableScaling, VariableScalingList
from ..optimization.optimization_variable import OptimizationVariable, OptimizationVariableList, OptimizationVariableContainer

class Parameter(OptimizationVariable):
    """
    A Parameter.
    """

    def __init__(self,
                 name: str,
                 mx: MX,
                 cx_start: list | None,
                 index: [range, list],
                 mapping: BiMapping = None,
                 parent_list=None,
                 function: Callable = None,
                 size: int = None,
                 cx_type: Callable | MX | SX = None,
                 scaling: VariableScaling = None,
                 **params: Any,
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
        super(Parameter, self).__init__(name, mx, cx_start, index, mapping, parent_list)
        self.function = function
        self.size = size
        self.cx_type = cx_type
        self.scaling = scaling
        self.params = params

    @property
    def cx(self):
        if self.parent_list is not None:
            return self.cx_start
        else:
            return self.cx_start()

    @property
    def cx_start(self):
        """
        The CX of the variable
        """

        if self.parent_list is None:
            raise RuntimeError(
                "Parameter must have been created by ParameterList to have a cx. "
                "Typically 'all' cannot be used"
            )
        return self.parent_list.cx_start

    def cx_mid(self):
        raise RuntimeError(
            "cx_mid is not available for parameters, only cx_start is accepted."
        )

    def cx_end(self):
        raise RuntimeError(
            "cx_end is not available for parameters, only cx_start is accepted."
        )

    def cx_intermediates_list(self):
        raise RuntimeError(
            "cx_intermediates_list is not available for parameters, only cx_start is accepted."
        )

class ParameterList(OptimizationVariableList):
    """
    A list of Parameters.
    """

    def __init__(self, cx_constructor, phase_dynamics):
        super(ParameterList, self).__init__(cx_constructor, phase_dynamics)
        self._cx_mid = None  # del ?
        self._cx_end = None
        self._cx_intermediates = None

    @property
    def current_cx_to_get(self):
        return self._current_cx_to_get

    @current_cx_to_get.setter
    def current_cx_to_get(self, index: int):
        raise RuntimeError(
            "current_cx_to_get is not changable for parameters, only cx_start is accepted (current_cx_to_get is always 0)."
        )

    # TODO
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
        """

        if len(cx) < 2:
            raise NotImplementedError("cx should be of dimension 2 (start, [mid], end)")

        index = range(self._cx_start.shape[0], self._cx_start.shape[0] + cx[0].shape[0])
        self._cx_start = vertcat(self._cx_start, cx[0])
        if len(cx) > 2:
            self._cx_mid = vertcat(self._cx_mid, cx[(len(cx) - 1) // 2])
        self._cx_end = vertcat(self._cx_end, cx[-1])

        for i, c in enumerate(cx[1:-1]):
            if i >= len(self._cx_intermediates):
                self._cx_intermediates.append(c)
            else:
                self._cx_intermediates[i] = vertcat(self._cx_intermediates[i], c)

        self.mx_reduced = vertcat(self.mx_reduced, MX.sym("var", cx[0].shape[0]))
        self.elements.append(OptimizationVariable(name, mx, cx, index, bimapping, parent_list=self))

    # TODO
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

        self.mx_reduced = scaled_optimization_variable.mx_reduced
        var = scaled_optimization_variable[name]
        self.elements.append(OptimizationVariable(name, var.mx, cx, var.index, var.mapping, self))


    def cx_mid(self):
        raise RuntimeError(
            "cx_mid is not available for parameters, only cx_start is accepted."
        )

    def cx_end(self):
        raise RuntimeError(
            "cx_end is not available for parameters, only cx_start is accepted."
        )

    def cx_intermediates_list(self):
        raise RuntimeError(
            "cx_intermediates_list is not available for parameters, only cx_start is accepted."
        )


class ParameterContainer(OptimizationVariableContainer):
    """
    A parameter container (i.e., the list of scaled parameters and a list of unscaled parameters).
    """
    def __init__(self, phase_dynamics):
        super(ParameterContainer, self).__init__(phase_dynamics)
        self._scaled: list[ParameterList, ...] = []
        self._unscaled: list[ParameterList, ...] = []

    @property
    def node_index(self):
        return self._node_index

    @node_index.setter
    def node_index(self, value):
        raise RuntimeError(
            "node_index is not changable for parameters since it is the same for every node."
        )

    def initialize(self, cx):
        # TODO
        return

    def initialize_from_shooting(self, n_shooting: int, cx: Callable):
        raise RuntimeError(
            "initialize_from_shooting is not available for parameters, only initialize is accepted."
        )

    @property
    def unscaled(self):
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._unscaled

    @property
    def scaled(self):
        """
        This method allows to intercept the scaled item and return the current node_index
        """
        return self._scaled

    @property
    def cx_intermediates_list(self):
        raise RuntimeError(
            "cx_intermediates_list is not available for parameters, only cx_start is accepted."
        )

    @property
    def cx_mid(self):
        raise RuntimeError(
            "cx_mid is not available for parameters, only cx_start is accepted."
        )

    @property
    def cx_end(self):
        raise RuntimeError(
            "cx_end is not available for parameters, only cx_start is accepted."
        )


# class Parameter(PenaltyOption):
#     """
#     A placeholder for a parameter
#
#     Attributes
#     ----------
#     function: Callable[OptimalControlProgram, MX]
#             The user defined function that modify the model
#     quadratic: bool
#         If the objective is squared [True] or not [False]
#     size: int
#         The number of variables this parameter has
#     cx: MX | SX
#         The type of casadi variable
#     mx: MX
#         The MX vector of the parameter
#     """
#
#     def __init__(
#         self,
#         function: Callable = None,
#         quadratic: bool = True,
#         size: int = None,
#         cx: Callable | MX | SX = None,
#         scaling: VariableScaling = None,
#         **params: Any,
#     ):
#         """
#         Parameters
#         ----------
#         function: Callable[OptimalControlProgram, MX]
#             The user defined function that modify the model
#         quadratic: bool
#             If the objective is squared [True] or not [False]
#         index: list
#             The indices of the parameter in the decision vector list
#         size: int
#             The number of variables this parameter has
#         cx: MX | SX
#             The type of casadi variable
#         params: dict
#             Any parameters to pass to the function
#         """
#
#         super(Parameter, self).__init__(Parameters, **params)
#         self.function.append(function)
#         self.size = size
#
#         if scaling is None:
#             scaling = VariableScaling(self.name, np.ones((self.size, 1)))
#         if not isinstance(scaling, VariableScaling):
#             raise ValueError("Parameter scaling must be a VariableScaling")
#
#         if scaling.shape[0] != self.size:
#             raise ValueError(f"Parameter scaling must be of size {self.size}, not {scaling.shape[0]}.")
#         if scaling.shape[1] != 1:
#             raise ValueError(f"Parameter scaling must have exactly one column, not {scaling.shape[1]}.")
#
#         self.scaling = scaling
#
#         self.quadratic = quadratic
#         self.index = None
#         self.cx = None
#         self.mx = None
#         self.declare_symbolic(cx)
#
#     def declare_symbolic(self, cx):
#         self.cx = cx.sym(self.name, self.size, 1)
#         self.mx = MX.sym(self.name, self.size, 1)
#
#     @property
#     def shape(self):
#         return self.cx.shape[0]
#
#     def add_or_replace_to_penalty_pool(self, ocp, penalty):
#         """
#         This allows to add a parameter penalty to the penalty function pool.
#         Parameters
#         ----------
#         ocp: OptimalControlProgram
#             A reference to the ocp
#         penalty: PenaltyOption
#             The penalty to add
#         """
#
#         if not penalty.name:
#             if penalty.type.name == "CUSTOM":
#                 penalty.name = penalty.custom_function.__name__
#             else:
#                 penalty.name = penalty.type.name
#
#         fake_penalty_controller = PenaltyController(ocp, ocp.nlp[0], [], [], [], [], [], ocp.parameters.cx, [], [])
#         penalty_function = penalty.type(penalty, fake_penalty_controller, **penalty.params)
#         self.set_penalty(ocp, fake_penalty_controller, penalty, penalty_function, penalty.expand)
#
#     def set_penalty(
#         self,
#         ocp,
#         controller: PenaltyController,
#         penalty: PenaltyOption,
#         penalty_function: MX | SX,
#         expand: bool = False,
#     ):
#         penalty.node_idx = [0]
#         penalty.dt = 1
#         penalty.multi_thread = False
#         self._set_penalty_function(ocp, controller, penalty, penalty_function, expand)
#
#     def _set_penalty_function(self, ocp, controller, penalty, penalty_function: MX | SX, expand: bool = False):
#         """
#         This method actually created the penalty function and adds it to the pool.
#
#         Parameters
#         ----------
#         ocp: OptimalControlProgram
#             A reference to the ocp
#         controller: PenaltyController
#             A reference to the penalty controller
#         penalty: PenaltyOption
#             The penalty to add
#         penalty_function: MX | SX
#             The penalty function
#         expand: bool
#             If the penalty function should be expanded or not
#         """ ""
#
#         # Do not use nlp.add_casadi_func because all functions must be registered
#         time_cx = ocp.cx(0, 0)
#         dt = ocp.dt_parameter.cx
#         state_cx = ocp.cx(0, 0)
#         control_cx = ocp.cx(0, 0)
#         param_cx = ocp.parameters.cx
#         algebraic_states_cx = ocp.cx(0, 0)
#
#         penalty.function.append(
#             Function(
#                 f"{self.name}",
#                 [time_cx, dt, state_cx, control_cx, param_cx, algebraic_states_cx],
#                 [penalty_function],
#                 ["t", "dt", "x", "u", "p", "a"],
#                 ["val"],
#             )
#         )
#
#         modified_fcn = penalty.function[0](time_cx, dt, state_cx, control_cx, param_cx, algebraic_states_cx)
#
#         weight_cx = ocp.cx.sym("weight", 1, 1)
#         target_cx = ocp.cx.sym("target", modified_fcn.shape)
#
#         modified_fcn = modified_fcn - target_cx
#         modified_fcn = modified_fcn**2 if penalty.quadratic else modified_fcn
#
#         penalty.weighted_function.append(
#             Function(  # Do not use nlp.add_casadi_func because all of them must be registered
#                 f"{self.name}",
#                 [time_cx, dt, state_cx, control_cx, param_cx, algebraic_states_cx, weight_cx, target_cx],
#                 [weight_cx * modified_fcn],
#                 ["t", "dt", "x", "u", "p", "a", "weight", "target"],
#                 ["val"],
#             )
#         )
#
#         if expand:
#             penalty.function[0] = penalty.function[0].expand()
#             penalty.weighted_function[0] = penalty.weighted_function[0].expand()
#
#         pool = controller.ocp.J
#         pool.append(penalty)

#
# class ParameterList(UniquePerProblemOptionList):
#     """
#     A list of Parameter
#
#     Methods
#     -------
#     add(
#         self,
#         parameter_name: str | Parameter,
#         function: Callable = None,
#         size: int = None,
#         phase: int = 0,
#         **extra_arguments
#     ):
#         Add a new Parameter to the list
#     print(self)
#         Print the ParameterList to the console
#     __contains__(self, item: str) -> bool
#         Allow for `str in ParameterList`
#     names(self) -> list:
#         Get all the name of the Parameter in the List
#     index(self, item: str) -> int
#         Get the index of a specific Parameter in the list
#     scaling(self)
#         The scaling of the parameters
#     cx(self)
#         The CX vector of all parameters
#     mx(self)
#         The MX vector of all parameters
#     shape(self)
#         The size of all parameters vector
#     """
#
#     def __init__(self):
#         super(ParameterList, self).__init__()
#
#         # This cx_type was introduced after Casadi changed the behavior of vertcat which now returns a DM.
#         self.cx_type = MX  # Assume MX for now, if needed, optimal control program will set this properly
#
#     def add(
#         self,
#         parameter_name: str | Parameter,
#         function: Callable = None,
#         size: int = None,
#         list_index: int = -1,
#         scaling: VariableScaling = None,
#         allow_reserved_name: bool = False,
#         **extra_arguments: Any,
#     ):
#         """
#         Add a new Parameter to the list
#
#         Parameters
#         ----------
#         parameter_name: str | Parameter
#             If str, the name of the parameter. This name will be used for plotting purpose. It must be unique
#             If Parameter, the parameter is copied
#         function: Callable[OptimalControlProgram, MX]
#             The user defined function that modify the model
#         size: int
#             The number of variables this parameter has
#         list_index: int
#             The index of the parameter in the parameters list
#         scaling: float
#             The scaling of the parameter
#         allow_reserved_name: bool
#             Overrides the restriction to reserved key. This is for internal purposes and should not be used by users
#             as it will result in undefined behavior
#         extra_arguments: dict
#             Any argument that should be passed to the user defined functions
#         """
#
#         if not allow_reserved_name and parameter_name == "dt":
#             raise KeyError("It is not possible to declare a parameter with the key 'dt' as it is a reserved name")
#
#         if isinstance(parameter_name, Parameter):
#             # case it is not a parameter name but trying to copy another parameter
#             self.copy(parameter_name)
#             if parameter_name.name != "dt":
#                 self[parameter_name.name].declare_symbolic(self.cx_type)
#         else:
#             if "phase" in extra_arguments:
#                 raise ValueError(
#                     "Parameters are declared for all phases at once. You must therefore not use "
#                     "'phase' but 'list_index' instead"
#                 )
#             super(ParameterList, self)._add(
#                 option_type=Parameter,
#                 list_index=list_index,
#                 function=function,
#                 name=parameter_name,
#                 size=size,
#                 scaling=scaling,
#                 cx=self.cx_type,
#                 **extra_arguments,
#             )
#
#     @property
#     def scaling(self) -> VariableScalingList:
#         """
#         The scaling of the parameters
#         """
#         out = VariableScalingList()
#         for p in self:
#             out.add(p.name, p.scaling.scaling)
#         return out
#
#     def __contains__(self, item: str) -> bool:
#         """
#         Allow for `str in ParameterList`
#
#         Parameters
#         ----------
#         item: str
#             The element to search
#
#         Returns
#         -------
#         If the element is the list
#         """
#
#         for p in self:
#             if p.name == item:
#                 return True
#         return False
#
#     def print(self):
#         # TODO: Print all elements in the console
#         raise NotImplementedError("Printing of ParameterList is not ready yet")
#
#     @property
#     def names(self) -> list:
#         """
#         Get all the name of the Parameter in the List
#
#         Returns
#         -------
#         A list of all names
#         """
#
#         n = []
#         for p in self:
#             n.append(p.name)
#         return n
#
#     def index(self, item: str) -> int:
#         """
#         Get the index of a specific Parameter in the list
#
#         Parameters
#         ----------
#         item: str
#             The name of the parameter to find
#
#         Returns
#         -------
#         The index of the Parameter in the list
#         """
#
#         return self.names.index(item)
#
#     def keys(self):
#         return [p.name for p in self.options[0]]
#
#     @property
#     def cx(self):
#         return self.cx_start
#
#     @property
#     def cx_start(self):
#         """
#         The CX vector of all parameters
#         """
#         out = vertcat(*[p.cx for p in self])
#         if isinstance(out, DM):
#             # Force the type if it is a DM (happens if self is empty)
#             out = self.cx_type(out)
#         return out
#
#     @property
#     def mx(self):
#         """
#         The MX vector of all parameters
#         """
#         out = vertcat(*[p.mx for p in self])
#         if isinstance(out, DM):
#             # Force the type if it is a DM (happens if self is empty)
#             out = MX(out)
#         return out
#
#     @property
#     def shape(self):
#         """ """
#         return sum([p.shape for p in self])

#
# class Parameters:
#     """
#     Emulation of the base class PenaltyFunctionAbstract so Parameters can be used as Objective and Constraints
#
#     Methods
#     -------
#     get_type()
#         Returns the type of the penalty
#     penalty_nature() -> str
#         Get the nature of the penalty
#     """
#
#     @staticmethod
#     def get_type():
#         """
#         Returns the type of the penalty
#         """
#
#         return Parameters
#
#     @staticmethod
#     def penalty_nature() -> str:
#         """
#         Get the nature of the penalty
#
#         Returns
#         -------
#         The nature of the penalty
#         """
#
#         return "parameter_objectives"
