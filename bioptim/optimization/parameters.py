from typing import Callable, Any

from casadi import MX, SX, DM, vertcat, Function
import numpy as np

from ..limits.penalty_controller import PenaltyController
from ..limits.penalty import PenaltyOption
from ..misc.options import UniquePerProblemOptionList
from ..optimization.non_linear_program import NonLinearProgram


class Parameter(PenaltyOption):
    """
    A placeholder for a parameter

    Attributes
    ----------
    function: Callable[OptimalControlProgram, MX]
            The user defined function that modify the model
    quadratic: bool
        If the objective is squared [True] or not [False]
    size: int
        The number of variables this parameter has
    cx: MX | SX
        The type of casadi variable
    mx: MX
        The MX vector of the parameter
    """

    def __init__(
        self,
        function: Callable = None,
        quadratic: bool = True,
        size: int = None,
        cx: Callable | MX | SX = None,
        scaling: np.ndarray = None,
        **params: Any,
    ):
        """
        Parameters
        ----------
        function: Callable[OptimalControlProgram, MX]
            The user defined function that modify the model
        quadratic: bool
            If the objective is squared [True] or not [False]
        index: list
            The indices of the parameter in the decision vector list
        size: int
            The number of variables this parameter has
        cx: MX | SX
            The type of casadi variable
        params: dict
            Any parameters to pass to the function
        """

        super(Parameter, self).__init__(Parameters, **params)
        self.function.append(function)

        if scaling is None:
            scaling = np.array([1.0])
        elif not isinstance(scaling, np.ndarray):
            raise ValueError("Parameter scaling must be a numpy array")

        if not (scaling > 0).all():
            raise ValueError("Parameter scaling must contain only positive values")

        if len(scaling.shape) == 0:
            raise ValueError("Parameter scaling must be a 1- or 2- dimensional array")
        elif len(scaling.shape) == 1:
            self.scaling = scaling[:, np.newaxis]
            if self.scaling.shape[0] != size and self.scaling.shape[0] == 1:
                self.scaling = np.repeat(self.scaling, size, 0)
            elif self.scaling.shape[0] != size and self.scaling.shape[0] != 1:
                raise ValueError(
                    f"The shape ({scaling.shape[0]}) of the scaling of parameter {params['name']} "
                    f"does not match the params shape."
                )
        elif len(scaling.shape) == 2:
            if scaling.shape[1] != 1:
                raise ValueError(
                    f"Invalid ncols for Parameter Scaling "
                    f"(ncols = {scaling.shape[1]}), the expected number of column is 1"
                )
        else:
            raise ValueError("Parameter scaling must be a 1- or 2- dimensional numpy array")

        self.quadratic = quadratic
        self.size = size
        self.index = None
        self.cx = None
        self.mx = None
        self.declare_symbolic(cx)

    def declare_symbolic(self, cx):
        self.cx = cx.sym(self.name, self.size, 1)
        self.mx = MX.sym(self.name, self.size, 1)

    @property
    def shape(self):
        return self.cx.shape[0]

    def add_or_replace_to_penalty_pool(self, ocp, penalty):
        """
        This allows to add a parameter penalty to the penalty function pool.
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        penalty: PenaltyOption
            The penalty to add
        """

        if not penalty.name:
            if penalty.type.name == "CUSTOM":
                penalty.name = penalty.custom_function.__name__
            else:
                penalty.name = penalty.type.name

        fake_penalty_controller = PenaltyController(ocp, ocp.nlp[0], [], [], [], [], [], ocp.parameters.cx, [], [])
        penalty_function = penalty.type(penalty, fake_penalty_controller, **penalty.params)
        self.set_penalty(ocp, fake_penalty_controller, penalty, penalty_function, penalty.expand)

    def set_penalty(
        self,
        ocp,
        controller: PenaltyController,
        penalty: PenaltyOption,
        penalty_function: MX | SX,
        expand: bool = False,
    ):
        penalty.node_idx = [0]
        penalty.dt = 1
        penalty.multi_thread = False
        self._set_penalty_function(ocp, controller, penalty, penalty_function, expand)

    def _set_penalty_function(self, ocp, controller, penalty, penalty_function: MX | SX, expand: bool = False):
        """
        This method actually created the penalty function and adds it to the pool.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        controller: PenaltyController
            A reference to the penalty controller
        penalty: PenaltyOption
            The penalty to add
        penalty_function: MX | SX
            The penalty function
        expand: bool
            If the penalty function should be expanded or not
        """ ""

        # Do not use nlp.add_casadi_func because all functions must be registered
        state_cx = ocp.cx(0, 0)
        control_cx = ocp.cx(0, 0)
        param_cx = ocp.parameters.cx
        stochastic_cx = ocp.cx(0, 0)
        motor_noise_cx = ocp.cx(0, 0)
        sensory_noise_cx = ocp.cx(0, 0)

        penalty.function.append(
            NonLinearProgram.to_casadi_func(
                f"{self.name}", penalty_function, state_cx, control_cx, param_cx, stochastic_cx, motor_noise_cx, sensory_noise_cx, expand=expand
            )
        )

        modified_fcn = penalty.function[0](state_cx, control_cx, param_cx, stochastic_cx, motor_noise_cx, sensory_noise_cx)

        dt_cx = ocp.cx.sym("dt", 1, 1)
        weight_cx = ocp.cx.sym("weight", 1, 1)
        target_cx = ocp.cx.sym("target", modified_fcn.shape)

        modified_fcn = modified_fcn - target_cx
        modified_fcn = modified_fcn**2 if penalty.quadratic else modified_fcn

        penalty.weighted_function.append(
            Function(  # Do not use nlp.add_casadi_func because all of them must be registered
                f"{self.name}",
                [state_cx, control_cx, param_cx, stochastic_cx, motor_noise_cx, sensory_noise_cx, weight_cx, target_cx, dt_cx],
                [weight_cx * modified_fcn * dt_cx],
            )
        )

        if expand:
            penalty.function[0].expand()
            penalty.weighted_function[0].expand()

        pool = controller.ocp.J
        pool.append(penalty)


class ParameterList(UniquePerProblemOptionList):
    """
    A list of Parameter

    Methods
    -------
    add(
        self,
        parameter_name: str | Parameter,
        function: Callable = None,
        size: int = None,
        phase: int = 0,
        **extra_arguments
    ):
        Add a new Parameter to the list
    print(self)
        Print the ParameterList to the console
    __contains__(self, item: str) -> bool
        Allow for `str in ParameterList`
    names(self) -> list:
        Get all the name of the Parameter in the List
    index(self, item: str) -> int
        Get the index of a specific Parameter in the list
    scaling(self)
        The scaling of the parameters
    cx(self)
        The CX vector of all parameters
    mx(self)
        The MX vector of all parameters
    shape(self)
        The size of all parameters vector
    """

    def __init__(self):
        super(ParameterList, self).__init__()

        # This cx_type was introduced after Casadi changed the behavior of vertcat which now returns a DM.
        self.cx_type = MX  # Assume MX for now, if needed, optimal control program will set this properly

    def add(
        self,
        parameter_name: str | Parameter,
        function: Callable = None,
        size: int = None,
        list_index: int = -1,
        scaling: np.ndarray = np.array([1.0]),
        allow_reserved_name: bool = False,
        **extra_arguments: Any,
    ):
        """
        Add a new Parameter to the list

        Parameters
        ----------
        parameter_name: str | Parameter
            If str, the name of the parameter. This name will be used for plotting purpose. It must be unique
            If Parameter, the parameter is copied
        function: Callable[OptimalControlProgram, MX]
            The user defined function that modify the model
        size: int
            The number of variables this parameter has
        list_index: int
            The index of the parameter in the parameters list
        scaling: float
            The scaling of the parameter
        allow_reserved_name: bool
            Overrides the restriction to reserved key. This is for internal purposes and should not be used by users
            as it will result in undefined behavior
        extra_arguments: dict
            Any argument that should be passed to the user defined functions
        """

        if not allow_reserved_name and parameter_name == "time":
            raise KeyError("It is not possible to declare a parameter with the key 'time' as it is a reserved name")

        if isinstance(parameter_name, Parameter):
            # case it is not a parameter name but trying to copy another parameter
            self.copy(parameter_name)
            if parameter_name.name != "time":
                self[parameter_name.name].declare_symbolic(self.cx_type)
        else:
            if "phase" in extra_arguments:
                raise ValueError(
                    "Parameters are declared for all phases at once. You must therefore not use "
                    "'phase' but 'list_index' instead"
                )
            super(ParameterList, self)._add(
                option_type=Parameter,
                list_index=list_index,
                function=function,
                name=parameter_name,
                size=size,
                scaling=scaling,
                cx=self.cx_type,
                **extra_arguments,
            )

    def __contains__(self, item: str) -> bool:
        """
        Allow for `str in ParameterList`

        Parameters
        ----------
        item: str
            The element to search

        Returns
        -------
        If the element is the list
        """

        for p in self:
            if p.name == item:
                return True
        return False

    def print(self):
        # TODO: Print all elements in the console
        raise NotImplementedError("Printing of ParameterList is not ready yet")

    @property
    def names(self) -> list:
        """
        Get all the name of the Parameter in the List

        Returns
        -------
        A list of all names
        """

        n = []
        for p in self:
            n.append(p.name)
        return n

    def index(self, item: str) -> int:
        """
        Get the index of a specific Parameter in the list

        Parameters
        ----------
        item: str
            The name of the parameter to find

        Returns
        -------
        The index of the Parameter in the list
        """

        return self.names.index(item)

    def keys(self):
        return [p.name for p in self.options[0]]

    @property
    def scaling(self):
        """
        The scaling of all parameters
        """

        return np.vstack([p.scaling for p in self]) if len(self) else np.array([[1.0]])

    @property
    def cx(self):
        return self.cx_start

    @property
    def cx_start(self):
        """
        The CX vector of all parameters
        """
        out = vertcat(*[p.cx for p in self])
        if isinstance(out, DM):
            # Force the type if it is a DM (happens if self is empty)
            out = self.cx_type(out)
        return out

    @property
    def mx(self):
        """
        The MX vector of all parameters
        """
        out = vertcat(*[p.mx for p in self])
        if isinstance(out, DM):
            # Force the type if it is a DM (happens if self is empty)
            out = MX(out)
        return out

    @property
    def shape(self):
        """ """
        return sum([p.shape for p in self])


class Parameters:
    """
    Emulation of the base class PenaltyFunctionAbstract so Parameters can be used as Objective and Constraints

    Methods
    -------
    get_type()
        Returns the type of the penalty
    penalty_nature() -> str
        Get the nature of the penalty
    """

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return Parameters

    @staticmethod
    def penalty_nature() -> str:
        """
        Get the nature of the penalty

        Returns
        -------
        The nature of the penalty
        """

        return "parameter_objectives"
