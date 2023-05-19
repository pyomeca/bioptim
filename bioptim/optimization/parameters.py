from typing import Callable, Any

from casadi import MX, SX, DM, vertcat, Function
import numpy as np

from ..limits.objective_functions import ObjectiveFcn, Objective, ObjectiveList
from ..limits.path_conditions import InitialGuess, InitialGuessList, Bounds, BoundsList
from ..limits.penalty_controller import PenaltyController
from ..limits.penalty import PenaltyOption
from ..misc.enums import InterpolationType, Node
from ..misc.options import UniquePerProblemOptionList
from ..optimization.non_linear_program import NonLinearProgram


class Parameter(PenaltyOption):
    """
    A placeholder for a parameter

    Attributes
    ----------
    function: Callable[OptimalControlProgram, MX]
            The user defined function that modify the model
    initial_guess: InitialGuess | InitialGuessList
        The list of initial guesses associated with this parameter
    bounds: Bounds | BoundsList
        The list of bounds associated with this parameter
    quadratic: bool
        If the objective is squared [True] or not [False]
    size: int
        The number of variables this parameter has
    penalty_list: Objective | ObjectiveList
        The list of objective associated with this parameter
    cx: MX | SX
        The type of casadi variable
    mx: MX
        The MX vector of the parameter
    """

    def __init__(
        self,
        function: Callable = None,
        initial_guess: InitialGuess | InitialGuessList = None,
        bounds: Bounds | BoundsList = None,
        quadratic: bool = True,
        size: int = None,
        penalty_list: Objective | ObjectiveList = None,
        cx: Callable | MX | SX = None,
        scaling: np.ndarray = None,
        **params: Any,
    ):
        """
        Parameters
        ----------
        function: Callable[OptimalControlProgram, MX]
            The user defined function that modify the model
        initial_guess: InitialGuess | InitialGuessList
            The list of initial guesses associated with this parameter
        bounds: Bounds | BoundsList
            The list of bounds associated with this parameter
        quadratic: bool
            If the objective is squared [True] or not [False]
        size: int
            The number of variables this parameter has
        penalty_list: Objective | ObjectiveList
            The list of objective associated with this parameter
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

        self.initial_guess = initial_guess
        self.bounds = bounds
        self.quadratic = quadratic
        self.size = size
        if isinstance(penalty_list, Objective):
            penalty_list_tp = ObjectiveList()
            penalty_list_tp.add(penalty_list)
            penalty_list = penalty_list_tp
        self.penalty_list = penalty_list
        self.cx = cx
        self.mx = None

    @property
    def shape(self):
        return self.cx.shape[0]

    def add_or_replace_to_penalty_pool(self, ocp, _):
        """
        Doing some configuration on the parameter and add it to the list of parameter_to_optimize

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        _: Any
            The placeholder for what is supposed to be nlp
        """

        old_parameter_cx = ocp.v.parameters_in_list.cx_start
        ocp.v.add_parameter(self)

        # Express the previously defined parameters with the new param set
        state_cx = ocp.cx()
        controls_cx = ocp.cx()
        parameter_cx = ocp.v.parameters_in_list.cx_start
        for p in ocp.v.parameters_in_list:
            if p.penalty_list is None:
                continue
            for p_list in p.penalty_list[0]:
                if not p_list.weighted_function:
                    continue

                dt_cx = ocp.cx.sym("dt", 1, 1)
                weight_cx = ocp.cx.sym("weight", 1, 1)
                target_cx = ocp.cx.sym("target", p_list.weighted_function[0].numel_out(), 1)

                p_list.function[0] = Function(
                    p_list.function[0].name(),
                    [state_cx, controls_cx, parameter_cx],
                    [p_list.function[0](state_cx, controls_cx, old_parameter_cx)],
                )
                p_list.weighted_function[0] = Function(
                    p_list.function[0].name(),
                    [state_cx, controls_cx, parameter_cx, weight_cx, target_cx, dt_cx],
                    [p_list.weighted_function[0](state_cx, controls_cx, old_parameter_cx, weight_cx, target_cx, dt_cx)],
                )

        if self.penalty_list:
            if ocp.phase_transitions:
                raise NotImplementedError("Updating parameters while having phase_transition is not supported yet")

            if isinstance(self.penalty_list, Objective):
                penalty_list_tp = ObjectiveList()
                penalty_list_tp.add(self.penalty_list)
                self.penalty_list = penalty_list_tp
            elif not isinstance(self.penalty_list, ObjectiveList):
                raise RuntimeError("penalty_list should be built from an Objective or ObjectiveList")

            if len(self.penalty_list) > 1 or len(self.penalty_list[0]) > 1:
                raise NotImplementedError("Parameters with more that one penalty is not implemented yet")

            for penalty in self.penalty_list[0]:
                # Sanity check
                if not isinstance(penalty.type, ObjectiveFcn.Parameter):
                    raise RuntimeError("Parameters should be declared custom_type=ObjectiveFcn.Parameters")
                if penalty.node != Node.DEFAULT:
                    raise RuntimeError("Parameters are timeless optimization, node=Node.DEFAULT should be declared")

                func = penalty.custom_function

                controller = PenaltyController(ocp, None, [], [], [], [], [], [])
                val = func(ocp, self.cx * self.scaling, **penalty.params)
                self.set_penalty(ocp, penalty, val, target_ns=1)
                penalty.ensure_penalty_sanity(ocp, None)
                penalty._add_penalty_to_pool(controller)

    def set_penalty(
        self,
        ocp,
        objective: Objective,
        penalty: MX | SX,
        combine_to: str = None,
        target_ns: int = -1,
        expand: bool = False,
    ):
        objective.rows = self._set_dim_idx(self.rows, penalty.rows())
        objective.cols = self._set_dim_idx(self.cols, penalty.columns())
        objective.node_idx = [0]
        objective.dt = 1
        if objective.target is not None:
            objective._check_target_dimensions(None, target_ns)
            if combine_to is not None:
                objective.add_target_to_plot(None, combine_to)
        self._set_penalty_function(ocp, objective, penalty, expand)

    def _set_penalty_function(self, ocp, objective, fcn: MX | SX, expand: bool = False):
        # Do not use nlp.add_casadi_func because all functions must be registered
        state_cx = ocp.cx(0, 0)
        control_cx = ocp.cx(0, 0)
        param_cx = ocp.v.parameters_in_list.cx_start

        objective.function.append(
            NonLinearProgram.to_casadi_func(
                f"{self.name}", fcn[objective.rows, objective.cols], state_cx, control_cx, param_cx, expand=expand
            )
        )

        modified_fcn = objective.function[0](state_cx, control_cx, param_cx)

        dt_cx = ocp.cx.sym("dt", 1, 1)
        weight_cx = ocp.cx.sym("weight", fcn.shape[0], 1)
        target_cx = ocp.cx.sym("target", modified_fcn.shape)

        modified_fcn = modified_fcn - target_cx
        modified_fcn = modified_fcn**2 if objective.quadratic else modified_fcn

        objective.weighted_function.append(
            Function(  # Do not use nlp.add_casadi_func because all of them must be registered
                f"{self.name}",
                [state_cx, control_cx, param_cx, weight_cx, target_cx, dt_cx],
                [weight_cx * modified_fcn * dt_cx],
            )
        )

        if expand:
            objective.function[0].expand()
            objective.weighted_function[0].expand()


class ParameterList(UniquePerProblemOptionList):
    """
    A list of Parameter

    Methods
    -------
    add(
        self,
        parameter_name: str | Parameter,
        function: Callable = None,
        initial_guess: InitialGuess | InitialGuessList = None,
        bounds: Bounds | BoundsList = None,
        size: int = None,
        phase: int = 0,
        penalty_list: Objective | ObjectiveList = None,
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
    bounds(self)
        The bounds of all parameters
    initial_guess(self)
        The initial guess of all parameters
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
        initial_guess: InitialGuess | InitialGuessList = None,
        bounds: Bounds | BoundsList = None,
        size: int = None,
        list_index: int = -1,
        penalty_list: Objective | ObjectiveList = None,
        scaling: np.ndarray = np.array([1.0]),
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
        initial_guess: InitialGuess | InitialGuessList
            The list of initial guesses associated with this parameter
        bounds: Bounds | BoundsList
            The list of bounds associated with this parameter
        size: int
            The number of variables this parameter has
        list_index: int
            The index of the parameter in the parameters list
        penalty_list: Objective | ObjectiveList
            The objective function associate with the parameter
        scaling: float
            The scaling of the parameter
        extra_arguments: dict
            Any argument that should be passed to the user defined functions
        """

        if isinstance(parameter_name, Parameter):
            self.copy(parameter_name)
        else:
            if not function or not initial_guess or not bounds or not size:
                raise RuntimeError(
                    "function, initial_guess, bounds and size are mandatory elements to declare a parameter"
                )
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
                initial_guess=initial_guess,
                bounds=bounds,
                size=size,
                penalty_list=penalty_list,
                scaling=scaling,
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
    def bounds(self):
        """
        The bounds of all parameters
        """

        _bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for p in self:
            _bounds.concatenate(p.bounds)
        return _bounds

    @property
    def initial_guess(self):
        """
        The initial guess of all parameters
        """

        _init = InitialGuessList(interpolation=InterpolationType.CONSTANT)
        for p in self:
            _init[p.name] = p.initial_guess
        return _init

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

        return "parameters"
