from typing import Callable, Union, Any

from casadi import vertcat

from .enums import Node
from ..limits.objective_functions import ObjectiveFcn, ObjectiveFunction, Objective, ObjectiveList
from ..limits.path_conditions import InitialGuess, InitialGuessList, Bounds, BoundsList
from .options import OptionList, OptionGeneric


class Parameter(OptionGeneric):
    """
    A placeholder for a parameter
    function: Callable[OptimalControlProgram, MX]
            The user defined function that modify the model
    initial_guess: Union[InitialGuess, InitialGuessList]
        The list of initial guesses associated with this parameter
    bounds: Union[Bounds, BoundsList]
        The list of bounds associated with this parameter
    quadratic: bool
        If the objective is squared [True] or not [False]
    size: int
        The number of variables this parameter has
    penalty_list: Union[Objective, ObjectiveList]
        The list of objective associated with this parameter
    cx: Union[MX, SX]
        The type of casadi variable
    """

    def __init__(
        self,
        function: Callable = None,
        initial_guess: Union[InitialGuess, InitialGuessList] = None,
        bounds: Union[Bounds, BoundsList] = None,
        quadratic: bool = True,
        size: int = None,
        penalty_list: Union[Objective, ObjectiveList] = None,
        cx: Callable = None,
        **params
    ):
        """
        Parameters
        ----------
        function: Callable[OptimalControlProgram, MX]
            The user defined function that modify the model
        initial_guess: Union[InitialGuess, InitialGuessList]
            The list of initial guesses associated with this parameter
        bounds: Union[Bounds, BoundsList]
            The list of bounds associated with this parameter
        quadratic: bool
            If the objective is squared [True] or not [False]
        size: int
            The number of variables this parameter has
        penalty_list: Union[Objective, ObjectiveList]
            The list of objective associated with this parameter
        cx: Union[MX, SX]
            The type of casadi variable
        params: dict
            Any parameters to pass to the function
        """

        super(Parameter, self).__init__(type=Parameters, **params)
        self.function = function
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.quadratic = quadratic
        self.size = size
        self.penalty_list = penalty_list
        self.cx = cx


class ParameterList(OptionList):
    """
    A list of Parameter

    Methods
    -------
    def add(
        self,
        parameter_name: str,
        function: Callable = None,
        initial_guess: Union[InitialGuess, InitialGuessList] = None,
        bounds: Union[Bounds, BoundsList] = None,
        size: int = None,
        phase: int = 0,
        penalty_list: Union[Objective, ObjectiveList] = None,
        **extra_arguments
    ):
        Add a new Parameter to the list
    print(self)
        Print the ParameterList to the console
    """

    def add(
        self,
        parameter_name: str,
        function: Callable = None,
        initial_guess: Union[InitialGuess, InitialGuessList] = None,
        bounds: Union[Bounds, BoundsList] = None,
        size: int = None,
        phase: int = 0,
        penalty_list: Union[Objective, ObjectiveList] = None,
        **extra_arguments: Any
    ):
        """
        Add a new Parameter to the list

        Parameters
        ----------
        parameter_name: str
            The name of the parameter. This name will be used for plotting purpose. It must be unique
        function: Callable[OptimalControlProgram, MX]
            The user defined function that modify the model
        initial_guess: Union[InitialGuess, InitialGuessList]
            The list of initial guesses associated with this parameter
        bounds: Union[Bounds, BoundsList]
            The list of bounds associated with this parameter
        size: int
            The number of variables this parameter has
        phase: int
            The phase model the parameter should be associated with
        penalty_list: Union[Objective, ObjectiveList]
            The objective function associate with the parameter
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

            super(ParameterList, self)._add(
                option_type=Parameter,
                phase=phase,
                function=function,
                name=parameter_name,
                initial_guess=initial_guess,
                bounds=bounds,
                size=size,
                penalty_list=penalty_list,
                **extra_arguments
            )

    def print(self):
        """
        Print the ParameterList to the console
        """
        # TODO: Print all elements in the console
        raise NotImplementedError("Printing of ParameterList is not ready yet")


class Parameters:
    """
    Emulation of the base class PenaltyFunctionAbstract so Parameters can be used as Objective and Constraints

    Methods
    -------
    add_or_replace(ocp, _, parameter: Parameter)
        Doing some configuration on the parameter and add it to the list of parameter_to_optimize
    _add_to_v(
            ocp, name: str, size: int, function: Callable, bounds: Union[Bounds, BoundsList],
            initial_guess: Union[InitialGuess, InitialGuessList], cx: Callable = None, **extra_params) -> Callable
        Add a parameter the vector of all variables (V)
    get_type()
        Returns the type of the penalty
    penalty_nature() -> str
        Get the nature of the penalty
    """

    @staticmethod
    def add_or_replace(ocp, _, parameter: Parameter):
        """
        Doing some configuration on the parameter and add it to the list of parameter_to_optimize

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        _: Any
            The place holder for what is supposed to be nlp
        parameter: PenaltyOption
            The actual parameter to declare
        """

        param_name = parameter.name
        pre_dynamic_function = parameter.function
        initial_guess = parameter.initial_guess
        bounds = parameter.bounds
        n_elements = parameter.size
        penalty_list = parameter.penalty_list

        cx = Parameters._add_to_v(
            ocp, param_name, n_elements, pre_dynamic_function, bounds, initial_guess, **parameter.params
        )

        if penalty_list:
            if ocp.state_transitions:
                raise NotImplementedError("Updating parameters while having state_transition is not supported yet")

            if isinstance(penalty_list, Objective):
                penalty_list_tp = ObjectiveList()
                penalty_list_tp.add(penalty_list)
                penalty_list = penalty_list_tp
            elif not isinstance(penalty_list, ObjectiveList):
                raise RuntimeError("penalty_list should be built from an Objective or ObjectiveList")

            if len(penalty_list) > 1 or len(penalty_list[0]) > 1:
                raise NotImplementedError("Parameters with more that one penalty is not implemented yet")
            penalty = penalty_list[0][0]

            # Sanity check
            if not isinstance(penalty.type, ObjectiveFcn.Parameter):
                raise RuntimeError("Parameters should be declared custom_type=ObjectiveFcn.Parameters")
            if penalty.node != Node.DEFAULT:
                raise RuntimeError("Parameters are timeless optimization, node=Node.DEFAULT should be declared")

            func = penalty.custom_function

            val = func(ocp, cx, **penalty.params)
            penalty.sliced_target = penalty.target
            ObjectiveFunction.ParameterFunction.clear_penalty(ocp, None, penalty)
            ObjectiveFunction.ParameterFunction.add_to_penalty(ocp, None, val, penalty)

    @staticmethod
    def _add_to_v(
        ocp,
        name: str,
        size: int,
        function: Union[Callable, None],
        bounds: Union[Bounds, BoundsList],
        initial_guess: Union[InitialGuess, InitialGuessList],
        cx: Callable = None,
        **extra_params
    ) -> Callable:
        """
        Add a parameter the vector of all variables (V)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        name: str
            The name of the parameter
        size: int
            The number of variables this parameter has
        function: Callable[OptimalControlProgram, MX]
                The user defined function that modify the model
        bounds: Union[Bounds, BoundsList]
            The list of bounds associated with this parameter
        initial_guess: Union[InitialGuess, InitialGuessList]
            The list of initial guesses associated with this parameter
        cx: Union[MX, SX]
            The type of casadi variable
        extra_params: dict
            Any parameters to pass to the function

        Returns
        -------
        The cx type
        """
        if cx is None:
            cx = ocp.CX.sym(name, size, 1)

        ocp.V = vertcat(ocp.V, cx)
        param_to_store = Parameter(
            cx=cx, function=function, size=size, bounds=bounds, initial_guess=initial_guess, **extra_params
        )

        if name in ocp.param_to_optimize:
            p = ocp.param_to_optimize[name]
            p.cx = vertcat(p.cx, param_to_store.cx)
            if p.function != param_to_store.function:
                raise RuntimeError("Pre dynamic function of same parameters must be the same")
            p.size += param_to_store.size
            if p.params != param_to_store.params:
                raise RuntimeError("Extra parameters of same parameters must be the same")
            p.bounds.concatenate(param_to_store.bounds)
            p.initial_guess.concatenate(param_to_store.initial_guess)
        else:
            ocp.param_to_optimize[name] = param_to_store

        bounds.check_and_adjust_dimensions(size, 1)
        ocp.V_bounds.concatenate(bounds)

        initial_guess.check_and_adjust_dimensions(size, 1)
        ocp.V_init.concatenate(initial_guess)

        return cx

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
