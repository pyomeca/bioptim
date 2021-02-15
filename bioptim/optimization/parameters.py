from typing import Callable, Union, Any

from casadi import MX, SX

from ..misc.enums import Node
from ..limits.objective_functions import ObjectiveFcn, ObjectiveFunction, Objective, ObjectiveList
from ..limits.path_conditions import InitialGuess, InitialGuessList, Bounds, BoundsList
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric


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


class ParameterList(UniquePerPhaseOptionList):
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
        parameter_name: Union[str, Parameter],
        function: Callable = None,
        initial_guess: Union[InitialGuess, InitialGuessList] = None,
        bounds: Union[Bounds, BoundsList] = None,
        size: int = None,
        phase: int = -1,
        penalty_list: Union[Objective, ObjectiveList] = None,
        **extra_arguments: Any
    ):
        """
        Add a new Parameter to the list

        Parameters
        ----------
        parameter_name: Union[str, Parameter
            If str, the name of the parameter. This name will be used for plotting purpose. It must be unique
            If Parameter, the parameter is copied
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

    def __contains__(self, item):
        for p in self:
            if p.name == item:
                return True
        return False

    def print(self):
        """
        Print the ParameterList to the console
        """
        # TODO: Print all elements in the console
        raise NotImplementedError("Printing of ParameterList is not ready yet")

    @property
    def names(self):
        n = []
        for p in self:
            n.append(p.name)
        return n

    def index(self, item):
        return self.names.index(item)


class Parameters:
    """
    Emulation of the base class PenaltyFunctionAbstract so Parameters can be used as Objective and Constraints

    Methods
    -------
    add_or_replace(ocp, _, parameter: Parameter)
        Doing some configuration on the parameter and add it to the list of parameter_to_optimize
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

        ocp.v.add_parameter(parameter)

        penalty_list = parameter.penalty_list
        if penalty_list:
            if ocp.phase_transitions:
                raise NotImplementedError("Updating parameters while having phase_transition is not supported yet")

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

            val = func(ocp, parameter.cx, **penalty.params)
            penalty.sliced_target = penalty.target
            ObjectiveFunction.ParameterFunction.clear_penalty(ocp, None, penalty)
            ObjectiveFunction.ParameterFunction.add_to_penalty(ocp, None, val, penalty)

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
