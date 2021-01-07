from casadi import vertcat

from .enums import Node
from ..limits.objective_functions import ObjectiveFcn, ObjectiveFunction, Objective, ObjectiveList
from .options_lists import OptionList, OptionGeneric


class Parameter(OptionGeneric):
    def __init__(
        self,
        function=None,
        initial_guess=None,
        bounds=None,
        quadratic=True,
        size=None,
        penalty_list=None,
        cx=None,
        **params
    ):
        super(Parameter, self).__init__(**params)
        self.function = function
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.size = size
        self.penalty_list = penalty_list
        self.quadratic = quadratic
        self.cx = cx


class ParameterList(OptionList):
    def add(
        self,
        parameter_name,
        function=None,
        initial_guess=None,
        bounds=None,
        size=None,
        phase=0,
        penalty_list=None,
        **extra_arguments
    ):
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


class Parameters:
    @staticmethod
    def add_or_replace(ocp, parameter):
        param_name = parameter.name
        pre_dynamic_function = parameter.function
        initial_guess = parameter.initial_guess
        bounds = parameter.bounds
        nb_elements = parameter.size
        penalty_list = parameter.penalty_list

        cx = Parameters._add_to_v(
            ocp, param_name, nb_elements, pre_dynamic_function, bounds, initial_guess, **parameter.params
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
    def _add_to_v(ocp, name, size, function, bounds, initial_guess, cx=None, **extra_params):
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
