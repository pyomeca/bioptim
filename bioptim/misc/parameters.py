from casadi import vertcat
import numpy as np
from .enums import Instant
from ..limits.constraints import Bounds
from ..limits.path_conditions import InitialGuess
from ..limits.objective_functions import Objective, ObjectiveFunction, ObjectiveOption, ObjectiveList
from .options_lists import OptionList, OptionGeneric


class ParameterOption(OptionGeneric):
    def __init__(self, function=None, initial_guess=None, bounds=None, size=None, penalty_list=None, **params):
        super(ParameterOption, self).__init__(**params)
        self.function = function
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.size = size
        self.penalty_list = penalty_list


class ParameterList(OptionList):
    def add(
        self,
        parameter_name,
        function=None,
        initial_guess=InitialGuess(),
        bounds=Bounds(),
        size=None,
        phase=0,
        penalty_list=None,
        **extra_arguments
    ):
        if isinstance(parameter_name, ParameterOption):
            self.copy(parameter_name)
        else:
            if not function or not initial_guess or not bounds or not size:
                raise RuntimeError(
                    "function, initial_guess, bounds and size are mandatory elements to declare a parameter"
                )

            super(ParameterList, self)._add(
                option_type=ParameterOption,
                function=function,
                phase=phase,
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

            if isinstance(penalty_list, ObjectiveOption):
                penalty_list_tp = ObjectiveList()
                penalty_list_tp.add(penalty_list)
                penalty_list = penalty_list_tp
            elif not isinstance(penalty_list, ObjectiveList):
                raise RuntimeError("penalty_list should be built from an ObjectiveOption or ObjectiveList")

            if len(penalty_list) > 1 or len(penalty_list[0]) > 1:
                raise NotImplementedError("Parameters with more that one penalty is not implemented yet")
            penalty = penalty_list[0][0]

            # Sanity check
            if not isinstance(penalty.type, Objective.Parameter):
                raise RuntimeError("Parameters should be declared custom_type=Objective.Parameters")
            if penalty.instant != Instant.DEFAULT:
                raise RuntimeError("Parameters are timeless optimization, instant=Instant.DEFAULT should be declared")

            func = penalty.custom_function
            weight = penalty.weight
            quadratic = False if penalty.quadratic is None else penalty.quadratic

            if "target" in penalty.params:
                target = penalty.params["target"]
                del penalty.params["target"]
            else:
                target = None
            val = func(ocp, cx, **penalty.params)
            ObjectiveFunction.ParameterFunction.clear_penalty(ocp, None, penalty)
            ObjectiveFunction.ParameterFunction.add_to_penalty(
                ocp, None, val, penalty, target=target, weight=weight, quadratic=quadratic
            )

    @staticmethod
    def _add_to_v(ocp, name, size, function, bounds, initial_guess, cx=None, **extra_params):
        if cx is None:
            cx = ocp.CX.sym(name, size, 1)

        ocp.V = vertcat(ocp.V, cx)
        param_to_store = {
            "cx": cx,
            "func": function,
            "size": size,
            "extra_params": extra_params,
            "bounds": bounds,
            "initial_guess": initial_guess,
        }
        if name in ocp.param_to_optimize:
            p = ocp.param_to_optimize[name]
            p["cx"] = vertcat(p["cx"], param_to_store["cx"])
            if p["func"] != param_to_store["func"]:
                raise RuntimeError("Pre dynamic function of same parameters must be the same")
            p["size"] += param_to_store["size"]
            if p["extra_params"] != param_to_store["extra_params"]:
                raise RuntimeError("Extra parameters of same parameters must be the same")
            p["bounds"].concatenate(param_to_store["bounds"])
            p["initial_guess"].concatenate(param_to_store["initial_guess"])
        else:
            ocp.param_to_optimize[name] = param_to_store

        bounds.check_and_adjust_dimensions(size, 1)
        ocp.V_bounds.concatenate(bounds)

        initial_guess.check_and_adjust_dimensions(size, 1)
        ocp.V_init.concatenate(initial_guess)

        return cx
