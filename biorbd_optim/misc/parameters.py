from casadi import vertcat

from .enums import Instant
from ..limits.objective_functions import Objective, ObjectiveFunction


class Parameters:
    @staticmethod
    def add_or_replace(ocp, parameter, penalty_idx):
        param_name = parameter["name"]
        pre_dynamic_function = parameter["function"]
        initial_guess = parameter["initial_guess"]
        bounds = parameter["bounds"]
        nb_elements = parameter["size"]
        del (
            parameter["name"],
            parameter["function"],
            parameter["bounds"],
            parameter["initial_guess"],
            parameter["size"],
        )

        penalty_set = None
        if "penalty_set" in parameter:
            penalty_set = parameter["penalty_set"]
            del parameter["penalty_set"]

        cx = Parameters._add_to_v(ocp, param_name, nb_elements, pre_dynamic_function, bounds, initial_guess, **parameter)
        if penalty_set:
            if len(penalty_set) > 1 or len(penalty_set[0]) > 1:
                raise NotImplementedError("Parameters with more that one penalty is not implemented yet")
            penalty = penalty_set[0][0]

            func = penalty["custom_function"]
            del penalty["custom_function"]

            if not isinstance(penalty["type"], Objective.Parameter):
                raise RuntimeError("Parameters should be declared custom_type=Objective.Parameters")
            del penalty["type"]

            if penalty["instant"] != Instant.DEFAULT:
                raise RuntimeError("Parameters are timeless optimization, instant=Instant.DEFAULT should be declared")
            del penalty["instant"]

            weight = penalty["weight"]
            del penalty["weight"]

            quadratic = False
            if penalty["quadratic"] is not None:
                quadratic = penalty["quadratic"]
            del penalty["quadratic"]

            val = func(ocp, cx, **penalty)
            penalty_idx = ObjectiveFunction.ParameterFunction._reset_penalty(ocp, None, penalty_idx)
            ObjectiveFunction.ParameterFunction._add_to_penalty(ocp, None, val, penalty_idx, weight=weight, quadratic=quadratic)

    @staticmethod
    def _add_to_v(ocp, param_name, nb_elements, pre_dynamic_function, bounds, initial_guess, cx=None, **extra_params):
        if cx is None:
            cx = ocp.CX.sym(param_name, nb_elements, 1)

        ocp.V = vertcat(ocp.V, cx)
        param_to_store = {
            "cx": cx,
            "func": pre_dynamic_function,
            "size": nb_elements,
            "extra_params": extra_params,
        }
        if param_name in ocp.param_to_optimize:
            p = ocp.param_to_optimize[param_name]
            p["cx"] = vertcat(p["cx"], param_to_store["cx"])
            if p["func"] != param_to_store["func"]:
                raise RuntimeError("Pre dynamic function of same parameters must be the same")
            p["size"] += param_to_store["size"]
            if p["extra_params"] != param_to_store["extra_params"]:
                raise RuntimeError("Extra parameters of same parameters must be the same")
        else:
            ocp.param_to_optimize[param_name] = param_to_store

        bounds.check_and_adjust_dimensions(nb_elements, 1)
        ocp.V_bounds.concatenate(bounds)

        initial_guess.check_and_adjust_dimensions(nb_elements, 1)
        ocp.V_init.concatenate(initial_guess)

        return cx
