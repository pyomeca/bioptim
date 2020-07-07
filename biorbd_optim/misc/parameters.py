from casadi import vertcat


class Parameters:
    @staticmethod
    def add_or_replace(ocp, parameter, penalty_idx):
        param_name = parameter["name"]
        pre_dynamic_function = parameter["function"]
        target_function = None
        if "target_function" in parameter:
            target_function = parameter["target_function"]
            del parameter["target_function"]
        if "type" in parameter:
            penalty_type = parameter["type"]._get_type()
            del parameter["type"]
        weight = 1
        bounds = parameter["bounds"]
        initial_guess = parameter["initial_guess"]
        nb_elements = parameter["size"]
        if "weight" in parameter:
            weight = parameter["weight"]
            del parameter["weight"]
        quadratic = False
        if "quadratic" in parameter:
            quadratic = parameter["quadratic"]
            del parameter["quadratic"]
        del (
            parameter["name"],
            parameter["function"],
            parameter["bounds"],
            parameter["initial_guess"],
            parameter["size"],
        )

        cx = Parameters.add_to_V(ocp, param_name, nb_elements, pre_dynamic_function, bounds, initial_guess, **parameter)
        if target_function:
            val = target_function(ocp, cx, **parameter)
            penalty_idx = penalty_type._reset_penalty(ocp, None, penalty_idx)
            penalty_type._add_to_penalty(ocp, None, val, penalty_idx, weight=weight, quadratic=quadratic)

    @staticmethod
    def add_to_V(ocp, param_name, nb_elements, pre_dynamic_function, bounds, initial_guess, cx=None, **extra_params):
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
