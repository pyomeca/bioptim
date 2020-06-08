from casadi import MX, vertcat

class Parameters:
    @staticmethod
    def add_or_replace(ocp, parameter, penalty_idx):
        param_name = parameter["name"]
        pre_dynamic_function = parameter["function"]
        target_function = None
        if "target_function" in parameter:
            target_function = parameter["target_function"]
            del parameter["target_function"]
        penalty_type = parameter["type"]._get_type()
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
        del parameter["name"], parameter["function"], parameter["type"], parameter["bounds"], parameter["initial_guess"], parameter["size"]

        mx = Parameters.add_to_V(ocp, param_name, nb_elements, pre_dynamic_function, bounds, initial_guess, **parameter)
        if target_function:
            val = target_function(ocp, mx, **parameter)
            penalty_idx = penalty_type._reset_penalty(ocp, None, penalty_idx)
            penalty_type._add_to_penalty(ocp, None, val, penalty_idx, weight=weight, quadratic=quadratic)

    @staticmethod
    def add_to_V(ocp, param_name, nb_elements, pre_dynamic_function, bounds, initial_guess, mx_sym=None, **extra_params):
        if mx_sym is None:
            mx_sym = MX.sym(param_name, nb_elements, 1)

        ocp.V = vertcat(ocp.V, mx_sym)
        param_to_store = {"mx": mx_sym, "func": pre_dynamic_function, "size": nb_elements, "extra_params": extra_params}
        if param_name in ocp.param_to_optimize:
            ocp.param_to_optimize[param_name].append(param_to_store)
        else:
            ocp.param_to_optimize[param_name] = param_to_store

        bounds.check_and_adjust_dimensions(nb_elements, 1)
        ocp.V_bounds.concatenate(bounds)

        initial_guess.check_and_adjust_dimensions(nb_elements, 1)
        ocp.V_init.concatenate(initial_guess)

        return mx_sym
