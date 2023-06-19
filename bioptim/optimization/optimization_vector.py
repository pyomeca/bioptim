import numpy as np
from casadi import vertcat, DM

from ..misc.enums import ControlType, InterpolationType


class OptimizationVectorHelper:
    """
    Methods
    -------
    vector(self)
        Format the x, u and p so they are in one nice (and useful) vector
    bounds(self)
        Format the x, u and p bounds so they are in one nice (and useful) vector
    init(self)
        Format the x, u and p init so they are in one nice (and useful) vector
    extract_phase_time(self, data: np.ndarray | DM) -> list
        Get the phase time. If time is optimized, the MX/SX values are replaced by their actual optimized time
    to_dictionaries(self, data: np.ndarray | DM) -> tuple
        Convert a vector of solution in an easy to use dictionary, where are the variables are given their proper names
    define_ocp_shooting_points(self)
        Declare all the casadi variables with the right size to be used during a specific phase
    define_ocp_bounds(self)
        Declare and parse the bounds for all the variables (v vector)
    define_ocp_initial_guess(self)
        Declare and parse the initial guesses for all the variables (v vector)
    add_parameter(self, param: Parameter)
        Add a parameter to the parameters pool
    """

    @staticmethod
    def declare_ocp_shooting_points(ocp):
        """
        Declare all the casadi variables with the right size to be used during a specific phase
        """
        x = []
        x_scaled = []
        u = []
        u_scaled = []
        for nlp in ocp.nlp:
            x.append([])
            x_scaled.append([])
            u.append([])
            u_scaled.append([])
            if nlp.control_type not in (ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS, ControlType.NONE):
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for k in range(nlp.ns + 1):
                OptimizationVectorHelper._set_node_index(nlp, k)
                if nlp.phase_idx == nlp.use_states_from_phase_idx:
                    if k != nlp.ns and nlp.ode_solver.is_direct_collocation:
                        x_scaled[nlp.phase_idx].append(
                            nlp.cx.sym(
                                "X_scaled_" + str(nlp.phase_idx) + "_" + str(k),
                                nlp.states.scaled.shape,
                                nlp.ode_solver.polynomial_degree + 1,
                            )
                        )
                    else:
                        x_scaled[nlp.phase_idx].append(
                            nlp.cx.sym("X_scaled_" + str(nlp.phase_idx) + "_" + str(k), nlp.states.scaled.shape, 1)
                        )
                    x[nlp.phase_idx].append(
                        x_scaled[nlp.phase_idx][k]
                        * np.concatenate([nlp.x_scaling[key].scaling for key in nlp.states.keys()])
                    )
                else:
                    x_scaled[nlp.phase_idx] = x_scaled[nlp.use_states_from_phase_idx]
                    x[nlp.phase_idx] = x[nlp.use_states_from_phase_idx]

                if nlp.phase_idx == nlp.use_controls_from_phase_idx:
                    if nlp.control_type != ControlType.CONSTANT or (
                        nlp.control_type == ControlType.CONSTANT and k != nlp.ns
                    ):
                        u_scaled[nlp.phase_idx].append(
                            nlp.cx.sym("U_scaled_" + str(nlp.phase_idx) + "_" + str(k), nlp.controls.scaled.shape, 1)
                        )
                        if nlp.controls.keys():
                            u[nlp.phase_idx].append(
                                u_scaled[nlp.phase_idx][0]
                                * np.concatenate([nlp.u_scaling[key].scaling for key in nlp.controls.keys()])
                            )
                else:
                    u_scaled[nlp.phase_idx] = u_scaled[nlp.use_controls_from_phase_idx]
                    u[nlp.phase_idx] = u[nlp.use_controls_from_phase_idx]

            nlp.X_scaled = x_scaled[nlp.phase_idx]
            nlp.X = x[nlp.phase_idx]

            nlp.U_scaled = u_scaled[nlp.phase_idx]
            nlp.U = u[nlp.phase_idx]

    @staticmethod
    def vector(ocp):
        """
        Format the x, u and p so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all variables
        """

        x_scaled = []
        u_scaled = []
        for nlp in ocp.nlp:
            if nlp.ode_solver.is_direct_collocation:
                x_scaled += [x.reshape((-1, 1)) for x in nlp.X_scaled]
            else:
                x_scaled += nlp.X_scaled
            u_scaled += nlp.U_scaled

        return vertcat(*x_scaled, *u_scaled, ocp.parameters.cx)

    @staticmethod
    def bounds_vectors(ocp) -> tuple[np.ndarray, np.ndarray]:
        """
        Format the x, u and p bounds so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all bounds (min, max)
        """
        v_bounds_min = np.ndarray((0, 1))
        v_bounds_max = np.ndarray((0, 1))

        # For states
        for i_phase in range(ocp.n_phases):
            current_nlp = ocp.nlp[i_phase]

            repeat = 1
            if current_nlp.ode_solver.is_direct_collocation:
                repeat += current_nlp.ode_solver.polynomial_degree

            nlp = ocp.nlp[current_nlp.use_states_from_phase_idx]
            OptimizationVectorHelper._set_node_index(nlp, 0)
            for key in nlp.states:
                nlp.x_bounds[key].check_and_adjust_dimensions(nlp.states[key].cx.shape[0], nlp.ns)

            for k in range(nlp.ns + 1):
                OptimizationVectorHelper._set_node_index(nlp, k)
                for p in range(repeat if k != nlp.ns else 1):
                    point = k if k != 0 else 0 if p == 0 else 1

                    collapsed_values_min = np.ndarray((nlp.states.shape, 1))
                    collapsed_values_max = np.ndarray((nlp.states.shape, 1))
                    for key in nlp.states:
                        if key in nlp.x_bounds.keys():
                            value_min = (
                                                nlp.x_bounds[key].min.evaluate_at(shooting_point=point) /
                                                nlp.x_scaling[key].scaling
                                        )[:, np.newaxis]
                            value_max = (
                                                nlp.x_bounds[key].max.evaluate_at(shooting_point=point) /
                                                nlp.x_scaling[key].scaling
                                        )[:, np.newaxis]
                        else:
                            value_min = -np.inf
                            value_max = np.inf
                        # Organize the controls according to the correct indices
                        collapsed_values_min[nlp.states[key].index, :] = value_min
                        collapsed_values_max[nlp.states[key].index, :] = value_max

                    v_bounds_min = np.concatenate((v_bounds_min, np.reshape(collapsed_values_min.T, (-1, 1))))
                    v_bounds_max = np.concatenate((v_bounds_max, np.reshape(collapsed_values_max.T, (-1, 1))))

        # For controls
        for i_phase in range(ocp.n_phases):
            current_nlp = ocp.nlp[i_phase]
            nlp = ocp.nlp[current_nlp.use_controls_from_phase_idx]
            OptimizationVectorHelper._set_node_index(nlp, 0)
            if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                ns = nlp.ns
                for key in nlp.controls.keys():
                    nlp.u_bounds[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns)
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                ns = nlp.ns + 1
                for key in nlp.controls.keys():
                    nlp.u_bounds[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns - 1)
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for k in range(ns):
                OptimizationVectorHelper._set_node_index(nlp, k)
                collapsed_values_min = np.ndarray((nlp.controls.shape, 1))
                collapsed_values_max = np.ndarray((nlp.controls.shape, 1))
                for key in nlp.controls:
                    if key in nlp.u_bounds.keys():
                        value_min = (
                            nlp.u_bounds[key].min.evaluate_at(shooting_point=k) / nlp.u_scaling[key].scaling
                        )
                        value_max = (
                            nlp.u_bounds[key].max.evaluate_at(shooting_point=k) / nlp.u_scaling[key].scaling
                        )
                    else:
                        value_min = -np.inf
                        value_max = np.inf

                    # Organize the controls according to the correct indices
                    collapsed_values_min[nlp.controls[key].index, 0] = value_min
                    collapsed_values_max[nlp.controls[key].index, 0] = value_max

                v_bounds_min = np.concatenate((v_bounds_min, np.reshape(collapsed_values_min.T, (-1, 1))))
                v_bounds_max = np.concatenate((v_bounds_max, np.reshape(collapsed_values_max.T, (-1, 1))))

        for key in ocp.parameters.keys():
            if key not in ocp.parameter_bounds.keys():
                v_bounds_min = np.concatenate((v_bounds_min, np.ones((ocp.parameters[key].size, 1)) * -np.inf))
                v_bounds_max = np.concatenate((v_bounds_max, np.ones((ocp.parameters[key].size, 1)) * np.inf))
                continue

            param_bound = ocp.parameter_bounds[key]
            bound = param_bound.scale(ocp.parameters[key].scaling)
            v_bounds_min = np.concatenate((v_bounds_min, bound.min))
            v_bounds_max = np.concatenate((v_bounds_max, bound.max))

        return v_bounds_min, v_bounds_max

    @staticmethod
    def init_vector(ocp):
        """
        Format the x, u and p bounds so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all bounds (min, max)
        """
        v_init = np.ndarray((0, 1))

        for i_phase in range(len(ocp.nlp)):
            current_nlp = ocp.nlp[i_phase]

            repeat = 1
            if current_nlp.ode_solver.is_direct_collocation:
                repeat += current_nlp.ode_solver.polynomial_degree

            # For states
            nlp = ocp.nlp[current_nlp.use_states_from_phase_idx]
            OptimizationVectorHelper._set_node_index(nlp, 0)
            for key in nlp.states:
                n_points = OptimizationVectorHelper._nb_points(nlp, nlp.x_init[key].type)
                nlp.x_init[key].check_and_adjust_dimensions(nlp.states[key].cx.shape[0], n_points)

            for k in range(nlp.ns + 1):
                OptimizationVectorHelper._set_node_index(nlp, k)
                for p in range(repeat if k != nlp.ns else 1):
                    point = k if k != 0 else 0 if p == 0 else 1

                    collapsed_values = np.ndarray((nlp.states.shape, 1))
                    for key in nlp.states:
                        if key in nlp.x_init.keys():
                            point_to_eval = point
                            if nlp.x_init[key].type == InterpolationType.ALL_POINTS:
                                point_to_eval = k * repeat + p
                            value = (
                                    nlp.x_init[key].init.evaluate_at(shooting_point=point_to_eval) /
                                    nlp.x_scaling[key].scaling
                            )[:, np.newaxis]
                        else:
                            value = 0
                        # Organize the controls according to the correct indices
                        collapsed_values[nlp.states[key].index, :] = value

                    v_init = np.concatenate((v_init, np.reshape(collapsed_values.T, (-1, 1))))

            # For controls
            nlp = ocp.nlp[current_nlp.use_controls_from_phase_idx]
            OptimizationVectorHelper._set_node_index(nlp, 0)
            if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                ns = nlp.ns
                for key in nlp.controls.keys():
                    nlp.u_init[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns - 1)
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                ns = nlp.ns + 1
                for key in nlp.controls.keys():
                    nlp.u_init[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns)
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for k in range(ns):
                OptimizationVectorHelper._set_node_index(nlp, k)
                collapsed_values = np.ndarray((nlp.controls.shape, 1))
                for key in nlp.controls:
                    if key in nlp.u_init.keys():
                        value = (
                            nlp.u_init[key].init.evaluate_at(shooting_point=k) / nlp.u_scaling[key].scaling
                        )
                    else:
                        value = 0

                    # Organize the controls according to the correct indices
                    collapsed_values[nlp.controls[key].index, 0] = value

                v_init = np.concatenate((v_init, np.reshape(collapsed_values.T, (-1, 1))))

        for key in ocp.parameters.keys():
            if key not in ocp.parameter_init.keys():
                v_init = np.concatenate((v_init, np.zeros((ocp.parameters[key].size, 1))))
                continue

            param_init = ocp.parameter_init[key]
            init = param_init.scale(ocp.parameters[key].scaling)
            v_init = np.concatenate((v_init, init.init))

        return v_init

    @staticmethod
    def extract_phase_time(ocp, data: np.ndarray | DM) -> list:
        """
        Get the phase time. If time is optimized, the MX/SX values are replaced by their actual optimized time

        Parameters
        ----------
        data: np.ndarray | DM
            The solution in a vector

        Returns
        -------
        The phase time
        """

        n_optimized_time = ocp.parameters["time"].shape if "time" in ocp.parameters.names else 0
        offset = data.shape[0] - n_optimized_time  # parameters are at the end of the vector
        data_time_optimized = []
        if "time" in ocp.parameters.names:
            for param in ocp.parameters:
                if param.name == "time":
                    data_time_optimized = list(np.array(data[offset : offset + param.shape])[:, 0])
                    break
                offset += param.shape

        # Starts at zero
        phase_time = [0] + [nlp.tf for nlp in ocp.nlp]
        if data_time_optimized:
            cmp = 0
            for i in range(len(phase_time)):
                if not isinstance(phase_time[i], (int, float)):
                    phase_time[i] = data_time_optimized[cmp]
                    cmp += 1
        return phase_time

    @staticmethod
    def to_dictionaries(ocp, data: np.ndarray | DM) -> tuple:
        """
        Convert a vector of solution in an easy to use dictionary, where are the variables are given their proper names

        Parameters
        ----------
        data: np.ndarray | DM
            The solution in a vector

        Returns
        -------
        The solution in a tuple of dictionaries format (tuple => each phase)
        """

        v_array = np.array(data).squeeze()

        data_states = []
        data_controls = []
        for p in range(ocp.n_phases):
            nlp = ocp.nlp[p]
            n_points = nlp.ns * (1 if nlp.ode_solver.is_direct_shooting else (nlp.ode_solver.polynomial_degree + 1)) + 1
            data_states.append({key: np.ndarray((nlp.states[key].shape, n_points)) for key in nlp.states})
            data_controls.append({key: np.ndarray((nlp.controls[key].shape, nlp.ns + (1 if nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ) else 0))) for key in ocp.nlp[p].controls})
        data_parameters = {key: np.ndarray((0, 1)) for key in ocp.parameters.keys()}

        offset = 0
        p_idx = 0
        for p in range(ocp.n_phases):
            nlp = ocp.nlp[p]
            nx = nlp.states.shape

            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                repeat = (nlp.ode_solver.polynomial_degree + 1) if nlp.ode_solver.is_direct_collocation else 1
                for k in range((nlp.ns * repeat) + 1):
                    nlp.states.node_index = k // repeat
                    x_array = v_array[offset : offset + nx].reshape((nlp.states.scaled.shape, -1), order="F")
                    for key in nlp.states:
                        data_states[p_idx][key][:, k:k+1] = x_array[nlp.states[key].index, :]
                    offset += nx
                p_idx += 1

        p_idx = 0
        for p in range(ocp.n_phases):
            nlp = ocp.nlp[p]
            nu = nlp.controls.shape

            if nlp.control_type in (ControlType.NONE,):
                continue

            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                for k in range(nlp.ns):
                    nlp.controls.node_index = k
                    u_array = v_array[offset: offset + nu].reshape((nlp.controls.scaled.shape, -1), order="F")
                    for key in nlp.controls:
                        data_controls[p_idx][key][:, k:k + 1] = u_array[nlp.controls[key].index, :]
                    offset += nu
                p_idx += 1

        for param in ocp.parameters:
            data_parameters[param.name] = v_array[offset : offset + param.shape, np.newaxis] * param.scaling
            offset += param.shape

        return data_states, data_controls, data_parameters

    @staticmethod
    def _nb_points(nlp, interpolation_type):
        n_points = nlp.ns

        if nlp.ode_solver.is_direct_shooting:
            if interpolation_type == InterpolationType.ALL_POINTS:
                raise ValueError("InterpolationType.ALL_POINTS must only be used with direct collocation")

        if nlp.ode_solver.is_direct_collocation:
            if interpolation_type != InterpolationType.EACH_FRAME:
                n_points *= nlp.ode_solver.steps + 1
        return n_points

    @staticmethod
    def _set_node_index(nlp, node):
        nlp.states.node_index = node
        nlp.states_dot.node_index = node
        nlp.controls.node_index = node
