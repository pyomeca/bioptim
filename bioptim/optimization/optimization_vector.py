import numpy as np
from casadi import vertcat, DM, SX, MX

from ..misc.enums import ControlType, InterpolationType


class OptimizationVectorHelper:
    """
    Methods
    -------
    vector(self)
        Format the x, u, p and s so they are in one nice (and useful) vector
    bounds(self)
        Format the x, u, p and s bounds so they are in one nice (and useful) vector
    init(self)
        Format the x, u, p and s init so they are in one nice (and useful) vector
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
        s = []
        s_scaled = []
        for nlp in ocp.nlp:
            x.append([])
            x_scaled.append([])
            u.append([])
            u_scaled.append([])
            s.append([])
            s_scaled.append([])
            if nlp.control_type not in (
                ControlType.CONSTANT,
                ControlType.CONSTANT_WITH_LAST_NODE,
                ControlType.LINEAR_CONTINUOUS,
                ControlType.NONE,
            ):
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for k in range(nlp.ns + 1):
                OptimizationVectorHelper._set_node_index(nlp, k)
                if nlp.phase_idx == nlp.use_states_from_phase_idx:
                    
                    n_col = nlp.n_states_decision_steps(k)
                    x_scaled[nlp.phase_idx].append(
                        nlp.cx.sym(f"X_scaled_{nlp.phase_idx}_{k}", nlp.states.scaled.shape, n_col)
                    )
                    
                    x[nlp.phase_idx].append(
                        x_scaled[nlp.phase_idx][k] * np.repeat(np.concatenate([nlp.x_scaling[key].scaling for key in nlp.states.keys()]), n_col, axis=1)
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

                s_scaled[nlp.phase_idx].append(
                    nlp.cx.sym("S_scaled_" + str(nlp.phase_idx) + "_" + str(k), nlp.stochastic_variables.shape, 1)
                )
                if nlp.stochastic_variables.keys():
                    s[nlp.phase_idx].append(
                        s_scaled[nlp.phase_idx][0]
                        * np.concatenate([nlp.s_scaling[key].scaling for key in nlp.stochastic_variables.keys()])
                    )
                else:
                    s[nlp.phase_idx].append(s_scaled[nlp.phase_idx][0])

            OptimizationVectorHelper._set_node_index(nlp, 0)

            nlp.X_scaled = x_scaled[nlp.phase_idx]
            nlp.X = x[nlp.phase_idx]

            nlp.U_scaled = u_scaled[nlp.phase_idx]
            nlp.U = u[nlp.phase_idx]

            nlp.S_scaled = s_scaled[nlp.phase_idx]
            nlp.S = s[nlp.phase_idx]

    @staticmethod
    def vector(ocp):
        """
        Format the x, u, p and s so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all variables
        """

        t_scaled = ocp.dt_parameter.cx
        x_scaled = []
        u_scaled = []
        s_scaled = []

        for nlp in ocp.nlp:
            if nlp.ode_solver.is_direct_collocation:
                x_scaled += [x.reshape((-1, 1)) for x in nlp.X_scaled]
            else:
                x_scaled += nlp.X_scaled
            u_scaled += nlp.U_scaled
            s_scaled += nlp.S_scaled

        return vertcat(t_scaled, *x_scaled, *u_scaled, ocp.parameters.cx, *s_scaled)

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

        # For time
        v_bounds_min = np.concatenate((v_bounds_min, ocp.dt_parameter_bounds.min))
        v_bounds_max = np.concatenate((v_bounds_max, ocp.dt_parameter_bounds.max))

        # For states
        for i_phase in range(ocp.n_phases):
            current_nlp = ocp.nlp[i_phase]

            nlp = ocp.nlp[current_nlp.use_states_from_phase_idx]
            repeat = nlp.n_states_decision_steps(0)
            OptimizationVectorHelper._set_node_index(nlp, 0)
            for key in nlp.states:
                if key in nlp.x_bounds.keys():
                    if nlp.x_bounds[key].type == InterpolationType.ALL_POINTS:
                        nlp.x_bounds[key].check_and_adjust_dimensions(nlp.states[key].cx.shape[0], nlp.ns * repeat)
                    else:
                        nlp.x_bounds[key].check_and_adjust_dimensions(nlp.states[key].cx.shape[0], nlp.ns)

            for k in range(nlp.n_states_nodes):
                OptimizationVectorHelper._set_node_index(nlp, k)
                repeat = nlp.n_states_decision_steps(k)

                for p in range(repeat if k != nlp.ns else 1):
                    # This allows CONSTANT_WITH_FIRST_AND_LAST to work in collocations, but is flawed for the other ones
                    # point refers to the column to use in the bounds matrix
                    point = k if k != 0 else 0 if p == 0 else 1

                    collapsed_values_min = np.ndarray((nlp.states.shape, 1))
                    collapsed_values_max = np.ndarray((nlp.states.shape, 1))
                    for key in nlp.states:
                        if key in nlp.x_bounds.keys():
                            value_min = (
                                nlp.x_bounds[key].min.evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
                                / nlp.x_scaling[key].scaling
                            )
                            value_max = (
                                nlp.x_bounds[key].max.evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
                                / nlp.x_scaling[key].scaling
                            )
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
            elif nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                ns = nlp.ns + 1
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for key in nlp.controls.keys():
                if key in nlp.u_bounds.keys():
                    nlp.u_bounds[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], ns - 1)

            for k in range(ns):
                OptimizationVectorHelper._set_node_index(nlp, k)
                collapsed_values_min = np.ndarray((nlp.controls.shape, 1))
                collapsed_values_max = np.ndarray((nlp.controls.shape, 1))
                for key in nlp.controls:
                    if key in nlp.u_bounds.keys():
                        value_min = nlp.u_bounds[key].min.evaluate_at(shooting_point=k)[:, np.newaxis] / nlp.u_scaling[key].scaling
                        value_max = nlp.u_bounds[key].max.evaluate_at(shooting_point=k)[:, np.newaxis] / nlp.u_scaling[key].scaling
                        value_min = value_min[:, 0]
                        value_max = value_max[:, 0]
                    else:
                        value_min = -np.inf
                        value_max = np.inf

                    # Organize the controls according to the correct indices
                    collapsed_values_min[nlp.controls[key].index, 0] = value_min
                    collapsed_values_max[nlp.controls[key].index, 0] = value_max

                v_bounds_min = np.concatenate((v_bounds_min, np.reshape(collapsed_values_min.T, (-1, 1))))
                v_bounds_max = np.concatenate((v_bounds_max, np.reshape(collapsed_values_max.T, (-1, 1))))

        # For parameters
        collapsed_values_min = np.ones((ocp.parameters.shape, 1)) * -np.inf
        collapsed_values_max = np.ones((ocp.parameters.shape, 1)) * np.inf
        for key in ocp.parameters.keys():
            if key not in ocp.parameter_bounds.keys():
                continue

            scaled_bounds = ocp.parameter_bounds[key].scale(ocp.parameters[key].scaling.scaling)
            collapsed_values_min[ocp.parameters[key].index, :] = scaled_bounds.min
            collapsed_values_max[ocp.parameters[key].index, :] = scaled_bounds.max
        v_bounds_min = np.concatenate((v_bounds_min, np.reshape(collapsed_values_min.T, (-1, 1))))
        v_bounds_max = np.concatenate((v_bounds_max, np.reshape(collapsed_values_max.T, (-1, 1))))

        # For stochastic variables
        for i_phase in range(ocp.n_phases):
            nlp = ocp.nlp[i_phase]
            OptimizationVectorHelper._set_node_index(nlp, 0)
            for key in nlp.stochastic_variables.keys():
                if key in nlp.s_bounds.keys():
                    nlp.s_bounds[key].check_and_adjust_dimensions(nlp.stochastic_variables[key].cx.shape[0], nlp.ns)

            for k in range(nlp.ns + 1):
                OptimizationVectorHelper._set_node_index(nlp, k)
                collapsed_values_min = np.ndarray((nlp.stochastic_variables.shape, 1))
                collapsed_values_max = np.ndarray((nlp.stochastic_variables.shape, 1))
                for key in nlp.stochastic_variables.keys():
                    if key in nlp.s_bounds.keys():
                        value_min = nlp.s_bounds[key].min.evaluate_at(shooting_point=k) / nlp.s_scaling[key].scaling
                        value_max = nlp.s_bounds[key].max.evaluate_at(shooting_point=k) / nlp.s_scaling[key].scaling
                    else:
                        value_min = -np.inf
                        value_max = np.inf

                    # Organize the stochastic variables according to the correct indices
                    collapsed_values_min[nlp.stochastic_variables[key].index, :] = np.reshape(value_min, (-1, 1))
                    collapsed_values_max[nlp.stochastic_variables[key].index, :] = np.reshape(value_max, (-1, 1))

                v_bounds_min = np.concatenate((v_bounds_min, np.reshape(collapsed_values_min.T, (-1, 1))))
                v_bounds_max = np.concatenate((v_bounds_max, np.reshape(collapsed_values_max.T, (-1, 1))))

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

        # For time
        v_init = np.concatenate((v_init, ocp.dt_parameter_initial_guess.init))

        # For states
        for i_phase in range(len(ocp.nlp)):
            current_nlp = ocp.nlp[i_phase]

            nlp = ocp.nlp[current_nlp.use_states_from_phase_idx]
            OptimizationVectorHelper._set_node_index(nlp, 0)
            repeat = nlp.n_states_decision_steps(0)
            for key in nlp.states:
                if key in nlp.x_init.keys():
                    if nlp.x_init[key].type == InterpolationType.ALL_POINTS:
                        nlp.x_init[key].check_and_adjust_dimensions(nlp.states[key].cx.shape[0], nlp.ns * repeat)
                    else:
                        nlp.x_init[key].check_and_adjust_dimensions(nlp.states[key].cx.shape[0], nlp.ns)

            for k in range(nlp.ns + 1):
                OptimizationVectorHelper._set_node_index(nlp, k)
                repeat = nlp.n_states_decision_steps(k)

                for p in range(repeat if k != nlp.ns else 1):
                    point = k if k != 0 else 0 if p == 0 else 1

                    collapsed_values = np.ndarray((nlp.states.shape, 1))
                    for key in nlp.states:
                        if key in nlp.x_init.keys():
                            point_to_eval = point
                            if nlp.x_init[key].type == InterpolationType.ALL_POINTS:
                                point_to_eval = k * repeat + p
                            value = (
                                nlp.x_init[key].init.evaluate_at(shooting_point=point_to_eval, repeat=repeat)[:, np.newaxis]
                                / nlp.x_scaling[key].scaling
                            )
                        else:
                            value = 0
                        # Organize the controls according to the correct indices
                        collapsed_values[nlp.states[key].index, :] = value

                    v_init = np.concatenate((v_init, np.reshape(collapsed_values.T, (-1, 1))))

        # For controls
        for i_phase in range(len(ocp.nlp)):
            current_nlp = ocp.nlp[i_phase]

            nlp = ocp.nlp[current_nlp.use_controls_from_phase_idx]
            OptimizationVectorHelper._set_node_index(nlp, 0)
            if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                ns = nlp.ns
            elif nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                ns = nlp.ns + 1
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for key in nlp.controls.keys():
                if key in nlp.u_init.keys():
                    nlp.u_init[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns - 1)

            for k in range(ns):
                OptimizationVectorHelper._set_node_index(nlp, k)
                collapsed_values = np.ndarray((nlp.controls.shape, 1))
                for key in nlp.controls:
                    if key in nlp.u_init.keys():
                        value = nlp.u_init[key].init.evaluate_at(shooting_point=k)[:, np.newaxis] / nlp.u_scaling[key].scaling
                        value = value[:, 0]
                    else:
                        value = 0

                    # Organize the controls according to the correct indices
                    collapsed_values[nlp.controls[key].index, 0] = value

                v_init = np.concatenate((v_init, np.reshape(collapsed_values.T, (-1, 1))))

        # For parameters
        collapsed_values = np.zeros((ocp.parameters.shape, 1))
        for key in ocp.parameters.keys():
            if key not in ocp.parameter_init.keys():
                v_init = np.concatenate((v_init, np.zeros((ocp.parameters[key].size, 1))))
                continue

            scaled_init = ocp.parameter_init[key].scale(ocp.parameters[key].scaling.scaling)
            collapsed_values[ocp.parameters[key].index, :] = scaled_init.init
        v_init = np.concatenate((v_init, np.reshape(collapsed_values.T, (-1, 1))))

        # For stochastic variables
        for i_phase in range(len(ocp.nlp)):
            nlp = ocp.nlp[i_phase]
            OptimizationVectorHelper._set_node_index(nlp, 0)

            repeat = nlp.n_states_decision_steps(0)
            for key in nlp.stochastic_variables.keys():
                if key in nlp.s_init.keys():
                    if nlp.s_init[key].type == InterpolationType.ALL_POINTS:
                        nlp.s_init[key].check_and_adjust_dimensions(nlp.stochastic_variables[key].cx.shape[0], nlp.ns * repeat)
                    else:
                        nlp.x_init[key].check_and_adjust_dimensions(nlp.stochastic_variables[key].cx.shape[0], nlp.ns)

            for k in range(nlp.ns + 1):
                OptimizationVectorHelper._set_node_index(nlp, k)
                collapsed_values = np.ndarray((nlp.stochastic_variables.shape, 1))
                for key in nlp.stochastic_variables:
                    if key in nlp.s_init.keys():
                        value = nlp.s_init[key].init.evaluate_at(shooting_point=k) / nlp.s_scaling[key].scaling
                    else:
                        value = 0

                    # Organize the stochastic variables according to the correct indices
                    collapsed_values[nlp.stochastic_variables[key].index, 0] = value

                v_init = np.concatenate((v_init, np.reshape(collapsed_values.T, (-1, 1))))

        return v_init

    @staticmethod
    def extract_phase_dt(ocp, data: np.ndarray | DM) -> list:
        """
        Get the dt values

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        data: np.ndarray | DM
            The solution in a vector

        Returns
        -------
        The dt values
        """
        out = data[ocp.dt_parameter.index]
        if isinstance(out, (DM, SX, MX)):
            return out.toarray()[:, 0].tolist()
        return list(out[:, 0])

    @staticmethod
    def extract_step_times(ocp, data: np.ndarray | DM) -> list:
        """
        Get the phase time. If time is optimized, the MX/SX values are replaced by their actual optimized time

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        data: np.ndarray | DM
            The solution in a vector

        Returns
        -------
        The phase time
        """

        phase_dt = OptimizationVectorHelper.extract_phase_dt(ocp, data)

        # Starts at zero
        out = []
        for dt, nlp in zip(phase_dt, ocp.nlp):
            phase_step_times = []
            for node in range(nlp.ns):
                phase_step_times.append(nlp.dynamics[node].step_times_from_dt(vertcat(dt * node, dt)))
            phase_step_times.append(DM(dt * nlp.ns))
            out.append(phase_step_times)
        return out

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
        data_stochastic = []
        for nlp in ocp.nlp:
            # using state nodes ensures for all ensures the dimensions of controls are coherent with states
            data_states.append({key: [None] * nlp.n_states_nodes for key in nlp.states.keys()})
            data_controls.append({key: [None] * nlp.n_states_nodes for key in nlp.controls.keys()})
            data_stochastic.append({key: [None] * nlp.n_states_nodes for key in nlp.stochastic_variables.keys()})
        data_parameters = {key: None for key in ocp.parameters.keys()}

        # For states
        offset = ocp.dt_parameter.size
        for p in range(ocp.n_phases):
            nlp = ocp.nlp[p]
            nx = nlp.states.shape

            if nlp.use_states_from_phase_idx != nlp.phase_idx:
                data_states[p] = data_states[nlp.use_states_from_phase_idx]
                continue
            for node in range(nlp.n_states_nodes):
                nlp.states.node_index = node
                n_cols = nlp.n_states_decision_steps(node)
                x_array = v_array[offset : offset + nx * n_cols].reshape((nx, -1), order="F")
                for key in nlp.states.keys():
                    data_states[p][key][node] = x_array[nlp.states[key].index, :]
                offset += nx * n_cols

        # For controls
        for p in range(ocp.n_phases):
            nlp = ocp.nlp[p]
            nu = nlp.controls.shape

            if nlp.use_controls_from_phase_idx != nlp.phase_idx:
                data_controls[p] = data_controls[nlp.use_controls_from_phase_idx]
                continue
            for node in range(nlp.n_states_nodes):  # Using n_states_nodes on purpose see higher
                n_cols = nlp.n_controls_steps(node)
                
                if n_cols == 0 or node >= nlp.n_controls_nodes:
                    u_array = np.ndarray((nu, n_cols)) * np.nan
                else:
                    u_array = v_array[offset : offset + nu * n_cols].reshape((nu, -1), order="F")
                    offset += nu

                for key in nlp.controls.keys():
                    data_controls[p][key][node] = u_array[nlp.controls.key_index(key), :]

        # For parameters
        for param in ocp.parameters:
            # The list is to simulate the node so it has the same structure as the states and controls
            data_parameters[param.name] = [v_array[[offset + i for i in param.index], np.newaxis]]
        data_parameters = [data_parameters]
        offset += len(ocp.parameters)

        # For stochastic variables
        if nlp.stochastic_variables.shape > 0:
            for p in range(ocp.n_phases):
                # TODO
                raise NotImplementedError("Stochastic variables not implemented yet")

        return data_states, data_controls, data_parameters, data_stochastic

    @staticmethod
    def _set_node_index(nlp, node):
        nlp.states.node_index = node
        nlp.states_dot.node_index = node
        nlp.controls.node_index = node
        nlp.stochastic_variables.node_index = node
