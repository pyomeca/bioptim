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
        # states
        x = []
        x_scaled = []
        # controls
        u = []
        u_scaled = []
        # algebraic states
        a = []
        a_scaled = []

        for nlp in ocp.nlp:
            x.append([])
            x_scaled.append([])
            u.append([])
            u_scaled.append([])
            a.append([])
            a_scaled.append([])
            if nlp.control_type not in (
                ControlType.CONSTANT,
                ControlType.CONSTANT_WITH_LAST_NODE,
                ControlType.LINEAR_CONTINUOUS,
            ):
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for k in range(nlp.ns + 1):
                _set_node_index(nlp, k)
                if nlp.phase_idx == nlp.use_states_from_phase_idx:
                    n_col = nlp.n_states_decision_steps(k)
                    x_scaled[nlp.phase_idx].append(
                        nlp.cx.sym(f"X_scaled_{nlp.phase_idx}_{k}", nlp.states.scaled.shape, n_col)
                    )

                    x[nlp.phase_idx].append(
                        x_scaled[nlp.phase_idx][k]
                        * np.repeat(
                            np.concatenate([nlp.x_scaling[key].scaling for key in nlp.states.keys()]), n_col, axis=1
                        )
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

                n_col = nlp.n_algebraic_states_decision_steps(k)
                a_scaled[nlp.phase_idx].append(
                    nlp.cx.sym(f"A_scaled_{nlp.phase_idx}_{k}", nlp.algebraic_states.scaled.shape, n_col)
                )

                if nlp.algebraic_states.keys():
                    a[nlp.phase_idx].append(
                        a_scaled[nlp.phase_idx][k]
                        * np.repeat(
                            np.concatenate([nlp.a_scaling[key].scaling for key in nlp.algebraic_states.keys()]),
                            n_col,
                            axis=1,
                        )
                    )

            _set_node_index(nlp, 0)

            nlp.X_scaled = x_scaled[nlp.phase_idx]
            nlp.X = x[nlp.phase_idx]

            nlp.U_scaled = u_scaled[nlp.phase_idx]
            nlp.U = u[nlp.phase_idx]

            nlp.A_scaled = a_scaled[nlp.phase_idx]
            nlp.A = a[nlp.phase_idx]

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
        a_scaled = []
        p_scaled = ocp.parameters.scaled.cx

        for nlp in ocp.nlp:
            x_scaled += [x.reshape((-1, 1)) for x in nlp.X_scaled]
            u_scaled += nlp.U_scaled
            a_scaled += [a.reshape((-1, 1)) for a in nlp.A_scaled]

        return vertcat(t_scaled, *x_scaled, *u_scaled, p_scaled, *a_scaled)

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
            min_bounds, max_bounds = _dispatch_state_bounds(
                nlp, nlp.states, nlp.x_bounds, nlp.x_scaling, lambda n: nlp.n_states_decision_steps(n)
            )

            v_bounds_min = np.concatenate((v_bounds_min, min_bounds))
            v_bounds_max = np.concatenate((v_bounds_max, max_bounds))

        # For controls
        for i_phase in range(ocp.n_phases):
            current_nlp = ocp.nlp[i_phase]
            nlp = ocp.nlp[current_nlp.use_controls_from_phase_idx]
            _set_node_index(nlp, 0)
            if nlp.control_type in (ControlType.CONSTANT,):
                ns = nlp.ns
            elif nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                ns = nlp.ns + 1
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for key in nlp.controls.keys():
                if key in nlp.u_bounds.keys():
                    nlp.u_bounds[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], ns - 1)

            for k in range(ns):
                _set_node_index(nlp, k)
                collapsed_values_min = np.ndarray((nlp.controls.shape, 1))
                collapsed_values_max = np.ndarray((nlp.controls.shape, 1))
                for key in nlp.controls:
                    if key in nlp.u_bounds.keys():
                        value_min = (
                            nlp.u_bounds[key].min.evaluate_at(shooting_point=k)[:, np.newaxis]
                            / nlp.u_scaling[key].scaling
                        )
                        value_max = (
                            nlp.u_bounds[key].max.evaluate_at(shooting_point=k)[:, np.newaxis]
                            / nlp.u_scaling[key].scaling
                        )
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

        # For algebraic_states variables
        for i_phase in range(ocp.n_phases):
            current_nlp = ocp.nlp[i_phase]

            nlp = ocp.nlp[current_nlp.use_states_from_phase_idx]
            min_bounds, max_bounds = _dispatch_state_bounds(
                nlp,
                nlp.algebraic_states,
                nlp.a_bounds,
                nlp.a_scaling,
                lambda n: nlp.n_algebraic_states_decision_steps(n),
            )

            v_bounds_min = np.concatenate((v_bounds_min, min_bounds))
            v_bounds_max = np.concatenate((v_bounds_max, max_bounds))

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
            init = _dispatch_state_initial_guess(
                nlp, nlp.states, nlp.x_init, nlp.x_scaling, lambda n: nlp.n_states_decision_steps(n)
            )
            v_init = np.concatenate((v_init, init))

        # For controls
        for i_phase in range(len(ocp.nlp)):
            current_nlp = ocp.nlp[i_phase]

            nlp = ocp.nlp[current_nlp.use_controls_from_phase_idx]
            _set_node_index(nlp, 0)
            if nlp.control_type in (ControlType.CONSTANT,):
                ns = nlp.ns - 1
            elif nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                ns = nlp.ns
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for key in nlp.controls.keys():
                if key in nlp.u_init.keys():
                    nlp.u_init[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], ns)

            for k in range(ns + 1):
                _set_node_index(nlp, k)
                collapsed_values = np.ndarray((nlp.controls.shape, 1))
                for key in nlp.controls:
                    if key in nlp.u_init.keys():
                        value = (
                            nlp.u_init[key].init.evaluate_at(shooting_point=k)[:, np.newaxis]
                            / nlp.u_scaling[key].scaling
                        )
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

        # For algebraic_states variables
        for i_phase in range(len(ocp.nlp)):
            current_nlp = ocp.nlp[i_phase]

            nlp = ocp.nlp[current_nlp.use_states_from_phase_idx]
            init = _dispatch_state_initial_guess(
                nlp, nlp.algebraic_states, nlp.a_init, nlp.a_scaling, lambda n: nlp.n_algebraic_states_decision_steps(n)
            )

            v_init = np.concatenate((v_init, init))

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
        out = data[range(len(ocp.time_phase_mapping.to_first.map_idx))]
        if isinstance(out, (DM, SX, MX)):
            return ocp.time_phase_mapping.to_second.map(out.toarray()[:, 0].tolist())
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
            The solution in a vector, if no data is provided, dummy data is used (it can be useful getting the dimensions)

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
        data_algebraic_states = []
        for nlp in ocp.nlp:
            # using state nodes ensures for all ensures the dimensions of controls are coherent with states
            data_states.append({key: [None] * nlp.n_states_nodes for key in nlp.states.keys()})
            data_controls.append({key: [None] * nlp.n_controls_nodes for key in nlp.controls.keys()})
            data_algebraic_states.append({key: [None] * nlp.n_states_nodes for key in nlp.algebraic_states.keys()})
        data_parameters = {key: None for key in ocp.parameters.keys()}

        # For states
        offset = len(ocp.time_phase_mapping.to_first.map_idx)
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
            for node in range(nlp.n_controls_nodes):  # Using n_states_nodes on purpose see higher
                n_cols = nlp.n_controls_steps(node)

                if nu == 0 or node >= nlp.n_controls_nodes:
                    u_array = np.ndarray((0, 1))
                else:
                    u_array = v_array[offset : offset + nu * n_cols].reshape((nu, -1), order="F")
                    offset += nu

                for key in nlp.controls.keys():
                    data_controls[p][key][node] = u_array[nlp.controls.key_index(key), :]

        # For parameters
        for key in ocp.parameters.keys():
            # The list is to simulate the node so it has the same structure as the states and controls
            data_parameters[key] = [v_array[[offset + i for i in ocp.parameters[key].index], np.newaxis]]
        data_parameters = [data_parameters]
        offset += sum([ocp.parameters[key].shape for key in ocp.parameters.keys()])

        # For algebraic_states variables
        for p in range(ocp.n_phases):
            nlp = ocp.nlp[p]
            na = nlp.algebraic_states.shape

            if nlp.use_states_from_phase_idx != nlp.phase_idx:
                data_algebraic_states[p] = data_algebraic_states[nlp.use_states_from_phase_idx]
                continue
            for node in range(nlp.n_states_nodes):
                nlp.algebraic_states.node_index = node
                n_cols = nlp.n_algebraic_states_decision_steps(node)

                if na == 0:
                    a_array = np.ndarray((0, 1))
                else:
                    a_array = v_array[offset : offset + na * n_cols].reshape((na, -1), order="F")

                for key in nlp.algebraic_states.keys():
                    data_algebraic_states[p][key][node] = a_array[nlp.algebraic_states[key].index, :]
                offset += na * n_cols

        return data_states, data_controls, data_parameters, data_algebraic_states


def _set_node_index(nlp, node):
    nlp.states.node_index = node
    nlp.states_dot.node_index = node
    nlp.controls.node_index = node
    nlp.algebraic_states.node_index = node


def _dispatch_state_bounds(nlp, states, states_bounds, states_scaling, n_steps_callback):
    states.node_index = 0
    repeat = n_steps_callback(0)

    for key in states.keys():
        if key in states_bounds.keys():
            if states_bounds[key].type == InterpolationType.ALL_POINTS:
                states_bounds[key].check_and_adjust_dimensions(states[key].cx.shape[0], nlp.ns * repeat)
            else:
                states_bounds[key].check_and_adjust_dimensions(states[key].cx.shape[0], nlp.ns)

    v_bounds_min = np.ndarray((0, 1))
    v_bounds_max = np.ndarray((0, 1))
    for k in range(nlp.n_states_nodes):
        states.node_index = k

        for p in range(repeat if k != nlp.ns else 1):
            collapsed_values_min = np.ndarray((states.shape, 1))
            collapsed_values_max = np.ndarray((states.shape, 1))
            for key in states:
                if key in states_bounds.keys():
                    if states_bounds[key].type == InterpolationType.ALL_POINTS:
                        point = k * n_steps_callback(0) + p
                    else:
                        # This allows CONSTANT_WITH_FIRST_AND_LAST to work in collocations, but is flawed for the other ones
                        # point refers to the column to use in the bounds matrix
                        point = k if k != 0 else 0 if p == 0 else 1

                    value_min = (
                        states_bounds[key].min.evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
                        / states_scaling[key].scaling
                    )
                    value_max = (
                        states_bounds[key].max.evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
                        / states_scaling[key].scaling
                    )
                else:
                    value_min = -np.inf
                    value_max = np.inf
                # Organize the controls according to the correct indices
                collapsed_values_min[states[key].index, :] = value_min
                collapsed_values_max[states[key].index, :] = value_max

            v_bounds_min = np.concatenate((v_bounds_min, np.reshape(collapsed_values_min.T, (-1, 1))))
            v_bounds_max = np.concatenate((v_bounds_max, np.reshape(collapsed_values_max.T, (-1, 1))))

    return v_bounds_min, v_bounds_max


def _dispatch_state_initial_guess(nlp, states, states_init, states_scaling, n_steps_callback):
    states.node_index = 0
    repeat = n_steps_callback(0)

    for key in states.keys():
        if key in states_init.keys():
            if states_init[key].type == InterpolationType.ALL_POINTS:
                states_init[key].check_and_adjust_dimensions(states[key].cx.shape[0], nlp.ns * repeat)
            else:
                states_init[key].check_and_adjust_dimensions(states[key].cx.shape[0], nlp.ns)

    v_init = np.ndarray((0, 1))
    for k in range(nlp.n_states_nodes):
        states.node_index = k

        for p in range(repeat if k != nlp.ns else 1):
            collapsed_values_init = np.ndarray((states.shape, 1))
            for key in states:
                if key in states_init.keys():
                    if states_init[key].type == InterpolationType.ALL_POINTS:
                        point = k * n_steps_callback(0) + p
                    else:
                        # This allows CONSTANT_WITH_FIRST_AND_LAST to work in collocations, but is flawed for the other ones
                        # point refers to the column to use in the bounds matrix
                        point = k if k != 0 else 0 if p == 0 else 1

                    value_init = (
                        states_init[key].init.evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
                        / states_scaling[key].scaling
                    )

                else:
                    value_init = 0

                # Organize the controls according to the correct indices
                collapsed_values_init[states[key].index, :] = value_init

            v_init = np.concatenate((v_init, np.reshape(collapsed_values_init.T, (-1, 1))))

    return v_init
