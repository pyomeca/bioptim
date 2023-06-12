import numpy as np
from casadi import vertcat, DM, MX, SX

from .parameters import ParameterList, Parameter
from ..limits.path_conditions import Bounds, InitialGuess, InitialGuessList, NoisedInitialGuess
from ..misc.enums import ControlType, InterpolationType


class OptimizationVector:
    """
    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    parameters_in_list: ParameterList
        A list of all the parameters in the ocp
    x_bounds: list
        A list of state bounds for each phase
    x_init: list
        A list of states initial guesses for each phase
    n_all_x: int
        The number of states of all the phases
    n_phase_x: list
        The number of states per phases
    u_bounds: list
        A list of control bounds for each phase
    u_init: list
        A list of control initial guesses for each phase
    n_all_u: int
        The number of controls of all the phases
    n_phase_u: list
        The number of controls per phases

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

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """

        self.ocp = ocp

        self.parameters_in_list = ParameterList()
        self.parameters_in_list.cx_type = ocp.cx

        self.x_scaled: MX | SX | list = []
        self.x_bounds = []
        self.x_init = []
        self.n_all_x = 0
        self.n_phase_x = []

        self.u_scaled: MX | SX | list = []
        self.u_bounds = []
        self.u_init = []
        self.n_all_u = 0
        self.n_phase_u = []

        for _ in range(self.ocp.n_phases):
            self.x_scaled.append([])
            self.x_bounds.append(Bounds("x_bounds", interpolation=InterpolationType.CONSTANT))
            self.x_init.append(InitialGuess("x_init", interpolation=InterpolationType.CONSTANT))
            self.n_phase_x.append(0)

            self.u_scaled.append([])
            self.u_bounds.append(Bounds("u_bounds", interpolation=InterpolationType.CONSTANT))
            self.u_init.append(InitialGuess("u_init", interpolation=InterpolationType.CONSTANT))
            self.n_phase_u.append(0)

    @property
    def vector(self):
        """
        Format the x, u and p so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all variables
        """

        x_scaled = []
        u_scaled = []
        for nlp in self.ocp.nlp:
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                x_scaled += [self.x_scaled[nlp.phase_idx].reshape((-1, 1))]
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                u_scaled += [self.u_scaled[nlp.phase_idx]]
        return vertcat(*x_scaled, *u_scaled, self.parameters_in_list.cx_start)

    @property
    def bounds_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Format the x, u and p bounds so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all bounds (min, max)
        """
        v_bounds_min = np.ndarray((0, 1))
        v_bounds_max = np.ndarray((0, 1))

        for phase, x_bound in enumerate(self.x_bounds):
            nlp = self.ocp.nlp[phase]
            repeat = 1
            if nlp.ode_solver.is_direct_collocation:
                repeat += nlp.ode_solver.polynomial_degree

            for k in range(self.ocp.nlp[phase].ns + 1):
                x_slice = slice(repeat * k, repeat * (k + 1), None)
                v_bounds_min = np.concatenate((v_bounds_min, np.reshape(x_bound.min[:, x_slice].T, (-1, 1))))
                v_bounds_max = np.concatenate((v_bounds_max, np.reshape(x_bound.max[:, x_slice].T, (-1, 1))))

        for phase, u_bound in enumerate(self.u_bounds):
            for k in range(self.ocp.nlp[phase].ns + 1):
                if u_bound.min[:, k : k + 1].shape[1] != 0:
                    v_bounds_min = np.concatenate((v_bounds_min, u_bound.min[:, k : k + 1]))
                    v_bounds_max = np.concatenate((v_bounds_max, u_bound.max[:, k : k + 1]))

        for param in self.parameters_in_list:
            # TODO Benjamin harmonize
            bound = param.bounds.scale(param.scaling)
            v_bounds_min = np.concatenate((v_bounds_min, bound.min))
            v_bounds_max = np.concatenate((v_bounds_max, bound.max))

        return v_bounds_min, v_bounds_max

    @property
    def init_vector(self):
        """
        Format the x, u and p init so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all init
        """
        v_init = np.ndarray((0, 1))
        for phase, x_init in enumerate(self.x_init):
            nlp = self.ocp.nlp[phase]
            repeat = 1
            if nlp.ode_solver.is_direct_collocation:
                repeat += nlp.ode_solver.polynomial_degree

            for k in range(self.ocp.nlp[phase].ns + 1):
                x_slice = slice(repeat * k, repeat * (k + 1), None)
                v_init = np.concatenate((v_init, np.reshape(x_init.init[:, x_slice].T, (-1, 1))))

        for phase, u_init in enumerate(self.u_init):
            for k in range(self.ocp.nlp[phase].ns + 1):
                if u_init.init[:, k : k + 1].shape[1] != 0:
                    v_init = np.concatenate((v_init, u_init.init[:, k : k + 1]))

        for param in self.parameters_in_list:
            # TODO Harmonize Benjamin
            v_init = np.concatenate((v_init, param.initial_guess.scale(param.scaling).init))

        return v_init

    def _init_linear_interpolation(self, phase: int) -> InitialGuessList:
        """
        Perform linear interpolation between shooting nodes so that initial guess values are defined for each
        collocation point

        Parameters
        ----------
        phase: int
            The phase index

        Returns
        -------
        The initial guess for the states variables for all collocation points

        """
        nlp = self.ocp.nlp[phase]
        n_points = nlp.ode_solver.polynomial_degree + 1
        x_init_vector = np.zeros((nlp.states.scaled.shape, self.n_phase_x[phase] // nlp.states.scaled.shape))
        init_values = (
            self.ocp.original_values["x_init"][phase].init
            if isinstance(self.ocp.original_values["x_init"], InitialGuessList)
            else self.ocp.original_values["x_init"].init
        )

        for idx_state, state in enumerate(init_values):
            for frame in range(nlp.ns):
                # the linear interpolation is performed at the given time steps from the ode solver
                steps = np.array(
                    nlp.ode_solver.integrator(self.ocp, nlp, node_index=0 if self.ocp.assume_phase_dynamics else frame)[
                        0
                    ].step_time
                )

                x_init_vector[idx_state, frame * n_points : (frame + 1) * n_points] = (
                    state[frame] + (state[frame + 1] - state[frame]) * steps
                )

            x_init_vector[idx_state, -1] = state[nlp.ns]

        x_init_reshaped = x_init_vector.reshape((1, -1), order="F").T
        out = InitialGuessList()
        out.add("x_init_linear", x_init_reshaped)
        return out

    def extract_phase_time(self, data: np.ndarray | DM) -> list:
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

        offset = self.n_all_x + self.n_all_u
        data_time_optimized = []
        if "time" in self.parameters_in_list.names:
            for param in self.parameters_in_list:
                if param.name == "time":
                    data_time_optimized = list(np.array(data[offset : offset + param.size])[:, 0])
                    break
                offset += param.size

        phase_time = [0] + [nlp.tf for nlp in self.ocp.nlp]
        if data_time_optimized:
            cmp = 0
            for i in range(len(phase_time)):
                if isinstance(phase_time[i], self.ocp.cx):
                    phase_time[i] = data_time_optimized[self.ocp.parameter_mappings["time"].to_second.map_idx[cmp]]
                    cmp += 1
        return phase_time

    def to_dictionaries(self, data: np.ndarray | DM) -> tuple:
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

        ocp = self.ocp
        v_array = np.array(data).squeeze()

        data_states = []
        data_controls = []
        for _ in range(self.ocp.n_phases):
            data_states.append({})
            data_controls.append({})
        data_parameters = {}

        offset = 0
        p_idx = 0
        for p in range(self.ocp.n_phases):
            ocp.nlp[p].states.node_index = 0
            if self.ocp.nlp[p].use_states_from_phase_idx == self.ocp.nlp[p].phase_idx:
                x_array = v_array[offset : offset + self.n_phase_x[p]].reshape(
                    (ocp.nlp[p].states.scaled.shape, -1), order="F"
                )
                offset_var = 0
                for var in ocp.nlp[p].states.scaled:
                    data_states[p_idx][var] = x_array[
                        offset_var : offset_var + len(ocp.nlp[p].states.scaled[var]),
                        :,
                    ]
                    offset_var += len(ocp.nlp[p].states.scaled[var])
                p_idx += 1
                offset += self.n_phase_x[p]

        offset = self.n_all_x
        p_idx = 0

        if self.ocp.nlp[0].control_type in (ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS):
            for p in range(self.ocp.n_phases):
                ocp.nlp[p].controls.node_index = 0
                if self.ocp.nlp[p].use_controls_from_phase_idx == self.ocp.nlp[p].phase_idx:
                    u_array = v_array[offset : offset + self.n_phase_u[p]].reshape(
                        (ocp.nlp[p].controls.scaled.shape, -1), order="F"
                    )
                    offset_var = 0
                    for var in ocp.nlp[p].controls.scaled:
                        data_controls[p_idx][var] = u_array[
                            offset_var : offset_var + len(ocp.nlp[p].controls.scaled[var]),
                            :,
                        ]
                        offset_var += len(ocp.nlp[p].controls.scaled[var])
                    p_idx += 1
                    offset += self.n_phase_u[p]

        offset = self.n_all_x + self.n_all_u
        scaling_offset = 0
        data_parameters["all"] = v_array[offset:, np.newaxis] * ocp.nlp[0].parameters.scaling
        if len(data_parameters["all"].shape) == 1:
            data_parameters["all"] = data_parameters["all"][:, np.newaxis]
        for param in self.parameters_in_list:
            data_parameters[param.name] = v_array[offset : offset + param.size, np.newaxis] * param.scaling
            offset += param.size
            scaling_offset += param.size
            if len(data_parameters[param.name].shape) == 1:
                data_parameters[param.name] = data_parameters[param.name][:, np.newaxis]

        return data_states, data_controls, data_parameters

    def define_ocp_shooting_points(self):
        """
        Declare all the casadi variables with the right size to be used during a specific phase
        """
        x = []
        x_scaled = []
        u = []
        u_scaled = []
        for nlp in self.ocp.nlp:
            x.append([])
            x_scaled.append([])
            u.append([])
            u_scaled.append([])
            if nlp.control_type not in (ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS, ControlType.NONE):
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for k in range(nlp.ns + 1):
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
            self.x_scaled[nlp.phase_idx] = vertcat(
                *[x_tp.reshape((-1, 1)) for x_tp in x_scaled[nlp.use_states_from_phase_idx]]
            )
            self.n_phase_x[nlp.phase_idx] = (
                self.x_scaled[nlp.phase_idx].size()[0] if nlp.phase_idx == nlp.use_states_from_phase_idx else 0
            )
            nlp.U_scaled = u_scaled[nlp.phase_idx]
            nlp.U = u[nlp.phase_idx]
            self.u_scaled[nlp.phase_idx] = vertcat(*u_scaled[nlp.use_controls_from_phase_idx])
            self.n_phase_u[nlp.phase_idx] = (
                self.u_scaled[nlp.phase_idx].size()[0] if nlp.phase_idx == nlp.use_controls_from_phase_idx else 0
            )

        self.n_all_x = sum(self.n_phase_x)
        self.n_all_u = sum(self.n_phase_u)

    def define_ocp_bounds(self):
        """
        Declare and parse the bounds for all the variables (v vector)
        """

        ocp = self.ocp

        # Sanity check
        for nlp in ocp.nlp:
            for key in nlp.states:
                if key not in nlp.x_bounds:
                    continue

                if nlp.use_states_from_phase_idx == nlp.phase_idx:
                    nlp.x_bounds[key].check_and_adjust_dimensions(nlp.states[key].cx.shape[0], nlp.ns)
            for key in nlp.controls:
                if key not in nlp.u_bounds:
                    continue

                if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                    if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                        nlp.u_bounds[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns - 1)
                    elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                        nlp.u_bounds[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns)
                    else:
                        raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                repeat = 1
                if nlp.ode_solver.is_direct_collocation:
                    repeat += nlp.ode_solver.polynomial_degree

                collapsed_values_min = np.ndarray((nlp.states.shape, (nlp.ns * repeat) + 1))
                collapsed_values_max = np.ndarray((nlp.states.shape, (nlp.ns * repeat) + 1))
                for k in range(nlp.ns + 1):
                    for p in range(repeat if k != nlp.ns else 1):
                        point = k if k != 0 else 0 if p == 0 else 1
                        x_slice = slice(repeat * k + p, repeat * k + p + 1, None)
                        for key in nlp.states:
                            if key in nlp.x_bounds:
                                value_min = (
                                    nlp.x_bounds[key].min.evaluate_at(shooting_point=point) / nlp.x_scaling[key].scaling
                                )[:, np.newaxis]
                                value_max = (
                                    nlp.x_bounds[key].max.evaluate_at(shooting_point=point) / nlp.x_scaling[key].scaling
                                )[:, np.newaxis]
                            else:
                                value_min = -np.inf
                                value_max = np.inf
                            collapsed_values_min[nlp.states[key].index, x_slice] = value_min
                            collapsed_values_max[nlp.states[key].index, x_slice] = value_max
                self.x_bounds[i_phase] = Bounds(
                    "x_bounds",
                    min_bound=collapsed_values_min,
                    max_bound=collapsed_values_max,
                    interpolation=InterpolationType.EACH_FRAME,
                )

            # For controls
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                    ns = nlp.ns
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    ns = nlp.ns + 1
                else:
                    raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

                collapsed_values_min = np.ndarray((nlp.controls.shape, ns))
                collapsed_values_max = np.ndarray((nlp.controls.shape, ns))
                for k in range(ns):
                    for key in nlp.controls:
                        if key in nlp.u_bounds:
                            value_min = (
                                nlp.u_bounds[key].min.evaluate_at(shooting_point=k) / nlp.u_scaling[key].scaling
                            )
                            value_max = (
                                nlp.u_bounds[key].max.evaluate_at(shooting_point=k) / nlp.u_scaling[key].scaling
                            )
                        else:
                            value_min = -np.inf
                            value_max = np.inf

                        collapsed_values_min[nlp.controls[key].index, k] = value_min
                        collapsed_values_max[nlp.controls[key].index, k] = value_max

                self.u_bounds[i_phase] = Bounds(
                    "u_bounds",
                    min_bound=collapsed_values_min,
                    max_bound=collapsed_values_max,
                    interpolation=InterpolationType.EACH_FRAME,
                )

    def get_ns(self, phase: int, interpolation_type: InterpolationType) -> int:
        """
        Define the number of shooting nodes and collocation points

        Parameters
        ----------
        phase: int
            The index of the current phase of the ocp
        interpolation_type: InterpolationType
            The interpolation type of x_init

        Returns
        -------
        ns: int
            The number of shooting nodes and collocation points
        """
        ocp = self.ocp
        ns = ocp.nlp[phase].ns
        if ocp.nlp[phase].ode_solver.is_direct_collocation:
            if interpolation_type != InterpolationType.EACH_FRAME:
                ns *= ocp.nlp[phase].ode_solver.steps + 1
        return ns

    def define_ocp_initial_guess(self):
        """
        Declare and parse the initial guesses for all the variables (v vector)
        """

        ocp = self.ocp
        # Sanity check
        for nlp in ocp.nlp:
            nlp.states.node_index = 0
            nlp.states_dot.node_index = 0
            nlp.controls.node_index = 0

            interpolation = nlp.x_init.type
            ns = self.get_ns(phase=nlp.phase_idx, interpolation_type=interpolation)
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                if nlp.ode_solver.is_direct_shooting:
                    if nlp.x_init.type == InterpolationType.ALL_POINTS:
                        raise ValueError("InterpolationType.ALL_POINTS must only be used with direct collocation")
                for key in nlp.states:
                    nlp.x_init[key].check_and_adjust_dimensions(nlp.states[key].cx.shape[0], ns)

            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                for key in nlp.controls:
                    if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                        nlp.u_init[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns - 1)
                    elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                        nlp.u_init[key].check_and_adjust_dimensions(nlp.controls[key].cx.shape[0], nlp.ns)
                    else:
                        raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                repeat = 1
                if nlp.ode_solver.is_direct_collocation:
                    repeat += nlp.ode_solver.polynomial_degree

                collapsed_values = np.ndarray((nlp.states.shape, (nlp.ns * repeat) + 1))
                for k in range(nlp.ns + 1):
                    for p in range(repeat if k != nlp.ns else 1):
                        x_slice = slice(repeat * k + p, repeat * k + p + 1, None)
                        point = k if k != 0 else 0 if p == 0 else 1
                        for key in nlp.states:
                            if isinstance(nlp.x_init, NoisedInitialGuess):
                                if nlp.x_init.type == InterpolationType.ALL_POINTS:
                                    point = k * repeat + p
                            elif (
                                isinstance(nlp.x_init, InitialGuess) and nlp.x_init.type == InterpolationType.EACH_FRAME
                            ):
                                point = k * repeat + p
                            collapsed_values[nlp.states[key].index, x_slice] = (
                                nlp.x_init[key].init.evaluate_at(shooting_point=point) / nlp.x_scaling[key].scaling
                            )[:, np.newaxis]
                self.x_init[i_phase] = InitialGuess(
                    "x_init", initial_guess=collapsed_values, interpolation=InterpolationType.EACH_FRAME
                )

            # For controls
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                    ns = nlp.ns
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    ns = nlp.ns + 1
                else:
                    raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

                collapsed_values = np.ndarray((nlp.controls.shape, ns))
                for k in range(ns):
                    for key in nlp.controls:
                        collapsed_values[nlp.controls[key].index, k] = (
                            nlp.u_init[key].init.evaluate_at(shooting_point=k) / nlp.u_scaling[key].scaling
                        )
                self.u_init[i_phase] = InitialGuess(
                    "u_init", initial_guess=collapsed_values, interpolation=InterpolationType.EACH_FRAME
                )

    def add_parameter(self, param: Parameter):
        """
        Add a parameter to the parameters pool

        Parameters
        ----------
        param: Parameter
            The new parameter to add to the pool
        """

        ocp = self.ocp
        param.cx = param.cx if param.cx is not None else ocp.cx.sym(param.name, param.size, 1)
        param.mx = MX.sym(f"{param.name}_MX", param.size, 1)

        if param.name in self.parameters_in_list:
            # Sanity check, you can only add a parameter with the same name if they do the same thing
            i = self.parameters_in_list.index(param.name)

            if param.function != self.parameters_in_list[i].function:
                raise RuntimeError("Pre dynamic function of same parameters must be the same")
            self.parameters_in_list[i].size += param.size
            self.parameters_in_list[i].cx = vertcat(self.parameters_in_list[i].cx, param.cx)
            self.parameters_in_list[i].mx = vertcat(self.parameters_in_list[i].mx, param.mx)
            self.parameters_in_list[i].scaling = vertcat(self.parameters_in_list[i].scaling, param.scaling)
            if param.params != self.parameters_in_list[i].params:
                raise RuntimeError("Extra parameters of same parameters must be the same")
            self.parameters_in_list[i].bounds.concatenate(param.bounds)
            self.parameters_in_list[i].initial_guess.concatenate(param.initial_guess)
        else:
            self.parameters_in_list.add(param)
