import numpy as np
from casadi import vertcat, DM, MX, SX

from .parameters import ParameterList, Parameter
from ..limits.path_conditions import Bounds, InitialGuess, InitialGuessList, NoisedInitialGuess
from ..misc.enums import ControlType, InterpolationType
from ..dynamics.ode_solver import OdeSolver


class OptimizationVector:
    """
    Attributes
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    parameters_in_list: ParameterList
        A list of all the parameters in the ocp
    x: MX, SX
        The optimization variable for the states
    x_bounds: list
        A list of state bounds for each phase
    x_init: list
        A list of states initial guesses for each phase
    n_all_x: int
        The number of states of all the phases
    n_phase_x: list
        The number of states per phases
    u: MX, SX
        The optimization variable for the controls
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
            self.x_bounds.append(Bounds(interpolation=InterpolationType.CONSTANT))
            self.x_init.append(InitialGuess(interpolation=InterpolationType.CONSTANT))
            self.n_phase_x.append(0)

            self.u_scaled.append([])
            self.u_bounds.append(Bounds(interpolation=InterpolationType.CONSTANT))
            self.u_init.append(InitialGuess(interpolation=InterpolationType.CONSTANT))
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
    def bounds(self):
        """
        Format the x, u and p bounds so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all bounds
        """

        if isinstance(self.ocp.nlp[0].ode_solver, OdeSolver.COLLOCATION) and not isinstance(
            self.ocp.nlp[0].ode_solver, OdeSolver.IRK
        ):
            n_steps = self.ocp.nlp[0].ode_solver.steps + 1
        else:
            n_steps = 1

        v_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for phase, x_bound in enumerate(self.x_bounds):
            v_bounds.concatenate(
                x_bound.scale(self.ocp.nlp[phase].x_scaling["all"].to_vector(self.ocp.nlp[phase].ns * n_steps + 1))
            )

        for phase, u_bound in enumerate(self.u_bounds):
            if self.ocp.nlp[0].control_type == ControlType.LINEAR_CONTINUOUS:
                ns = self.ocp.nlp[phase].ns + 1
            else:
                ns = self.ocp.nlp[phase].ns
            v_bounds.concatenate(u_bound.scale(self.ocp.nlp[phase].u_scaling["all"].to_vector(ns)))

        for param in self.parameters_in_list:
            v_bounds.concatenate(param.bounds.scale(param.scaling))

        return v_bounds

    @property
    def init(self):
        """
        Format the x, u and p init so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all init
        """
        v_init = InitialGuess(interpolation=InterpolationType.CONSTANT)
        if isinstance(self.ocp.nlp[0].ode_solver, OdeSolver.COLLOCATION) and not isinstance(
            self.ocp.nlp[0].ode_solver, OdeSolver.IRK
        ):
            steps = self.ocp.nlp[0].ode_solver.steps + 1
        else:
            steps = 1

        for phase, x_init in enumerate(self.x_init):
            nlp = self.ocp.nlp[phase]

            if isinstance(self.ocp.original_values["x_init"], InitialGuessList):
                original_x_init = self.ocp.original_values["x_init"][phase]
            else:
                original_x_init = self.ocp.original_values["x_init"]
            interpolation_type = None if original_x_init is None else original_x_init.type

            if nlp.ode_solver.is_direct_collocation and interpolation_type == InterpolationType.EACH_FRAME:
                v_init.concatenate(
                    self._init_linear_interpolation(phase=phase).scale(
                        self.ocp.nlp[phase].x_scaling["all"].to_vector(self.ocp.nlp[phase].ns * steps + 1),
                    )
                )
            else:
                v_init.concatenate(
                    x_init.scale(self.ocp.nlp[phase].x_scaling["all"].to_vector(self.ocp.nlp[phase].ns * steps + 1))
                )

        for phase, u_init in enumerate(self.u_init):
            if self.ocp.nlp[0].control_type == ControlType.LINEAR_CONTINUOUS:
                ns = self.ocp.nlp[phase].ns + 1
            else:
                ns = self.ocp.nlp[phase].ns
            v_init.concatenate(u_init.scale(self.ocp.nlp[phase].u_scaling["all"].to_vector(ns)))

        for param in self.parameters_in_list:
            v_init.concatenate(param.initial_guess.scale(param.scaling))

        return v_init

    def _init_linear_interpolation(self, phase: int) -> InitialGuess:
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
        x_init_vector = np.zeros(
            (nlp.states.scaled[0].shape, self.n_phase_x[phase] // nlp.states.scaled[0].shape)
        )  # TODO: [0] to [node_index]
        init_values = (
            self.ocp.original_values["x_init"][phase].init
            if isinstance(self.ocp.original_values["x_init"], InitialGuessList)
            else self.ocp.original_values["x_init"].init
        )
        # the linear interpolation is performed at the given time steps from the ode solver
        steps = np.array(
            nlp.ode_solver.integrator(self.ocp, nlp, node_index=None)[0].step_time
        )  # TODO: Change node_index

        for idx_state, state in enumerate(init_values):
            for frame in range(nlp.ns):
                x_init_vector[idx_state, frame * n_points : (frame + 1) * n_points] = (
                    state[frame] + (state[frame + 1] - state[frame]) * steps
                )

            x_init_vector[idx_state, -1] = state[nlp.ns]

        x_init_reshaped = x_init_vector.reshape((1, -1), order="F").T
        return InitialGuess(x_init_reshaped)

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
            if self.ocp.nlp[p].use_states_from_phase_idx == self.ocp.nlp[p].phase_idx:
                x_array = v_array[offset : offset + self.n_phase_x[p]].reshape(
                    (ocp.nlp[p].states.scaled[0].shape, -1), order="F"  # TODO: [0] to [node_index]
                )
                data_states[p_idx]["all"] = x_array
                offset_var = 0
                for var in ocp.nlp[p].states.scaled[0]:  # TODO: [0] to [node_index]
                    data_states[p_idx][var] = x_array[
                        offset_var : offset_var + len(ocp.nlp[p].states.scaled[0][var]),
                        :,  # TODO: [0] to [node_index]
                    ]
                    offset_var += len(ocp.nlp[p].states.scaled[0][var])
                p_idx += 1
                offset += self.n_phase_x[p]

        offset = self.n_all_x
        p_idx = 0

        if self.ocp.nlp[0].control_type in (ControlType.CONSTANT, ControlType.LINEAR_CONTINUOUS):
            for p in range(self.ocp.n_phases):
                if self.ocp.nlp[p].use_controls_from_phase_idx == self.ocp.nlp[p].phase_idx:
                    u_array = v_array[offset : offset + self.n_phase_u[p]].reshape(
                        (ocp.nlp[p].controls.scaled[0].shape, -1), order="F"  # TODO: [0] to [node_index]
                    )
                    data_controls[p_idx]["all"] = u_array
                    offset_var = 0
                    for var in ocp.nlp[p].controls.scaled[0]:  # TODO: [0] to [node_index]
                        data_controls[p_idx][var] = u_array[
                            offset_var : offset_var + len(ocp.nlp[p].controls.scaled[0][var]),
                            :,  # TODO: [0] to [node_index]
                        ]
                        offset_var += len(ocp.nlp[p].controls.scaled[0][var])  # TODO: [0] to [node_index]
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
                                nlp.states.scaled[0].shape,  # TODO: [0] to [node_index]
                                nlp.ode_solver.polynomial_degree + 1,
                            )
                        )
                    else:
                        x_scaled[nlp.phase_idx].append(
                            nlp.cx.sym(
                                "X_scaled_" + str(nlp.phase_idx) + "_" + str(k), nlp.states.scaled[0].shape, 1
                            )  # TODO: [0] to [node_index]
                        )
                    x[nlp.phase_idx].append(x_scaled[nlp.phase_idx][k] * nlp.x_scaling["all"].scaling)
                else:
                    x_scaled[nlp.phase_idx] = x_scaled[nlp.use_states_from_phase_idx]
                    x[nlp.phase_idx] = x[nlp.use_states_from_phase_idx]

                if nlp.phase_idx == nlp.use_controls_from_phase_idx:
                    if nlp.control_type != ControlType.CONSTANT or (
                        nlp.control_type == ControlType.CONSTANT and k != nlp.ns
                    ):
                        u_scaled[nlp.phase_idx].append(
                            nlp.cx.sym(
                                "U_scaled_" + str(nlp.phase_idx) + "_" + str(k), nlp.controls.scaled[0].shape, 1
                            )  # TODO: [0] to [node_index]
                        )
                        u[nlp.phase_idx].append(u_scaled[nlp.phase_idx][0] * nlp.u_scaling["all"].scaling)
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
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                nlp.x_bounds.check_and_adjust_dimensions(nlp.states[0].shape, nlp.ns)  # TODO: [0] to [node_index]
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                    nlp.u_bounds.check_and_adjust_dimensions(
                        nlp.controls[0].shape, nlp.ns - 1
                    )  # TODO: [0] to [node_index]
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    nlp.u_bounds.check_and_adjust_dimensions(nlp.controls[0].shape, nlp.ns)
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                nx = nlp.states[0].shape  # TODO: [0] to [node_index]
                if nlp.ode_solver.is_direct_collocation:
                    all_nx = nx * nlp.ns * (nlp.ode_solver.polynomial_degree + 1) + nx
                    outer_offset = nx * (nlp.ode_solver.polynomial_degree + 1)
                    repeat = nlp.ode_solver.polynomial_degree + 1
                else:
                    all_nx = nx * (nlp.ns + 1)
                    outer_offset = nx
                    repeat = 1
                x_bounds = Bounds([0] * all_nx, [0] * all_nx, interpolation=InterpolationType.CONSTANT)
                for k in range(nlp.ns + 1):
                    for p in range(repeat if k != nlp.ns else 1):
                        span = slice(k * outer_offset + p * nx, k * outer_offset + (p + 1) * nx)
                        point = k if k != 0 else 0 if p == 0 else 1
                        x_bounds.min[span, 0] = nlp.x_bounds.min.evaluate_at(shooting_point=point)
                        x_bounds.max[span, 0] = nlp.x_bounds.max.evaluate_at(shooting_point=point)

                self.x_bounds[i_phase] = x_bounds

            # For controls
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                    ns = nlp.ns
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    ns = nlp.ns + 1
                else:
                    raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
                nu = nlp.controls[0].shape  # TODO: [0] to [node_index]
                all_nu = nu * ns
                u_bounds = Bounds([0] * all_nu, [0] * all_nu, interpolation=InterpolationType.CONSTANT)
                for k in range(ns):
                    u_bounds.min[k * nu : (k + 1) * nu, 0] = nlp.u_bounds.min.evaluate_at(shooting_point=k)
                    u_bounds.max[k * nu : (k + 1) * nu, 0] = nlp.u_bounds.max.evaluate_at(shooting_point=k)

                self.u_bounds[i_phase] = u_bounds

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
            interpolation = nlp.x_init.type
            ns = self.get_ns(phase=nlp.phase_idx, interpolation_type=interpolation)
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                if nlp.ode_solver.is_direct_shooting:
                    if nlp.x_init.type == InterpolationType.ALL_POINTS:
                        raise ValueError("InterpolationType.ALL_POINTS must only be used with direct collocation")
                nlp.x_init.check_and_adjust_dimensions(nlp.states[0].shape, ns)  # TODO: [0] to [node_index]

            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                    nlp.u_init.check_and_adjust_dimensions(
                        nlp.controls[0].shape, nlp.ns - 1
                    )  # TODO: [0] to [node_index]
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    nlp.u_init.check_and_adjust_dimensions(nlp.controls[0].shape, nlp.ns)
                else:
                    raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            if nlp.use_states_from_phase_idx == nlp.phase_idx:
                nx = nlp.states[0].shape  # TODO: [0] to [node_index]
                if nlp.ode_solver.is_direct_collocation and nlp.x_init.type != InterpolationType.EACH_FRAME:
                    all_nx = nx * nlp.ns * (nlp.ode_solver.polynomial_degree + 1) + nx
                    outer_offset = nx * (nlp.ode_solver.polynomial_degree + 1)
                    repeat = nlp.ode_solver.polynomial_degree + 1
                else:
                    all_nx = nx * (nlp.ns + 1)
                    outer_offset = nx
                    repeat = 1

                x_init = InitialGuess([0] * all_nx, interpolation=InterpolationType.CONSTANT)
                for k in range(nlp.ns + 1):
                    for p in range(repeat if k != nlp.ns else 1):
                        span = slice(k * outer_offset + p * nx, k * outer_offset + (p + 1) * nx)
                        point = k if k != 0 else 0 if p == 0 else 1
                        if isinstance(nlp.x_init, NoisedInitialGuess):
                            if nlp.x_init.type == InterpolationType.ALL_POINTS:
                                point = k * repeat + p
                        elif isinstance(nlp.x_init, InitialGuess) and nlp.x_init.type == InterpolationType.EACH_FRAME:
                            point = k * repeat + p
                        x_init.init[span, 0] = nlp.x_init.init.evaluate_at(shooting_point=point)
                self.x_init[i_phase] = x_init

            # For controls
            if nlp.use_controls_from_phase_idx == nlp.phase_idx:
                if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                    ns = nlp.ns
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    ns = nlp.ns + 1
                else:
                    raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
                nu = nlp.controls[0].shape  # TODO: [0] to [node_index]
                all_nu = nu * ns
                u_init = InitialGuess([0] * all_nu, interpolation=InterpolationType.CONSTANT)
                for k in range(ns):
                    u_init.init[k * nu : (k + 1) * nu, 0] = nlp.u_init.init.evaluate_at(shooting_point=k)

                self.u_init[i_phase] = u_init

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
