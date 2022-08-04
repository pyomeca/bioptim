from typing import Union

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
    extract_phase_time(self, data: Union[np.array, DM]) -> list
        Get the phase time. If time is optimized, the MX/SX values are replaced by their actual optimized time
    to_dictionaries(self, data: Union[np.array, DM]) -> tuple
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

        self.x: Union[MX, SX, list] = []
        self.x_bounds = []
        self.x_init = []
        self.n_all_x = 0
        self.n_phase_x = []

        self.u: Union[MX, SX, list] = []
        self.u_bounds = []
        self.u_init = []
        self.n_all_u = 0
        self.n_phase_u = []

        for _ in range(self.ocp.n_phases):
            self.x.append([])
            self.x_bounds.append(Bounds(interpolation=InterpolationType.CONSTANT))
            self.x_init.append(InitialGuess(interpolation=InterpolationType.CONSTANT))
            self.n_phase_x.append(0)

            self.u.append([])
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

        x = [x.reshape((-1, 1)) for x in self.x]
        return vertcat(*x, *self.u, self.parameters_in_list.cx)

    @property
    def bounds(self):
        """
        Format the x, u and p bounds so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all bounds
        """

        v_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for x_bound in self.x_bounds:
            v_bounds.concatenate(x_bound)
        for u_bound in self.u_bounds:
            v_bounds.concatenate(u_bound)
        v_bounds.concatenate(self.parameters_in_list.bounds)
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
        nlp = self.ocp.nlp[0]
        for phase, x_init in enumerate(self.x_init):
            if type(self.ocp.original_values["x_init"]) == InitialGuess:
                interpolation_type = self.ocp.original_values["x_init"].type
            elif type(self.ocp.original_values["x_init"]) == InitialGuessList:
                interpolation_type = self.ocp.original_values["x_init"][phase].type
            else:
                interpolation_type = None
            if nlp.ode_solver.is_direct_collocation and interpolation_type == InterpolationType.EACH_FRAME:
                n_points = nlp.ode_solver.polynomial_degree + 1
                x_init_vector = np.zeros(self.n_all_x)
                init_values = self.ocp.original_values["x_init"].init
                for index, state in enumerate(init_values):
                    for frame in range(nlp.ns):
                        point = (index * nlp.ns + frame) * n_points
                        x_init_vector[point : point + n_points] = np.linspace(state[frame], state[frame + 1], n_points)
                v_init.concatenate(InitialGuess(x_init_vector))
            else:
                v_init.concatenate(x_init)
        for u_init in self.u_init:
            v_init.concatenate(u_init)
        v_init.concatenate(self.parameters_in_list.initial_guess)
        return v_init

    def extract_phase_time(self, data: Union[np.array, DM]) -> list:
        """
        Get the phase time. If time is optimized, the MX/SX values are replaced by their actual optimized time

        Parameters
        ----------
        data: Union[np.array, DM]
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
                    phase_time[i] = data_time_optimized[cmp]
                    cmp += 1
        return phase_time

    def to_dictionaries(self, data: Union[np.array, DM]) -> tuple:
        """
        Convert a vector of solution in an easy to use dictionary, where are the variables are given their proper names

        Parameters
        ----------
        data: Union[np.array, DM]
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
            x_array = v_array[offset : offset + self.n_phase_x[p]].reshape((ocp.nlp[p].states.shape, -1), order="F")
            data_states[p_idx]["all"] = x_array
            offset_var = 0
            for var in ocp.nlp[p].states:
                data_states[p_idx][var] = x_array[offset_var : offset_var + len(ocp.nlp[p].states[var]), :]
                offset_var += len(ocp.nlp[p].states[var])
            p_idx += 1
            offset += self.n_phase_x[p]

        offset = self.n_all_x
        p_idx = 0
        for p in range(self.ocp.n_phases):
            u_array = v_array[offset : offset + self.n_phase_u[p]].reshape((ocp.nlp[p].controls.shape, -1), order="F")
            data_controls[p_idx]["all"] = u_array
            offset_var = 0
            for var in ocp.nlp[p].controls:
                data_controls[p_idx][var] = u_array[offset_var : offset_var + len(ocp.nlp[p].controls[var]), :]
                offset_var += len(ocp.nlp[p].controls[var])
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

        for nlp in self.ocp.nlp:
            x = []
            u = []
            if nlp.control_type != ControlType.CONSTANT and nlp.control_type != ControlType.LINEAR_CONTINUOUS:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")

            for k in range(nlp.ns + 1):
                if k != nlp.ns and nlp.ode_solver.is_direct_collocation:
                    x.append(
                        nlp.cx.sym(
                            "X_" + str(nlp.phase_idx) + "_" + str(k),
                            nlp.states.shape,
                            nlp.ode_solver.polynomial_degree + 1,
                        )
                    )
                else:
                    x.append(nlp.cx.sym("X_" + str(nlp.phase_idx) + "_" + str(k), nlp.states.shape, 1))

                if nlp.control_type != ControlType.CONSTANT or (
                    nlp.control_type == ControlType.CONSTANT and k != nlp.ns
                ):
                    u.append(nlp.cx.sym("U_" + str(nlp.phase_idx) + "_" + str(k), nlp.controls.shape, 1))

            nlp.X = x
            self.x[nlp.phase_idx] = vertcat(*[x_tp.reshape((-1, 1)) for x_tp in x])
            self.n_phase_x[nlp.phase_idx] = self.x[nlp.phase_idx].size()[0]

            nlp.U = u
            self.u[nlp.phase_idx] = vertcat(*u)
            self.n_phase_u[nlp.phase_idx] = self.u[nlp.phase_idx].size()[0]

        self.n_all_x = sum(self.n_phase_x)
        self.n_all_u = sum(self.n_phase_u)

    def define_ocp_bounds(self):
        """
        Declare and parse the bounds for all the variables (v vector)
        """

        ocp = self.ocp

        # Sanity check
        for nlp in ocp.nlp:
            nlp.x_bounds.check_and_adjust_dimensions(nlp.states.shape, nlp.ns)
            if nlp.control_type == ControlType.CONSTANT:
                nlp.u_bounds.check_and_adjust_dimensions(nlp.controls.shape, nlp.ns - 1)
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                nlp.u_bounds.check_and_adjust_dimensions(nlp.controls.shape, nlp.ns)
            else:
                raise NotImplementedError(f"Plotting {nlp.control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            nx = nlp.states.shape
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

            # For controls
            if nlp.control_type == ControlType.CONSTANT:
                ns = nlp.ns
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                ns = nlp.ns + 1
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
            nu = nlp.controls.shape
            all_nu = nu * ns
            u_bounds = Bounds([0] * all_nu, [0] * all_nu, interpolation=InterpolationType.CONSTANT)
            for k in range(ns):
                u_bounds.min[k * nu : (k + 1) * nu, 0] = nlp.u_bounds.min.evaluate_at(shooting_point=k)
                u_bounds.max[k * nu : (k + 1) * nu, 0] = nlp.u_bounds.max.evaluate_at(shooting_point=k)

            self.x_bounds[i_phase] = x_bounds
            self.u_bounds[i_phase] = u_bounds

    def define_ocp_initial_guess(self):
        """
        Declare and parse the initial guesses for all the variables (v vector)
        """

        ocp = self.ocp

        # Sanity check
        for i in range(ocp.n_phases):
            if ocp.nlp[i].x_init.type == InterpolationType.ALL_POINTS:
                if not ocp.nlp[i].ode_solver.is_direct_collocation:
                    raise ValueError("InterpolationType.ALL_POINTS must only be used with direct collocation")
            if type(self.ocp.original_values["x_init"]) == InitialGuess or \
                    (type(self.ocp.original_values["x_init"]) == NoisedInitialGuess and
                     ocp.nlp[i].x_init.type == InterpolationType.ALL_POINTS):
                interpolation_type = ocp.original_values["x_init"].type
            elif type(self.ocp.original_values["x_init"]) == InitialGuessList:
                interpolation_type = ocp.original_values["x_init"][i].type
            else:
                interpolation_type = None
            ns = (
                ocp.nlp[i].ns * (ocp.nlp[i].ode_solver.steps + 1)
                if (ocp.nlp[i].ode_solver.is_direct_collocation
                    and (interpolation_type != InterpolationType.EACH_FRAME
                    and type(ocp.nlp[i].x_init) != NoisedInitialGuess))
                   or (interpolation_type == InterpolationType.ALL_POINTS
                       and type(ocp.nlp[i].x_init) == NoisedInitialGuess)
                else ocp.nlp[i].ns
            )
            ocp.nlp[i].x_init.check_and_adjust_dimensions(ocp.nlp[i].states.shape, ns)
            if ocp.nlp[i].control_type == ControlType.CONSTANT:
                ocp.nlp[i].u_init.check_and_adjust_dimensions(ocp.nlp[i].controls.shape, ocp.nlp[i].ns - 1)
            elif ocp.nlp[i].control_type == ControlType.LINEAR_CONTINUOUS:
                ocp.nlp[i].u_init.check_and_adjust_dimensions(ocp.nlp[i].controls.shape, ocp.nlp[i].ns)
            else:
                raise NotImplementedError(f"Plotting {ocp.nlp[i].control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            nx = nlp.states.shape
            if nlp.ode_solver.is_direct_collocation and interpolation_type != InterpolationType.EACH_FRAME:
                all_nx = nx * nlp.ns * (nlp.ode_solver.polynomial_degree + 1) + nx
                outer_offset = nx * (nlp.ode_solver.polynomial_degree + 1)
                repeat = nlp.ode_solver.polynomial_degree + 1
            else:
                all_nx = nx * (nlp.ns + 1)
                outer_offset = nx
                repeat = 1

            x_init = InitialGuess([0] * all_nx, interpolation=InterpolationType.CONSTANT)
            for k in range(nlp.ns + 1):  # todo
                for p in range(repeat if k != nlp.ns else 1):
                    span = slice(k * outer_offset + p * nx, k * outer_offset + (p + 1) * nx)
                    if (nlp.x_init.type == InterpolationType.EACH_FRAME and type(nlp.x_init) != NoisedInitialGuess) or \
                            (nlp.x_init.type == InterpolationType.ALL_POINTS and type(nlp.x_init) == NoisedInitialGuess):
                        point = k * repeat + p
                    else:
                        point = k if k != 0 else 0 if p == 0 else 1
                    x_init.init[span, 0] = nlp.x_init.init.evaluate_at(shooting_point=point)

            # For controls
            if nlp.control_type == ControlType.CONSTANT:
                ns = nlp.ns
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                ns = nlp.ns + 1
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
            nu = nlp.controls.shape
            all_nu = nu * ns
            u_init = InitialGuess([0] * all_nu, interpolation=InterpolationType.CONSTANT)
            for k in range(ns):
                u_init.init[k * nu : (k + 1) * nu, 0] = nlp.u_init.init.evaluate_at(shooting_point=k)

            self.x_init[i_phase] = x_init
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
