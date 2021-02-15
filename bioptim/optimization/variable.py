import numpy as np
from casadi import vertcat

from .parameters import ParameterList, Parameter
from ..limits.path_conditions import Bounds, InitialGuess
from ..misc.enums import ControlType, InterpolationType


class OptimizationVariable:

    def __init__(self, ocp):
        self.ocp = ocp

        self.parameters_in_list = ParameterList()
        self.params = Parameter(cx=ocp.CX(), bounds=Bounds(interpolation=InterpolationType.CONSTANT), initial_guess=InitialGuess())

        self.x = []
        self.x_bounds = []
        self.x_init = []
        self.n_all_x = 0
        self.n_phase_x = []

        self.u = []
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
        Format the p, x and u so they are in one nice vector

        Returns
        -------
        The vector of all variables
        """

        return vertcat(*self.x, *self.u, self.params.cx)

    @property
    def bounds(self):
        """
        Format the p, x and u bounds so they are in one nice vector

        Returns
        -------
        The vector of all bounds
        """

        v_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        for x_bound in self.x_bounds:
            v_bounds.concatenate(x_bound)
        for u_bound in self.u_bounds:
            v_bounds.concatenate(u_bound)
        v_bounds.concatenate(self.params.bounds)
        return v_bounds

    @property
    def init(self):
        """
        Format the p, x and u init so they are in one nice vector

        Returns
        -------
        The vector of all init
        """

        v_init = InitialGuess(interpolation=InterpolationType.CONSTANT)
        for x_init in self.x_init:
            v_init.concatenate(x_init)
        for u_init in self.u_init:
            v_init.concatenate(u_init)
        v_init.concatenate(self.params.initial_guess)
        return v_init

    def to_dictionaries(self, data, phase_idx):
        ocp = self.ocp
        v_array = np.array(data).squeeze()

        if phase_idx is None:
            phase_idx = range(len(ocp.nlp))
        elif isinstance(phase_idx, int):
            phase_idx = [phase_idx]
        data_states = []
        data_controls = []
        for _ in range(len(phase_idx)):
            data_states.append({})
            data_controls.append({})
        data_parameters = {}

        offset = 0
        p_idx = 0
        for p in range(self.ocp.n_phases):
            if p in phase_idx:
                x_array = v_array[offset:offset + self.n_phase_x[p]].reshape((ocp.nlp[p].nx, -1), order='F')
                data_states[p_idx]["all"] = x_array
                offset_var = 0
                for var in ocp.nlp[p].var_states:
                    data_states[p_idx][var] = x_array[offset_var : offset_var + ocp.nlp[p].var_states[var], :]
                    offset_var += ocp.nlp[p].var_states[var]
                p_idx += 1
            offset += self.n_phase_x[p]

        offset = self.n_all_x
        p_idx = 0
        for p in range(self.ocp.n_phases):
            if p in phase_idx:
                u_array = v_array[offset:offset + self.n_phase_u[p]].reshape((ocp.nlp[p].nu, -1), order='F')
                data_controls[p_idx]["all"] = u_array
                offset_var = 0
                for var in ocp.nlp[p].var_controls:
                    data_controls[p_idx][var] = u_array[offset_var: offset_var + ocp.nlp[p].var_controls[var], :]
                    offset_var += ocp.nlp[p].var_controls[var]
                p_idx += 1
            offset += self.n_phase_u[p]

        offset = self.n_all_x + self.n_all_u
        data_parameters["all"] = v_array[offset:]
        if len(data_parameters["all"].shape) == 1:
            data_parameters["all"] = data_parameters["all"][:, np.newaxis]
        for param in self.parameters_in_list:
            data_parameters[param.name] = v_array[offset:offset + param.size]
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
                x.append(nlp.CX.sym("X_" + str(nlp.phase_idx) + "_" + str(k), nlp.nx))

                if nlp.control_type != ControlType.CONSTANT or (nlp.control_type == ControlType.CONSTANT and k != nlp.ns):
                    u.append(nlp.CX.sym("U_" + str(nlp.phase_idx) + "_" + str(k), nlp.nu, 1))

            nlp.X = x
            self.x[nlp.phase_idx] = vertcat(*x)
            self.n_phase_x[nlp.phase_idx] = self.x[nlp.phase_idx].size()[0]

            nlp.U = u
            self.u[nlp.phase_idx] = vertcat(*u)
            self.n_phase_u[nlp.phase_idx] = self.u[nlp.phase_idx].size()[0]

        self.n_all_x = sum(self.n_phase_x)
        self.n_all_u = sum(self.n_phase_u)

    def define_ocp_bounds(self):
        """
        Declare and parse the bounds for all the variables (V vector)
        """

        ocp = self.ocp

        # Sanity check
        for i in range(ocp.n_phases):
            ocp.nlp[i].x_bounds.check_and_adjust_dimensions(ocp.nlp[i].nx, ocp.nlp[i].ns)
            if ocp.nlp[i].control_type == ControlType.CONSTANT:
                ocp.nlp[i].u_bounds.check_and_adjust_dimensions(ocp.nlp[i].nu, ocp.nlp[i].ns - 1)
            elif ocp.nlp[i].control_type == ControlType.LINEAR_CONTINUOUS:
                ocp.nlp[i].u_bounds.check_and_adjust_dimensions(ocp.nlp[i].nu, ocp.nlp[i].ns)
            else:
                raise NotImplementedError(f"Plotting {ocp.nlp[i].control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            nx = nlp.nx * (nlp.ns + 1)
            x_bounds = Bounds([0] * nx, [0] * nx, interpolation=InterpolationType.CONSTANT)
            for k in range(nlp.ns + 1):
                x_bounds.min[k*nlp.nx : (k+1)*nlp.nx, 0] = nlp.x_bounds.min.evaluate_at(shooting_point=k)
                x_bounds.max[k*nlp.nx : (k+1)*nlp.nx, 0] = nlp.x_bounds.max.evaluate_at(shooting_point=k)

            # For controls
            if nlp.control_type == ControlType.CONSTANT:
                ns = nlp.ns
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                ns = (nlp.ns + 1)
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
            nu = nlp.nu * ns
            u_bounds = Bounds([0] * nu, [0] * nu, interpolation=InterpolationType.CONSTANT)
            for k in range(ns):
                u_bounds.min[k*nlp.nu : (k+1)*nlp.nu, 0] = nlp.u_bounds.min.evaluate_at(shooting_point=k)
                u_bounds.max[k*nlp.nu : (k+1)*nlp.nu, 0] = nlp.u_bounds.max.evaluate_at(shooting_point=k)

            self.x_bounds[i_phase] = x_bounds
            self.u_bounds[i_phase] = u_bounds

    def define_ocp_initial_guess(self):
        """
        Declare and parse the initial guesses for all the variables (V vector)
        """
        ocp = self.ocp

        # Sanity check
        for i in range(ocp.n_phases):
            ocp.nlp[i].x_init.check_and_adjust_dimensions(ocp.nlp[i].nx, ocp.nlp[i].ns)
            if ocp.nlp[i].control_type == ControlType.CONSTANT:
                ocp.nlp[i].u_init.check_and_adjust_dimensions(ocp.nlp[i].nu, ocp.nlp[i].ns - 1)
            elif ocp.nlp[i].control_type == ControlType.LINEAR_CONTINUOUS:
                ocp.nlp[i].u_init.check_and_adjust_dimensions(ocp.nlp[i].nu, ocp.nlp[i].ns)
            else:
                raise NotImplementedError(f"Plotting {ocp.nlp[i].control_type} is not implemented yet")

        # Declare phases dimensions
        for i_phase, nlp in enumerate(ocp.nlp):
            # For states
            nx = nlp.nx * (nlp.ns + 1)
            x_init = InitialGuess([0] * nx, interpolation=InterpolationType.CONSTANT)
            for k in range(nlp.ns + 1):
                x_init.init[k * nlp.nx: (k + 1) * nlp.nx, 0] = nlp.x_init.init.evaluate_at(shooting_point=k)

            # For controls
            if nlp.control_type == ControlType.CONSTANT:
                ns = nlp.ns
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                ns = (nlp.ns + 1)
            else:
                raise NotImplementedError(f"Multiple shooting problem not implemented yet for {nlp.control_type}")
            nu = nlp.nu * ns
            u_init = InitialGuess([0] * nu, interpolation=InterpolationType.CONSTANT)
            for k in range(ns):
                u_init.init[k*nlp.nu : (k+1)*nlp.nu, 0] = nlp.u_init.init.evaluate_at(shooting_point=k)

            self.x_init[i_phase] = x_init
            self.u_init[i_phase] = u_init

    def add_parameter(self, param: Parameter):
        ocp = self.ocp
        param.cx = param.cx if param.cx is not None else ocp.CX.sym(param.name, param.size, 1)

        if param.name in self.parameters_in_list:
            # Sanity check, you can only add a parameter with the same name if they do the same thing
            i = self.parameters_in_list.index(param.name)

            if param.function != self.parameters_in_list[i].function:
                raise RuntimeError("Pre dynamic function of same parameters must be the same")
            self.parameters_in_list[i].size += param.size
            if param.params != self.parameters_in_list[i].params:
                raise RuntimeError("Extra parameters of same parameters must be the same")
            self.parameters_in_list[i].bounds.concatenate(param.bounds)
            self.parameters_in_list[i].initial_guess.concatenate(param.initial_guess)
        else:
            self.parameters_in_list.add(param)

        # Reinitialize param vector (Allows to regroup Parameter of same name together)
        self.params = Parameter(cx=ocp.CX(), bounds=Bounds(interpolation=InterpolationType.CONSTANT), initial_guess=InitialGuess())
        for p in self.parameters_in_list:
            self.params.cx = vertcat(self.params.cx, p.cx)
            self.params.size = p.size

            p.bounds.check_and_adjust_dimensions(p.size, 1)
            self.params.bounds.concatenate(p.bounds)

            p.initial_guess.check_and_adjust_dimensions(p.size, 1)
            self.params.initial_guess.concatenate(p.initial_guess)
