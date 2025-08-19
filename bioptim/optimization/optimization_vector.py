import numpy as np
from casadi import vertcat, DM, SX, MX

from ..misc.parameters_types import (
    AnyTuple,
    CX,
    DMList,
    FloatList,
    NpArray,
    DoubleNpArrayTuple,
)

from ..misc.enums import ControlType
from .bound_vector import _dispatch_state_bounds, _dispatch_control_bounds
from .init_vector import _dispatch_state_initial_guess, _dispatch_control_initial_guess
from .vector_utils import DEFAULT_INITIAL_GUESS, DEFAULT_MIN_BOUND, DEFAULT_MAX_BOUND


class OptimizationVectorHelper:
    """
    Methods
    -------
    declare_ocp_shooting_points(self)
        Declare all the casadi variables with the right size to be used during a specific phase
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
    def declare_ocp_shooting_points(ocp: "OptimalControlProgram") -> None:
        """
        Declare all the casadi variables with the right size to be used during a specific phase
        """
        for nlp in ocp.nlp:
            nlp.declare_shooting_points()

    @staticmethod
    def vector(ocp: "OptimalControlProgram") -> CX:
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
    def bounds_vectors(ocp: "OptimalControlProgram") -> DoubleNpArrayTuple:
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
        for nlp in ocp.nlp:
            min_bounds, max_bounds = _dispatch_state_bounds(
                nlp, nlp.states, nlp.x_bounds, nlp.x_scaling, nlp.n_states_decision_steps(0)
            )

            v_bounds_min = np.concatenate((v_bounds_min, min_bounds))
            v_bounds_max = np.concatenate((v_bounds_max, max_bounds))

        # For controls
        for nlp in ocp.nlp:
            min_bounds, max_bounds = _dispatch_control_bounds(nlp, nlp.controls, nlp.u_bounds, nlp.u_scaling)

            v_bounds_min = np.concatenate((v_bounds_min, min_bounds))
            v_bounds_max = np.concatenate((v_bounds_max, max_bounds))

        # For parameters
        collapsed_values_min = np.ones((ocp.parameters.shape, 1)) * DEFAULT_MIN_BOUND
        collapsed_values_max = np.ones((ocp.parameters.shape, 1)) * DEFAULT_MAX_BOUND
        for key in ocp.parameters.keys():
            if key not in ocp.parameter_bounds.keys():
                continue

            scaled_bounds = ocp.parameter_bounds[key].scale(ocp.parameters[key].scaling.scaling)
            collapsed_values_min[ocp.parameters[key].index, :] = scaled_bounds.min
            collapsed_values_max[ocp.parameters[key].index, :] = scaled_bounds.max
        v_bounds_min = np.concatenate((v_bounds_min, np.reshape(collapsed_values_min.T, (-1, 1))))
        v_bounds_max = np.concatenate((v_bounds_max, np.reshape(collapsed_values_max.T, (-1, 1))))

        # For algebraic_states variables
        for nlp in ocp.nlp:
            min_bounds, max_bounds = _dispatch_state_bounds(
                nlp,
                nlp.algebraic_states,
                nlp.a_bounds,
                nlp.a_scaling,
                nlp.n_algebraic_states_decision_steps(0),
            )

            v_bounds_min = np.concatenate((v_bounds_min, min_bounds))
            v_bounds_max = np.concatenate((v_bounds_max, max_bounds))

        return v_bounds_min, v_bounds_max

    @staticmethod
    def init_vector(ocp: "OptimalControlProgram") -> NpArray:
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
        for nlp in ocp.nlp:
            init = _dispatch_state_initial_guess(
                nlp, nlp.states, nlp.x_init, nlp.x_scaling, nlp.n_states_decision_steps(0)
            )
            v_init = np.concatenate((v_init, init))

        # For controls
        for nlp in ocp.nlp:
            init = _dispatch_control_initial_guess(nlp, nlp.controls, nlp.u_init, nlp.u_scaling)
            v_init = np.concatenate((v_init, init))

        # For parameters
        collapsed_values = np.zeros((ocp.parameters.shape, 1))
        for key in ocp.parameters.keys():
            if key not in ocp.parameter_init.keys():
                v_init = np.concatenate((v_init, DEFAULT_INITIAL_GUESS * np.ones((ocp.parameters[key].size, 1))))
                continue

            scaled_init = ocp.parameter_init[key].scale(ocp.parameters[key].scaling.scaling)
            collapsed_values[ocp.parameters[key].index, :] = scaled_init.init
        v_init = np.concatenate((v_init, np.reshape(collapsed_values.T, (-1, 1))))

        # For algebraic_states variables
        for nlp in ocp.nlp:
            init = _dispatch_state_initial_guess(
                nlp,
                nlp.algebraic_states,
                nlp.a_init,
                nlp.a_scaling,
                nlp.n_algebraic_states_decision_steps(0),
            )

            v_init = np.concatenate((v_init, init))

        return v_init

    @staticmethod
    def extract_phase_dt(ocp: "OptimalControlProgram", data: NpArray | DM) -> FloatList:
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
    def extract_step_times(ocp: "OptimalControlProgram", data: NpArray | DM) -> list[DMList]:
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
    def to_dictionaries(ocp: "OptimalControlProgram", data: NpArray | DM) -> AnyTuple:
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
        for p, nlp in enumerate(ocp.nlp):
            nx = nlp.states.shape
            for node in range(nlp.n_states_nodes):
                nlp.states.node_index = node
                n_cols = nlp.n_states_decision_steps(node)
                x_array = v_array[offset : offset + nx * n_cols].reshape((nx, -1), order="F")
                for key in nlp.states.keys():
                    data_states[p][key][node] = x_array[nlp.states[key].index, :]
                offset += nx * n_cols

        # For controls
        for p, nlp in enumerate(ocp.nlp):
            nu = nlp.controls.shape

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
        for p, nlp in enumerate(ocp.nlp):
            na = nlp.algebraic_states.shape

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
