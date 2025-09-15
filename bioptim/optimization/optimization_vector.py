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
from .vector_layout import VectorLayout


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
    define_ocp_shooting_points(self)
        Declare all the casadi variables with the right size to be used during a specific phase
    define_ocp_bounds(self)
        Declare and parse the bounds for all the variables (v vector)
    define_ocp_initial_guess(self)
        Declare and parse the initial guesses for all the variables (v vector)
    add_parameter(self, param: Parameter)
        Add a parameter to the parameters pool
    control_duplication(controls: dict, nlps: list["NonLinearProgram"]) -> dict
        Duplicate the controls for linear continuous control types.
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

        time, states, controls, algebraic_states, parameters = ocp.get_decision_variables()

        return ocp.vector_layout.stack(
            time, states, controls, algebraic_states, parameters, ocp.vector_layout.query_function
        )

    @staticmethod
    def bounds_vectors(ocp: "OptimalControlProgram") -> DoubleNpArrayTuple:
        """
        Format the x, u and p bounds so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all bounds (min, max)
        """
        states_min_bounds = []
        states_max_bounds = []
        for nlp in ocp.nlp:
            min_bounds_i, max_bounds_i = _dispatch_state_bounds(
                nlp, nlp.states, nlp.x_bounds, nlp.x_scaling, nlp.n_states_decision_steps(0)
            )
            states_min_bounds.append(min_bounds_i)
            states_max_bounds.append(max_bounds_i)

        controls_min_bounds = []
        controls_max_bounds = []
        for nlp in ocp.nlp:
            min_bounds_i, max_bounds_i = _dispatch_control_bounds(nlp, nlp.controls, nlp.u_bounds, nlp.u_scaling)
            controls_min_bounds.append(min_bounds_i)
            controls_max_bounds.append(max_bounds_i)

        # For parameters
        collapsed_values_min = np.ones((ocp.parameters.shape, 1)) * DEFAULT_MIN_BOUND
        collapsed_values_max = np.ones((ocp.parameters.shape, 1)) * DEFAULT_MAX_BOUND
        for key in ocp.parameters.keys():
            if key not in ocp.parameter_bounds.keys():
                continue

            scaled_bounds = ocp.parameter_bounds[key].scale(ocp.parameters[key].scaling.scaling)
            collapsed_values_min[ocp.parameters[key].index, :] = scaled_bounds.min
            collapsed_values_max[ocp.parameters[key].index, :] = scaled_bounds.max

        parameters_min_bounds = np.reshape(collapsed_values_min.T, (-1, 1))
        parameters_max_bounds = np.reshape(collapsed_values_max.T, (-1, 1))

        algebraic_states_min_bounds = []
        algebraic_states_max_bounds = []
        for nlp in ocp.nlp:
            min_bounds_i, max_bounds_i = _dispatch_state_bounds(
                nlp,
                nlp.algebraic_states,
                nlp.a_bounds,
                nlp.a_scaling,
                nlp.n_algebraic_states_decision_steps(0),
            )
            algebraic_states_min_bounds.append(min_bounds_i)
            algebraic_states_max_bounds.append(max_bounds_i)

        v_bounds_min = ocp.vector_layout.stack(
            time=ocp.dt_parameter_bounds.min,
            states=states_min_bounds,
            controls=controls_min_bounds,
            algebraics=algebraic_states_min_bounds,
            parameters=parameters_min_bounds,
            query_function=ocp.vector_layout.query_function,
        )

        v_bounds_max = ocp.vector_layout.stack(
            time=ocp.dt_parameter_bounds.max,
            states=states_max_bounds,
            controls=controls_max_bounds,
            algebraics=algebraic_states_max_bounds,
            parameters=parameters_max_bounds,
            query_function=ocp.vector_layout.query_function,
        )
        return v_bounds_min, v_bounds_max

    @staticmethod
    def init_vector(ocp: "OptimalControlProgram") -> NpArray:
        """
        Format the x, u and p bounds so they are in one nice (and useful) vector

        Returns
        -------
        The vector of all bounds (min, max)
        """

        # For states
        states_init = []
        for nlp in ocp.nlp:
            init = _dispatch_state_initial_guess(
                nlp, nlp.states, nlp.x_init, nlp.x_scaling, nlp.n_states_decision_steps(0)
            )
            states_init.append(init)

        # For controls
        controls_init = []
        for nlp in ocp.nlp:
            init = _dispatch_control_initial_guess(nlp, nlp.controls, nlp.u_init, nlp.u_scaling)
            controls_init.append(init)

        # For parameters
        collapsed_values = np.ones((ocp.parameters.shape, 1)) * DEFAULT_INITIAL_GUESS
        for key in ocp.parameters.keys():
            if key not in ocp.parameter_init.keys():
                continue

            scaled_init = ocp.parameter_init[key].scale(ocp.parameters[key].scaling.scaling)
            collapsed_values[ocp.parameters[key].index, :] = scaled_init.init
        parameters_init = np.reshape(collapsed_values.T, (-1, 1))

        # For algebraic_states variables
        algebraic_states_init = []
        for nlp in ocp.nlp:
            init = _dispatch_state_initial_guess(
                nlp,
                nlp.algebraic_states,
                nlp.a_init,
                nlp.a_scaling,
                nlp.n_algebraic_states_decision_steps(0),
            )
            algebraic_states_init.append(init)

        v_init = ocp.vector_layout.stack(
            time=ocp.dt_parameter_initial_guess.init,
            states=states_init,
            controls=controls_init,
            algebraics=algebraic_states_init,
            parameters=parameters_init,
            query_function=ocp.vector_layout.query_function,
        )

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
    def control_duplication(controls: dict, nlps: list["NonLinearProgram"]) -> dict:
        """Duplicate the controls for linear continuous control types."""
        for p, (controls_p, nlp) in enumerate(zip(controls, nlps)):
            for key in controls_p.keys():
                controls_p_key = controls_p[key]
                for node in range(nlp.n_controls_nodes):

                    # NOTE: hardcoded that phases are sequential 0->1->2 ... not 0->2->3 + 0->1
                    is_last_phase = p == (len(nlps) - 1)
                    is_last_node = node == (nlp.n_controls_nodes - 1)

                    # NOTE: only different of 1 for ControlType.LINEAR_CONTINUOUS
                    n_cols = nlp.control_type.nb_interpolation_points(
                        has_no_successor_phase=is_last_phase,
                        is_last_node=is_last_node,
                    )

                    # NOTE: hardcoded that the next phase is the next in the list. Not a graph.
                    if is_last_node and not is_last_phase:
                        phase_next_node = p + 1
                        next_node = 0
                    else:
                        phase_next_node = p
                        next_node = node + 1

                    current_control = controls_p_key[node]
                    for i in range(1, n_cols):
                        next_control = controls[phase_next_node][key][next_node]
                        current_control = np.hstack((current_control, next_control))

                    controls_p_key[node] = current_control
                controls_p[key] = controls_p_key
            controls[p] = controls_p

        return controls
