from casadi import vertcat, Function
import numpy as np

from ...dynamics.ode_solver import OdeSolver
from ...misc.enums import ControlType, PhaseDynamics
from ..non_linear_program import NonLinearProgram
from ..optimization_variable import OptimizationVariableList, OptimizationVariable


class SimplifiedOptimizationVariable:
    """
    Simplified version of OptimizationVariable (compatible with pickle)
    """

    def __init__(self, other: OptimizationVariable):
        self.name = other.name
        self.index = other.index
        self.mapping = other.mapping

    def __len__(self):
        return len(self.index)


class SimplifiedOptimizationVariableList:
    """
    Simplified version of OptimizationVariableList (compatible with pickle)
    """

    def __init__(self, other: OptimizationVariableList):
        self.elements = []
        if isinstance(other, SimplifiedOptimizationVariableList):
            self.shape = other.shape
        else:
            self.shape = other.cx_start.shape[0]
        for elt in other:
            self.append(other[elt])

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.elements[item]
        elif isinstance(item, str):
            for elt in self.elements:
                if item == elt.name:
                    return elt
            raise KeyError(f"{item} is not in the list")
        else:
            raise ValueError("OptimizationVariableList can be sliced with int or str only")

    def append(self, other: OptimizationVariable):
        self.elements.append(SimplifiedOptimizationVariable(other))

    def __contains__(self, item):
        for elt in self.elements:
            if item == elt.name:
                return True
        else:
            return False

    def keys(self):
        return [elt.name for elt in self]

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self[self._iter_idx - 1].name


class SimplifiedNLP:
    """
    A simplified version of the NonLinearProgram structure (compatible with pickle)

    Methods
    -------
    get_integrated_values(self, states: dict, controls: dict, parameters: dict, stochastic_variables: dict) -> dict
        TODO
    _generate_time(self, time_phase: np.ndarray, keep_intermediate_points: bool = None,
        shooting_type: Shooting = None) -> np.ndarray
        Generate time vector steps for a phase considering all the phase final time
    _define_step_times(self, dynamics_step_time: list, ode_solver_steps: int,
        keep_intermediate_points: bool = None, continuous: bool = True,
        is_direct_collocation: bool = None, duplicate_collocation_starting_point: bool = False) -> np.ndarray
        Define the time steps for the integration of the whole phase
    _define_step_times(self, dynamics_step_time: list, ode_solver_steps: int,
        keep_intermediate_points: bool = None, continuous: bool = True,
        is_direct_collocation: bool = None, duplicate_collocation_starting_point: bool = False) -> np.ndarray
        Define the time steps for the integration of the whole phase
    _complete_controls(self, controls: dict[str, np.ndarray]) -> dict[str, np.ndarray]
        Controls don't necessarily have dimensions that matches the states. This method aligns them


    Attributes
    ----------
    phase_idx: int
        The index of the phase
    use_states_from_phase_idx: int
        The index of the phase from which the states are taken
    use_controls_from_phase_idx: int
        The index of the phase from which the controls are taken
    time_cx: MX.sym
        The time of the phase
    states: OptimizationVariableList
        The states of the phase
    states_dot: OptimizationVariableList
        The derivative of the states of the phase
    controls: OptimizationVariableList
        The controls of the phase
    stochastic_variables: OptimizationVariableList
        The stochastic variables of the phase
    integrated_values: dict
        The integrated values of the phase
    dynamics: list[OdeSolver]
        All the dynamics for each of the node of the phase
    dynamics_func: list[Function]
        All the dynamics function for each of the node of the phase
    ode_solver: OdeSolverBase
        The number of finite element of the RK
    control_type: ControlType
        The control type for the current nlp
    J: list[list[Objective]]
        All the objectives at each of the node of the phase
    J_internal: list[list[Objective]]
        All the objectives at each of the node of the phase
    g: list[list[Constraint]]
        All the constraints at each of the node of the phase
    g_internal: list[list[Constraint]]
        All the constraints at each of the node of the phase (not built by the user)
    g_implicit: list[list[Constraint]]
        All the implicit constraints at each of the node of the phase (mostly implicit dynamics)
    ns: int
        The number of shooting points
    parameters: OptimizationVariableList
        The parameters of the phase
    x_scaling: VariableScalingList
        The scaling of the states
    u_scaling: VariableScalingList
        The scaling of the controls
    s_scaling: VariableScalingList
        The scaling of the stochastic variables
    phase_dynamics: PhaseDynamics
        The dynamics of the phase such as PhaseDynamics.ONE_PER_NODE
    """

    def __init__(self, nlp: NonLinearProgram):
        """
        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the NonLinearProgram to strip
        """

        self.dt = nlp.dt
        self.time_index = nlp.time_index
        self.phase_idx = nlp.phase_idx
        self.use_states_from_phase_idx = nlp.use_states_from_phase_idx
        self.use_controls_from_phase_idx = nlp.use_controls_from_phase_idx
        self.model = nlp.model
        self.time_cx = nlp.time_cx
        self.states = nlp.states
        self.states_dot = nlp.states_dot
        self.controls = nlp.controls
        self.stochastic_variables = nlp.stochastic_variables
        self.integrated_values = nlp.integrated_values
        self.dynamics = nlp.dynamics
        self.dynamics_func = nlp.dynamics_func
        self.ode_solver = nlp.ode_solver
        self.variable_mappings = nlp.variable_mappings
        self.control_type = nlp.control_type
        self.J = nlp.J
        self.J_internal = nlp.J_internal
        self.g = nlp.g
        self.g_internal = nlp.g_internal
        self.g_implicit = nlp.g_implicit
        self.ns = nlp.ns
        self.parameters = nlp.parameters
        self.x_scaling = nlp.x_scaling
        self.u_scaling = nlp.u_scaling
        self.s_scaling = nlp.s_scaling
        self.phase_dynamics = nlp.phase_dynamics

    def get_integrated_values(
        self,
        states: dict[str, np.ndarray],
        controls: dict[str, np.ndarray],
        parameters: dict[str, np.ndarray],
        stochastic_variables: dict[str, np.ndarray],
    ) -> dict:
        """
        TODO :

        Parameters
        ----------
        states: dict
        controls: dict
        parameters: dict
        stochastic_variables: dict

        Returns
        -------
        dict

        """

        integrated_values_num = {}

        self.states.node_index = 0
        self.controls.node_index = 0
        self.parameters.node_index = 0
        self.stochastic_variables.node_index = 0
        for key in self.integrated_values:
            states_cx = self.states.cx_start
            controls_cx = self.controls.cx_start
            stochastic_variables_cx = self.stochastic_variables.cx_start
            integrated_values_cx = self.integrated_values[key].cx_start

            states_num = np.array([])
            for key_tempo in states.keys():
                states_num = np.concatenate((states_num, states[key_tempo][:, 0]))

            controls_num = np.array([])
            for key_tempo in controls.keys():
                controls_num = np.concatenate((controls_num, controls[key_tempo][:, 0]))

            stochastic_variables_num = np.array([])
            for key_tempo in stochastic_variables.keys():
                stochastic_variables_num = np.concatenate(
                    (stochastic_variables_num, stochastic_variables[key_tempo][:, 0])
                )

            for i_node in range(1, self.ns):
                self.states.node_index = i_node
                self.controls.node_index = i_node
                self.stochastic_variables.node_index = i_node
                self.integrated_values.node_index = i_node

                states_cx = vertcat(states_cx, self.states.cx_start)
                controls_cx = vertcat(controls_cx, self.controls.cx_start)
                stochastic_variables_cx = vertcat(stochastic_variables_cx, self.stochastic_variables.cx_start)
                integrated_values_cx = vertcat(integrated_values_cx, self.integrated_values[key].cx_start)
                states_num_tempo = np.array([])
                for key_tempo in states.keys():
                    states_num_tempo = np.concatenate((states_num_tempo, states[key_tempo][:, i_node]))
                states_num = vertcat(states_num, states_num_tempo)

                controls_num_tempo = np.array([])
                for key_tempo in controls.keys():
                    controls_num_tempo = np.concatenate((controls_num_tempo, controls[key_tempo][:, i_node]))
                controls_num = vertcat(controls_num, controls_num_tempo)

                stochastic_variables_num_tempo = np.array([])
                if len(stochastic_variables) > 0:
                    for key_tempo in stochastic_variables.keys():
                        stochastic_variables_num_tempo = np.concatenate(
                            (
                                stochastic_variables_num_tempo,
                                stochastic_variables[key_tempo][:, i_node],
                            )
                        )
                    stochastic_variables_num = vertcat(stochastic_variables_num, stochastic_variables_num_tempo)

            time_tempo = np.array([])
            parameters_tempo = np.array([])
            if len(parameters) > 0:
                for key_tempo in parameters.keys():
                    parameters_tempo = np.concatenate((parameters_tempo, parameters[key_tempo]))
            casadi_func = Function(
                "integrate_values",
                [self.time_cx, states_cx, controls_cx, self.parameters.cx_start, stochastic_variables_cx],
                [integrated_values_cx],
            )
            integrated_values_this_time = casadi_func(
                time_tempo, states_num, controls_num, parameters_tempo, stochastic_variables_num
            )
            nb_elements = self.integrated_values[key].cx_start.shape[0]
            integrated_values_data = np.zeros((nb_elements, self.ns))
            for i_node in range(self.ns):
                integrated_values_data[:, i_node] = np.reshape(
                    integrated_values_this_time[i_node * nb_elements : (i_node + 1) * nb_elements],
                    (nb_elements,),
                )
            integrated_values_num[key] = integrated_values_data

        return integrated_values_num

    def _complete_controls(self, controls: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Controls don't necessarily have dimensions that matches the states. This method aligns them
        E.g. if the control is constant, it will add a column of nan to match the states
        But if the control is linear, it won't do anything

        Parameters
        ----------
        controls: dict[str, np.ndarray]
            Control either scaled or unscaled it doesn't matter here.

        Returns
        -------
        controls: dict[str, np.ndarray]
            Controls with the extra NaN if relevant
        """

        def add_nan_column(matrix):
            nan_column = np.nan * np.zeros((matrix.shape[0], 1))
            return np.concatenate((matrix, nan_column), axis=1)

        if self.control_type in (ControlType.CONSTANT, ControlType.NONE):
            controls = {key: add_nan_column(matrix) for key, matrix in controls.items()}
        elif self.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
            pass
        else:
            raise NotImplementedError(f"ControlType {self.control_type} is not implemented in _complete_control")

        return controls


class SimplifiedOCP:
    """
    A simplified version of the NonLinearProgram structure (compatible with pickle)

    Methods
    -------
    get_integrated_values(self, states: list[np.ndarray], controls: list[np.ndarray], parameters: np.ndarray,
                            stochastic_variables: list[np.ndarray]) -> list[dict]
        TODO
    _generate_time(self, time_phase: list[float], keep_intermediate_points: bool = None,
                    merge_phases: bool = False, shooting_type: Shooting = None) -> np.ndarray | list[np.ndarray]
        Generate time integration vector
    _complete_controls(self, controls: dict[str, list[dict[str, np.ndarray]]]) -> dict[str, list[dict[str, np.ndarray]]]
        Controls don't necessarily have dimensions that matches the states. This method aligns them

    Attributes
    ----------
    nlp: Solution.SimplifiedNLP
        All the phases of the ocp
    parameters: dict
        The parameters of the ocp
    n_phases: int
        The number of phases
    J: list
        Objective values that are not phase dependent (mostly parameters)
    J_internal: list
        Objective values that are phase dependent
    g: list
        Constraints that are not phase dependent, made by the user
    g_internal: list
        Constraints that are phase dependent, not made by the user (mostly dynamics)
    g_implicit: list
        Constraints that are phase dependent, not made by the user (mostly implciit dynamics)
    phase_transitions: list[PhaseTransition]
        The list of transition constraint between phases
    prepare_plots: Callable
        The function to call to prepare the PlotOCP
    time_phase_mapping: list
        The mapping between the time and the phase
    n_threads: int
        The number of threads to use for the parallelization
    """

    def __init__(self, ocp):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to strip
        """

        self.nlp = [SimplifiedNLP(nlp) for nlp in ocp.nlp]
        self.parameters = ocp.parameters
        self.n_phases = len(self.nlp)
        self.J = ocp.J
        self.J_internal = ocp.J_internal
        self.g = ocp.g
        self.g_internal = ocp.g_internal
        self.g_implicit = ocp.g_implicit
        self.phase_transitions = ocp.phase_transitions
        self.prepare_plots = ocp.prepare_plots
        self.time_phase_mapping = ocp.time_phase_mapping
        self.n_threads = ocp.n_threads

    def get_integrated_values(
        self,
        states: list[dict[str, np.ndarray], ...],
        controls: list[[str, np.ndarray], ...],
        parameters: dict[str, np.ndarray],
        stochastic_variables: list[[str, np.ndarray], ...],
    ):
        """
        TODO:

        Parameters
        ----------
        states: list[dict]
            The states of the ocp
        controls: list[dict]
            The controls of the ocp
        parameters: dict
            The parameters of the ocp
        stochastic_variables: list[dict]
            The stochastic variables of the ocp

        Returns
        -------
        list[dict]
        """

        integrated_values_num = [{} for _ in self.nlp]
        for i_phase, nlp in enumerate(self.nlp):
            integrated_values_num[i_phase] = nlp.get_integrated_values(
                states[i_phase],
                controls[i_phase],
                parameters,
                stochastic_variables[i_phase],
            )
        return integrated_values_num

    def generate_node_times(
        self,
        dt_times: list[float],
        phase_end_times: list[float],
    ) -> list[list[np.ndarray, ...], ...]:
        """
        Generate time integration vector

        Parameters
        ----------
        dt_times: list[float]
            The time step for each phase
        phase_end_times: list[float]
            list of end time for each phase

        Returns
        -------
        t_integrated: np.ndarray or list of np.ndarray
        The time vector for each phase at each shooting node at each steps of the shooting
        """

        node_times = []
        for p, nlp in enumerate(self.nlp):
            phase_node_times = []
            for ns in range(nlp.ns):
                starting_phase_time = 0 if p == 0 else phase_end_times[p - 1]
                ns = 0 if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE else ns

                phase_node_times.append(np.array(nlp.dynamics[ns].step_times_from_dt([starting_phase_time, dt_times[p]])))
            node_times.append(phase_node_times)
        return node_times

    def complete_controls(
        self, controls: dict[str, list[dict[str, np.ndarray]]]
    ) -> dict[str, list[dict[str, np.ndarray]]]:
        """
        Controls don't necessarily have dimensions that matches the states. This method aligns them
        E.g. if the control is constant, it will add a column of nan to match the states
        But if the control is linear, it won't do anything

        Parameters
        ----------
        controls: dict[str, list[dict[str, np.ndarray]]]
            Controls of the optimal control problem

        Returns
        -------
        controls: dict[str, list[dict[str, np.ndarray]]]
            Completed controls with the extra nan if relevant

        """

        for p, nlp in enumerate(self.nlp):
            controls["scaled"][p] = nlp._complete_controls(controls["scaled"][p])
            controls["unscaled"][p] = nlp._complete_controls(controls["unscaled"][p])

        return controls
