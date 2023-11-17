from casadi import vertcat, Function
import numpy as np

from ...dynamics.ode_solver import OdeSolver
from ...misc.enums import ControlType, Shooting
from ..non_linear_program import NonLinearProgram
from ..optimization_variable import OptimizationVariableList, OptimizationVariable

from .utils import concatenate_optimization_variables


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

    def get_integrated_values(self, states: dict, controls: dict, parameters: dict, stochastic_variables: dict) -> dict:
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

    def _generate_time(
        self,
        step_time: float,
        time_phase: np.ndarray,
        keep_intermediate_points: bool = None,
        shooting_type: Shooting = None,
    ):
        """
        Generate time vector steps for a phase considering all the phase final time

        Parameters
        ----------
        time_phase: np.ndarray
            The time of each phase
        keep_intermediate_points: bool
            If the integration should return the intermediate values of the integration [False]
            or only keep the node [True] effective keeping the initial size of the states
        shooting_type: Shooting
            Which type of integration such as Shooting.SINGLE_CONTINUOUS or Shooting.MULTIPLE,
            default is None but behaves as Shooting.SINGLE.

        Returns
        -------
        np.ndarray
        """
        is_direct_collocation = self.ode_solver.is_direct_collocation
        duplicate_collocation_starting_point = False
        if is_direct_collocation:
            duplicate_collocation_starting_point = self.ode_solver.duplicate_collocation_starting_point

        step_times = self._define_step_times(
            dynamics_step_time=step_time,
            ode_solver_steps=self.ode_solver.steps,
            is_direct_collocation=is_direct_collocation,
            duplicate_collocation_starting_point=duplicate_collocation_starting_point,
            keep_intermediate_points=keep_intermediate_points,
            continuous=shooting_type == Shooting.SINGLE,
        )

        if shooting_type == Shooting.SINGLE_DISCONTINUOUS_PHASE:
            # discard the last time step because continuity concerns only the end of the phases
            # and not the end of each interval
            step_times = step_times[:-1]

        dt_ns = float(time_phase[self.phase_idx + 1] / self.ns)
        time = [(step_times * dt_ns + i * dt_ns).tolist() for i in range(self.ns)]

        if shooting_type == Shooting.MULTIPLE:
            # keep all the intervals in separate lists
            flat_time = [np.array(sub_time) for sub_time in time]
        else:
            # flatten the list of list into a list of floats
            flat_time = [st for sub_time in time for st in sub_time]

        # add the final time of the phase
        if shooting_type == Shooting.MULTIPLE:
            flat_time.append(np.array([self.ns * dt_ns]))
        if shooting_type == Shooting.SINGLE or shooting_type == Shooting.SINGLE_DISCONTINUOUS_PHASE:
            flat_time += [self.ns * dt_ns]

        return sum(time_phase[: self.phase_idx + 1]) + np.array(flat_time, dtype=object)

    @staticmethod
    def _define_step_times(
        dynamics_step_time: float | list,
        ode_solver_steps: int,
        keep_intermediate_points: bool = None,
        continuous: bool = True,
        is_direct_collocation: bool = None,
        duplicate_collocation_starting_point: bool = False,
    ) -> np.ndarray:
        """
        Define the time steps for the integration of the whole phase

        Parameters
        ----------
        dynamics_step_time: list
            The step time of the dynamics function
        ode_solver_steps: int
            The number of steps of the ode solver
        keep_intermediate_points: bool
            If the integration should return the intermediate values of the integration [False]
            or only keep the node [True] effective keeping the initial size of the states
        continuous: bool
            If the arrival value of a node should be discarded [True] or kept [False]. The value of an integrated
            arrival node and the beginning of the next one are expected to be almost equal when the problem converged
        is_direct_collocation: bool
            If the ode solver is direct collocation
        duplicate_collocation_starting_point
            If the ode solver is direct collocation and an additional collocation point at the shooting node was used

        Returns
        -------
        step_times: np.ndarray
            The time steps for each interval of the phase of ocp
        """

        if keep_intermediate_points is None:
            keep_intermediate_points = True if is_direct_collocation else False

        if is_direct_collocation:
            # time is not linear because of the collocation points
            if keep_intermediate_points:
                step_times = np.array(dynamics_step_time + [1])

                if duplicate_collocation_starting_point:
                    step_times = np.array([0] + step_times)
            else:
                step_times = np.array(dynamics_step_time + [1])[[0, -1]]

        else:
            # time is linear in the case of direct multiple shooting
            step_times = np.linspace(0, 1, ode_solver_steps + 1) if keep_intermediate_points else np.array([0, 1])
        # it does not take the last nodes of each interval
        if continuous:
            step_times = step_times[:-1]

        return step_times

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
        states: list[np.ndarray],
        controls: list[np.ndarray],
        parameters: np.ndarray,
        stochastic_variables: list[np.ndarray],
    ):
        """
        TODO:

        Parameters
        ----------
        states: list[np.ndarray]
        controls: list[np.ndarray]
        parameters: np.ndarray
        stochastic_variables: list[np.ndarray]

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

    def _generate_time(
        self,
        time_phase: list[float],
        keep_intermediate_points: bool = None,
        merge_phases: bool = False,
        shooting_type: Shooting = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Generate time integration vector

        Parameters
        ----------
        time_phase: list[float]
            list of time phase for each phase
        keep_intermediate_points
            If the integration should return the intermediate values of the integration [False]
            or only keep the node [True] effective keeping the initial size of the states
        merge_phases: bool
            If the phase should be merged in a unique phase
        shooting_type: Shooting
            Which type of integration such as Shooting.SINGLE_CONTINUOUS or Shooting.MULTIPLE,
            default is None but behaves as Shooting.SINGLE.

        Returns
        -------
        t_integrated: np.ndarray or list of np.ndarray
        The time vector
        """
        if shooting_type is None:
            shooting_type = Shooting.SINGLE_DISCONTINUOUS_PHASE


        time_vector = []
        for p, nlp in enumerate(self.nlp):
            if isinstance(self.nlp[0].ode_solver, OdeSolver.COLLOCATION):
                step_time = nlp.dynamics[0].step_time
            else: 
                step_time = time_phase[p] / nlp.ns
            phase_time_vector = nlp._generate_time(step_time, time_phase, keep_intermediate_points, shooting_type)
            time_vector.append(phase_time_vector)

        if merge_phases:
            return concatenate_optimization_variables(time_vector, continuous_phase=shooting_type == Shooting.SINGLE)
        else:
            return time_vector

    def _complete_controls(
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
