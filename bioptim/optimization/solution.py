from typing import Any
from copy import deepcopy

import numpy as np
from scipy import interpolate as sci_interp
from scipy.interpolate import interp1d
from casadi import vertcat, DM, Function, MX, horzcat
from matplotlib import pyplot as plt

from ..limits.objective_functions import ObjectiveFcn
from ..limits.path_conditions import InitialGuess, InitialGuessList
from ..misc.enums import (
    ControlType,
    CostType,
    Shooting,
    InterpolationType,
    SolverType,
    SolutionIntegrator,
    Node,
    QuadratureRule,
)
from ..misc.utils import check_version
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.optimization_variable import OptimizationVariableList, OptimizationVariable
from ..optimization.optimization_vector import OptimizationVectorHelper
from ..dynamics.ode_solver import OdeSolver
from ..interfaces.solve_ivp_interface import solve_ivp_interface, solve_ivp_bioptim_interface
from ..interfaces.biorbd_model import BiorbdModel


class Solution:
    """
    Data manipulation, graphing and storage

    Attributes
    ----------
    ocp: SimplifiedOCP
        The OCP simplified
    ns: list
        The number of shooting point for each phase
    is_interpolated: bool
        If the current structure is interpolated
    is_integrated: bool
        If the current structure is integrated
    is_merged: bool
        If the phases were merged
    vector: np.ndarray
        The data in the vector format
    _cost: float
        The value of the cost function
    constraints: list
        The values of the constraint
    lam_g: list
        The Lagrange multiplier of the constraints
    lam_p: list
        The Lagrange multiplier of the parameters
    lam_x: list
        The Lagrange multiplier of the states and controls
    inf_pr: list
        The unscaled constraint violation at each iteration
    inf_du: list
        The scaled dual infeasibility at each iteration
    solver_time_to_optimize: float
        The total time to solve the program
    iterations: int
        The number of iterations that were required to solve the program
    status: int
        Optimization success status (Ipopt: 0=Succeeded, 1=Failed)
    _states: list
        The data structure that holds the states
    _controls: list
        The data structure that holds the controls
    parameters: dict
        The data structure that holds the parameters
    _stochastic_variables: list
        The data structure that holds the stochastic variables
    _integrated_values: list
        The data structure that holds the update values
    phase_time: list
        The total time for each phases

    Methods
    -------
    copy(self, skip_data: bool = False) -> Any
        Create a deepcopy of the Solution
    @property
    states(self) -> list | dict
        Returns the state scaled and unscaled in list if more than one phases, otherwise it returns the only dict
    @property
    states_scaled_no_intermediate(self) -> list | dict
        Returns the state scaled in list if more than one phases, otherwise it returns the only dict
        and removes the intermediate states scaled if Collocation solver is used
    @property
    states_no_intermediate(self) -> list | dict
        Returns the state unscaled in list if more than one phases, otherwise it returns the only dict
        and removes the intermediate states unscaled if Collocation solver is used
    @property
    controls(self) -> list | dict
        Returns the controls scaled and unscaled in list if more than one phases, otherwise it returns the only dict
    integrate(self, shooting_type: Shooting = Shooting.MULTIPLE, keep_intermediate_points: bool = True,
              merge_phases: bool = False, continuous: bool = True) -> Solution
        Integrate the states unscaled
    interpolate(self, n_frames: int | list | tuple) -> Solution
        Interpolate the states unscaled
    merge_phases(self) -> Solution
        Get a data structure where all the phases are merged into one
    _merge_phases(self, skip_states: bool = False, skip_controls: bool = False) -> tuple
        Actually performing the phase merging
    _complete_control(self)
        Controls don't necessarily have dimensions that matches the states. This method aligns them
    graphs(self, automatically_organize: bool, show_bounds: bool,
           show_now: bool, shooting_type: Shooting)
        Show the graphs of the simulation
    animate(self, n_frames: int = 0, show_now: bool = True, **kwargs: Any) -> None | list
        Animate the simulation
    print(self, cost_type: CostType = CostType.ALL)
        Print the objective functions and/or constraints to the console
    """

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
            if isinstance(other, Solution.SimplifiedOptimizationVariableList):
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
            self.elements.append(Solution.SimplifiedOptimizationVariable(other))

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

        Attributes
        ----------
        control_type: ControlType
            The control type for the current nlp
        dynamics: list[OdeSolver]
            All the dynamics for each of the node of the phase
        g: list[list[Constraint]]
            All the constraints at each of the node of the phase
        J: list[list[Objective]]
            All the objectives at each of the node of the phase
        model: BioModel
            A reference to the biorbd BioModel
        variable_mappings: dict
            All the BiMapping of the states and controls
        ode_solver: OdeSolverBase
            The number of finite element of the RK
        ns: int
            The number of shooting points
        """

        def __init__(self, nlp: NonLinearProgram):
            """
            Parameters
            ----------
            nlp: NonLinearProgram
                A reference to the NonLinearProgram to strip
            """

            self.tf = nlp.tf
            self.phase_idx = nlp.phase_idx
            self.use_states_from_phase_idx = nlp.use_states_from_phase_idx
            self.use_controls_from_phase_idx = nlp.use_controls_from_phase_idx
            self.model = nlp.model
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
            self.assume_phase_dynamics = nlp.assume_phase_dynamics

    class SimplifiedOCP:
        """
        A simplified version of the NonLinearProgram structure (compatible with pickle)

        Attributes
        ----------
        g: list
            Constraints that are not phase dependent (mostly parameters and continuity constraints)
        J: list
            Objective values that are not phase dependent (mostly parameters)
        nlp: NLP
            All the phases of the ocp
        phase_transitions: list[PhaseTransition]
            The list of transition constraint between phases
        prepare_plots: Callable
            The function to call to prepare the PlotOCP
        The variable optimization holder
        """

        def __init__(self, ocp):
            """
            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp to strip
            """

            self.nlp = [Solution.SimplifiedNLP(nlp) for nlp in ocp.nlp]
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
            self.assume_phase_dynamics = ocp.assume_phase_dynamics
            self.n_threads = ocp.n_threads

    def __init__(self, ocp, sol: dict | list | tuple | np.ndarray | DM | None):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to strip
        sol: dict | list | tuple | np.ndarray | DM | None
            The values of a solution
        """

        self.ocp = Solution.SimplifiedOCP(ocp) if ocp else None
        self.ns = [nlp.ns for nlp in self.ocp.nlp]

        # Current internal state of the data
        self.is_interpolated = False
        self.is_integrated = False
        self.is_merged = False
        self.recomputed_time_steps = False

        self.vector = None
        self._cost = None
        self.constraints = None
        self._detailed_cost = None

        self.lam_g = None
        self.lam_p = None
        self.lam_x = None
        self.inf_pr = None
        self.inf_du = None
        self.solver_time_to_optimize = None
        self.real_time_to_optimize = None
        self.iterations = None
        self.status = None
        self._time_vector = None

        # Extract the data now for further use
        self._states = {}
        self._controls = {}
        self.parameters = {}
        self._stochastic_variables = {}
        self._integrated_values = {}
        self.phase_time = []

        def get_integrated_values(states, controls, parameters, stochastic_variables):
            integrated_values_num = [{} for _ in self.ocp.nlp]
            for i_phase, nlp in enumerate(self.ocp.nlp):
                nlp.states.node_index = 0
                nlp.controls.node_index = 0
                nlp.parameters.node_index = 0
                nlp.stochastic_variables.node_index = 0
                for key in nlp.integrated_values:
                    states_cx = nlp.states.cx_start
                    controls_cx = nlp.controls.cx_start
                    stochastic_variables_cx = nlp.stochastic_variables.cx_start
                    integrated_values_cx = nlp.integrated_values[key].cx_start

                    states_num = np.array([])
                    for key_tempo in states[i_phase].keys():
                        states_num = np.concatenate((states_num, states[i_phase][key_tempo][:, 0]))

                    controls_num = np.array([])
                    for key_tempo in controls[i_phase].keys():
                        controls_num = np.concatenate((controls_num, controls[i_phase][key_tempo][:, 0]))

                    stochastic_variables_num = np.array([])
                    for key_tempo in stochastic_variables[i_phase].keys():
                        stochastic_variables_num = np.concatenate(
                            (stochastic_variables_num, stochastic_variables[i_phase][key_tempo][:, 0])
                        )

                    for i_node in range(1, nlp.ns):
                        nlp.states.node_index = i_node
                        nlp.controls.node_index = i_node
                        nlp.stochastic_variables.node_index = i_node
                        nlp.integrated_values.node_index = i_node

                        states_cx = vertcat(states_cx, nlp.states.cx_start)
                        controls_cx = vertcat(controls_cx, nlp.controls.cx_start)
                        stochastic_variables_cx = vertcat(stochastic_variables_cx, nlp.stochastic_variables.cx_start)
                        integrated_values_cx = vertcat(integrated_values_cx, nlp.integrated_values[key].cx_start)
                        states_num_tempo = np.array([])
                        for key_tempo in states[i_phase].keys():
                            states_num_tempo = np.concatenate((states_num_tempo, states[i_phase][key_tempo][:, i_node]))
                        states_num = vertcat(states_num, states_num_tempo)

                        controls_num_tempo = np.array([])
                        for key_tempo in controls[i_phase].keys():
                            controls_num_tempo = np.concatenate(
                                (controls_num_tempo, controls[i_phase][key_tempo][:, i_node])
                            )
                        controls_num = vertcat(controls_num, controls_num_tempo)

                        stochastic_variables_num_tempo = np.array([])
                        if len(stochastic_variables[i_phase]) > 0:
                            for key_tempo in stochastic_variables[i_phase].keys():
                                stochastic_variables_num_tempo = np.concatenate(
                                    (
                                        stochastic_variables_num_tempo,
                                        stochastic_variables[i_phase][key_tempo][:, i_node],
                                    )
                                )
                            stochastic_variables_num = vertcat(stochastic_variables_num, stochastic_variables_num_tempo)

                    parameters_tempo = np.array([])
                    if len(parameters) > 0:
                        for key_tempo in parameters[i_phase].keys():
                            parameters_tempo = np.concatenate((parameters_tempo, parameters[i_phase][key_tempo]))
                    casadi_func = Function(
                        "integrate_values",
                        [states_cx, controls_cx, nlp.parameters.cx_start, stochastic_variables_cx],
                        [integrated_values_cx],
                    )
                    integrated_values_this_time = casadi_func(
                        states_num, controls_num, parameters_tempo, stochastic_variables_num
                    )
                    nb_elements = nlp.integrated_values[key].cx_start.shape[0]
                    integrated_values_data = np.zeros((nb_elements, nlp.ns))
                    for i_node in range(nlp.ns):
                        integrated_values_data[:, i_node] = np.reshape(
                            integrated_values_this_time[i_node * nb_elements : (i_node + 1) * nb_elements],
                            (nb_elements,),
                        )
                    integrated_values_num[i_phase][key] = integrated_values_data
                return integrated_values_num

        def init_from_dict(_sol: dict):
            """
            Initialize all the attributes from an Ipopt-like dictionary data structure

            Parameters
            ----------
            _sol: dict
                The solution in a Ipopt-like dictionary
            """

            self.vector = _sol["x"]
            if _sol["solver"] == SolverType.IPOPT.value:
                self._cost = _sol["f"]
                self.constraints = _sol["g"]

                self.lam_g = _sol["lam_g"]
                self.lam_p = _sol["lam_p"]
                self.lam_x = _sol["lam_x"]
                self.inf_pr = _sol["inf_pr"]
                self.inf_du = _sol["inf_du"]

            self.solver_time_to_optimize = _sol["solver_time_to_optimize"]
            self.real_time_to_optimize = _sol["real_time_to_optimize"]
            self.iterations = _sol["iter"]
            self.status = _sol["status"]

            # Extract the data now for further use
            (
                self._states["scaled"],
                self._controls["scaled"],
                self.parameters,
                self._stochastic_variables["scaled"],
            ) = OptimizationVectorHelper.to_dictionaries(self.ocp, self.vector)
            (
                self._states["unscaled"],
                self._controls["unscaled"],
                self._stochastic_variables["unscaled"],
            ) = self._to_unscaled_values(
                self._states["scaled"], self._controls["scaled"], self._stochastic_variables["scaled"]
            )
            self._complete_control()
            self.phase_time = OptimizationVectorHelper.extract_phase_time(self.ocp, self.vector)
            self._time_vector = self._generate_time()
            self._integrated_values = get_integrated_values(
                self._states["unscaled"],
                self._controls["unscaled"],
                self.parameters,
                self._stochastic_variables["unscaled"],
            )

        def init_from_initial_guess(_sol: list):
            """
            Initialize all the attributes from a list of initial guesses (states, controls)

            Parameters
            ----------
            _sol: list
                The list of initial guesses
            """

            n_param = len(ocp.parameters)

            # Sanity checks
            for i in range(len(_sol)):  # Convert to list if necessary and copy for as many phases there are
                if isinstance(_sol[i], InitialGuess):
                    tp = InitialGuessList()
                    for _ in range(len(self.ns)):
                        tp.add(deepcopy(_sol[i].init), interpolation=_sol[i].init.type)
                    _sol[i] = tp
            if sum([isinstance(s, InitialGuessList) for s in _sol]) != 4:
                raise ValueError(
                    "solution must be a solution dict, "
                    "an InitialGuess[List] of len 3 or 4 (states, controls, parameters, stochastic_variables), "
                    "or a None"
                )
            if sum([len(s) != len(self.ns) if p != 3 else False for p, s in enumerate(_sol)]) != 0:
                raise ValueError("The InitialGuessList len must match the number of phases")
            if n_param != 0:
                if len(_sol) != 3 and len(_sol[2]) != 1 and _sol[2][0].shape != (n_param, 1):
                    raise ValueError(
                        "The 3rd element is the InitialGuess of the parameter and "
                        "should be a unique vector of size equal to n_param"
                    )

            self.vector = np.ndarray((0, 1))
            sol_states, sol_controls, sol_stochastic_variables = _sol[0], _sol[1], _sol[3]

            # For states
            for p, ss in enumerate(sol_states):
                repeat = 1
                if isinstance(self.ocp.nlp[p].ode_solver, OdeSolver.COLLOCATION):
                    repeat = self.ocp.nlp[p].ode_solver.polynomial_degree + 1
                for key in ss.keys():
                    ns = (
                        self.ocp.nlp[p].ns + 1
                        if ss[key].init.type != InterpolationType.EACH_FRAME
                        else self.ocp.nlp[p].ns
                    )
                    ss[key].init.check_and_adjust_dimensions(len(self.ocp.nlp[p].states[key]), ns, "states")

                for i in range(self.ns[p] + 1):
                    for key in ss.keys():
                        self.vector = np.concatenate((self.vector, ss[key].init.evaluate_at(i, repeat)[:, np.newaxis]))

            # For controls
            for p, ss in enumerate(sol_controls):
                control_type = self.ocp.nlp[p].control_type
                if control_type == ControlType.CONSTANT:
                    off = 0
                elif control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                    off = 1
                else:
                    raise NotImplementedError(f"control_type {control_type} is not implemented in Solution")

                for key in ss.keys():
                    self.ocp.nlp[p].controls[key].node_index = 0
                    ss[key].init.check_and_adjust_dimensions(len(self.ocp.nlp[p].controls[key]), self.ns[p], "controls")

                for i in range(self.ns[p] + off):
                    for key in ss.keys():
                        self.vector = np.concatenate((self.vector, ss[key].init.evaluate_at(i)[:, np.newaxis]))

            # For parameters
            if n_param:
                sol_params = _sol[2]
                for p, ss in enumerate(sol_params):
                    self.vector = np.concatenate((self.vector, np.repeat(ss.init, self.ns[p] + 1)[:, np.newaxis]))

            # For stochastic variables
            for p, ss in enumerate(sol_stochastic_variables):
                for key in ss.keys():
                    self.ocp.nlp[p].stochastic_variables[key].node_index = 0
                    ss[key].init.check_and_adjust_dimensions(
                        len(self.ocp.nlp[p].stochastic_variables[key]), self.ns[p], "stochastic_variables"
                    )

                for i in range(self.ns[p] + 1):
                    for key in ss.keys():
                        self.vector = np.concatenate((self.vector, ss[key].init.evaluate_at(i)[:, np.newaxis]))

            (
                self._states["scaled"],
                self._controls["scaled"],
                self.parameters,
                self._stochastic_variables["scaled"],
            ) = OptimizationVectorHelper.to_dictionaries(self.ocp, self.vector)
            (
                self._states["unscaled"],
                self._controls["unscaled"],
                self._stochastic_variables["unscaled"],
            ) = self._to_unscaled_values(
                self._states["scaled"], self._controls["scaled"], self._stochastic_variables["scaled"]
            )
            self._complete_control()
            self.phase_time = OptimizationVectorHelper.extract_phase_time(self.ocp, self.vector)
            self._time_vector = self._generate_time()
            self._integrated_values = get_integrated_values(
                self._states["unscaled"],
                self._controls["unscaled"],
                self.parameters,
                self._stochastic_variables["unscaled"],
            )

        def init_from_vector(_sol: np.ndarray | DM):
            """
            Initialize all the attributes from a vector of solution

            Parameters
            ----------
            _sol: np.ndarray | DM
                The solution in vector format
            """

            self.vector = _sol
            (
                self._states["scaled"],
                self._controls["scaled"],
                self.parameters,
                self._stochastic_variables["scaled"],
            ) = OptimizationVectorHelper.to_dictionaries(self.ocp, self.vector)
            (
                self._states["unscaled"],
                self._controls["unscaled"],
                self._stochastic_variables["unscaled"],
            ) = self._to_unscaled_values(
                self._states["scaled"], self._controls["scaled"], self._stochastic_variables["scaled"]
            )
            self._complete_control()
            self.phase_time = OptimizationVectorHelper.extract_phase_time(self.ocp, self.vector)
            self._integrated_values = get_integrated_values(
                self._states["unscaled"],
                self._controls["unscaled"],
                self.parameters,
                self._stochastic_variables["unscaled"],
            )

        if isinstance(sol, dict):
            init_from_dict(sol)
        elif isinstance(sol, (list, tuple)) and len(sol) == 4:
            init_from_initial_guess(sol)
        elif isinstance(sol, (np.ndarray, DM)):
            init_from_vector(sol)
        elif sol is None:
            self.ns = []
        else:
            raise ValueError("Solution called with unknown initializer")

    def _to_unscaled_values(self, states_scaled, controls_scaled, stochastic_variables_scaled) -> tuple:
        """
        Convert values of scaled solution to unscaled values
        """

        ocp = self.ocp

        states = [{} for _ in range(len(states_scaled))]
        controls = [{} for _ in range(len(controls_scaled))]
        stochastic_variables = [{} for _ in range(len(stochastic_variables_scaled))]
        for phase in range(len(states_scaled)):
            states[phase] = {}
            controls[phase] = {}
            stochastic_variables[phase] = {}
            for key, value in states_scaled[phase].items():
                states[phase][key] = value * ocp.nlp[phase].x_scaling[key].to_array(states_scaled[phase][key].shape[1])
            for key, value in controls_scaled[phase].items():
                controls[phase][key] = value * ocp.nlp[phase].u_scaling[key].to_array(
                    controls_scaled[phase][key].shape[1]
                )
            for key, value in stochastic_variables_scaled[phase].items():
                stochastic_variables[phase][key] = value * ocp.nlp[phase].s_scaling[key].to_array(
                    stochastic_variables_scaled[phase][key].shape[1]
                )

        return states, controls, stochastic_variables

    @property
    def cost(self):
        if self._cost is None:
            self._cost = 0
            for J in self.ocp.J:
                _, val_weighted = self._get_penalty_cost(None, J)
                self._cost += val_weighted

            for idx_phase, nlp in enumerate(self.ocp.nlp):
                for J in nlp.J:
                    _, val_weighted = self._get_penalty_cost(nlp, J)
                    self._cost += val_weighted
            self._cost = DM(self._cost)
        return self._cost

    def copy(self, skip_data: bool = False) -> Any:
        """
        Create a deepcopy of the Solution

        Parameters
        ----------
        skip_data: bool
            If data should be ignored in the copy

        Returns
        -------
        Return a Solution data structure
        """

        new = Solution(self.ocp, None)

        new.vector = deepcopy(self.vector)
        new._cost = deepcopy(self._cost)
        new.constraints = deepcopy(self.constraints)

        new.lam_g = deepcopy(self.lam_g)
        new.lam_p = deepcopy(self.lam_p)
        new.lam_x = deepcopy(self.lam_x)
        new.inf_pr = deepcopy(self.inf_pr)
        new.inf_du = deepcopy(self.inf_du)
        new.solver_time_to_optimize = deepcopy(self.solver_time_to_optimize)
        new.real_time_to_optimize = deepcopy(self.real_time_to_optimize)
        new.iterations = deepcopy(self.iterations)

        new.is_interpolated = deepcopy(self.is_interpolated)
        new.is_integrated = deepcopy(self.is_integrated)
        new.is_merged = deepcopy(self.is_merged)

        new.phase_time = deepcopy(self.phase_time)
        new.ns = deepcopy(self.ns)

        new._time_vector = deepcopy(self._time_vector)

        if skip_data:
            new._states["unscaled"], new._controls["unscaled"], new._stochastic_variables["unscaled"] = [], [], []
            new._states["scaled"], new._controls["scaled"], new.parameters, new._stochastic_variables["scaled"] = (
                [],
                [],
                {},
                [],
            )
        else:
            new._states["scaled"] = deepcopy(self._states["scaled"])
            new._controls["scaled"] = deepcopy(self._controls["scaled"])
            new.parameters = deepcopy(self.parameters)
            new._states["unscaled"] = deepcopy(self._states["unscaled"])
            new._controls["unscaled"] = deepcopy(self._controls["unscaled"])
            new._stochastic_variables["scaled"] = deepcopy(self._stochastic_variables["scaled"])
            new._stochastic_variables["unscaled"] = deepcopy(self._stochastic_variables["unscaled"])
        return new

    @property
    def states(self) -> list | dict:
        """
        Returns the state in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The states data
        """

        return self._states["unscaled"] if len(self._states["unscaled"]) > 1 else self._states["unscaled"][0]

    @property
    def states_scaled(self) -> list | dict:
        """
        Returns the state in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The states data
        """

        return self._states["scaled"] if len(self._states["scaled"]) > 1 else self._states["scaled"][0]

    def _no_intermediate(self, states) -> list | dict:
        """
        Returns the state in list if more than one phases, otherwise it returns the only dict
        it removes the intermediate states in the case COLLOCATION Solver is used

        Returns
        -------
        The states data without intermediate states in the case of collocation
        """

        if self.is_merged:
            idx_no_intermediate = []
            for i, nlp in enumerate(self.ocp.nlp):
                if type(nlp.ode_solver) is OdeSolver.COLLOCATION:
                    idx_no_intermediate.append(
                        list(
                            range(
                                0,
                                nlp.ns * (nlp.ode_solver.polynomial_degree + 1) + 1,
                                nlp.ode_solver.polynomial_degree + 1,
                            )
                        )
                    )
                else:
                    idx_no_intermediate.append(list(range(0, self.ocp.nlp[i].ns + 1, 1)))

            # merge the index of the intermediate states
            all_intermediate_idx = []
            previous_end = (
                -1 * (self.ocp.nlp[0].ode_solver.polynomial_degree + 1)
                if type(self.ocp.nlp[0].ode_solver) is OdeSolver.COLLOCATION
                else -1
            )

            idx = -1
            offset = 0
            for p, idx in enumerate(idx_no_intermediate):
                offset = (
                    (self.ocp.nlp[p].ode_solver.polynomial_degree + 1)
                    if type(self.ocp.nlp[p].ode_solver) is OdeSolver.COLLOCATION
                    else 1
                )
                if p == 0:
                    all_intermediate_idx.extend([*idx[:-1]])
                else:
                    previous_end = all_intermediate_idx[-1]
                    new_idx = [previous_end + i + offset for i in idx[0:-1]]
                    all_intermediate_idx.extend(new_idx)
            all_intermediate_idx.append(previous_end + idx[-1] + offset)  # add the last index

            # build the new states dictionary for each key
            states_no_intermediate = dict()
            for key in states[0].keys():
                # keep one value each five values
                states_no_intermediate[key] = states[0][key][:, all_intermediate_idx]

            return states_no_intermediate

        else:
            states_no_intermediate = []
            for i, nlp in enumerate(self.ocp.nlp):
                if type(nlp.ode_solver) is OdeSolver.COLLOCATION:
                    states_no_intermediate.append(dict())
                    for key in states[i].keys():
                        # keep one value each five values
                        states_no_intermediate[i][key] = states[i][key][:, :: nlp.ode_solver.polynomial_degree + 1]
                else:
                    states_no_intermediate.append(states[i])

            return states_no_intermediate[0] if len(states_no_intermediate) == 1 else states_no_intermediate

    @property
    def states_scaled_no_intermediate(self) -> list | dict:
        """
        Returns the state in list if more than one phases, otherwise it returns the only dict
        it removes the intermediate states in the case COLLOCATION Solver is used

        Returns
        -------
        The states data without intermediate states in the case of collocation
        """
        return self._no_intermediate(self._states["scaled"])

    @property
    def states_no_intermediate(self) -> list | dict:
        """
        Returns the state in list if more than one phases, otherwise it returns the only dict
        it removes the intermediate states in the case COLLOCATION Solver is used

        Returns
        -------
        The states data without intermediate states in the case of collocation
        """
        return self._no_intermediate(self._states["unscaled"])

    @property
    def controls(self) -> list | dict:
        """
        Returns the controls in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The controls data
        """

        if not self._controls["unscaled"]:
            raise RuntimeError(
                "There is no controls in the solution. "
                "This may happen in "
                "previously integrated and interpolated structure"
            )

        return self._controls["unscaled"] if len(self._controls["unscaled"]) > 1 else self._controls["unscaled"][0]

    @property
    def controls_scaled(self) -> list | dict:
        """
        Returns the controls in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The controls data
        """

        return self._controls["scaled"] if len(self._controls["scaled"]) > 1 else self._controls["scaled"][0]

    @property
    def stochastic_variables(self) -> list | dict:
        """
        Returns the stochastic variables in list if more than one phases, otherwise it returns the only dict
        Returns
        -------
        The stochastic variables data
        """

        return (
            self._stochastic_variables["unscaled"]
            if len(self._stochastic_variables["unscaled"]) > 1
            else self._stochastic_variables["unscaled"][0]
        )

    @property
    def stochastic_variables_scaled(self) -> list | dict:
        """
        Returns the scaled stochastic variables in list if more than one phases, otherwise it returns the only dict
        Returns
        -------
        The stochastic variables data
        """

        return (
            self._stochastic_variables["scaled"]
            if len(self._stochastic_variables["scaled"]) > 1
            else self._stochastic_variables["scaled"][0]
        )

    @property
    def integrated_values(self) -> list | dict:
        """
        Returns the update values in list if more than one phases, otherwise it returns the only dict
        Returns
        -------
        The update values data
        """

        return self._integrated_values if len(self._integrated_values) > 1 else self._integrated_values[0]

    @property
    def time(self) -> list | dict | np.ndarray:
        """
        Returns the time vector in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The time instant vector
        """

        if self._time_vector is None:
            raise RuntimeError(
                "There is no time vector in the solution. "
                "This may happen in "
                "previously integrated and interpolated structure"
            )
        return self._time_vector[0] if len(self._time_vector) == 1 else self._time_vector

    def __integrate_sanity_checks(
        self,
        shooting_type,
        keep_intermediate_points,
        integrator,
    ):
        """
        Sanity checks for the integrate method

        Parameters
        ----------
        shooting_type: Shooting
            The shooting type
        keep_intermediate_points: bool
            If True, the intermediate points are kept
        integrator: Integrator
            The integrator to use such as SolutionIntegrator.OCP, SolutionIntegrator.SCIPY_RK45, etc...
        """
        if self.is_integrated:
            raise RuntimeError("Cannot integrate twice")
        if self.is_interpolated:
            raise RuntimeError("Cannot integrate after interpolating")
        if self.is_merged:
            raise RuntimeError("Cannot integrate after merging phases")

        if shooting_type == Shooting.MULTIPLE and not keep_intermediate_points:
            raise ValueError(
                "shooting_type=Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously."
                "When using multiple shooting, the intermediate points should be kept."
            )

        n_direct_collocation = sum([nlp.ode_solver.is_direct_collocation for nlp in self.ocp.nlp])
        if n_direct_collocation > 0 and integrator == SolutionIntegrator.OCP:
            raise ValueError(
                "When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE"
            )

        n_trapezoidal = sum([isinstance(nlp.ode_solver, OdeSolver.TRAPEZOIDAL) for nlp in self.ocp.nlp])
        if n_trapezoidal > 0 and integrator == SolutionIntegrator.OCP:
            raise ValueError(
                "When the ode_solver of the Optimal Control Problem is OdeSolver.TRAPEZOIDAL, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
            )

    def integrate(
        self,
        shooting_type: Shooting = Shooting.SINGLE,
        keep_intermediate_points: bool = False,
        merge_phases: bool = False,
        integrator: SolutionIntegrator = SolutionIntegrator.SCIPY_RK45,
    ) -> Any:
        """
        Integrate the states

        Parameters
        ----------
        shooting_type: Shooting
            Which type of integration
        keep_intermediate_points: bool
            If the integration should return the intermediate values of the integration [False]
            or only keep the node [True] effective keeping the initial size of the states
        merge_phases: bool
            If the phase should be merged in a unique phase
        integrator: SolutionIntegrator
            Use the scipy integrator RK45 by default, you can use any integrator provided by scipy or the OCP integrator

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        self.__integrate_sanity_checks(
            shooting_type=shooting_type,
            keep_intermediate_points=keep_intermediate_points,
            integrator=integrator,
        )

        out = self.__perform_integration(
            shooting_type=shooting_type,
            keep_intermediate_points=keep_intermediate_points,
            integrator=integrator,
        )

        if merge_phases:
            out.is_merged = True
            out.phase_time = [out.phase_time[0], sum(out.phase_time[1:])]
            out.ns = sum(out.ns)

            if shooting_type == Shooting.SINGLE:
                out._states["unscaled"] = concatenate_optimization_variables_dict(out._states["unscaled"])
                out._time_vector = [concatenate_optimization_variables(out._time_vector)]

            else:
                out._states["unscaled"] = concatenate_optimization_variables_dict(
                    out._states["unscaled"], continuous=False
                )
                out._time_vector = [
                    concatenate_optimization_variables(
                        out._time_vector, continuous_phase=False, continuous_interval=False
                    )
                ]

        elif shooting_type == Shooting.MULTIPLE:
            out._time_vector = concatenate_optimization_variables(
                out._time_vector, continuous_phase=False, continuous_interval=False, merge_phases=merge_phases
            )

        out.is_integrated = True

        return out

    def noisy_integrate(
        self,
        shooting_type: Shooting = Shooting.SINGLE,
        keep_intermediate_points: bool = False,
        merge_phases: bool = False,
        integrator: SolutionIntegrator = SolutionIntegrator.SCIPY_RK45,
        n_random: int = 30,
    ) -> Any:
        """
        Integrate the states

        Parameters
        ----------
        shooting_type: Shooting
            Which type of integration
        keep_intermediate_points: bool
            If the integration should return the intermediate values of the integration [False]
            or only keep the node [True] effective keeping the initial size of the states
        merge_phases: bool
            If the phase should be merged in a unique phase
        integrator: SolutionIntegrator
            Use the scipy integrator RK45 by default, you can use any integrator provided by scipy or the OCP integrator

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        self.__integrate_sanity_checks(
            shooting_type=shooting_type,
            keep_intermediate_points=keep_intermediate_points,
            integrator=integrator,
        )

        out = self.__perform_noisy_integration(
            shooting_type=shooting_type,
            keep_intermediate_points=keep_intermediate_points,
            integrator=integrator,
            n_random=n_random,
        )

        if merge_phases:
            out.is_merged = True
            out.phase_time = [out.phase_time[0], sum(out.phase_time[1:])]
            out.ns = sum(out.ns)

            if shooting_type == Shooting.SINGLE:
                out._states["unscaled"] = concatenate_optimization_variables_dict(out._states["unscaled"])
                out._time_vector = [concatenate_optimization_variables(out._time_vector)]

            else:
                out._states["unscaled"] = concatenate_optimization_variables_dict(
                    out._states["unscaled"], continuous=False
                )
                out._time_vector = [
                    concatenate_optimization_variables(
                        out._time_vector, continuous_phase=False, continuous_interval=False
                    )
                ]

        elif shooting_type == Shooting.MULTIPLE:
            out._time_vector = concatenate_optimization_variables(
                out._time_vector, continuous_phase=False, continuous_interval=False, merge_phases=merge_phases
            )

        out.is_integrated = True

        return out

    def _generate_time(
        self,
        keep_intermediate_points: bool = None,
        merge_phases: bool = False,
        shooting_type: Shooting = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Generate time integration vector

        Parameters
        ----------
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
        time_phase = self.phase_time
        for p, nlp in enumerate(self.ocp.nlp):
            is_direct_collocation = nlp.ode_solver.is_direct_collocation

            step_times = self._define_step_times(
                dynamics_step_time=nlp.dynamics[0].step_time,
                ode_solver_steps=nlp.ode_solver.steps,
                is_direct_collocation=is_direct_collocation,
                keep_intermediate_points=keep_intermediate_points,
                continuous=shooting_type == Shooting.SINGLE,
            )

            if shooting_type == Shooting.SINGLE_DISCONTINUOUS_PHASE:
                # discard the last time step because continuity concerns only the end of the phases
                # and not the end of each interval
                step_times = step_times[:-1]

            dt_ns = time_phase[p + 1] / nlp.ns
            time = [(step_times * dt_ns + i * dt_ns).tolist() for i in range(nlp.ns)]

            if shooting_type == Shooting.MULTIPLE:
                # keep all the intervals in separate lists
                flat_time = [np.array(sub_time) for sub_time in time]
            else:
                # flatten the list of list into a list of floats
                flat_time = [st for sub_time in time for st in sub_time]

            # add the final time of the phase
            if shooting_type == Shooting.MULTIPLE:
                flat_time.append(np.array([nlp.ns * dt_ns]))
            if shooting_type == Shooting.SINGLE or shooting_type == Shooting.SINGLE_DISCONTINUOUS_PHASE:
                flat_time += [nlp.ns * dt_ns]

            time_vector.append(sum(time_phase[: p + 1]) + np.array(flat_time, dtype=object))

        if merge_phases:
            return concatenate_optimization_variables(time_vector, continuous_phase=shooting_type == Shooting.SINGLE)
        else:
            return time_vector

    @staticmethod
    def _define_step_times(
        dynamics_step_time: list,
        ode_solver_steps: int,
        keep_intermediate_points: bool = None,
        continuous: bool = True,
        is_direct_collocation: bool = None,
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

        Returns
        -------
        step_times: np.ndarray
            The time steps for each interval of the phase of ocp
        """

        if keep_intermediate_points is None:
            keep_intermediate_points = True if is_direct_collocation else False

        if is_direct_collocation:
            # time is not linear because of the collocation points
            step_times = (
                np.array(dynamics_step_time + [1])
                if keep_intermediate_points
                else np.array(dynamics_step_time + [1])[[0, -1]]
            )

        else:
            # time is linear in the case of direct multiple shooting
            step_times = np.linspace(0, 1, ode_solver_steps + 1) if keep_intermediate_points else np.array([0, 1])
        # it does not take the last nodes of each interval
        if continuous:
            step_times = step_times[:-1]

        return step_times

    def _get_first_frame_states(self, sol, shooting_type: Shooting, phase: int) -> np.ndarray:
        """
        Get the first frame of the states for a given phase,
        according to the shooting type, the integrator and the phase of the ocp

        Parameters
        ----------
        sol: Solution
            The initial state of the phase
        shooting_type: Shooting
            The shooting type to use
        phase: int
            The phase of the ocp to consider

        Returns
        -------
        np.ndarray
            Shape is n_states x 1 if Shooting.SINGLE_CONTINUOUS or Shooting.SINGLE
            Shape is n_states x n_shooting if Shooting.MULTIPLE
        """
        # Get the first frame of the phase
        if shooting_type == Shooting.SINGLE:
            if phase == 0:
                return np.hstack([self._states["unscaled"][0][key][:, 0] for key in self.ocp.nlp[phase].states])

            x0 = np.concatenate(
                [self._states["unscaled"][phase - 1][key][:, -1] for key in self.ocp.nlp[phase - 1].states]
            )
            u0 = np.concatenate(
                [self._controls["unscaled"][phase - 1][key][:, -1] for key in self.ocp.nlp[phase - 1].controls]
            )
            if self.ocp.assume_phase_dynamics or not np.isnan(u0).any():
                u0 = vertcat(u0, u0)
            params = []
            s0 = []
            if len(self.ocp.nlp[phase - 1].stochastic_variables) > 0:
                s0 = np.concatenate(
                    [
                        self.stochastic_variables["unscaled"][phase - 1][key][:, -1]
                        for key in self.ocp.nlp[phase - 1].stochastic_variables["unscaled"]
                    ]
                )
            if self.parameters.keys():
                params = np.vstack([self.parameters[key] for key in self.parameters])
            val = self.ocp.phase_transitions[phase - 1].function[-1](vertcat(x0, x0), u0, params, s0, 0, 0)
            if val.shape[0] != x0.shape[0]:
                raise RuntimeError(
                    f"Phase transition must have the same number of states ({val.shape[0]}) "
                    f"when integrating with Shooting.SINGLE_CONTINUOUS. If it is not possible, "
                    f"please integrate with Shooting.SINGLE"
                )
            x0 += np.array(val)[:, 0]
            return x0

        elif shooting_type == Shooting.SINGLE_DISCONTINUOUS_PHASE:
            return np.vstack([self._states["unscaled"][phase][key][:, 0:1] for key in self.ocp.nlp[phase].states])[:, 0]

        elif shooting_type == Shooting.MULTIPLE:
            return np.vstack(
                [
                    (
                        self.states_no_intermediate[phase][key][:, :-1]
                        if len(self.ocp.nlp) > 1
                        else self.states_no_intermediate[key][:, :-1]
                    )
                    for key in self.ocp.nlp[phase].states
                ]
            )
        else:
            raise NotImplementedError(f"Shooting type {shooting_type} is not implemented")

    def __perform_integration(
        self,
        shooting_type: Shooting,
        keep_intermediate_points: bool,
        integrator: SolutionIntegrator,
    ):
        """
        This function performs the integration of the system dynamics
        with different options using scipy or the default integrator

        Parameters
        ----------
        shooting_type: Shooting
            Which type of integration (SINGLE_CONTINUOUS, MULTIPLE, SINGLE)
        keep_intermediate_points: bool
            If the integration should return the intermediate values of the integration
        integrator
            Use the ode solver defined by the OCP or use a separate integrator provided by scipy such as RK45 or DOP853

        Returns
        -------
        Solution
            A Solution data structure with the states integrated. The controls are removed from this structure
        """

        # Copy the data
        out = self.copy(skip_data=True)
        out.recomputed_time_steps = integrator != SolutionIntegrator.OCP
        out._states["unscaled"] = [dict() for _ in range(len(self._states["unscaled"]))]
        out._time_vector = self._generate_time(
            keep_intermediate_points=keep_intermediate_points,
            merge_phases=False,
            shooting_type=shooting_type,
        )

        params = vertcat(*[self.parameters[key] for key in self.parameters])

        for p, (nlp, t_eval) in enumerate(zip(self.ocp.nlp, out._time_vector)):
            self.ocp.nlp[p].controls.node_index = 0

            states_phase_idx = self.ocp.nlp[p].use_states_from_phase_idx
            controls_phase_idx = self.ocp.nlp[p].use_controls_from_phase_idx
            param_scaling = nlp.parameters.scaling
            x0 = self._get_first_frame_states(out, shooting_type, phase=p)
            u = (
                np.array([])
                if nlp.control_type == ControlType.NONE
                else np.concatenate(
                    [
                        self._controls["unscaled"][controls_phase_idx][key]
                        for key in self.ocp.nlp[controls_phase_idx].controls
                    ]
                )
            )
            if self.ocp.nlp[p].stochastic_variables.keys():
                s = np.concatenate(
                    [self._stochastic_variables["unscaled"][p][key] for key in self.ocp.nlp[p].stochastic_variables]
                )
            else:
                s = np.array([])
            if integrator == SolutionIntegrator.OCP:
                integrated_sol = solve_ivp_bioptim_interface(
                    dynamics_func=nlp.dynamics,
                    keep_intermediate_points=keep_intermediate_points,
                    x0=x0,
                    u=u,
                    s=s,
                    params=params,
                    param_scaling=param_scaling,
                    shooting_type=shooting_type,
                    control_type=nlp.control_type,
                )
            else:
                integrated_sol = solve_ivp_interface(
                    dynamics_func=nlp.dynamics_func,
                    keep_intermediate_points=keep_intermediate_points,
                    t_eval=t_eval[:-1] if shooting_type == Shooting.MULTIPLE else t_eval,
                    x0=x0,
                    u=u,
                    s=s,
                    params=params,
                    method=integrator.value,
                    control_type=nlp.control_type,
                )

            for key in nlp.states:
                out._states["unscaled"][states_phase_idx][key] = integrated_sol[nlp.states[key].index, :]

                if shooting_type == Shooting.MULTIPLE:
                    # last node of the phase is not integrated but do exist as an independent node
                    out._states["unscaled"][states_phase_idx][key] = np.concatenate(
                        (
                            out._states["unscaled"][states_phase_idx][key],
                            self._states["unscaled"][states_phase_idx][key][:, -1:],
                        ),
                        axis=1,
                    )

        return out

    def __perform_noisy_integration(
        self,
        shooting_type: Shooting,
        keep_intermediate_points: bool,
        integrator: SolutionIntegrator,
        n_random: int,
    ):
        """
        This function performs the integration of the system dynamics in a noisy environment
        with different options using scipy or the default integrator

        Parameters
        ----------
        shooting_type: Shooting
            Which type of integration (SINGLE_CONTINUOUS, MULTIPLE, SINGLE)
        keep_intermediate_points: bool
            If the integration should return the intermediate values of the integration
        integrator
            Use the ode solver defined by the OCP or use a separate integrator provided by scipy such as RK45 or DOP853

        Returns
        -------
        Solution
            A Solution data structure with the states integrated. The controls are removed from this structure
        """

        # Copy the data
        out = self.copy(skip_data=True)
        out.recomputed_time_steps = integrator != SolutionIntegrator.OCP
        out._states["unscaled"] = [dict() for _ in range(len(self._states["unscaled"]))]
        out._time_vector = self._generate_time(
            keep_intermediate_points=keep_intermediate_points,
            merge_phases=False,
            shooting_type=shooting_type,
        )

        params = vertcat(*[self.parameters[key] for key in self.parameters])

        for i_phase, (nlp, t_eval) in enumerate(zip(self.ocp.nlp, out._time_vector)):
            self.ocp.nlp[i_phase].controls.node_index = 0

            states_phase_idx = self.ocp.nlp[i_phase].use_states_from_phase_idx
            controls_phase_idx = self.ocp.nlp[i_phase].use_controls_from_phase_idx
            param_scaling = nlp.parameters.scaling
            x0 = self._get_first_frame_states(out, shooting_type, phase=i_phase)
            u = (
                np.array([])
                if nlp.control_type == ControlType.NONE
                else np.concatenate(
                    [
                        self._controls["unscaled"][controls_phase_idx][key]
                        for key in self.ocp.nlp[controls_phase_idx].controls
                    ]
                )
            )

            if self.ocp.nlp[i_phase].stochastic_variables.keys():
                s = np.concatenate(
                    [self._stochastic_variables[i_phase][key] for key in self.ocp.nlp[i_phase].stochastic_variables]
                )
            else:
                s = np.array([])
            if integrator == SolutionIntegrator.OCP:
                integrated_sol = solve_ivp_bioptim_interface(
                    dynamics_func=nlp.dynamics,
                    keep_intermediate_points=keep_intermediate_points,
                    x0=x0,
                    u=u,
                    s=s,
                    params=params,
                    param_scaling=param_scaling,
                    shooting_type=shooting_type,
                    control_type=nlp.control_type,
                )
            else:
                integrated_sol = solve_ivp_interface(
                    dynamics_func=nlp.dynamics_func,
                    keep_intermediate_points=keep_intermediate_points,
                    t_eval=t_eval[:-1] if shooting_type == Shooting.MULTIPLE else t_eval,
                    x0=x0,
                    u=u,
                    s=s,
                    params=params,
                    method=integrator.value,
                    control_type=nlp.control_type,
                )

            for key in nlp.states:
                out._states["unscaled"][states_phase_idx][key] = integrated_sol[nlp.states[key].index, :]

                if shooting_type == Shooting.MULTIPLE:
                    # last node of the phase is not integrated but do exist as an independent node
                    out._states["unscaled"][states_phase_idx][key] = np.concatenate(
                        (
                            out._states["unscaled"][states_phase_idx][key],
                            self._states["unscaled"][states_phase_idx][key][:, -1:],
                        ),
                        axis=1,
                    )

        return out

    def interpolate(self, n_frames: int | list | tuple) -> Any:
        """
        Interpolate the states

        Parameters
        ----------
        n_frames: int | list | tuple
            If the value is an int, the Solution returns merges the phases,
            otherwise, it interpolates them independently

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        out = self.copy(skip_data=True)

        t_all = []
        for p, data in enumerate(self._states["unscaled"]):
            nlp = self.ocp.nlp[p]
            if nlp.ode_solver.is_direct_collocation and not self.recomputed_time_steps:
                time_offset = sum(out.phase_time[: p + 1])
                step_time = np.array(nlp.dynamics[0].step_time)
                dt = out.phase_time[p + 1] / nlp.ns
                t_tp = np.array([step_time * dt + s * dt + time_offset for s in range(nlp.ns)]).reshape(-1, 1)
                t_all.append(np.concatenate((t_tp, [[t_tp[-1, 0]]]))[:, 0])
            else:
                t_all.append(np.linspace(sum(out.phase_time[: p + 1]), sum(out.phase_time[: p + 2]), out.ns[p] + 1))

        if isinstance(n_frames, int):
            _, data_states, _, _, _, _, out.phase_time, out.ns = self._merge_phases(skip_controls=True)
            t_all = [np.concatenate((np.concatenate([_t[:-1] for _t in t_all]), [t_all[-1][-1]]))]
            n_frames = [n_frames]
            out.is_merged = True
        elif isinstance(n_frames, (list, tuple)) and len(n_frames) == len(self._states["unscaled"]):
            data_states = self._states["unscaled"]
        else:
            raise ValueError(
                "n_frames should either be a int to merge_phases phases "
                "or a list of int of the number of phases dimension"
            )

        out._states["unscaled"] = []
        for _ in range(len(data_states)):
            out._states["unscaled"].append({})
        for p in range(len(data_states)):
            for key in self.ocp.nlp[0].states:
                x_phase = data_states[p][key]
                n_elements = x_phase.shape[0]

                t_phase = t_all[p]
                t_phase, time_index = np.unique(t_phase, return_index=True)
                t_int = np.linspace(t_phase[0], t_phase[-1], n_frames[p])
                x_interpolate = np.ndarray((n_elements, n_frames[p]))
                for j in range(n_elements):
                    s = sci_interp.splrep(t_phase, x_phase[j, time_index], k=1)
                    x_interpolate[j, :] = sci_interp.splev(t_int, s)
                out._states["unscaled"][p][key] = x_interpolate

        out.is_interpolated = True
        return out

    def merge_phases(self) -> Any:
        """
        Get a data structure where all the phases are merged into one

        Returns
        -------
        The new data structure with the phases merged
        """

        new = self.copy(skip_data=True)
        new.parameters = deepcopy(self.parameters)
        (
            new._states["scaled"],
            new._states["unscaled"],
            new._controls["scaled"],
            new._controls["unscaled"],
            new._stochastic_variables["scaled"],
            new._stochastic_variables["unscaled"],
            new.phase_time,
            new.ns,
        ) = self._merge_phases()
        new._time_vector = [np.array(concatenate_optimization_variables(self._time_vector))]
        new.is_merged = True
        return new

    def _merge_phases(
        self,
        skip_states: bool = False,
        skip_controls: bool = False,
        skip_stochastic: bool = False,
        continuous: bool = True,
    ) -> tuple:
        """
        Actually performing the phase merging

        Parameters
        ----------
        skip_states: bool
            If the merge should ignore the states
        skip_controls: bool
            If the merge should ignore the controls
        continuous: bool
            If the last frame of each phase should be kept [False] or discard [True]

        Returns
        -------
        A tuple containing the new states, new controls, the recalculated phase time
        and the new number of shooting points
        """

        if self.is_merged:
            return (
                deepcopy(self._states["scaled"]),
                deepcopy(self._states["unscaled"]),
                deepcopy(self._controls["scaled"]),
                deepcopy(self._controls["unscaled"]),
                deepcopy(self._stochastic_variables["scaled"]),
                deepcopy(self._stochastic_variables["unscaled"]),
                deepcopy(self.phase_time),
                deepcopy(self.ns),
            )

        def _merge(data: list, is_control: bool) -> list | dict:
            """
            Merge the phases of a states or controls data structure

            Parameters
            ----------
            data: list
                The data to structure to merge the phases
            is_control: bool
                If the current data is a control

            Returns
            -------
            The data merged
            """

            if isinstance(data, dict):
                return data

            # Sanity check (all phases must contain the same keys with the same dimensions)
            keys = data[0].keys()
            sizes = [data[0][d].shape[0] for d in data[0]]
            for d in data:
                if d.keys() != keys or [d[key].shape[0] for key in d] != sizes:
                    raise RuntimeError("Program dimension must be coherent across phases to merge_phases them")

            data_out = [{}]
            for i, key in enumerate(keys):
                data_out[0][key] = np.ndarray((sizes[i], 0))

            add = 0 if is_control or continuous else 1
            for p in range(len(data)):
                d = data[p]
                for key in d:
                    if self.ocp.nlp[p].ode_solver.is_direct_collocation and not is_control:
                        steps = self.ocp.nlp[p].ode_solver.steps + 1
                        data_out[0][key] = np.concatenate(
                            (data_out[0][key], d[key][:, : self.ns[p] * steps + add]), axis=1
                        )
                    else:
                        data_out[0][key] = np.concatenate((data_out[0][key], d[key][:, : self.ns[p] + add]), axis=1)
            if add == 0:
                for key in data[-1]:
                    data_out[0][key] = np.concatenate((data_out[0][key], data[-1][key][:, -1][:, np.newaxis]), axis=1)

            return data_out

        if len(self._states["scaled"]) == 1:
            out_states_scaled = deepcopy(self._states["scaled"])
            out_states = deepcopy(self._states["unscaled"])
        else:
            out_states_scaled = (
                _merge(self._states["scaled"], is_control=False) if not skip_states and self._states["scaled"] else None
            )
            out_states = _merge(self._states["unscaled"], is_control=False) if not skip_states else None

        if len(self._controls["scaled"]) == 1:
            out_controls_scaled = deepcopy(self._controls["scaled"])
            out_controls = deepcopy(self._controls["unscaled"])
        else:
            out_controls_scaled = (
                _merge(self._controls["scaled"], is_control=True)
                if not skip_controls and self._controls["scaled"]
                else None
            )
            out_controls = _merge(self._controls["unscaled"], is_control=True) if not skip_controls else None
        phase_time = [0] + [sum([self.phase_time[i + 1] for i in range(len(self.phase_time) - 1)])]
        ns = [sum(self.ns)]

        if len(self._stochastic_variables["scaled"]) == 1:
            out_stochastic_variables_scaled = deepcopy(self._stochastic_variables["scaled"])
            out_stochastic_variables = deepcopy(self._stochastic_variables["unscaled"])
        else:
            out_stochastic_variables_scaled = (
                _merge(self._stochastic_variables["scaled"], is_control=False)
                if not skip_stochastic and self._stochastic_variables["scaled"]
                else None
            )
            out_stochastic_variables = (
                _merge(self._stochastic_variables["unscaled"], is_control=False) if not skip_stochastic else None
            )

        return (
            out_states_scaled,
            out_states,
            out_controls_scaled,
            out_controls,
            out_stochastic_variables_scaled,
            out_stochastic_variables,
            phase_time,
            ns,
        )

    def _complete_control(self):
        """
        Controls don't necessarily have dimensions that matches the states. This method aligns them
        """

        for p, nlp in enumerate(self.ocp.nlp):
            if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
                for key in self._controls["scaled"][p]:
                    self._controls["scaled"][p][key] = np.concatenate(
                        (
                            self._controls["scaled"][p][key],
                            np.nan * np.zeros((self._controls["scaled"][p][key].shape[0], 1)),
                        ),
                        axis=1,
                    )
                    self._controls["unscaled"][p][key] = np.concatenate(
                        (
                            self._controls["unscaled"][p][key],
                            np.nan * np.zeros((self._controls["unscaled"][p][key].shape[0], 1)),
                        ),
                        axis=1,
                    )
            elif nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                pass
            else:
                raise NotImplementedError(f"ControlType {nlp.control_type} is not implemented  in _complete_control")

    def graphs(
        self,
        automatically_organize: bool = True,
        show_bounds: bool = False,
        show_now: bool = True,
        shooting_type: Shooting = Shooting.MULTIPLE,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
    ):
        """
        Show the graphs of the simulation

        Parameters
        ----------
        automatically_organize: bool
            If the figures should be spread on the screen automatically
        show_bounds: bool
            If the plot should adapt to bounds (True) or to data (False)
        show_now: bool
            If the show method should be called. This is blocking
        shooting_type: Shooting
            The type of interpolation
        integrator: SolutionIntegrator
            Use the scipy solve_ivp integrator for RungeKutta 45 instead of currently defined integrator
        """

        if self.is_merged or self.is_interpolated or self.is_integrated:
            raise NotImplementedError("It is not possible to graph a modified Solution yet")

        plot_ocp = self.ocp.prepare_plots(automatically_organize, show_bounds, shooting_type, integrator)
        plot_ocp.update_data(self.vector)
        if show_now:
            plt.show()

    def animate(
        self,
        n_frames: int = 0,
        shooting_type: Shooting = None,
        show_now: bool = True,
        show_tracked_markers: bool = False,
        **kwargs: Any,
    ) -> None | list:
        """
        Animate the simulation

        Parameters
        ----------
        n_frames: int
            The number of frames to interpolate to. If the value is 0, the data are merged to a one phase if possible.
            If the value is -1, the data is not merge in one phase
        shooting_type: Shooting
            The Shooting type to animate
        show_now: bool
            If the bioviz exec() function should be called automatically. This is blocking method
        show_tracked_markers: bool
            If the tracked markers should be displayed
        kwargs: Any
            Any parameters to pass to bioviz

        Returns
        -------
            A list of bioviz structures (one for each phase). So one can call exec() by hand
        """

        data_to_animate = self.integrate(shooting_type=shooting_type) if shooting_type else self.copy()
        if n_frames == 0:
            try:
                data_to_animate = data_to_animate.interpolate(sum(self.ns) + 1)
            except RuntimeError:
                pass

        elif n_frames > 0:
            data_to_animate = data_to_animate.interpolate(n_frames)

        if show_tracked_markers and len(self.ocp.nlp) == 1:
            tracked_markers = self._prepare_tracked_markers_for_animation(n_shooting=n_frames)
        elif show_tracked_markers and len(self.ocp.nlp) > 1:
            raise NotImplementedError(
                "Tracking markers is not implemented for multiple phases. "
                "Set show_tracked_markers to False such that sol.animate(show_tracked_markers=False)."
            )
        else:
            tracked_markers = [None for _ in range(len(self.ocp.nlp))]

        # assuming that all the models or the same type.
        self._check_models_comes_from_same_super_class()
        self.ocp.nlp[0].model.animate(
            solution=data_to_animate,
            show_now=show_now,
            tracked_markers=tracked_markers,
            **kwargs,
        )

    def _check_models_comes_from_same_super_class(self):
        """Check that all the models comes from the same super class"""
        for i, nlp in enumerate(self.ocp.nlp):
            model_super_classes = nlp.model.__class__.mro()[:-1]  # remove object class
            nlps = self.ocp.nlp.copy()
            del nlps[i]
            for j, sub_nlp in enumerate(nlps):
                if not any([isinstance(sub_nlp.model, super_class) for super_class in model_super_classes]):
                    raise RuntimeError(
                        f"The animation is only available for compatible models. "
                        f"Here, the model of phase {i} is of type {nlp.model.__class__.__name__} and the model of "
                        f"phase {j + 1 if i < j else j} is of type {sub_nlp.model.__class__.__name__} and "
                        f"they don't share the same super class."
                    )

    def _prepare_tracked_markers_for_animation(self, n_shooting: int = None) -> list:
        """Prepare the markers which are tracked to the animation"""

        n_frames = sum(self.ns) + 1 if n_shooting is None else n_shooting + 1

        all_tracked_markers = []

        for phase, nlp in enumerate(self.ocp.nlp):
            tracked_markers = None
            for objective in nlp.J:
                if objective.target is not None:
                    if objective.type in (
                        ObjectiveFcn.Mayer.TRACK_MARKERS,
                        ObjectiveFcn.Lagrange.TRACK_MARKERS,
                    ) and objective.node[0] in (Node.ALL, Node.ALL_SHOOTING):
                        tracked_markers = np.full((3, nlp.model.nb_markers, self.ns[phase] + 1), np.nan)
                        for i in range(len(objective.rows)):
                            tracked_markers[objective.rows[i], objective.cols, :] = objective.target[0][i, :, :]
                        missing_row = np.where(np.isnan(tracked_markers))[0]
                        if missing_row.size > 0:
                            tracked_markers[missing_row, :, :] = 0

            # interpolation
            if n_frames > 0 and tracked_markers is not None:
                x = np.linspace(0, self.ns[phase], self.ns[phase] + 1)
                xnew = np.linspace(0, self.ns[phase], n_frames)
                f = interp1d(x, tracked_markers, kind="cubic")
                tracked_markers = f(xnew)

            all_tracked_markers.append(tracked_markers)

        return all_tracked_markers

    def _get_penalty_cost(self, nlp, penalty):
        phase_idx = nlp.phase_idx
        steps = nlp.ode_solver.steps + 1 if nlp.ode_solver.is_direct_collocation else 1
        nlp.controls.node_index = 0

        val = []
        val_weighted = []
        p = vertcat(*[self.parameters[key] / self.ocp.parameters[key].scaling for key in self.parameters.keys()])

        dt = (
            Function("time", [nlp.parameters.cx], [penalty.dt])(self.parameters["time"])
            if "time" in self.parameters
            else penalty.dt
        )

        for idx in penalty.node_idx:
            x = []
            u = []
            s = []
            target = []
            if nlp is not None:
                if penalty.transition:

                    _x_0 = np.array(())
                    _u_0 = np.array(())
                    _s_0 = np.array(())
                    for key in nlp.states:
                        _x_0 = np.concatenate((_x_0, self._states["scaled"][penalty.nodes_phase[0]][key][:, penalty.multinode_idx[0]]))
                    for key in nlp.controls:
                        # Make an exception to the fact that U is not available for the last node
                        _u_0 = np.concatenate((_u_0, self._controls["scaled"][penalty.nodes_phase[0]][key][:, penalty.multinode_idx[0]]))
                    for key in nlp.stochastic_variables:
                        _s_0 = np.concatenate((_s_0, self._stochastic_variables["scaled"][penalty.nodes_phase[0]][key][:, penalty.multinode_idx[0]]))

                    _x_1 = np.array(())
                    _u_1 = np.array(())
                    _s_1 = np.array(())
                    for key in nlp.states:
                        _x_1 = np.concatenate((_x_1, self._states["scaled"][penalty.nodes_phase[1]][key][:,
                                                     penalty.multinode_idx[1]]))
                    for key in nlp.controls:
                        # Make an exception to the fact that U is not available for the last node
                        _u_1 = np.concatenate((_u_1, self._controls["scaled"][penalty.nodes_phase[1]][key][:,
                                                     penalty.multinode_idx[1]]))
                    for key in nlp.stochastic_variables:
                        _s_1 = np.concatenate((_s_1,
                                               self._stochastic_variables["scaled"][penalty.nodes_phase[1]][key][:,
                                               penalty.multinode_idx[1]]))

                    x = np.hstack((_x_0, _x_1))
                    u = np.hstack((_u_0, _u_1))
                    s = np.hstack((_s_0, _s_1))

                elif penalty.multinode_penalty:
                    x = np.array(())
                    u = np.array(())
                    s = np.array(())
                    for i in range(len(penalty.nodes_phase)):
                        node_idx = penalty.multinode_idx[i]
                        phase_idx = penalty.nodes_phase[i]

                        _x = np.array(())
                        _u = np.array(())
                        _s = np.array(())
                        for key in nlp.states:
                            _x = np.concatenate((_x, self._states["scaled"][phase_idx][key][:, node_idx]))
                        for key in nlp.controls:
                            # Make an exception to the fact that U is not available for the last node
                            _u = np.concatenate((_u, self._controls["scaled"][phase_idx][key][:, node_idx]))
                        for key in nlp.stochastic_variables:
                            _s = np.concatenate((_s, self._stochastic_variables["scaled"][phase_idx][key][:, node_idx]))
                        x = np.vstack((x, _x)) if x.size else _x
                        u = np.vstack((u, _u)) if u.size else _u
                        s = np.vstack((s, _s)) if s.size else _s
                    x = x.T
                    u = u.T
                    s = s.T
                elif (
                    "Lagrange" not in penalty.type.__str__()
                    and "Mayer" not in penalty.type.__str__()
                    and "MultinodeObjectiveFcn" not in penalty.type.__str__()
                    and "ConstraintFcn" not in penalty.type.__str__()
                ):
                    if penalty.target is not None:
                        target = penalty.target[0]
                else:
                    if penalty.integrate or nlp.ode_solver.is_direct_collocation:
                        if idx != nlp.ns:
                            col_x_idx = list(range(idx * steps, (idx + 1) * steps))
                        else:
                            col_x_idx = [idx * steps]
                    else:
                        col_x_idx = [idx]
                    col_u_idx = [idx]
                    col_s_idx = [idx]
                    if (
                        penalty.derivative
                        or penalty.explicit_derivative
                        or penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                    ):
                        col_x_idx.append((idx + 1) * (steps if nlp.ode_solver.is_direct_shooting else 1))

                        if (
                            penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                        ) or nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                            col_u_idx.append((idx + 1))
                    elif penalty.integration_rule == QuadratureRule.TRAPEZOIDAL:
                        if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                            col_u_idx.append((idx + 1))

                    x = np.ndarray((nlp.states.shape, len(col_x_idx)))
                    for key in nlp.states:
                        x[nlp.states[key].index, :] = self._states["scaled"][phase_idx][key][:, col_x_idx]

                    u = np.ndarray((nlp.controls.shape, len(col_u_idx)))
                    for key in nlp.controls:
                        u[nlp.controls[key].index, :] = (
                            []
                            if nlp.control_type == ControlType.NONE
                            else self._controls["scaled"][phase_idx][key][:, col_u_idx]
                        )

                    s = np.ndarray((nlp.stochastic_variables.shape, len(col_s_idx)))
                    for key in nlp.stochastic_variables:
                        s[nlp.stochastic_variables[key].index, :] = self._stochastic_variables["scaled"][phase_idx][
                            key
                        ][:, col_s_idx]

                    if penalty.target is None:
                        target = []
                    elif (
                        penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                        or penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
                    ):
                        target = np.vstack(
                            (
                                penalty.target[0][:, penalty.node_idx.index(idx)],
                                penalty.target[1][:, penalty.node_idx.index(idx)],
                            )
                        ).T
                    else:
                        target = penalty.target[0][..., penalty.node_idx.index(idx)]

            # Deal with final node which sometime is nan (meaning it should be removed to fit the dimensions of the
            # casadi function
            if not self.ocp.assume_phase_dynamics and ((isinstance(u, list) and u != []) or isinstance(u, np.ndarray)):
                u = u[:, ~np.isnan(np.sum(u, axis=0))]
            if (
                penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                or penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
            ):
                val.append(penalty.function[idx](x[:, 0], u[:, 0], p, s[:, 0], 0, 0))
            else:
                x_reshaped = x.T.reshape((-1, 1)) if len(x.shape) > 1 and x.shape[1] != 1 else x
                u_reshaped = u.T.reshape((-1, 1)) if len(u.shape) > 1 and u.shape[1] != 1 else u
                s_reshaped = s.T.reshape((-1, 1)) if len(s.shape) > 1 and s.shape[1] != 1 else s
                val.append(penalty.function[idx](x_reshaped, u_reshaped, p, s_reshaped, 0, 0))

            x_reshaped = x.T.reshape((-1, 1)) if len(x.shape) > 1 and x.shape[1] != 1 else x
            u_reshaped = u.T.reshape((-1, 1)) if len(u.shape) > 1 and u.shape[1] != 1 else u
            s_reshaped = s.T.reshape((-1, 1)) if len(s.shape) > 1 and s.shape[1] != 1 else s
            val_weighted.append(
                penalty.weighted_function[idx](x_reshaped, u_reshaped, p, s_reshaped, 0, 0, penalty.weight, target, dt)
            )

        val = np.nansum(val)
        val_weighted = np.nansum(val_weighted)

        return val, val_weighted

    @property
    def detailed_cost(self):
        if self._detailed_cost is None:
            self._compute_detailed_cost()
        return self._detailed_cost

    def _compute_detailed_cost(self):
        """
        Adds the detailed objective functions and/or constraints values to sol

        Parameters
        ----------
        """
        if self.ocp.n_threads > 1:
            raise NotImplementedError("Computing detailed cost with n_thread > 1 is not implemented yet")

        self._detailed_cost = []

        for nlp in self.ocp.nlp:
            for penalty in nlp.J_internal + nlp.J:
                if not penalty:
                    continue
                val, val_weighted = self._get_penalty_cost(nlp, penalty)
                self._detailed_cost += [
                    {"name": penalty.type.__str__(), "cost_value_weighted": val_weighted, "cost_value": val}
                ]
        for penalty in self.ocp.J:
            val, val_weighted = self._get_penalty_cost(self.ocp.nlp[0], penalty)
            self._detailed_cost += [
                {"name": penalty.type.__str__(), "cost_value_weighted": val_weighted, "cost_value": val}
            ]
        return

    def print_cost(self, cost_type: CostType = CostType.ALL):
        """
        Print the objective functions and/or constraints to the console

        Parameters
        ----------
        cost_type: CostType
            The type of cost to console print
        """

        def print_penalty_list(nlp, penalties, print_only_weighted):
            running_total = 0

            for penalty in penalties:
                if not penalty:
                    continue

                val, val_weighted = self._get_penalty_cost(nlp, penalty)
                running_total += val_weighted

                if penalty.node in [Node.MULTINODES, Node.TRANSITION]:
                    node_name = penalty.node.name
                else:
                    node_name = f"{penalty.node[0]}" if isinstance(penalty.node[0], int) else penalty.node[0].name

                if self._detailed_cost is not None:
                    self._detailed_cost += [
                        {
                            "name": penalty.type.__str__(),
                            "penalty": penalty.type.__str__().split(".")[0],
                            "function": penalty.name,
                            "cost_value_weighted": val_weighted,
                            "cost_value": val,
                            "params": penalty.params,
                            "derivative": penalty.derivative,
                            "explicit_derivative": penalty.explicit_derivative,
                            "integration_rule": penalty.integration_rule.name,
                            "weight": penalty.weight,
                            "expand": penalty.expand,
                            "node": node_name,
                        }
                    ]
                if print_only_weighted:
                    print(f"{penalty.type}: {val_weighted}")
                else:
                    print(f"{penalty.type}: {val_weighted} (non weighted {val: .2f})")

            return running_total

        def print_objective_functions(ocp):
            """
            Print the values of each objective function to the console
            """
            print(f"\n---- COST FUNCTION VALUES ----")
            running_total = print_penalty_list(None, ocp.J_internal, False)
            running_total += print_penalty_list(None, ocp.J, False)
            if running_total:
                print("")

            for nlp in ocp.nlp:
                print(f"PHASE {nlp.phase_idx}")
                running_total += print_penalty_list(nlp, nlp.J_internal, False)
                running_total += print_penalty_list(nlp, nlp.J, False)
                print("")

            print(f"Sum cost functions: {running_total}")
            print(f"------------------------------")

        def print_constraints(ocp, sol):
            """
            Print the values of each constraint with its lagrange multiplier to the console
            """

            if sol.constraints is None:
                return

            # Todo, min/mean/max
            print(f"\n--------- CONSTRAINTS ---------")
            if (
                print_penalty_list(None, ocp.g_internal, True)
                + print_penalty_list(None, ocp.g_implicit, True)
                + print_penalty_list(None, ocp.g, True)
            ):
                print("")

            for idx_phase, nlp in enumerate(ocp.nlp):
                print(f"PHASE {idx_phase}")
                print_penalty_list(nlp, nlp.g_internal, True)
                print_penalty_list(nlp, nlp.g_implicit, True)
                print_penalty_list(nlp, nlp.g, True)
                print("")
            print(f"------------------------------")

        if cost_type == CostType.OBJECTIVES:
            print_objective_functions(self.ocp)
        elif cost_type == CostType.CONSTRAINTS:
            print_constraints(self.ocp, self)
        elif cost_type == CostType.ALL:
            print(
                f"Solver reported time: {self.solver_time_to_optimize} sec\n"
                f"Real time: {self.real_time_to_optimize} sec"
            )
            self.print_cost(CostType.OBJECTIVES)
            self.print_cost(CostType.CONSTRAINTS)
        else:
            raise ValueError("print can only be called with CostType.OBJECTIVES or CostType.CONSTRAINTS")


def concatenate_optimization_variables_dict(
    variable: list[dict[np.ndarray]], continuous: bool = True
) -> list[dict[np.ndarray]]:
    """
    This function concatenates the decision variables of the phases of the system
    into a single array, omitting the last element of each phase except for the last one.

    Parameters
    ----------
    variable : list or dict
        list of decision variables of the phases of the system
    continuous: bool
        If the arrival value of a node should be discarded [True] or kept [False].

    Returns
    -------
    z_concatenated : np.ndarray or dict
        array of the decision variables of the phases of the system concatenated
    """
    if isinstance(variable, list):
        if isinstance(variable[0], dict):
            variable_dict = dict()
            for key in variable[0].keys():
                variable_dict[key] = [v_i[key] for v_i in variable]
                final_tuple = [
                    y[:, :-1] if i < (len(variable_dict[key]) - 1) and continuous else y
                    for i, y in enumerate(variable_dict[key])
                ]
                variable_dict[key] = np.hstack(final_tuple)
            return [variable_dict]
    else:
        raise ValueError("the input must be a list")


def concatenate_optimization_variables(
    variable: list[np.ndarray] | np.ndarray,
    continuous_phase: bool = True,
    continuous_interval: bool = True,
    merge_phases: bool = True,
) -> np.ndarray | list[dict[np.ndarray]]:
    """
    This function concatenates the decision variables of the phases of the system
    into a single array, omitting the last element of each phase except for the last one.

    Parameters
    ----------
    variable : list or dict
        list of decision variables of the phases of the system
    continuous_phase: bool
        If the arrival value of a node should be discarded [True] or kept [False]. The value of an integrated
    continuous_interval: bool
        If the arrival value of a node of each interval should be discarded [True] or kept [False].
        Only useful in direct multiple shooting
    merge_phases: bool
        If the decision variables of each phase should be merged into a single array [True] or kept separated [False].

    Returns
    -------
    z_concatenated : np.ndarray or dict
        array of the decision variables of the phases of the system concatenated
    """
    if len(variable[0].shape):
        if isinstance(variable[0][0], np.ndarray):
            z_final = []
            for zi in variable:
                z_final.append(concatenate_optimization_variables(zi, continuous_interval))

            if merge_phases:
                return concatenate_optimization_variables(z_final, continuous_phase)
            else:
                return z_final
        else:
            final_tuple = []
            for i, y in enumerate(variable):
                if i < (len(variable) - 1) and continuous_phase:
                    final_tuple.append(y[:, :-1] if len(y.shape) == 2 else y[:-1])
                else:
                    final_tuple.append(y)

        return np.hstack(final_tuple)
