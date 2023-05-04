from typing import Any
from copy import deepcopy

import numpy as np
from scipy import interpolate as sci_interp
from casadi import vertcat, DM, Function
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
    IntegralApproximation,
)
from ..misc.utils import check_version
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.optimization_variable import (
    OptimizationVariableList,
    OptimizationVariable,
    OptimizationVariableContainer,
)
from ..dynamics.ode_solver import OdeSolver
from ..interfaces.solve_ivp_interface import solve_ivp_interface, solve_ivp_bioptim_interface


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

            self.phase_idx = nlp.phase_idx
            self.use_states_from_phase_idx = nlp.use_states_from_phase_idx
            self.use_controls_from_phase_idx = nlp.use_controls_from_phase_idx
            self.model = nlp.model
            self.states = nlp.states
            self.controls = nlp.controls
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
        v: OptimizationVector
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
            self.v = ocp.v
            self.J = ocp.J
            self.J_internal = ocp.J_internal
            self.g = ocp.g
            self.g_internal = ocp.g_internal
            self.g_implicit = ocp.g_implicit
            self.phase_transitions = ocp.phase_transitions
            self.prepare_plots = ocp.prepare_plots

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
        self.detailed_cost = []

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
        self.phase_time = []

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
            self._states["scaled"], self._controls["scaled"], self.parameters = self.ocp.v.to_dictionaries(self.vector)
            self._states["unscaled"], self._controls["unscaled"] = self._to_unscaled_values(
                self._states["scaled"], self._controls["scaled"]
            )
            self._complete_control()
            self.phase_time = self.ocp.v.extract_phase_time(self.vector)
            self._time_vector = self._generate_time()

        def init_from_initial_guess(_sol: list):
            """
            Initialize all the attributes from a list of initial guesses (states, controls)

            Parameters
            ----------
            _sol: list
                The list of initial guesses
            """

            n_param = len(ocp.v.parameters_in_list)

            # Sanity checks
            for i in range(len(_sol)):  # Convert to list if necessary and copy for as many phases there are
                if isinstance(_sol[i], InitialGuess):
                    tp = InitialGuessList()
                    for _ in range(len(self.ns)):
                        tp.add(deepcopy(_sol[i].init), interpolation=_sol[i].init.type)
                    _sol[i] = tp
            if sum([isinstance(s, InitialGuessList) for s in _sol]) != 2:
                raise ValueError(
                    "solution must be a solution dict, "
                    "an InitialGuess[List] of len 2 or 3 (states, controls, parameters), "
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
            sol_states, sol_controls = _sol[0], _sol[1]
            for p, s in enumerate(sol_states):
                ns = self.ocp.nlp[p].ns + 1 if s.init.type != InterpolationType.EACH_FRAME else self.ocp.nlp[p].ns
                s.init.check_and_adjust_dimensions(
                    self.ocp.nlp[p].states[0]["scaled"].shape, ns, "states"
                )  # TODO: [0] to [node_index]
                for i in range(self.ns[p] + 1):
                    self.vector = np.concatenate((self.vector, s.init.evaluate_at(i)[:, np.newaxis]))
            for p, s in enumerate(sol_controls):
                control_type = self.ocp.nlp[p].control_type
                if control_type == ControlType.CONSTANT:
                    off = 0
                elif control_type == ControlType.LINEAR_CONTINUOUS:
                    off = 1
                else:
                    raise NotImplementedError(f"control_type {control_type} is not implemented in Solution")
                s.init.check_and_adjust_dimensions(
                    self.ocp.nlp[p].controls[0]["scaled"].shape, self.ns[p], "controls"
                )  # TODO: [0] to [node_index]
                for i in range(self.ns[p] + off):
                    self.vector = np.concatenate((self.vector, s.init.evaluate_at(i)[:, np.newaxis]))

            if n_param:
                sol_params = _sol[2]
                for p, s in enumerate(sol_params):
                    self.vector = np.concatenate((self.vector, np.repeat(s.init, self.ns[p] + 1)[:, np.newaxis]))

            self._states["scaled"], self._controls["scaled"], self.parameters = self.ocp.v.to_dictionaries(self.vector)
            self._states["unscaled"], self._controls["unscaled"] = self._to_unscaled_values(
                self._states["scaled"], self._controls["scaled"]
            )
            self._complete_control()
            self.phase_time = self.ocp.v.extract_phase_time(self.vector)
            self._time_vector = self._generate_time()

        def init_from_vector(_sol: np.ndarray | DM):
            """
            Initialize all the attributes from a vector of solution

            Parameters
            ----------
            _sol: np.ndarray | DM
                The solution in vector format
            """

            self.vector = _sol
            self._states["scaled"], self._controls["scaled"], self.parameters = self.ocp.v.to_dictionaries(self.vector)
            self._states["unscaled"], self._controls["unscaled"] = self._to_unscaled_values(
                self._states["scaled"], self._controls["scaled"]
            )
            self._complete_control()
            self.phase_time = self.ocp.v.extract_phase_time(self.vector)

        if isinstance(sol, dict):
            init_from_dict(sol)
        elif isinstance(sol, (list, tuple)) and len(sol) in (2, 3):
            init_from_initial_guess(sol)
        elif isinstance(sol, (np.ndarray, DM)):
            init_from_vector(sol)
        elif sol is None:
            self.ns = []
        else:
            raise ValueError("Solution called with unknown initializer")

    def _to_unscaled_values(self, states_scaled, controls_scaled) -> tuple:
        """
        Convert values of scaled solution to unscaled values
        """

        ocp = self.ocp

        states = [{} for _ in range(len(states_scaled))]
        controls = [{} for _ in range(len(states_scaled))]
        for phase in range(len(states_scaled)):
            states[phase] = {}
            controls[phase] = {}
            for key, value in states_scaled[phase].items():
                states[phase][key] = value * ocp.nlp[phase].x_scaling[key].to_array(
                    states_scaled[phase][key].shape[0], states_scaled[phase][key].shape[1]
                )
            for key, value in controls_scaled[phase].items():
                controls[phase][key] = value * ocp.nlp[phase].u_scaling[key].to_array(
                    controls_scaled[phase][key].shape[0], controls_scaled[phase][key].shape[1]
                )

        return states, controls

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
            new._states["unscaled"], new._controls["unscaled"] = [], []
            new._states["scaled"], new._controls["scaled"], new.parameters = [], [], {}
        else:
            new._states["scaled"] = deepcopy(self._states["scaled"])
            new._controls["scaled"] = deepcopy(self._controls["scaled"])
            new.parameters = deepcopy(self.parameters)
            new._states["unscaled"] = deepcopy(self._states["unscaled"])
            new._controls["unscaled"] = deepcopy(self._controls["unscaled"])

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
    def time(self) -> list | dict:
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
            if phase != 0:
                x0 = sol._states["unscaled"][phase - 1]["all"][:, -1]  # the last node of the previous phase
                u0 = self._controls["unscaled"][phase - 1]["all"][:, -1]
                params = self.parameters["all"]
                val = self.ocp.phase_transitions[phase - 1].function(vertcat(x0, x0), vertcat(u0, u0), params)
                if val.shape[0] != x0.shape[0]:
                    raise RuntimeError(
                        f"Phase transition must have the same number of states ({val.shape[0]}) "
                        f"when integrating with Shooting.SINGLE_CONTINUOUS. If it is not possible, "
                        f"please integrate with Shooting.SINGLE"
                    )
                x0 += np.array(val)[:, 0]
                return x0
            else:
                return self._states["unscaled"][phase]["all"][:, 0]

        elif shooting_type == Shooting.SINGLE_DISCONTINUOUS_PHASE:
            return self._states["unscaled"][phase]["all"][:, 0]

        elif shooting_type == Shooting.MULTIPLE:
            return (
                self.states_no_intermediate[phase]["all"][:, :-1]
                if len(self.ocp.nlp) > 1
                else self.states_no_intermediate["all"][:, :-1]
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

        params = self.parameters["all"]

        for p, (nlp, t_eval) in enumerate(zip(self.ocp.nlp, out._time_vector)):
            states_phase_idx = self.ocp.nlp[p].use_states_from_phase_idx
            controls_phase_idx = self.ocp.nlp[p].use_controls_from_phase_idx
            param_scaling = nlp.parameters.scaling
            x0 = self._get_first_frame_states(out, shooting_type, phase=p)
            u = (
                np.array([])
                if nlp.control_type == ControlType.NONE
                else self._controls["unscaled"][controls_phase_idx]["all"]
            )
            if integrator != SolutionIntegrator.OCP:
                out._states["unscaled"][states_phase_idx]["all"] = solve_ivp_interface(
                    dynamics_func=nlp.dynamics_func,
                    keep_intermediate_points=keep_intermediate_points,
                    t_eval=t_eval[:-1] if shooting_type == Shooting.MULTIPLE else t_eval,
                    x0=x0,
                    u=u,
                    params=params,
                    method=integrator.value,
                    control_type=nlp.control_type,
                )

            else:
                out._states["unscaled"][states_phase_idx]["all"] = solve_ivp_bioptim_interface(
                    dynamics_func=nlp.dynamics,
                    keep_intermediate_points=keep_intermediate_points,
                    x0=x0,
                    u=u,
                    params=params,
                    param_scaling=param_scaling,
                    shooting_type=shooting_type,
                    control_type=nlp.control_type,
                )

            if shooting_type == Shooting.MULTIPLE:
                # last node of the phase is not integrated but do exist as an independent node
                out._states["unscaled"][states_phase_idx]["all"] = np.concatenate(
                    (
                        out._states["unscaled"][states_phase_idx]["all"],
                        self._states["unscaled"][states_phase_idx]["all"][:, -1:],
                    ),
                    axis=1,
                )

            # Dispatch the integrated values to all the keys
            for key in nlp.states[0]:  # TODO: [0] to [node_index]
                out._states["unscaled"][states_phase_idx][key] = out._states["unscaled"][states_phase_idx]["all"][
                    nlp.states[0][key].index, :  # TODO: [0] to [node_index]
                ]

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
            _, data_states, _, _, out.phase_time, out.ns = self._merge_phases(skip_controls=True)
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
            x_phase = data_states[p]["all"]
            n_elements = x_phase.shape[0]

            t_phase = t_all[p]
            t_phase, time_index = np.unique(t_phase, return_index=True)
            t_int = np.linspace(t_phase[0], t_phase[-1], n_frames[p])
            x_interpolate = np.ndarray((n_elements, n_frames[p]))
            for j in range(n_elements):
                s = sci_interp.splrep(t_phase, x_phase[j, time_index], k=1)
                x_interpolate[j, :] = sci_interp.splev(t_int, s)
            out._states["unscaled"][p]["all"] = x_interpolate

            offset = 0
            for key in data_states[p]:
                if key == "all":
                    continue
                n_elements = data_states[p][key].shape[0]
                out._states["unscaled"][p][key] = out._states["unscaled"][p]["all"][offset : offset + n_elements]
                offset += n_elements

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
            new.phase_time,
            new.ns,
        ) = self._merge_phases()
        new._time_vector = [np.array(concatenate_optimization_variables(self._time_vector))]
        new.is_merged = True
        return new

    def _merge_phases(self, skip_states: bool = False, skip_controls: bool = False, continuous: bool = True) -> tuple:
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

        return out_states_scaled, out_states, out_controls_scaled, out_controls, phase_time, ns

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
            elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
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
        self, n_frames: int = 0, shooting_type: Shooting = None, show_now: bool = True, **kwargs: Any
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
        kwargs: Any
            Any parameters to pass to bioviz

        Returns
        -------
            A list of bioviz structures (one for each phase). So one can call exec() by hand
        """

        try:
            import bioviz
        except ModuleNotFoundError:
            raise RuntimeError("bioviz must be install to animate the model")

        from ..interfaces.biorbd_model import BiorbdModel

        check_version(bioviz, "2.3.0", "2.4.0")

        data_to_animate = self.integrate(shooting_type=shooting_type) if shooting_type else self.copy()
        if n_frames == 0:
            try:
                data_to_animate = data_to_animate.interpolate(sum(self.ns))
            except RuntimeError:
                pass

        elif n_frames > 0:
            data_to_animate = data_to_animate.interpolate(n_frames)

        states = data_to_animate.states
        if not isinstance(states, (list, tuple)):
            states = [states]

        all_bioviz = []
        for idx_phase, data in enumerate(states):
            if not isinstance(self.ocp.nlp[idx_phase].model, BiorbdModel):
                raise NotImplementedError("Animation is only implemented for biorbd models")

            # Convert parameters to actual values
            nlp = self.ocp.nlp[idx_phase]
            for param in nlp.parameters:
                if param.function:
                    param.function(nlp.model, self.parameters[param.name], **param.params)

            # noinspection PyTypeChecker
            biorbd_model: BiorbdModel = nlp.model

            all_bioviz.append(bioviz.Viz(biorbd_model.path, **kwargs))
            all_bioviz[-1].load_movement(self.ocp.nlp[idx_phase].variable_mappings["q"].to_second.map(data["q"]))
            for objective in self.ocp.nlp[idx_phase].J:
                if objective.target is not None:
                    if objective.type in (
                        ObjectiveFcn.Mayer.TRACK_MARKERS,
                        ObjectiveFcn.Lagrange.TRACK_MARKERS,
                    ) and objective.node[0] in (Node.ALL, Node.ALL_SHOOTING):
                        all_bioviz[-1].load_experimental_markers(objective.target[0])

        if show_now:
            b_is_visible = [True] * len(all_bioviz)
            while sum(b_is_visible):
                for i, b in enumerate(all_bioviz):
                    if b.vtk_window.is_active:
                        b.update()
                    else:
                        b_is_visible[i] = False
            return None
        else:
            return all_bioviz

    def _get_penalty_cost(self, nlp, penalty):
        phase_idx = nlp.phase_idx
        steps = nlp.ode_solver.steps + 1 if nlp.ode_solver.is_direct_collocation else 1

        val = []
        val_weighted = []
        p = self.parameters["all"]
        dt = (
            Function("time", [nlp.parameters.cx_start], [penalty.dt])(self.parameters["time"])
            if "time" in self.parameters
            else penalty.dt
        )

        if penalty.binode_constraint or penalty.allnode_constraint:
            penalty.node_idx = [penalty.node_idx]

        for idx in penalty.node_idx:
            x = []
            u = []
            target = []
            if nlp is not None:
                if penalty.transition:
                    phase_post = (phase_idx + 1) % len(self._states["scaled"])
                    x = np.concatenate(
                        (
                            self._states["scaled"][phase_idx]["all"][:, -1],
                            self._states["scaled"][phase_post]["all"][:, 0],
                        )
                    )
                    u = (
                        []
                        if nlp.control_type == ControlType.NONE
                        else np.concatenate(
                            (
                                self._controls["scaled"][phase_idx]["all"][:, -1],
                                self._controls["scaled"][phase_post]["all"][:, 0],
                            )
                        )
                    )
                elif penalty.binode_constraint:
                    x = np.concatenate(
                        (
                            self._states["scaled"][penalty.phase_first_idx]["all"][:, idx[0]],
                            self._states["scaled"][penalty.phase_second_idx]["all"][:, idx[1]],
                        )
                    )

                    # Make an exception to the fact that U is not available for the last node
                    mod_u0 = 1 if penalty.first_node == Node.END else 0
                    mod_u1 = 1 if penalty.second_node == Node.END else 0
                    u = np.concatenate(
                        (
                            self._controls["scaled"][penalty.phase_first_idx]["all"][:, idx[0] - mod_u0],
                            self._controls["scaled"][penalty.phase_second_idx]["all"][:, idx[1] - mod_u1],
                        )
                    )

                    # elif penalty.allnode_constraint:
                    #     x = np.concatenate(
                    #         (
                    #             self._states["scaled"][penalty.phase_idx]["all"][:, :],
                    #         )
                    #     )

                else:
                    col_x_idx = list(range(idx * steps, (idx + 1) * steps)) if penalty.integrate else [idx]
                    col_u_idx = [idx]
                    if (
                        penalty.derivative
                        or penalty.explicit_derivative
                        or penalty.integration_rule == IntegralApproximation.TRAPEZOIDAL
                    ):
                        col_x_idx.append((idx + 1) * (steps if nlp.ode_solver.is_direct_shooting else 1))

                        if (
                            penalty.integration_rule != IntegralApproximation.TRAPEZOIDAL
                        ) or nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                            col_u_idx.append((idx + 1))
                    elif penalty.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL:
                        if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                            col_u_idx.append((idx + 1))
                    if nlp.ode_solver.is_direct_collocation and (
                        "Lagrange" in penalty.type.__str__() or "Mayer" in penalty.type.__str__()
                    ):
                        x = (
                            self.states_no_intermediate["all"][:, col_x_idx]
                            if len(self.phase_time) - 1 == 1
                            else self.states_no_intermediate[0][phase_idx]["all"][:, col_x_idx]
                        )
                    else:
                        x = self._states["scaled"][phase_idx]["all"][:, col_x_idx]

                    u = (
                        []
                        if nlp.control_type == ControlType.NONE
                        else self._controls["scaled"][phase_idx]["all"][:, col_u_idx]
                    )
                    if penalty.target is None:
                        target = []
                    elif (
                        penalty.integration_rule == IntegralApproximation.TRAPEZOIDAL
                        or penalty.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
                    ):
                        target = np.vstack(
                            (
                                penalty.target[0][:, penalty.node_idx.index(idx)],
                                penalty.target[1][:, penalty.node_idx.index(idx)],
                            )
                        ).T
                    else:
                        target = penalty.target[0][..., penalty.node_idx.index(idx)]

            val.append(penalty.function_non_threaded(x, u, p))
            val_weighted.append(penalty.weighted_function_non_threaded(x, u, p, penalty.weight, target, dt))

        val = np.nansum(val)
        val_weighted = np.nansum(val_weighted)

        return val, val_weighted

    def detailed_cost_values(self):
        """
        Adds the detailed objective functions and/or constraints values to sol

        Parameters
        ----------
        cost_type: CostType
            The type of cost to console print
        """

        for nlp in self.ocp.nlp:
            for penalty in nlp.J_internal + nlp.J:
                if not penalty:
                    continue
                val, val_weighted = self._get_penalty_cost(nlp, penalty)
                self.detailed_cost += [
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

                if penalty.node != Node.TRANSITION:
                    node_name = f"{penalty.node[0]}" if isinstance(penalty.node[0], int) else penalty.node[0].name
                else:
                    node_name = penalty.node.name

                self.detailed_cost += [
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
