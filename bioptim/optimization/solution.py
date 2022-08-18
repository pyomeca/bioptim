from typing import Any, Union
from copy import deepcopy, copy

import biorbd_casadi as biorbd
import numpy as np
from scipy import interpolate as sci_interp
from scipy.integrate import solve_ivp
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
from ..optimization.optimization_variable import OptimizationVariableList, OptimizationVariable
from ..dynamics.ode_solver import OdeSolver


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
    states(self) -> Union[list, dict]
        Returns the state in list if more than one phases, otherwise it returns the only dict
    @property
    states_no_intermediate(self) -> Union[list, dict]
        Returns the state in list if more than one phases, otherwise it returns the only dict
        and removes the intermediate states if Collocation solver is used
    @property
    controls(self) -> Union[list, dict]
        Returns the controls in list if more than one phases, otherwise it returns the only dict
    integrate(self, shooting_type: Shooting = Shooting.MULTIPLE, keep_intermediate_points: bool = True,
              merge_phases: bool = False, continuous: bool = True) -> Solution
        Integrate the states
    interpolate(self, n_frames: Union[int, list, tuple]) -> Solution
        Interpolate the states
    merge_phases(self) -> Solution
        Get a data structure where all the phases are merged into one
    _merge_phases(self, skip_states: bool = False, skip_controls: bool = False) -> tuple
        Actually performing the phase merging
    _complete_control(self)
        Controls don't necessarily have dimensions that matches the states. This method aligns them
    graphs(self, automatically_organize: bool, show_bounds: bool,
           show_now: bool, shooting_type: Shooting)
        Show the graphs of the simulation
    animate(self, n_frames: int = 0, show_now: bool = True, **kwargs: Any) -> Union[None, list]
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

        def __init__(self, other: Union[OptimizationVariableList]):
            self.elements = []
            if isinstance(other, Solution.SimplifiedOptimizationVariableList):
                self.shape = other.shape
            else:
                self.shape = other.cx.shape[0]
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
        dynamics: list[ODE_SOLVER]
            All the dynamics for each of the node of the phase
        g: list[list[Constraint]]
            All the constraints at each of the node of the phase
        J: list[list[Objective]]
            All the objectives at each of the node of the phase
        model: biorbd.Model
            A reference to the biorbd Model
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
            self.model = nlp.model
            self.states = Solution.SimplifiedOptimizationVariableList(nlp.states)
            self.controls = Solution.SimplifiedOptimizationVariableList(nlp.controls)
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

    def __init__(self, ocp, sol: Union[dict, list, tuple, np.ndarray, DM, None]):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to strip
        sol: Union[dict, list, tuple, np.ndarray, DM]
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
        self.time_vector = None

        # Extract the data now for further use
        self._states, self._controls, self.parameters = {}, {}, {}
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
            if _sol["solver"] == SolverType.IPOPT:
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
            self._states, self._controls, self.parameters = self.ocp.v.to_dictionaries(self.vector)
            self._complete_control()
            self.phase_time = self.ocp.v.extract_phase_time(self.vector)

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
                s.init.check_and_adjust_dimensions(self.ocp.nlp[p].states.shape, ns, "states")
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
                s.init.check_and_adjust_dimensions(self.ocp.nlp[p].controls.shape, self.ns[p], "controls")
                for i in range(self.ns[p] + off):
                    self.vector = np.concatenate((self.vector, s.init.evaluate_at(i)[:, np.newaxis]))

            if n_param:
                sol_params = _sol[2]
                for p, s in enumerate(sol_params):
                    self.vector = np.concatenate((self.vector, np.repeat(s.init, self.ns[p] + 1)[:, np.newaxis]))

            self._states, self._controls, self.parameters = self.ocp.v.to_dictionaries(self.vector)
            self._complete_control()
            self.phase_time = self.ocp.v.extract_phase_time(self.vector)

        def init_from_vector(_sol: Union[np.ndarray, DM]):
            """
            Initialize all the attributes from a vector of solution

            Parameters
            ----------
            _sol: Union[np.ndarray, DM]
                The solution in vector format
            """

            self.vector = _sol
            self._states, self._controls, self.parameters = self.ocp.v.to_dictionaries(self.vector)
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

        new.time_vector = deepcopy(self.time_vector)

        if skip_data:
            new._states, new._controls, new.parameters = [], [], {}
        else:
            new._states = deepcopy(self._states)
            new._controls = deepcopy(self._controls)
            new.parameters = deepcopy(self.parameters)

        return new

    @property
    def states(self) -> Union[list, dict]:
        """
        Returns the state in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The states data
        """

        return self._states[0] if len(self._states) == 1 else self._states

    @property
    def states_no_intermediate(self) -> Union[list, dict]:
        """
        Returns the state in list if more than one phases, otherwise it returns the only dict
        it removes the intermediate states in the case COLLOCATION Solver is used

        Returns
        -------
        The states data without intermediate states in the case of collocation
        """
        states_no_intermediate = []
        for i, nlp in enumerate(self.ocp.nlp):
            if isinstance(nlp.ode_solver, OdeSolver.COLLOCATION) and not isinstance(nlp.ode_solver, OdeSolver.IRK):
                states_no_intermediate.append(dict())
                for key in self._states[i].keys():
                    # keep one value each five values
                    states_no_intermediate[i][key] = self._states[i][key][:, :: nlp.ode_solver.polynomial_degree + 1]
            else:
                states_no_intermediate.append(self._states[i])

        return states_no_intermediate[0] if len(states_no_intermediate) == 1 else states_no_intermediate

    @property
    def controls(self) -> Union[list, dict]:
        """
        Returns the controls in list if more than one phases, otherwise it returns the only dict

        Returns
        -------
        The controls data
        """

        if not self._controls:
            raise RuntimeError(
                "There is no controls in the solution. "
                "This may happen in "
                "previously integrated and interpolated structure"
            )
        return self._controls[0] if len(self._controls) == 1 else self._controls

    def integrate(
        self,
        shooting_type: Shooting = Shooting.SINGLE_CONTINUOUS,
        keep_intermediate_points: bool = False,
        merge_phases: bool = False,
        continuous: bool = True,
        integrator: SolutionIntegrator = SolutionIntegrator.DEFAULT,
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
        continuous: bool
            If the arrival value of a node should be discarded [True] or kept [False]. The value of an integrated
            arrival node and the beginning of the next one are expected to be almost equal when the problem converged
        integrator: SolutionIntegrator
            Use the ode defined by OCP or use a separate integrator provided by scipy

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        # Sanity check
        if self.is_integrated:
            raise RuntimeError("Cannot integrate twice")
        if self.is_interpolated:
            raise RuntimeError("Cannot integrate after interpolating")
        if self.is_merged:
            raise RuntimeError("Cannot integrate after merging phases")

        if shooting_type == Shooting.MULTIPLE and not keep_intermediate_points:
            raise ValueError(
                "Shooting.MULTIPLE and keep_intermediate_points=False cannot be used simultaneously "
                "since it would do nothing"
            )
        if shooting_type == Shooting.SINGLE_CONTINUOUS and not continuous:
            raise ValueError(
                "Shooting.SINGLE_CONTINUOUS and continuous=False cannot be used simultaneously it is a contradiction"
            )

        out = self.__perform_integration(shooting_type, keep_intermediate_points, continuous, merge_phases, integrator)

        if merge_phases:
            if continuous:
                out = out.interpolate(sum(out.ns) + 1)
            else:
                out._states, _, out.phase_time, out.ns = out._merge_phases(skip_controls=True, continuous=continuous)
                out.is_merged = True
        out.is_integrated = True

        return out

    def _generate_time_vector(
        self,
        time_phase,
        keep_intermediate_points: bool,
        continuous: bool,
        merge_phases: bool,
        integrator: SolutionIntegrator,
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Generate time integration vector

        Returns
        -------
        t_integrated: np.ndarray or list of np.ndarray
        The time vector
        """

        t_integrated = []
        last_t = 0
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            n_int_steps = (
                nlp.ode_solver.steps_scipy if integrator != SolutionIntegrator.DEFAULT else nlp.ode_solver.steps
            )
            dt_ns = time_phase[phase_idx + 1] / nlp.ns
            time_phase_integrated = []
            last_t_int = copy(last_t)
            for _ in range(nlp.ns):
                if nlp.ode_solver.is_direct_collocation and integrator == SolutionIntegrator.DEFAULT:
                    time_phase_integrated += (np.array(nlp.dynamics[0].step_time) * dt_ns + last_t_int).tolist()
                else:
                    time_interval = np.linspace(last_t_int, last_t_int + dt_ns, n_int_steps + 1)
                    if continuous and _ != nlp.ns - 1:
                        time_interval = time_interval[:-1]
                    if not keep_intermediate_points:
                        if _ == nlp.ns - 1:
                            time_interval = time_interval[[0, -1]]
                        else:
                            time_interval = np.array([time_interval[0]])
                    time_phase_integrated += time_interval.tolist()

                if not continuous and _ == nlp.ns - 1:
                    time_phase_integrated += [time_phase_integrated[-1]]

                last_t_int += dt_ns
            if continuous and merge_phases and phase_idx != len(self.ocp.nlp) - 1:
                t_integrated += time_phase_integrated[:-1]
            else:
                t_integrated += time_phase_integrated
            last_t += time_phase[phase_idx + 1]
        return t_integrated

    def __perform_integration(
        self,
        shooting_type: Shooting,
        keep_intermediate_points: bool,
        continuous: bool,
        merge_phases: bool,
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
        continuous: bool
            If the arrival value of a node should be discarded [True] or kept [False]. The value of an integrated
        merge_phases
            If the phase should be merged in a unique phase
        integrator
            Use the ode solver defined by the OCP or use a separate integrator provided by scipy such as RK45 or DOP853

        Returns
        -------
        Solution
            A Solution data structure with the states integrated. The controls are removed from this structure
        """

        n_direct_collocation = sum([nlp.ode_solver.is_direct_collocation for nlp in self.ocp.nlp])

        if n_direct_collocation > 0 and integrator == SolutionIntegrator.DEFAULT:
            if continuous:
                raise RuntimeError(
                    "Integration with direct collocation must be not continuous if a scipy integrator is used"
                )

            if shooting_type != Shooting.MULTIPLE:
                raise RuntimeError(
                    "Integration with direct collocation must using shooting_type=Shooting.MULTIPLE "
                    "if a scipy integrator is not used"
                )

        # Copy the data
        out = self.copy(skip_data=True)
        out.recomputed_time_steps = integrator != SolutionIntegrator.DEFAULT
        out._states = []
        out.time_vector = self._generate_time_vector(
            out.phase_time, keep_intermediate_points, continuous, merge_phases, integrator
        )
        for _ in range(len(self._states)):
            out._states.append({})

        sum_states_len = 0
        params = self.parameters["all"]
        x0 = self._states[0]["all"][:, 0]
        for p, nlp in enumerate(self.ocp.nlp):
            param_scaling = nlp.parameters.scaling
            n_states = self._states[p]["all"].shape[0]
            n_steps = nlp.ode_solver.steps_scipy if integrator != SolutionIntegrator.DEFAULT else nlp.ode_solver.steps
            if not continuous:
                n_steps += 1
            if keep_intermediate_points:
                out.ns[p] *= n_steps

            out._states[p]["all"] = np.ndarray((n_states, out.ns[p] + 1))

            # Get the first frame of the phase
            if shooting_type == Shooting.SINGLE_CONTINUOUS:
                if p != 0:
                    u0 = self._controls[p - 1]["all"][:, -1]
                    val = self.ocp.phase_transitions[p - 1].function(vertcat(x0, x0), vertcat(u0, u0), params)
                    if val.shape[0] != x0.shape[0]:
                        raise RuntimeError(
                            f"Phase transition must have the same number of states ({val.shape[0]}) "
                            f"when integrating with Shooting.SINGLE_CONTINUOUS. If it is not possible, "
                            f"please integrate with Shooting.SINGLE"
                        )
                    x0 += np.array(val)[:, 0]
            else:
                col = (
                    slice(0, n_steps)
                    if nlp.ode_solver.is_direct_collocation and integrator == SolutionIntegrator.DEFAULT
                    else 0
                )
                x0 = self._states[p]["all"][:, col]

            for s in range(self.ns[p]):
                if nlp.control_type == ControlType.CONSTANT:
                    u = self._controls[p]["all"][:, s]
                elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                    u = self._controls[p]["all"][:, s : s + 2]
                else:
                    raise NotImplementedError(f"ControlType {nlp.control_type} " f"not yet implemented in integrating")

                if integrator != SolutionIntegrator.DEFAULT:
                    t_init = sum(out.phase_time[:p]) / nlp.ns
                    t_end = sum(out.phase_time[: (p + 2)]) / nlp.ns
                    n_points = n_steps + 1 if continuous else n_steps
                    t_eval = np.linspace(t_init, t_end, n_points) if keep_intermediate_points else [t_init, t_end]
                    integrated = solve_ivp(
                        lambda t, x: np.array(nlp.dynamics_func(x, u, params))[:, 0],
                        [t_init, t_end],
                        x0,
                        t_eval=t_eval,
                        method=integrator.value,
                    ).y

                    next_state_col = (
                        (s + 1) * (nlp.ode_solver.steps + 1) if nlp.ode_solver.is_direct_collocation else s + 1
                    )
                    cols_in_out = [s * n_steps, (s + 1) * n_steps] if keep_intermediate_points else [s, s + 2]
                else:
                    if nlp.ode_solver.is_direct_collocation:
                        if keep_intermediate_points:
                            integrated = x0  # That is only for continuous=False
                            cols_in_out = [s * n_steps, (s + 1) * n_steps]
                        else:
                            integrated = x0[:, [0, -1]]
                            cols_in_out = [s, s + 2]
                        next_state_col = slice((s + 1) * n_steps, (s + 2) * n_steps)

                    else:
                        if keep_intermediate_points:
                            integrated = np.array(nlp.dynamics[s](x0=x0, p=u, params=params / param_scaling)["xall"])
                            cols_in_out = [s * n_steps, (s + 1) * n_steps]
                        else:
                            integrated = np.concatenate(
                                (x0[:, np.newaxis], nlp.dynamics[s](x0=x0, p=u, params=params / param_scaling)["xf"]),
                                axis=1,
                            )
                            cols_in_out = [s, s + 2]
                        next_state_col = s + 1

                cols_in_out = slice(
                    cols_in_out[0], cols_in_out[1] + 1 if continuous and keep_intermediate_points else cols_in_out[1]
                )
                out._states[p]["all"][:, cols_in_out] = integrated
                x0 = (
                    np.array(self._states[p]["all"][:, next_state_col])
                    if shooting_type == Shooting.MULTIPLE
                    else integrated[:, -1]
                )

            if not continuous:
                out._states[p]["all"][:, -1] = self._states[p]["all"][:, -1]

            # Dispatch the integrated values to all the keys
            for key in nlp.states:
                out._states[p][key] = out._states[p]["all"][nlp.states[key].index, :]

            sum_states_len += out._states[p]["all"].shape[1]

        return out

    def interpolate(self, n_frames: Union[int, list, tuple]) -> Any:
        """
        Interpolate the states

        Parameters
        ----------
        n_frames: Union[int, list, tuple]
            If the value is an int, the Solution returns merges the phases,
            otherwise, it interpolates them independently

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        out = self.copy(skip_data=True)

        t_all = []
        for p, data in enumerate(self._states):
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
            data_states, _, out.phase_time, out.ns = self._merge_phases(skip_controls=True)
            t_all = [np.concatenate((np.concatenate([_t[:-1] for _t in t_all]), [t_all[-1][-1]]))]

            n_frames = [n_frames]
            out.is_merged = True
        elif isinstance(n_frames, (list, tuple)) and len(n_frames) == len(self._states):
            data_states = self._states
        else:
            raise ValueError(
                "n_frames should either be a int to merge_phases phases "
                "or a list of int of the number of phases dimension"
            )

        out._states = []
        for _ in range(len(data_states)):
            out._states.append({})
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
            out._states[p]["all"] = x_interpolate

            offset = 0
            for key in data_states[p]:
                if key == "all":
                    continue
                n_elements = data_states[p][key].shape[0]
                out._states[p][key] = out._states[p]["all"][offset : offset + n_elements]
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
        new._states, new._controls, new.phase_time, new.ns = self._merge_phases()
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
            return deepcopy(self._states), deepcopy(self._controls), deepcopy(self.phase_time), deepcopy(self.ns)

        def _merge(data: list, is_control: bool) -> Union[list, dict]:
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

        if len(self._states) == 1:
            out_states = deepcopy(self._states)
        else:
            out_states = _merge(self.states, is_control=False) if not skip_states and self._states else None

        if len(self._controls) == 1:
            out_controls = deepcopy(self._controls)
        else:
            out_controls = _merge(self.controls, is_control=True) if not skip_controls and self._controls else None
        phase_time = [0] + [sum([self.phase_time[i + 1] for i in range(len(self.phase_time) - 1)])]
        ns = [sum(self.ns)]

        return out_states, out_controls, phase_time, ns

    def _complete_control(self):
        """
        Controls don't necessarily have dimensions that matches the states. This method aligns them
        """

        for p, nlp in enumerate(self.ocp.nlp):
            if nlp.control_type == ControlType.CONSTANT:
                for key in self._controls[p]:
                    self._controls[p][key] = np.concatenate(
                        (self._controls[p][key], np.nan * np.zeros((self._controls[p][key].shape[0], 1))), axis=1
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
        integrator: SolutionIntegrator = SolutionIntegrator.DEFAULT,
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
    ) -> Union[None, list]:
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
        check_version(bioviz, "2.1.1", "2.2.0")

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
            # Convert parameters to actual values
            nlp = self.ocp.nlp[idx_phase]
            for param in nlp.parameters:
                if param.function:
                    param.function(nlp.model, self.parameters[param.name], **param.params)

            all_bioviz.append(bioviz.Viz(self.ocp.nlp[idx_phase].model.path().absolutePath().to_string(), **kwargs))
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
            Function("time", [nlp.parameters.cx], [penalty.dt])(self.parameters["time"])
            if "time" in self.parameters
            else penalty.dt
        )

        if penalty.multinode_constraint:
            penalty.node_idx = [penalty.node_idx]

        for idx in penalty.node_idx:
            x = []
            u = []
            target = []
            if nlp is not None:
                if penalty.transition:
                    phase_post = (phase_idx + 1) % len(self._states)
                    x = np.concatenate((self._states[phase_idx]["all"][:, -1], self._states[phase_post]["all"][:, 0]))
                    u = np.concatenate(
                        (self._controls[phase_idx]["all"][:, -1], self._controls[phase_post]["all"][:, 0])
                    )
                elif penalty.multinode_constraint:

                    x = np.concatenate(
                        (
                            self._states[penalty.phase_first_idx]["all"][:, idx[0]],
                            self._states[penalty.phase_second_idx]["all"][:, idx[1]],
                        )
                    )
                    # Make an exception to the fact that U is not available for the last node
                    mod_u0 = 1 if penalty.first_node == Node.END else 0
                    mod_u1 = 1 if penalty.second_node == Node.END else 0
                    u = np.concatenate(
                        (
                            self._controls[penalty.phase_first_idx]["all"][:, idx[0] - mod_u0],
                            self._controls[penalty.phase_second_idx]["all"][:, idx[1] - mod_u1],
                        )
                    )

                else:
                    col_x_idx = list(range(idx * steps, (idx + 1) * steps)) if penalty.integrate else [idx]
                    col_u_idx = [idx]
                    if (
                        penalty.derivative
                        or penalty.explicit_derivative
                        or penalty.integration_rule == IntegralApproximation.TRAPEZOIDAL
                    ):
                        col_x_idx.append((idx + 1) * steps)
                        if (
                            penalty.integration_rule != IntegralApproximation.TRAPEZOIDAL
                        ) or nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                            col_u_idx.append((idx + 1))
                    elif penalty.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL:
                        if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                            col_u_idx.append((idx + 1))

                    x = self._states[phase_idx]["all"][:, col_x_idx]
                    u = self._controls[phase_idx]["all"][:, col_u_idx]
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
                self.detailed_cost += [{"name": penalty.name, "cost_value_weighted": val_weighted, "cost_value": val}]
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
                self.detailed_cost += [
                    {
                        "name": penalty.name,
                        "cost_value_weighted": val_weighted,
                        "cost_value": val,
                        "params": penalty.params,
                    }
                ]
                if print_only_weighted:
                    print(f"{penalty.name}: {val_weighted}")
                else:
                    print(f"{penalty.name}: {val: .2f} (weighted {val_weighted})")

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
            Print the values of each constraints with its lagrange multiplier to the console
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
