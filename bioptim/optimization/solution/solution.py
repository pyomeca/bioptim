from typing import Any
from copy import deepcopy

import numpy as np
from scipy import interpolate as sci_interp
from scipy.interpolate import interp1d
from casadi import vertcat, DM
from matplotlib import pyplot as plt

from .solution_data import SolutionData, SolutionMerge
from ..optimization_vector import OptimizationVectorHelper
from ...limits.objective_functions import ObjectiveFcn
from ...limits.path_conditions import InitialGuess, InitialGuessList
from ...limits.penalty_helpers import PenaltyHelpers
from ...misc.enums import ControlType, CostType, Shooting, InterpolationType, SolverType, SolutionIntegrator, Node
from ...dynamics.ode_solver import OdeSolver
from ...interfaces.solve_ivp_interface import solve_ivp_bioptim_interface



class Solution:
    """
    Data manipulation, graphing and storage

    Attributes
    ----------
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
    _decision_times: list
        The time at each node so the integration can be done (equivalent to t_span)
    _stepwise_times: list
        The time corresponding to _stepwise_states
    _decision_states: SolutionData
        A SolutionData based solely on the solution
    _stepwise_states: SolutionData
        A SolutionData based on the integrated solution directly from the bioptim integrator
    _stepwise_controls: SolutionData
        The data structure that holds the controls
    _parameters: SolutionData
        The data structure that holds the parameters
    _stochastic: SolutionData
        The data structure that holds the stochastic variables
    phases_dt: list
        The time step for each phases

    Methods
    -------
    copy(self, skip_data: bool = False) -> Any
        Create a deepcopy of the Solution
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

    def __init__(
        self,
        ocp,
        vector: np.ndarray | DM = None,
        cost: np.ndarray | DM = None,
        constraints: np.ndarray | DM = None,
        lam_g: np.ndarray | DM = None,
        lam_p: np.ndarray | DM = None,
        lam_x: np.ndarray | DM = None,
        inf_pr: np.ndarray | DM = None,
        inf_du: np.ndarray | DM = None,
        solver_time_to_optimize: float = None,
        real_time_to_optimize: float = None,
        iterations: int = None,
        status: int = None,
    ):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to strip
        vector: np.ndarray | DM
            The solution vector, containing all the states, controls, parameters and stochastic variables
        cost: np.ndarray | DM
            The cost value of the objective function
        constraints: np.ndarray | DM
            The constraints value
        lam_g: np.ndarray | DM
            The lagrange multipliers for the constraints
        lam_p: np.ndarray | DM
            The lagrange multipliers for the parameters
        lam_x: np.ndarray | DM
            The lagrange multipliers for the states
        lam_g: np.ndarray | DM
            The lagrange multipliers for the constraints
        inf_pr: np.ndarray | DM
            The primal infeasibility
        inf_du: np.ndarray | DM
            The dual infeasibility
        solver_time_to_optimize: float
            The time to optimize
        real_time_to_optimize: float
            The real time to optimize
        iterations: int
            The number of iterations
        status: int
            The status of the solution
        """

        self.ocp = ocp

        # Penalties
        self._cost, self._detailed_cost, self.constraints = cost, None, constraints

        # Solver options
        self.status, self.iterations = status, iterations
        self.lam_g, self.lam_p, self.lam_x, self.inf_pr, self.inf_du = lam_g, lam_p, lam_x, inf_pr, inf_du
        self.solver_time_to_optimize, self.real_time_to_optimize = solver_time_to_optimize, real_time_to_optimize

        # Extract the data now for further use
        self._decision_states = None
        self._stepwise_states = None
        self._stepwise_controls = None
        self._parameters = None
        self._stochastic = None

        self.vector = vector
        if self.vector is not None:
            self.phases_dt = OptimizationVectorHelper.extract_phase_dt(ocp, vector)
            self._stepwise_times = OptimizationVectorHelper.extract_step_times(ocp, vector)
            self._decision_times = [
                [vertcat(t[0], t[-1]) for t in self._stepwise_times[p]] for p in range(self.ocp.n_phases)
            ]
            
            x, u, p, s = OptimizationVectorHelper.to_dictionaries(ocp, vector)
            self._decision_states = SolutionData.from_scaled(ocp, x, "x")
            self._stepwise_controls = SolutionData.from_scaled(ocp, u, "u")
            self._parameters = SolutionData.from_scaled(ocp, p, "p")
            self._stochastic = SolutionData.from_scaled(ocp, s, "s")

    @classmethod
    def from_dict(cls, ocp, sol: dict):
        """
        Initialize all the attributes from an Ipopt-like dictionary data structure

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        sol: dict
            The solution in a Ipopt-like dictionary
        """

        if not isinstance(sol, dict):
            raise ValueError("The _sol entry should be a dictionary")

        is_ipopt = sol["solver"] == SolverType.IPOPT.value

        return cls(
            ocp=ocp,
            vector=sol["x"],
            cost=sol["f"] if is_ipopt else None,
            constraints=sol["g"] if is_ipopt else None,
            lam_g=sol["lam_g"] if is_ipopt else None,
            lam_p=sol["lam_p"] if is_ipopt else None,
            lam_x=sol["lam_x"] if is_ipopt else None,
            inf_pr=sol["inf_pr"] if is_ipopt else None,
            inf_du=sol["inf_du"] if is_ipopt else None,
            solver_time_to_optimize=sol["solver_time_to_optimize"],
            real_time_to_optimize=sol["real_time_to_optimize"],
            iterations=sol["iter"],
            status=sol["status"],
        )

    @classmethod
    def from_initial_guess(cls, ocp, sol: list):
        """
        Initialize all the attributes from a list of initial guesses (states, controls)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        sol: list
            The list of initial guesses
        """

        if not (isinstance(sol, (list, tuple)) and len(sol) == 5):
            raise ValueError("_sol should be a list of tuple and the length should be 5")

        n_param = len(ocp.parameters)
        all_ns = [nlp.ns for nlp in ocp.nlp]

        # Sanity checks
        for i in range(len(sol)):  # Convert to list if necessary and copy for as many phases there are
            if isinstance(sol[i], InitialGuess):
                tp = InitialGuessList()
                for _ in range(len(all_ns)):
                    tp.add(deepcopy(sol[i].init), interpolation=sol[i].init.type)
                sol[i] = tp
        if sum([isinstance(s, InitialGuessList) for s in sol]) != 4:
            raise ValueError(
                "solution must be a solution dict, "
                "an InitialGuess[List] of len 4 (states, controls, parameters, stochastic_variables), "
                "or a None"
            )
        if sum([len(s) != len(all_ns) if p != 3 else False for p, s in enumerate(sol)]) != 0:
            raise ValueError("The InitialGuessList len must match the number of phases")
        if n_param != 0:
            if len(sol) != 3 and len(sol[3]) != 1 and sol[3][0].shape != (n_param, 1):
                raise ValueError(
                    "The 3rd element is the InitialGuess of the parameter and "
                    "should be a unique vector of size equal to n_param"
                )

        dt, sol_states, sol_controls, sol_params, sol_stochastic_variables = sol

        vector = np.ndarray((0, 1))

        # For time
        if len(dt.shape) == 1:
            dt = dt[:, np.newaxis]
        vector = np.concatenate((vector, dt))

        # For states
        for p, ss in enumerate(sol_states):
            repeat = 1
            if isinstance(ocp.nlp[p].ode_solver, OdeSolver.COLLOCATION):
                repeat = ocp.nlp[p].ode_solver.polynomial_degree + 1
            for key in ss.keys():
                ns = ocp.nlp[p].ns + 1 if ss[key].init.type != InterpolationType.EACH_FRAME else ocp.nlp[p].ns
                ss[key].init.check_and_adjust_dimensions(len(ocp.nlp[p].states[key]), ns, "states")

            for i in range(all_ns[p] + 1):
                for key in ss.keys():
                    vector = np.concatenate((vector, ss[key].init.evaluate_at(i, repeat)[:, np.newaxis]))

        # For controls
        for p, ss in enumerate(sol_controls):
            control_type = ocp.nlp[p].control_type
            if control_type == ControlType.CONSTANT:
                off = 0
            elif control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                off = 1
            else:
                raise NotImplementedError(f"control_type {control_type} is not implemented in Solution")

            for key in ss.keys():
                ss[key].init.check_and_adjust_dimensions(len(ocp.nlp[p].controls[key]), all_ns[p], "controls")

            for i in range(all_ns[p] + off):
                for key in ss.keys():
                    vector = np.concatenate((vector, ss[key].init.evaluate_at(i)[:, np.newaxis]))

        # For parameters
        if n_param:
            for p, ss in enumerate(sol_params):
                vector = np.concatenate((vector, np.repeat(ss.init, all_ns[p] + 1)[:, np.newaxis]))

        # For stochastic variables
        for p, ss in enumerate(sol_stochastic_variables):
            for key in ss.keys():
                ss[key].init.check_and_adjust_dimensions(
                    len(ocp.nlp[p].stochastic_variables[key]), all_ns[p], "stochastic_variables"
                )

            for i in range(all_ns[p] + 1):
                for key in ss.keys():
                    vector = np.concatenate((vector, ss[key].init.evaluate_at(i)[:, np.newaxis]))

        return cls(ocp=ocp, vector=vector)

    @classmethod
    def from_vector(cls, ocp, sol: np.ndarray | DM):
        """
        Initialize all the attributes from a vector of solution

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        sol: np.ndarray | DM
            The solution in vector format
        """

        if not isinstance(sol, (np.ndarray, DM)):
            raise ValueError("The _sol entry should be a np.ndarray or a DM.")

        return cls(ocp=ocp, vector=sol)

    @classmethod
    def from_ocp(cls, ocp):
        """
        Initialize all the attributes from a vector of solution

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        """

        return cls(ocp=ocp)

    @property
    def times(self):
        """
        Returns the time vector at each phase

        Returns
        -------
        The time vector
        """

        out = deepcopy(self._stepwise_times)
        for p in range(len(out)):
            out[p] = np.concatenate(out[p])[:, 0]
        return out if len(out) > 1 else out[0]

    def phase_times(self, t_span: bool = False) -> list:
        """
        Returns the time vector at each node

        Parameters
        ----------
        t_span: bool
            If the time vector should correspond to stepwise_states (False) or to t_span (True). If you don't know what
            it means, you probably want the stepwise_states (False) version.

        Returns
        -------
        The time vector
        """

        out = deepcopy(self._decision_times if t_span else self._stepwise_times)
        return out if len(out) > 1 else out[0]

    @property
    def t_spans(self):
        """
        Returns the time vector at each node
        """
        return self.phase_times(t_span=True)
    
    @property
    def stepwise_times(self):
        """
        Returns the time vector at each node
        """
        return self.phase_times(t_span=False)

    @property
    def states(self):
        if self._stepwise_states is None:
            self._integrate_stepwise()
        
        out = self._stepwise_states.to_dict(to_merge=SolutionMerge.NODES, scaled=False)
        return out if len(out) > 1 else out[0]
    
    @property
    def controls(self):
        out = self._stepwise_controls.to_dict(to_merge=SolutionMerge.NODES, scaled=False)
        return out if len(out) > 1 else out[0]

    def decision_states(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge, ...] = None):
        """
        Returns the decision states

        Parameters
        ----------
        scaled: bool
            If the decision states should be scaled or not (note that scaled is as Ipopt received them, while unscaled
            is as the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge, ...]
            The type of merge to perform. If None, then no merge is performed. 

        Returns
        -------
        The decision variables
        """

        data = self._decision_states.to_dict(to_merge=to_merge, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    def stepwise_states(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge, ...] = None):
        """
        Returns the stepwise integrated states

        Parameters
        ----------
        scaled: bool
            If the states should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as the
            model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge, ...]
            The type of merge to perform. If None, then no merge is performed.

        Returns
        -------
        The stepwise integrated states
        """

        if self._stepwise_states is None:
            self._integrate_stepwise()

        data = self._stepwise_states.to_dict(to_merge=to_merge, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    def decision_controls(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge, ...] = None):
        return self.stepwise_controls(scaled=scaled, to_merge=to_merge)

    def stepwise_controls(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge, ...] = None):
        """
        Returns the controls. Note the final control is always present but set to np.nan if it is not defined

        Parameters
        ----------
        scaled: bool
            If the controls should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as
            the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge, ...]
            The type of merge to perform. If None, then no merge is performed.

        Returns
        -------
        The controls
        """

        data = self._stepwise_controls.to_dict(to_merge=to_merge, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    @property
    def parameters(self):
        """
        Returns the parameters
        """

        return self.decision_parameters(scaled=False)

    def decision_parameters(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge, ...] = None):
        """
        Returns the decision parameters

        Parameters
        ----------
        scaled: bool
            If the parameters should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as
            the model needs temps). If you don't know what it means, you probably want the unscaled version.

        Returns
        -------
        The decision parameters
        """
        if to_merge is None:
            to_merge = []

        if isinstance(to_merge, SolutionMerge):
            to_merge = [to_merge]

        if SolutionMerge.PHASES in to_merge:
            raise ValueError("Cannot merge phases for parameters as it is not bound to phases")
        if SolutionMerge.NODES in to_merge:
            raise ValueError("Cannot merge nodes for parameters as it is not bound to nodes")

        out = self._parameters.to_dict(scaled=scaled, to_merge=to_merge)

        # Remove the residual phases and nodes 
        if to_merge:
            out = out[0][0][:, 0]
        else:
            out = out[0]
            out = {key: out[key][0][:, 0] for key in out.keys()}
        
        return out

    def stochastic(self, scaled: bool = False, concatenate_keys: bool = False):
        """
        Returns the stochastic variables

        Parameters
        ----------
        scaled: bool
            If the stochastic variables should be scaled or not (note that scaled is as Ipopt received them, while
            unscaled is as the model needs temps). If you don't know what it means, you probably want the
            unscaled version.
        concatenate_keys: bool
            If the stochastic variables should be returned individually (False) or concatenated (True). If individual,
            then the return value does not contain the key dict.

        Returns
        -------
        The stochastic variables
        """

        data = self._stochastic.to_dict(to_merge=SolutionMerge.KEYS if concatenate_keys else None, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    def copy(self, skip_data: bool = False) -> "Solution":
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

        new = Solution.from_ocp(self.ocp)

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

        new.phases_dt = deepcopy(self.phases_dt)
        new._stepwise_times = deepcopy(self._stepwise_times)
        new._decision_times = deepcopy(self._decision_times)

        if not skip_data:
            new._decision_states = deepcopy(self._decision_states)
            new._stepwise_states = deepcopy(self._stepwise_states)

            new._stepwise_controls = deepcopy(self._stepwise_controls)

            new._stochastic = deepcopy(self._stochastic)
            new._parameters = deepcopy(self._parameters)
        return new

    def integrate(
            self, 
            shooting_type: Shooting = Shooting.SINGLE, 
            integrator: SolutionIntegrator = SolutionIntegrator.OCP, 
            to_merge: SolutionMerge | list[SolutionMerge, ...] = None,
        ):

        has_direct_collocation = sum([nlp.ode_solver.is_direct_collocation for nlp in self.ocp.nlp]) > 0
        if has_direct_collocation and integrator == SolutionIntegrator.OCP:
            raise ValueError(
                "When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE"
            )
        has_trapezoidal = sum([isinstance(nlp.ode_solver, OdeSolver.TRAPEZOIDAL) for nlp in self.ocp.nlp]) > 0
        if has_trapezoidal:
            raise ValueError(
                "When the ode_solver of the Optimal Control Problem is OdeSolver.TRAPEZOIDAL, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
            )

        params = self._parameters.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)[0][0]
        x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        s = self._stochastic.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)

        out: list = [None] * len(self.ocp.nlp)
        integrated_sol = None
        for p, nlp in enumerate(self.ocp.nlp):
            t = self._decision_times[p]

            next_x = self._states_for_phase_integration(shooting_type, p, integrated_sol, x, u, params, s)

            if integrator == SolutionIntegrator.OCP:
                integrated_sol = solve_ivp_bioptim_interface(
                    shooting_type=shooting_type, dynamics_func=nlp.dynamics, t=t, x=next_x, u=u[p], s=s[p], p=params
                )
            else:
                raise NotImplementedError(f"{integrator} is not implemented yet")
            
            out[p] = {}
            for key in nlp.states.keys():
                out[p][key] = [None] * nlp.n_states_nodes
                for ns, sol_ns in enumerate(integrated_sol):
                    out[p][key][ns] = sol_ns[nlp.states[key].index, :]

        if to_merge:
            out = SolutionData.from_unscaled(self.ocp, out, "x").to_dict(to_merge=to_merge, scaled=False)

        return out if len(out) > 1 else out[0]

    def _states_for_phase_integration(
        self,
        shooting_type: Shooting,
        phase_idx: int,
        integrated_states: np.ndarray,
        decision_states,
        decision_controls,
        params,
        decision_stochastic,
    ):
        """
        Returns the states to integrate for the phase_idx phase. If there was a phase transition, the last state of the
        previous phase is transformed into the first state of the next phase

        Parameters
        ----------
        shooting_type
            The shooting type to use
        phase_idx
            The phase index of the next phase to integrate
        integrated_states
            The states integrated from the previous phase
        decision_states
            The decision states merged with SolutionMerge.KEYS
        decision_controls
            The decision controls merged with SolutionMerge.KEYS
        params
            The parameters merged with SolutionMerge.KEYS
        decision_stochastic
            The stochastic merged with SolutionMerge.KEYS

        Returns
        -------
        The states to integrate
        """

        # In the case of multiple shootings, we don't need to do anything special
        if shooting_type == Shooting.MULTIPLE:
            return decision_states[phase_idx]

        # At first phase, return the normal decision states.
        if phase_idx == 0:
            return [decision_states[phase_idx][0]]

        penalty = self.ocp.phase_transitions[phase_idx - 1]

        times = DM([t[-1] for t in self._stepwise_times])
        t0 = PenaltyHelpers.t0(penalty, self.ocp, 0, lambda p, n: times[p])
        dt = PenaltyHelpers.phases_dt(penalty, self.ocp, lambda p: np.array([self.phases_dt[idx] for idx in p]))
        # Compute the error between the last state of the previous phase and the first state of the next phase
        # based on the phase transition objective or constraint function. That is why we need to concatenate
        # twice the last state
        x = PenaltyHelpers.states(penalty, self.ocp, 0, lambda controller_idx, p, n: integrated_states[-1])
        u = PenaltyHelpers.controls(penalty, self.ocp, 0, lambda controller_idx, p, n: decision_controls[p][n])
        s = PenaltyHelpers.stochastic_variables(penalty, self.ocp, 0, lambda controller_idx, p, n: decision_stochastic[p][n])

        dx = penalty.function[-1](t0, dt, x, u, params, s)
        if dx.shape[0] != decision_states[phase_idx][0].shape[0]:
            raise RuntimeError(
                f"Phase transition must have the same number of states ({dx.shape[0]}) "
                f"when integrating with Shooting.SINGLE. If it is not possible, "
                f"please integrate with Shooting.SINGLE_DISCONTINUOUS_PHASE."
            )

        return [decision_states[phase_idx][0] + dx]

    def _integrate_stepwise(self) -> None:
        """
        This method integrate to stepwise level the states. That is the states that are used in the dynamics and 
        continuity constraints.

        Returns
        -------
        dict
            The integrated data structure similar in structure to the original _decision_states
        """

        params = self._parameters.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)[0][0]
        x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        s = self._stochastic.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)

        unscaled: list = [None] * len(self.ocp.nlp)
        for p, nlp in enumerate(self.ocp.nlp):
            t = self._decision_times[p]

            integrated_sol = solve_ivp_bioptim_interface(
                shooting_type=Shooting.MULTIPLE, dynamics_func=nlp.dynamics, t=t, x=x[p], u=u[p], s=s[p], p=params
            )
            
            unscaled[p] = {}
            for key in nlp.states.keys():
                unscaled[p][key] = [None] * nlp.n_states_nodes
                for ns, sol_ns in enumerate(integrated_sol):
                    unscaled[p][key][ns] = sol_ns[nlp.states[key].index, :]

        self._stepwise_states = SolutionData.from_unscaled(self.ocp, unscaled, "x")

    def interpolate(self, n_frames: int | list | tuple, scaled: bool = False):
        """
        Interpolate the states

        Parameters
        ----------
        n_frames: int | list | tuple
            If the value is an int, the Solution returns merges the phases,
            otherwise, it interpolates them independently
        scaled: bool
            If the states should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as the
            model needs temps). If you don't know what it means, you probably want the unscaled version.

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        if self._stepwise_states is None:
            self._integrate_stepwise()

        # Get the states, but do not bother the duplicates now
        if isinstance(n_frames, int):  # So merge phases
            t_all = []
            last_t = 0
            for p in range(len(self.ocp.nlp)):
                t_all.append(np.concatenate(self._stepwise_times[p]) + last_t)
                last_t = t_all[-1][-1]
            t_all = [np.concatenate(t_all)]

            states = [self._stepwise_states.to_dict(scaled=scaled, to_merge=SolutionMerge.ALL)]
            n_frames = [n_frames]

        elif not isinstance(n_frames, (list, tuple)) or len(n_frames) != len(self._stepwise_states.unscaled):
            raise ValueError(
                "n_frames should either be a int to merge_phases phases "
                "or a list of int of the number of phases dimension"
            )
        else:
            t_all = [np.concatenate(self._stepwise_times[p]) for p in range(len(self.ocp.nlp))]
            states = self._stepwise_states._merge_nodes(scaled=scaled)

        data = []
        for p in range(len(states)):
            data.append({})

            nlp = self.ocp.nlp[p]

            # Now remove the duplicates
            t_round = np.round(t_all[p], decimals=8)  # Otherwise, there are some numerical issues with np.unique
            t, idx = np.unique(t_round, return_index=True)
            x = states[p][:, idx]
            
            x_interpolated = np.ndarray((x.shape[0], n_frames[p]))
            t_interpolated = np.linspace(t_round[0], t_round[-1], n_frames[p])
            for j in range(x.shape[0]):
                s = sci_interp.splrep(t, x[j, :], k=1)
                x_interpolated[j, :] = sci_interp.splev(t_interpolated, s)[:, 0]
            
            for key in nlp.states.keys():
                data[p][key] = x_interpolated[nlp.states[key].index, :]

        return data if len(data) > 1 else data[0]

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

        if shooting_type:
            self.integrate(shooting_type=shooting_type)

        for idx_phase in range(len(self.ocp.nlp)):
            for objective in self.ocp.nlp[idx_phase].J:
                if objective.target is not None:
                    if objective.type in (
                        ObjectiveFcn.Mayer.TRACK_MARKERS,
                        ObjectiveFcn.Lagrange.TRACK_MARKERS,
                    ) and objective.node[0] in (Node.ALL, Node.ALL_SHOOTING):
                        n_frames += objective.target[0].shape[2]
                        break

        if n_frames == 0:
            try:
                data_to_animate = self.interpolate(sum([nlp.ns for nlp in self.ocp.nlp]) + 1)
            except RuntimeError:
                pass
        elif n_frames > 0:
            data_to_animate = self.interpolate(n_frames)

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

        all_bioviz = self.ocp.nlp[0].model.animate(
            self.ocp,
            solution=data_to_animate,
            show_now=show_now,
            tracked_markers=tracked_markers,
            **kwargs,
        )

        return all_bioviz

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
            n_states_nodes = self.ocp.nlp[phase].n_states_nodes
            if type(nlp.ode_solver) == OdeSolver.COLLOCATION:
                n_states_nodes -= 1

            tracked_markers = None
            for objective in nlp.J:
                if objective.target is not None:
                    if objective.type in (
                        ObjectiveFcn.Mayer.TRACK_MARKERS,
                        ObjectiveFcn.Lagrange.TRACK_MARKERS,
                    ) and objective.node[0] in (Node.ALL, Node.ALL_SHOOTING):
                        tracked_markers = np.full((3, nlp.model.nb_markers, n_states_nodes), np.nan)

                        for i in range(len(objective.rows)):
                            tracked_markers[objective.rows[i], objective.cols, :] = objective.target[0][i, :, :]

                        missing_row = np.where(np.isnan(tracked_markers))[0]
                        if missing_row.size > 0:
                            tracked_markers[missing_row, :, :] = 0

            # interpolation
            if n_frames > 0 and tracked_markers is not None:
                x = np.linspace(0, n_states_nodes - 1, n_states_nodes)
                xnew = np.linspace(0, n_states_nodes - 1, n_frames)
                f = interp1d(x, tracked_markers, kind="cubic")
                tracked_markers = f(xnew)

            all_tracked_markers.append(tracked_markers)

        return all_tracked_markers

    def _get_penalty_cost(self, nlp, penalty):
        if nlp is None:
            raise NotImplementedError("penalty cost over the full ocp is not implemented yet")

        val = []
        val_weighted = []
        
        phases_dt = PenaltyHelpers.phases_dt(penalty, self.ocp, lambda p: np.array([self.phases_dt[idx] for idx in p]))
        params = PenaltyHelpers.parameters(penalty, lambda: np.array([self._parameters.scaled[0][key] for key in self._parameters.scaled[0].keys()]))

        merged_x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        merged_u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        merged_s = self._stochastic.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        for idx in range(len(penalty.node_idx)):
            t0 = PenaltyHelpers.t0(penalty, idx, lambda p_idx, n_idx: self._stepwise_times[p_idx][n_idx])
            x = PenaltyHelpers.states(penalty, idx, lambda p_idx, n_idx, sn_idx: merged_x[p_idx][n_idx][:, sn_idx])
            u = PenaltyHelpers.controls(penalty, idx, lambda p_idx, n_idx, sn_idx: merged_u[p_idx][n_idx][:, sn_idx])
            s = PenaltyHelpers.states(penalty, idx, lambda p_idx, n_idx, sn_idx: merged_s[p_idx][n_idx][:, sn_idx])
            weight = PenaltyHelpers.weight(penalty)
            target = PenaltyHelpers.target(penalty, idx)

            node_idx = penalty.node_idx[idx]
            val.append(penalty.function[node_idx](t0, phases_dt, x, u, params, s))
            val_weighted.append(penalty.weighted_function[node_idx](t0, phases_dt, x, u, params, s, weight, target))

        if self.ocp.n_threads > 1:
            val = [v[:, 0] for v in val]
            val_weighted = [v[:, 0] for v in val_weighted]

        val = np.nansum(val)
        val_weighted = np.nansum(val_weighted)

        return val, val_weighted

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
