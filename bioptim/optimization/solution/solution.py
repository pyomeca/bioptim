from typing import Any
from copy import deepcopy

import numpy as np
from scipy import interpolate as sci_interp
from scipy.interpolate import interp1d
from casadi import vertcat, DM
from matplotlib import pyplot as plt

from ...limits.objective_functions import ObjectiveFcn
from ...limits.path_conditions import InitialGuess, InitialGuessList
from ...limits.penalty_helpers import PenaltyHelpers
from ...misc.enums import ControlType, CostType, Shooting, InterpolationType, SolverType, SolutionIntegrator, Node
from ...dynamics.ode_solver import OdeSolver
from ...interfaces.solve_ivp_interface import solve_ivp_bioptim_interface

from ..optimization_vector import OptimizationVectorHelper


def _to_unscaled_values(scaled: list, ocp, variable_type: str) -> list:
    """
    Convert values of scaled solution to unscaled values

    Parameters
    ----------
    scaled: list
        The scaled values
    variable_type: str
        The type of variable to convert (x for states, u for controls, p for parameters, s for stochastic variables)
    """

    unscaled: list = [None for _ in range(len(scaled))]
    for phase in range(len(scaled)):
        unscaled[phase] = {}
        for key in scaled[phase].keys():
            scale_factor = getattr(ocp.nlp[phase], f"{variable_type}_scaling")[key]
            if isinstance(scaled[phase][key], list):  # Nodes are not merged
                unscaled[phase][key] = []
                for node in range(len(scaled[phase][key])):
                    value = scaled[phase][key][node]
                    unscaled[phase][key].append(value * scale_factor.to_array(value.shape[1]))
            elif isinstance(scaled[phase][key], np.ndarray):  # Nodes are merged
                value = scaled[phase][key]
                unscaled[phase][key] = value * scale_factor.to_array(value.shape[1])
            else:
                raise ValueError(f"Unrecognized type {type(scaled[phase][key])} for {key}")

    return unscaled


def _to_scaled_values(unscaled: list, ocp, variable_type: str) -> list:
    """
    Convert values of unscaled solution to scaled values

    Parameters
    ----------
    unscaled: list
        The unscaled values
    variable_type: str
        The type of variable to convert (x for states, u for controls, p for parameters, s for stochastic variables)
    """

    if not unscaled:
        return []

    scaled: list = [None for _ in range(len(unscaled))]
    for phase in range(len(unscaled)):
        scaled[phase] = {}
        for key in unscaled[phase].keys():
            scale_factor = getattr(ocp.nlp[phase], f"{variable_type}_scaling")[key]

            if isinstance(unscaled[phase][key], list):  # Nodes are not merged
                scaled[phase][key] = []
                for node in range(len(unscaled[phase][key])):
                    value = unscaled[phase][key][node]
                    scaled[phase][key].append(value / scale_factor.to_array(value.shape[1]))
            elif isinstance(unscaled[phase][key], np.ndarray):  # Nodes are merged
                value = unscaled[phase][key]
                scaled[phase][key] = value / scale_factor.to_array(value.shape[1])
            else:
                raise ValueError(f"Unrecognized type {type(unscaled[phase][key])} for {key}")

    return scaled


def _merge_dict_keys(data, ocp):
    """
    Merge the keys of a SolutionData.unscaled or SolutionData.scaled

    Parameters
    ----------
    data
        The data to merge
    ocp
        The ocp

    Returns
    -------
    The merged data
    """

    # This is a bit complex, but it's just to concatenate all the value of the keys together
    out = []
    has_empty = False
    for p in range(len(data)):
        tp_phases = []
        for n in range(ocp.nlp[p].n_states_nodes):
            tp_nodes = []
            for key in data[0].keys():
                tp_nodes.append(data[p][key][n])

            if not tp_nodes:
                has_empty = True
                tp_phases.append(np.ndarray((0, 1)))
                continue
            elif has_empty:
                raise RuntimeError("Cannot merge empty and non-empty structures")
            tp_phases.append(np.concatenate(tp_nodes))

        out.append(tp_phases)
    return out


class SolutionData:
    def __init__(self, unscaled, scaled, n_nodes: list[int, ...]):
        """
        Parameters
        ----------
        ... # TODO
        n_nodes: list
            The number of node at each phase
        """
        self.unscaled = unscaled
        self.scaled = scaled
        self.n_phases = len(self.unscaled)
        self.n_nodes = n_nodes  # This is painfully necessary to get from outside to merge key if no keys are available

    def __getitem__(self, **keys):
        phase = 0
        if len(self.unscaled) > 1:
            if "phase" not in keys:
                raise RuntimeError("You must specify the phase when more than one phase is present in the solution")
            phase = keys["phase"]
        key = keys["key"]
        return self.unscaled[phase][key]

    def keys(self, phase: int = 0):
        return self.unscaled[phase].keys()

    def merge_phases(self, scaled: bool = False):
        """
        Merge the phases by merging keys and nodes before. 
        This method does not remove the redundent nodes when merging the phase nor the nodes
        """

        return np.concatenate([self.merge_nodes(phase=phase, scaled=scaled) for phase in range(self.n_phases)], axis=1)

    def merge_nodes(self, phase: int, scaled: bool = False):
        """
        Merge the steps by merging keys before.
        """
        if not self.keys():
            return np.ndarray((0, 1))

        out = [self.merge_keys(phase=phase, node=node, scaled=scaled) for node in range(self.n_nodes[phase])]
        return np.concatenate(out, axis=1)
    
    def merge_keys(self, phase: int, node: int = None, scaled: bool = False):
        if not self.keys():
            return np.ndarray((0, 1))

        data = self.scaled if scaled else self.unscaled
        return np.concatenate([data[phase][key][node] for key in self.keys()], axis=0)
    


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
            n_phases_node = [nlp.n_states_nodes for nlp in self.ocp.nlp]
            self._decision_states = SolutionData(x, _to_unscaled_values(x, ocp, "x"), n_nodes=n_phases_node)
            self._stepwise_controls = SolutionData(u, _to_unscaled_values(u, ocp, "u"), n_nodes=n_phases_node)
            self._parameters = SolutionData(p, _to_unscaled_values(p, ocp, "p"), n_nodes=n_phases_node)
            self._stochastic = SolutionData(s, _to_unscaled_values(s, ocp, "s"), n_nodes=n_phases_node)

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

        if not (isinstance(sol, (list, tuple)) and len(sol) == 4):
            raise ValueError("_sol should be a list of tuple and the length should be 4")

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

        vector = np.ndarray((0, 1))
        sol_states, sol_controls, sol_params, sol_stochastic_variables = sol

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
                ocp.nlp[p].controls[key].node_index = 0
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
                ocp.nlp[p].stochastic_variables[key].node_index = 0
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

    def time(self, t_span: bool = False) -> list:
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

        return self._decision_times if t_span else self._stepwise_times

    def decision_states(self, scaled: bool = False, concatenate_keys: bool = False):
        """
        Returns the decision states

        Parameters
        ----------
        scaled: bool
            If the decision states should be scaled or not (note that scaled is as Ipopt received them, while unscaled
            is as the model needs temps). If you don't know what it means, you probably want the unscaled version.
        concatenate_keys: bool
            If the states should be returned individually (False) or concatenated (True). If individual, then the
            return value does not contain the key dict.

        Returns
        -------
        The decision variables
        """

        data = deepcopy(self._decision_states.scaled if scaled else self._decision_states.unscaled)
        if concatenate_keys:
            data = _merge_dict_keys(data, self.ocp)

        return data if len(data) > 1 else data[0]

    def stepwise_states(self, scaled: bool = False, concatenate_keys: bool = False):
        """
        Returns the stepwise integrated states

        Parameters
        ----------
        scaled: bool
            If the states should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as the
            model needs temps). If you don't know what it means, you probably want the unscaled version.
        concatenate_keys: bool
            If the states should be returned individually (False) or concatenated (True). If individual, then the
            return value does not contain the key dict.

        Returns
        -------
        The stepwise integrated states
        """

        if self._stepwise_states is None:
            self._integrate_stepwise()
        data = self._stepwise_states.scaled if scaled else self._stepwise_states.unscaled
        if concatenate_keys:
            data = _merge_dict_keys(data, self.ocp)
        return data if len(data) > 1 else data[0]

    def controls(self, scaled: bool = False, concatenate_keys: bool = False):
        """
        Returns the controls. Note the final control is always present but set to np.nan if it is not defined

        Parameters
        ----------
        scaled: bool
            If the controls should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as
            the model needs temps). If you don't know what it means, you probably want the unscaled version.
        concatenate_keys: bool
            If the controls should be returned individually (False) or concatenated (True). If individual, then the
            return value does not contain the key dict.

        Returns
        -------
        The controls
        """

        data = self._stepwise_controls.scaled if scaled else self._stepwise_controls.unscaled
        if concatenate_keys:
            data = _merge_dict_keys(data, self.ocp)
        return data if len(data) > 1 else data[0]

    def parameters(self, scaled: bool = False, concatenate_keys: bool = False):
        """
        Returns the parameters

        Parameters
        ----------
        scaled: bool
            If the parameters should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as
            the model needs temps). If you don't know what it means, you probably want the unscaled version.

        Returns
        -------
        The parameters
        """

        data = self._parameters.scaled[0] if scaled else self._parameters.unscaled[0]
        if concatenate_keys:
            data = _merge_dict_keys(data, self.ocp)
        return data

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

        data = self._stochastic.scaled if scaled else self._stochastic.unscaled
        if concatenate_keys:
            data = _merge_dict_keys(data, self.ocp)
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

        new._dt = deepcopy(self._dt)
        new._stepwise_times = deepcopy(self._stepwise_times)

        if not skip_data:
            new._decision_states = deepcopy(self._decision_states)
            new._stepwise_states = deepcopy(self._stepwise_states)

            new._stepwise_controls = deepcopy(self._stepwise_controls)

            new._stochastic = deepcopy(self._stochastic)
            new._parameters = deepcopy(self._parameters)
        return new

    def _integrate_stepwise(self) -> None:
        """
        This method integrate to stepwise level the states. That is the states that are used in the dynamics and 
        continuity constraints.

        Returns
        -------
        dict
            The integrated data structure similar in structure to the original _decision_states
        """

        # Prepare the output
        params = vertcat(*[self._parameters.unscaled[0][key] for key in self._parameters.keys()])

        unscaled: list = [None] * len(self.ocp.nlp)
        
        for p, nlp in enumerate(self.ocp.nlp):
            t = self._decision_times[p]

            x = [self._decision_states.merge_keys(phase=p, node=n) for n in range(nlp.n_states_nodes)]
            u = [self._stepwise_controls.merge_keys(phase=p, node=n) for n in range(nlp.n_states_nodes)]
            s = [self._stochastic.merge_keys(phase=p, node=n) for n in range(nlp.n_states_nodes)]

            integrated_sol = solve_ivp_bioptim_interface(
                shooting_type=Shooting.MULTIPLE, dynamics_func=nlp.dynamics, t=t, x=x, u=u, s=s, p=params
            )
            
            unscaled[p] = {}
            for key in nlp.states.keys():
                unscaled[p][key] = [None] * nlp.n_states_nodes
                for ns, sol_ns in enumerate(integrated_sol):
                    unscaled[p][key][ns] = sol_ns[nlp.states[key].index, :]

        n_nodes = self._decision_states.n_nodes
        self._stepwise_states = SolutionData(unscaled, _to_scaled_values(unscaled, self.ocp, "x"), n_nodes)

    def interpolate(self, n_frames: int | list | tuple, scaled: bool = False) -> SolutionData:
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
        t_all = [np.concatenate(self._stepwise_times[p]) for p in range(len(self.ocp.nlp))]
        if isinstance(n_frames, int):  # So merge phases
            states = [self._stepwise_states.merge_phases(scaled=scaled)]
            t_all = [np.concatenate(t_all)]
            n_frames = [n_frames]

        elif not isinstance(n_frames, (list, tuple)) or len(n_frames) != len(self._stepwise_states.unscaled):
            raise ValueError(
                "n_frames should either be a int to merge_phases phases "
                "or a list of int of the number of phases dimension"
            )
        else:
            states = self._stepwise_states.merge_nodes(scaled=scaled)

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

        time_tp = []
        for t in new._time_for_integration:
            time_tp.extend(t)
        new._time_for_integration = time_tp

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

        stepwise_times = []
        if len(self._stepwise_times) == 1:
            stepwise_times = deepcopy(self._stepwise_times)
        elif len(self._stepwise_times) > 1:
            raise NotImplementedError("Merging phases is not implemented yet")
    
        out_decision_states = []
        if len(self._decision_states["scaled"]) == 1:
            out_decision_states = deepcopy(self._decision_states)
        elif len(self._decision_states["scaled"]) > 1:
            raise NotImplementedError("Merging phases is not implemented yet")
            out_decision_states_scaled = (
                _merge(self._states["scaled"], is_control=False) if not skip_states and self._states["scaled"] else None
            )
            out_states = _merge(self._states["unscaled"], is_control=False) if not skip_states else None
        
        out_stepwise_states = []
        if len(self._stepwise_states.scaled) == 1:
            out_stepwise_states = deepcopy(self._stepwise_states)
        elif len(self._stepwise_states.scaled) > 1:
            raise NotImplementedError("Merging phases is not implemented yet")

        out_stepwise_controls = []
        if len(self._stepwise_controls["scaled"]) == 1:
            out_stepwise_controls = deepcopy(self._stepwise_controls["scaled"])
        elif len(self._stepwise_controls["scaled"]) > 1:
            raise NotImplementedError("Merging phases is not implemented yet")
            out_controls_scaled = (
                _merge(self._controls["scaled"], is_control=True)
                if not skip_controls and self._controls["scaled"]
                else None
            )
            out_controls = _merge(self._controls["unscaled"], is_control=True) if not skip_controls else None
        
        out_stochastic = []
        if len(self._stochastic["scaled"]) == 1:
            out_stochastic = deepcopy(self._stochastic)
        elif len(self._stochastic["scaled"]) > 1:
            raise NotImplementedError("Merging phases is not implemented yet")
            out_stochastic_variables_scaled = (
                _merge(self._stochastic_variables["scaled"], is_control=False)
                if not skip_stochastic and self._stochastic_variables["scaled"]
                else None
            )
            out_stochastic_variables = (
                _merge(self._stochastic_variables["unscaled"], is_control=False) if not skip_stochastic else None
            )

        return stepwise_times, out_decision_states, out_stepwise_states, out_stepwise_controls, out_stochastic

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
        if nlp is None:
            raise NotImplementedError("penalty cost over the full ocp is not implemented yet")

        val = []
        val_weighted = []
        
        phases_dt = PenaltyHelpers.phases_dt(penalty, lambda: np.array(self.phases_dt))
        params = PenaltyHelpers.parameters(penalty, lambda: np.array([self._parameters.scaled[0][key] for key in self._parameters.scaled[0].keys()]))
        for node_idx in penalty.node_idx:
            t0 = PenaltyHelpers.t0(penalty, node_idx, lambda p_idx, n_idx: self._stepwise_times[p_idx][n_idx])
            x = PenaltyHelpers.states(penalty, node_idx, lambda p_idx, n_idx: self._decision_states.merge_keys(phase=p_idx, node=n_idx, scaled=True))
            u = PenaltyHelpers.controls(penalty, node_idx, lambda p_idx, n_idx: self._stepwise_controls.merge_keys(phase=p_idx, node=n_idx, scaled=True))
            s = PenaltyHelpers.stochastic(penalty, node_idx, lambda p_idx, n_idx: self._stochastic.merge_keys(phase=p_idx, node=n_idx, scaled=True))
            weight = PenaltyHelpers.weight(penalty)
            target = PenaltyHelpers.target(penalty, node_idx)

            # PenaltyHelpers._get_x_u_s_at_idx(penalty, node_idx, x, u, s)

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
