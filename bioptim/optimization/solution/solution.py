from typing import Any
from copy import deepcopy

import numpy as np
from scipy import interpolate as sci_interp
from scipy.interpolate import interp1d
from casadi import vertcat, DM, Function
from matplotlib import pyplot as plt

from ...limits.objective_functions import ObjectiveFcn
from ...limits.path_conditions import InitialGuess, InitialGuessList
from ...misc.enums import (
    ControlType,
    CostType,
    Shooting,
    InterpolationType,
    SolverType,
    SolutionIntegrator,
    Node,
    QuadratureRule,
    PhaseDynamics,
)
from ...dynamics.ode_solver import OdeSolver
from ...interfaces.solve_ivp_interface import solve_ivp_interface, solve_ivp_bioptim_interface

from ..optimization_vector import OptimizationVectorHelper

from .utils import concatenate_optimization_variables_dict


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
    _decision_states: dict[list[dict[list[np.ndarray], ...], ...], list[dict[list[np.ndarray], ...], ...]]
        A dict ("unscaled", "scaled") of a list (n_phase) of dict (n_states) of list (n_shooting) of np.ndarray (n_elements), 
        based solely on the solution
    _stepwise_states: dict[list[dict[list[np.ndarray], ...], ...], list[dict[list[np.ndarray], ...], ...]]
        A dict ("unscaled", "scaled") of a list (n_phase) of dict (n_states) of list (n_shooting) of np.ndarray (n_steps),
        based on the integrated solution directly from the bioptim integrator
    _whole_interval_states: dict[list[dict[list[np.ndarray], ...], ...], list[dict[list[np.ndarray], ...], ...]]
        A dict ("unscaled", "scaled") of a list (n_phase) of dict (n_states) of list (n_shooting) of np.ndarray (n_steps + x),
        based on the integrated solution directly from the bioptim integrator that is 
        ensure to include the starting point and the final point of the shooting node
    _controls: list
        The data structure that holds the controls
    parameters: dict
        The data structure that holds the parameters
    _stochastic_variables: list
        The data structure that holds the stochastic variables
    _integrated_values: list
        The data structure that holds the update values
    _dt: list
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
        ocp: "OptimalControlProgram",
        ns: list[float] = None,
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
        ns: list[float]
            The number of shooting points for each phase
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
        _decision_states: list
            The states of the solution
        _stepwise_states: list
            The integrated states of the solution
        _whole_interval_states: list
            The fully integrated states of the solution
        _stepwise_controls: list
            The controls of the solution
        parameters: dict
            The parameters of the solution
        _stochastic_variables: list
            The stochastic variables of the solution
        """

        self.ocp = ocp

        # Penalties
        self._cost, self._detailed_cost, self.constraints = cost, None, constraints

        # Solver options
        self.status, self.iterations = status, iterations
        self.lam_g, self.lam_p, self.lam_x, self.inf_pr, self.inf_du = lam_g, lam_p, lam_x, inf_pr, inf_du
        self.solver_time_to_optimize, self.real_time_to_optimize = solver_time_to_optimize, real_time_to_optimize

        # Extract the data now for further use
        self._decision_states = {"scaled": [], "unscaled": []}
        self._stepwise_states = {"scaled": [], "unscaled": []}
        self._interpolated_states = {"scaled": [], "unscaled": []}
        self._whole_interval_states = {"scaled": [], "unscaled": []}
        self._stepwise_controls = {"scaled": [], "unscaled": []}
        self._parameters = {"scaled": [], "unscaled": []}
        self._stochastic = {"scaled": [], "unscaled": []}

        self.vector = vector
        if self.vector is not None:
            self._dt = OptimizationVectorHelper.extract_dt(ocp, vector)
            self._stepwise_times = OptimizationVectorHelper.extract_step_times(ocp, vector)
            
            x, c, p, s = OptimizationVectorHelper.to_dictionaries(ocp, vector)
            self._decision_states = {"scaled": x, "unscaled": self._to_unscaled_values(x, "x")}
            self._stepwise_controls = {"scaled": c, "unscaled": self._to_unscaled_values(c, "u")}
            self._parameters = {"scaled": p, "unscaled": self._to_unscaled_values(p, "p")}
            self._stochastic = {"scaled": s, "unscaled": self._to_unscaled_values(s, "s")}


    @classmethod
    def from_dict(cls, ocp, _sol: dict):
        """
        Initialize all the attributes from an Ipopt-like dictionary data structure

        Parameters
        ----------
        _ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        _sol: dict
            The solution in a Ipopt-like dictionary
        """

        if not isinstance(_sol, dict):
            raise ValueError("The _sol entry should be a dictionary")

        is_ipopt = _sol["solver"] == SolverType.IPOPT.value

        return cls(
            ocp=ocp,
            ns=[nlp.ns for nlp in ocp.nlp],
            vector=_sol["x"],
            cost=_sol["f"] if is_ipopt else None,
            constraints=_sol["g"] if is_ipopt else None,
            lam_g=_sol["lam_g"] if is_ipopt else None,
            lam_p=_sol["lam_p"] if is_ipopt else None,
            lam_x=_sol["lam_x"] if is_ipopt else None,
            inf_pr=_sol["inf_pr"] if is_ipopt else None,
            inf_du=_sol["inf_du"] if is_ipopt else None,
            solver_time_to_optimize=_sol["solver_time_to_optimize"],
            real_time_to_optimize=_sol["real_time_to_optimize"],
            iterations=_sol["iter"],
            status=_sol["status"],
        )

    @classmethod
    def from_initial_guess(cls, ocp, _sol: list):
        """
        Initialize all the attributes from a list of initial guesses (states, controls)

        Parameters
        ----------
        _ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        _sol: list
            The list of initial guesses
        """

        if not (isinstance(_sol, (list, tuple)) and len(_sol) == 4):
            raise ValueError("_sol should be a list of tuple and the length should be 4")

        n_param = len(ocp.parameters)
        all_ns = [nlp.ns for nlp in ocp.nlp]

        # Sanity checks
        for i in range(len(_sol)):  # Convert to list if necessary and copy for as many phases there are
            if isinstance(_sol[i], InitialGuess):
                tp = InitialGuessList()
                for _ in range(len(all_ns)):
                    tp.add(deepcopy(_sol[i].init), interpolation=_sol[i].init.type)
                _sol[i] = tp
        if sum([isinstance(s, InitialGuessList) for s in _sol]) != 4:
            raise ValueError(
                "solution must be a solution dict, "
                "an InitialGuess[List] of len 4 (states, controls, parameters, stochastic_variables), "
                "or a None"
            )
        if sum([len(s) != len(all_ns) if p != 3 else False for p, s in enumerate(_sol)]) != 0:
            raise ValueError("The InitialGuessList len must match the number of phases")
        if n_param != 0:
            if len(_sol) != 3 and len(_sol[3]) != 1 and _sol[3][0].shape != (n_param, 1):
                raise ValueError(
                    "The 3rd element is the InitialGuess of the parameter and "
                    "should be a unique vector of size equal to n_param"
                )

        vector = np.ndarray((0, 1))
        sol_states, sol_controls, sol_params, sol_stochastic_variables = _sol

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

        return cls(ocp=ocp, ns=[nlp.ns for nlp in ocp.nlp], vector=vector)

    @classmethod
    def from_vector(cls, ocp, _sol: np.ndarray | DM):
        """
        Initialize all the attributes from a vector of solution

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        _sol: np.ndarray | DM
            The solution in vector format
        """

        if not isinstance(_sol, (np.ndarray, DM)):
            raise ValueError("The _sol entry should be a np.ndarray or a DM.")

        return cls(ocp=ocp, ns=[nlp.ns for nlp in ocp.nlp], vector=_sol)

    @classmethod
    def from_ocp(cls, ocp):
        """
        Initialize all the attributes from a vector of solution

        Parameters
        ----------
        _ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        """

        return cls(ocp=ocp)

    @property
    def interpolated_states(self):
        data = self._interpolated_states["unscaled"]
        if data is None:
            raise RuntimeError("You must call interpolate() before accessing the interpolated states")

        return data[0] if len(data) == 1 else data

    def _to_unscaled_values(self, scaled, variable_type) -> list:
        """
        Convert values of scaled solution to unscaled values

        Parameters
        ----------
        scaled: list
            The scaled values
        variable_type: str
            The type of variable to convert (x for states, u for controls, p for parameters, s for stochastic variables)
        """
        
        if not scaled: 
            return []

        unscaled = [None for _ in range(len(scaled))]
        for phase in range(len(scaled)):
            unscaled[phase] = {}
            for key in scaled[phase].keys():
                scale_factor = getattr(self.ocp.nlp[phase], f"{variable_type}_scaling")[key]
                if isinstance(scaled[phase][key], list):
                    # This is if decision variables are sent
                    unscaled[phase][key] = []
                    for node in range(len(scaled[phase][key])):
                        value = scaled[phase][key][node]
                        unscaled[phase][key].append(value * scale_factor.to_array(value.shape[1]))
                elif isinstance(scaled[phase][key], np.ndarray):
                    # This is if interpolated values are sent
                    value = scaled[phase][key]
                    unscaled[phase][key] = value * scale_factor.to_array(value.shape[1])
                else: 
                    raise ValueError(f"Unrecognized type {type(scaled[phase][key])} for {key}")
                
        return unscaled
    
    def _to_scaled_values(self, unscaled, variable_type) -> list:
        """
        Convert values of unscaled solution to scaled values

        Parameters
        ----------
        scaled: list
            The unscaled values
        variable_type: str
            The type of variable to convert (x for states, u for controls, p for parameters, s for stochastic variables)
        """
        
        if not unscaled: 
            return []

        scaled = [None for _ in range(len(unscaled))]
        for phase in range(len(unscaled)):
            scaled[phase] = {}
            for key in unscaled[phase].keys():
                scale_factor = getattr(self.ocp.nlp[phase], f"{variable_type}_scaling")[key]

                if isinstance(unscaled[phase][key], list):
                    # This is if decision variables are sent
                    scaled[phase][key] = []
                    for node in range(len(unscaled[phase][key])):
                        value = unscaled[phase][key][node]
                        scaled[phase][key].append(value / scale_factor.to_array(value.shape[1]))
                elif isinstance(unscaled[phase][key], np.ndarray):
                    # This is if interpolated values are sent
                    value = unscaled[phase][key]
                    scaled[phase][key] = value / scale_factor.to_array(value.shape[1])
                else:
                    raise ValueError(f"Unrecognized type {type(unscaled[phase][key])} for {key}")

        return scaled

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
            new._decision_states["unscaled"] = deepcopy(self._decision_states["unscaled"])
            new._decision_states["scaled"] = deepcopy(self._decision_states["scaled"])

            new._stepwise_states["unscaled"] = deepcopy(self._stepwise_states["unscaled"])
            new._stepwise_states["scaled"] = deepcopy(self._stepwise_states["scaled"])

            new._interpolated_states["unscaled"] = deepcopy(self._interpolated_states["unscaled"])
            new._interpolated_states["scaled"] = deepcopy(self._interpolated_states["scaled"])

            new._whole_interval_states["unscaled"] = deepcopy(self._whole_interval_states["unscaled"])
            new._whole_interval_states["scaled"] = deepcopy(self._whole_interval_states["scaled"])
            
            new._stepwise_controls["unscaled"] = deepcopy(self._stepwise_controls["unscaled"])
            new._stepwise_controls["scaled"] = deepcopy(self._stepwise_controls["scaled"])
            
            new._stochastic["unscaled"] = deepcopy(self._stochastic["unscaled"])
            new._stochastic["scaled"] = deepcopy(self._stochastic["scaled"])
            
            new._parameters["unscaled"] = deepcopy(self._parameters["unscaled"])
            new._parameters["scaled"] = deepcopy(self._parameters["scaled"])
        return new

    def integrate(
        self,
        shooting_type: Shooting = Shooting.SINGLE,
        keep_intermediate_points: bool = False,
        integrator: SolutionIntegrator = SolutionIntegrator.SCIPY_RK45,
    ) -> "Solution":
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

        out = self._perform_integration(
            shooting_type=shooting_type,
            keep_intermediate_points=keep_intermediate_points,
            integrator=integrator,
        )

        return out

    def _integrate_stepwise(self) -> dict:
        """
        This method integrate to stepwise level the states. That is the states that are used in the dynamics and 
        continuity constraints.

        Returns
        -------
        dict
            The integrated data structure similar in structure to the original _decision_states
        """

        # Prepare the output
        out = {"unscaled": None, "scaled": None}

        params = vertcat(*[self._parameters[key][0] for key in self._parameters["unscaled"][0].keys()])

        out["unscaled"] = [None] * len(self.ocp.nlp)
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            t = [vertcat(t[0], t[-1]) for t in self._stepwise_times[phase_idx]]
            x = [
                self._add_node(None, self._decision_states["scaled"], phase_idx, n, nlp.states.keys()) 
                for n in range(nlp.n_states_nodes)
            ]
            u = [
                self._add_node(None, self._stepwise_controls["scaled"], phase_idx, n, nlp.controls.keys())
                for n in range(nlp.n_states_nodes)
            ]
            s = [
                self._add_node(None, self._stochastic["scaled"], phase_idx, n, nlp.stochastic_variables.keys())
                for n in range(nlp.n_states_nodes)
            ]

            integrated_sol = solve_ivp_bioptim_interface(
                shooting_type=Shooting.MULTIPLE, dynamics_func=nlp.dynamics, t=t, x=x, u=u, s=s, p=params
            )
            
            out["unscaled"][phase_idx] = {}
            for key in nlp.states.keys():
                out["unscaled"][phase_idx][key] = [None] * nlp.n_states_nodes
                for ns, sol_ns in enumerate(integrated_sol):
                    out["unscaled"][phase_idx][key][ns] = sol_ns[nlp.states[key].index, :]

        out["scaled"] = self._to_scaled_values(out["unscaled"], "x")
        return out

    def interpolate(self, n_frames: int | list | tuple) -> "Solution":
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
        if not self._stepwise_states["unscaled"]:
            out._stepwise_states = self._integrate_stepwise()

        
        t_all = [np.concatenate(self._stepwise_times[p]) for p in range(len(self.ocp.nlp))]
        if isinstance(n_frames, int):
            _, _, data_states, _, _ = out._merge_phases(skip_controls=True)
            t_all = [np.concatenate(t_all)]
            n_frames = [n_frames]
            
        elif isinstance(n_frames, (list, tuple)) and len(n_frames) == len(self._states["unscaled"]):
            data_states = self._states["unscaled"]
        else:
            raise ValueError(
                "n_frames should either be a int to merge_phases phases "
                "or a list of int of the number of phases dimension"
            )

        out._interpolated_states = {"unscaled": [], "scaled": []}
        for p in range(len(data_states["unscaled"])):
            out._interpolated_states["unscaled"].append({})

            nlp = self.ocp.nlp[p]
            # Get the states, but do not bother the duplicates
            x = None
            for node in range(nlp.n_states_nodes):
                x = self._add_node(x, data_states["scaled"], p, node, nlp.states.keys()) 
            
            # Now remove the duplicates
            t_round = np.round(t_all[p], decimals=8)  # Otherwise, there are some numerical issues with np.unique
            t, idx = np.unique(t_round, return_index = True)
            x = x[:, idx]
            
            x_interpolated = np.ndarray((x.shape[0], n_frames[p]))
            t_interpolated = np.linspace(t_round[0], t_round[-1], n_frames[p])
            for j in range(x.shape[0]):
                s = sci_interp.splrep(t, x[j, :], k=1)
                x_interpolated[j, :] = sci_interp.splev(t_interpolated, s)[:, 0]
            
            for key in nlp.states.keys():
                out._interpolated_states["unscaled"][p][key] = x_interpolated[nlp.states[key].index, :]

        out._interpolated_states["scaled"] = self._to_scaled_values(out._interpolated_states["unscaled"], "x")
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
        if len(self._stepwise_states["scaled"]) == 1:
            out_stepwise_states = deepcopy(self._stepwise_states)
        elif len(self._stepwise_states["scaled"]) > 1:
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
        
        data_to_animate = self.integrate(shooting_type=shooting_type) if shooting_type else self.copy()

        for idx_phase in range(len(data_to_animate.ocp.nlp)):
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
                data_to_animate = data_to_animate.interpolate(sum([nlp.ns for nlp in self.ocp.nlp]) + 1)
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

        all_bioviz = self.ocp.nlp[0].model.animate(
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

    @staticmethod
    def _add_node(concatenate_with: np.ndarray, data, phase_idx, node_idx, keys) -> np.ndarray:
        if not keys:
            return np.array(())
        out = np.concatenate([data[phase_idx][key][node_idx] for key in keys])
        if concatenate_with is None:
            return out
        else:
            return np.concatenate((concatenate_with, out), axis=1)

    def _get_penalty_cost(self, nlp, penalty):
        phase_idx = nlp.phase_idx
        nlp.controls.node_index = 0  # This is so "keys" is not empty

        val = []
        val_weighted = []
        p = vertcat(*[self._parameters["scaled"][0][key] for key in self._parameters["scaled"][0].keys()])

        phase_dt = self._dt[phase_idx]
        dt = penalty.dt_to_float(phase_dt)

        for node_idx in penalty.node_idx:
            if penalty.transition or penalty.multinode_penalty:
                phases = penalty.nodes_phase
                nodes = penalty.multinode_idx
            else:
                phases = [phase_idx]
                nodes = [node_idx]
                if penalty.explicit_derivative:
                    nodes.append(node_idx + 1)
            
            t = self._stepwise_times[phases[0]][nodes[0]][0]  # Starting time of the current shooting node
            
            x = None
            u = None
            s = None
            for p in phases:
                for n in nodes:
                    x = self._add_node(x, self._decision_states["scaled"], p, n, nlp.states.keys()) 
                    u = self._add_node(u, self._stepwise_controls["scaled"], p, n, nlp.controls.keys())
                    s = self._add_node(s, self._stochastic["scaled"], p, n, nlp.stochastic_variables.keys())
                
            x = x.reshape((-1, 1), order="F")
            u = u.reshape((-1, 1), order="F")
            s = s.reshape((-1, 1), order="F")
            val.append(penalty.function[node_idx](t, self._dt, x, u, p, s))

            target = []
            if penalty.target is not None:
                target = penalty.target[0][..., penalty.node_idx.index(node_idx)]

            val_weighted.append(
                penalty.weighted_function[node_idx](t, self._dt, x, u, p, s, penalty.weight, target, dt)
            )

        if self.ocp.n_threads > 1:
            val = [v[:, 0] for v in val]
            val_weighted = [v[:, 0] for v in val_weighted]

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
