import numpy as np
from casadi import vertcat, DM, Function
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy import interpolate as sci_interp
from typing import Any

from .solution_data import SolutionData, SolutionMerge, TimeAlignment, TimeResolution
from ..optimization_vector import OptimizationVectorHelper
from ...dynamics.ode_solvers import OdeSolver
from ...interfaces.solve_ivp_interface import solve_ivp_interface
from ...limits.path_conditions import InitialGuess, InitialGuessList
from ...limits.penalty_helpers import PenaltyHelpers
from ...misc.enums import (
    ControlType,
    CostType,
    Shooting,
    InterpolationType,
    SolverType,
    SolutionIntegrator,
    Node,
)
from ...models.protocols.stochastic_biomodel import StochasticBioModel


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
    _decision_algebraic_states: SolutionData
        The data structure that holds the algebraic_states variables
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
            The solution vector, containing all the states, controls, parameters and algebraic_states variables
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
        self._decision_algebraic_states = None

        self.vector = vector
        if self.vector is not None:
            self.phases_dt = OptimizationVectorHelper.extract_phase_dt(ocp, vector)
            self._stepwise_times = OptimizationVectorHelper.extract_step_times(ocp, vector)

            x, u, p, a = OptimizationVectorHelper.to_dictionaries(ocp, vector)
            self._decision_states = SolutionData.from_scaled(ocp, x, "x")
            self._stepwise_controls = SolutionData.from_scaled(ocp, u, "u")
            self._parameters = SolutionData.from_scaled(ocp, p, "p")
            self._decision_algebraic_states = SolutionData.from_scaled(ocp, a, "a")

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
                "an InitialGuess[List] of len 4 (states, controls, parameters, algebraic_states), "
                "or a None"
            )

        if len(sol[0]) != len(all_ns):
            raise ValueError("The time step dt array len must match the number of phases")

        is_right_size = [
            len(s) != len(all_ns) if p != 3 and len(sol[p + 1].keys()) != 0 else False for p, s in enumerate(sol[:1])
        ]

        if sum(is_right_size) != 0:
            raise ValueError("The InitialGuessList len must match the number of phases")

        if n_param != 0:
            if len(sol) != 3 and len(sol[3]) != 1 and sol[3][0].shape != (n_param, 1):
                raise ValueError(
                    "The 3rd element is the InitialGuess of the parameter and "
                    "should be a unique vector of size equal to n_param"
                )

        dt, sol_states, sol_controls, sol_params, sol_algebraic_states = sol

        vector = np.ndarray((0, 1))

        # For time
        if len(dt.shape) == 1:
            dt = dt[:, np.newaxis]
        vector = np.concatenate((vector, dt))

        # For states
        for p, ss in enumerate(sol_states):
            nb_intermediate_frames = 1
            if isinstance(ocp.nlp[p].dynamics_type.ode_solver, OdeSolver.COLLOCATION):
                nb_intermediate_frames = ocp.nlp[p].dynamics_type.ode_solver.polynomial_degree + 1
            for key in ss.keys():
                ns = (
                    ocp.nlp[p].ns * nb_intermediate_frames
                    if ss[key].init.type == InterpolationType.ALL_POINTS
                    else ocp.nlp[p].ns + 1 if ss[key].init.type != InterpolationType.EACH_FRAME else ocp.nlp[p].ns
                )
                ss[key].init.check_and_adjust_dimensions(len(ocp.nlp[p].states[key]), ns, "states")

            for i in range(all_ns[p] * nb_intermediate_frames + 1):
                for key in ss.keys():
                    vector = np.concatenate(
                        (vector, ss[key].init.evaluate_at(i, nb_intermediate_frames)[:, np.newaxis])
                    )

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
                ss[key].init.check_and_adjust_dimensions(len(ocp.nlp[p].controls[key]), all_ns[p] - 1 + off, "controls")

            for i in range(all_ns[p] + off):
                for key in ss.keys():
                    vector = np.concatenate((vector, ss[key].init.evaluate_at(i)[:, np.newaxis]))

        # For parameters
        if n_param:
            for p, ss in enumerate(sol_params):
                for key in ss.keys():
                    vector = np.concatenate((vector, np.repeat(ss[key].init, 1)[:, np.newaxis]))

        # For algebraic_states variables
        for p, ss in enumerate(sol_algebraic_states):
            for key in ss.keys():
                ss[key].init.check_and_adjust_dimensions(
                    len(ocp.nlp[p].algebraic_states[key]), all_ns[p], "algebraic_states"
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

    def t_span(
        self,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        time_alignment: TimeAlignment = TimeAlignment.STATES,
        continuous: bool = True,
    ):
        """
        Returns the time span at each node of each phases
        """
        return self._process_time_vector(
            time_resolution=TimeResolution.NODE_SPAN,
            to_merge=to_merge,
            time_alignment=time_alignment,
            continuous=continuous,
        )

    def decision_time(
        self,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        time_alignment: TimeAlignment = TimeAlignment.STATES,
        continuous: bool = True,
    ) -> list | np.ndarray:
        """
        Returns the time vector at each node that matches decision_states or decision_controls

        Parameters
        ----------
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed. It is often useful to merge NODES, but
            is completely useless to merge KEYS
        time_alignment: TimeAlignment
            The type of alignment to perform. If TimeAlignment.STATES, then the time vector is aligned with the states
            (i.e. all the subnodes and the last node time are present). If TimeAlignment.CONTROLS, then the time vector
            is aligned with the controls (i.e. only starting of the node without the last node if CONTROL constant).
        continuous: bool
            If the time should be continuous throughout the whole ocp. If False, then the time is reset at the
            beginning of each phase.
        """

        return self._process_time_vector(
            time_resolution=TimeResolution.DECISION,
            to_merge=to_merge,
            time_alignment=time_alignment,
            continuous=continuous,
        )

    def stepwise_time(
        self,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        time_alignment: TimeAlignment = TimeAlignment.STATES,
        continuous: bool = True,
        duplicated_times: bool = True,
    ) -> list | np.ndarray:
        """
        Returns the time vector at each node that matches stepwise_states or stepwise_controls

        Parameters
        ----------
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed. It is often useful to merge NODES, but
            is completely useless to merge KEYS
        time_alignment: TimeAlignment
            The type of alignment to perform. If TimeAlignment.STATES, then the time vector is aligned with the states
            (i.e. all the subnodes and the last node time are present). If TimeAlignment.CONTROLS, then the time vector
            is aligned with the controls (i.e. only starting of the node without the last node if CONTROL constant).
        continuous: bool
            If the time should be continuous throughout the whole ocp. If False, then the time is reset at the
            beginning of each phase.
        duplicated_times: bool
            If the times should be duplicated for each nodes.
            If False, then the returned time vector will not have any duplicated times

        Returns
        -------
        The time vector at each node that matches stepwise_states or stepwise_controls
        """

        return self._process_time_vector(
            time_resolution=TimeResolution.STEPWISE,
            to_merge=to_merge,
            time_alignment=time_alignment,
            continuous=continuous,
            duplicated_times=duplicated_times,
        )

    def _process_time_vector(
        self,
        time_resolution: TimeResolution,
        to_merge: SolutionMerge | list[SolutionMerge],
        time_alignment: TimeAlignment,
        continuous: bool,
        duplicated_times: bool = True,
    ):
        if to_merge is None or isinstance(to_merge, SolutionMerge):
            to_merge = [to_merge]

        # Make sure to not return internal structure
        times_tp = deepcopy(self._stepwise_times)

        # Select the appropriate time matrix
        phases_tf = []
        times = []
        for nlp in self.ocp.nlp:
            phases_tf.append(times_tp[nlp.phase_idx][-1])

            if time_resolution == TimeResolution.NODE_SPAN:
                if time_alignment == TimeAlignment.STATES:
                    times.append([t if t.shape == (1, 1) else t[[0, -1]] for t in times_tp[nlp.phase_idx]])
                elif time_alignment == TimeAlignment.CONTROLS:
                    times.append([t[[0, -1]] for t in times_tp[nlp.phase_idx][:-1]])
            else:
                if time_alignment == TimeAlignment.STATES:
                    if nlp.dynamics_type.ode_solver.is_direct_collocation:
                        if nlp.dynamics_type.ode_solver.duplicate_starting_point:
                            times.append(
                                [t if t.shape == (1, 1) else vertcat(t[0], t[:-1]) for t in times_tp[nlp.phase_idx]]
                            )
                        else:
                            times.append([t if t.shape == (1, 1) else t[:-1] for t in times_tp[nlp.phase_idx]])

                    else:
                        if time_resolution == TimeResolution.STEPWISE:
                            times.append(times_tp[nlp.phase_idx])

                        elif time_resolution == TimeResolution.DECISION:
                            times.append([t[0] for t in times_tp[nlp.phase_idx]])

                        else:
                            raise ValueError("Unrecognized time_resolution")

                elif time_alignment == TimeAlignment.CONTROLS:
                    if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                        times.append([(t if t.shape == (1, 1) else t[[0, -1]]) for t in times_tp[nlp.phase_idx]])
                        if len(times) < len(self.ocp.nlp):
                            # The point is duplicated for internal phases, but not the last one
                            times[-1][-1] = times[-1][-1][[0, 0]].T
                    elif nlp.control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                        times.append([t[0] for t in times_tp[nlp.phase_idx]])
                    elif nlp.control_type == ControlType.CONSTANT:
                        times.append([t[0] for t in times_tp[nlp.phase_idx]][:-1])
                    else:
                        raise ValueError(f"Unrecognized control type {nlp.control_type}")

                else:
                    raise ValueError("time_alignment should be either TimeAlignment.STATES or TimeAlignment.CONTROLS")

        if not duplicated_times:
            for i in range(len(times)):
                for j in range(len(times[i])):
                    # Last node of last phase is always kept
                    keep_condition = times[i][j].shape[0] == 1 and i == len(times) - 1
                    times[i][j] = times[i][j][:] if keep_condition else times[i][j][:-1]
                    if j == len(times[i]) - 1 and i != len(times) - 1:
                        del times[i][j]

        if continuous:
            for phase_idx, phase_time in enumerate(times):
                if phase_idx == 0:
                    continue
                previous_tf = sum(phases_tf[:phase_idx])
                times[phase_idx] = [t + previous_tf for t in phase_time]

        if SolutionMerge.NODES in to_merge or SolutionMerge.ALL in to_merge:
            for phase_idx in range(len(times)):
                np.concatenate((np.concatenate(times[phase_idx][:-1]), times[phase_idx][-1]))
                times[phase_idx] = np.concatenate((np.concatenate(times[phase_idx][:-1]), times[phase_idx][-1]))

        if (
            SolutionMerge.PHASES in to_merge and SolutionMerge.NODES not in to_merge
        ) and SolutionMerge.ALL not in to_merge:
            raise ValueError("Cannot merge phases without nodes")

        if SolutionMerge.PHASES in to_merge or SolutionMerge.ALL in to_merge:
            # NODES is necessarily in to_merge if PHASES is in to_merge
            times = np.concatenate(times)

        return times if len(times) > 1 else times[0]

    def decision_states(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
        """
        Returns the decision states

        Parameters
        ----------
        scaled: bool
            If the decision states should be scaled or not (note that scaled is as Ipopt received them, while unscaled
            is as the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.

        Returns
        -------
        The decision variables
        """

        data = self._decision_states.to_dict(to_merge=to_merge, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    def stepwise_states(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
        """
        Returns the stepwise integrated states

        Parameters
        ----------
        scaled: bool
            If the states should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as the
            model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge]
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

    def decision_controls(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
        """
        Returns the decision controls

        Parameters
        ----------
        scaled : bool
            If the decision controls should be scaled or not (note that scaled is as Ipopt received them, while unscaled
            is as the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge : SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.
        """
        return self.stepwise_controls(scaled=scaled, to_merge=to_merge)

    def stepwise_controls(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
        """
        Returns the controls. Note the final control is always present but set to np.nan if it is not defined

        Parameters
        ----------
        scaled: bool
            If the controls should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as
            the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge]
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

    def decision_parameters(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
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

    def decision_algebraic_states(self, scaled: bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
        """
        Returns the decision algebraic_states

        Parameters
        ----------
        scaled: bool
            If the decision states should be scaled or not (note that scaled is as Ipopt received them, while unscaled
            is as the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.

        Returns
        -------
        The decision variables
        """

        data = self._decision_algebraic_states.to_dict(to_merge=to_merge, scaled=scaled)
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

        if not skip_data:
            new._decision_states = deepcopy(self._decision_states)
            new._stepwise_states = deepcopy(self._stepwise_states)

            new._stepwise_controls = deepcopy(self._stepwise_controls)

            new._decision_algebraic_states = deepcopy(self._decision_algebraic_states)
            new._parameters = deepcopy(self._parameters)
        return new

    def _prepare_integrate(self, integrator: SolutionIntegrator):
        """
        Prepare the variables for the states integration and checks if the integrator is compatible with the ocp.

        Parameters
        ----------
        integrator: SolutionIntegrator
            The integrator to use for the integration
        """

        has_direct_collocation = sum([nlp.dynamics_type.ode_solver.is_direct_collocation for nlp in self.ocp.nlp]) > 0
        if has_direct_collocation and integrator == SolutionIntegrator.OCP:
            raise ValueError(
                "When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE"
            )

        has_trapezoidal = (
            sum([isinstance(nlp.dynamics_type.ode_solver, OdeSolver.TRAPEZOIDAL) for nlp in self.ocp.nlp]) > 0
        )
        if has_trapezoidal and integrator == SolutionIntegrator.OCP:
            raise ValueError(
                "When the ode_solver of the Optimal Control Problem is OdeSolver.TRAPEZOIDAL, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
            )

        params = self._parameters.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)[0][0]
        t_spans = self.t_span(time_alignment=TimeAlignment.CONTROLS)
        if len(self.ocp.nlp) == 1:
            t_spans = [t_spans]
        x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        a = self._decision_algebraic_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        return t_spans, x, u, params, a

    def integrate(
        self,
        shooting_type: Shooting = Shooting.SINGLE,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        duplicated_times: bool = True,
        return_time: bool = False,
    ):
        """
        Create a deepcopy of the Solution

        Parameters
        ----------
        shooting_type: Shooting
            The integration shooting type to use
        integrator: SolutionIntegrator
            The type of integrator to use
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.
        duplicated_times: bool
            If the times should be duplicated for each node.
            If False, then the returned time vector will not have any duplicated times.
            Default is True.
        return_time: bool
            If the time vector should be returned, default is False.

        Returns
        -------
        Return the integrated states
        """
        from ...interfaces.interface_utils import get_numerical_timeseries

        t_spans, x, u, params, a = self._prepare_integrate(integrator=integrator)

        out: list = [None] * len(self.ocp.nlp)
        integrated_sol = None
        for p, nlp in enumerate(self.ocp.nlp):
            first_x = self._states_for_phase_integration(shooting_type, p, integrated_sol, x, u, params, a)
            d = []
            for n_idx in range(nlp.ns + 1):
                d_tp = get_numerical_timeseries(self.ocp, p, n_idx, 0)
                if d_tp.shape == (0, 0):
                    d += [np.array([])]
                else:
                    d += [np.array(d_tp)]

            integrated_sol = solve_ivp_interface(
                list_of_dynamics=[nlp.dynamics_func] * nlp.ns,
                shooting_type=shooting_type,
                nlp=nlp,
                t=t_spans[p],
                x=first_x,
                u=u[p],
                a=a[p],
                d=d,
                p=params,
                method=integrator,
            )

            out[p] = {}
            for key in nlp.states.keys():
                out[p][key] = [None] * nlp.n_states_nodes
                for ns, sol_ns in enumerate(integrated_sol):
                    if duplicated_times:
                        out[p][key][ns] = sol_ns[nlp.states[key].index, :]
                    else:
                        # Last node of last phase is always kept
                        duplicated_times_condition = p == len(self.ocp.nlp) - 1 and ns == nlp.ns
                        out[p][key][ns] = (
                            sol_ns[nlp.states[key].index, :]
                            if duplicated_times_condition
                            else sol_ns[nlp.states[key].index, :-1]
                        )

        if to_merge:
            out = SolutionData.from_unscaled(self.ocp, out, "x").to_dict(to_merge=to_merge, scaled=False)

        if return_time:
            time_vector = self._return_time_vector(to_merge=to_merge, duplicated_times=duplicated_times)
            return out if len(out) > 1 else out[0], time_vector if len(time_vector) > 1 else time_vector[0]
        else:
            return out if len(out) > 1 else out[0]

    def noisy_integrate(
        self,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        size: int = 100,
    ):
        """
        Integrated the states with different noise values sampled from the covariance matrix.
        """
        from ...optimization.stochastic_optimal_control_program import StochasticOptimalControlProgram
        from ...interfaces.interface_utils import get_numerical_timeseries

        if not isinstance(self.ocp, StochasticOptimalControlProgram):
            raise ValueError("This method is only available for StochasticOptimalControlProgram.")

        t_spans, x, u, params, a = self._prepare_integrate(integrator=integrator)

        cov_index = self.ocp.nlp[0].controls["cov"].index
        n_sub_nodes = x[0][0].shape[1]
        motor_noise_index = self.ocp.nlp[0].parameters["motor_noise"].index
        sensory_noise_index = (
            self.ocp.nlp[0].parameters["sensory_noise"].index
            if len(list(self.ocp.nlp[0].parameters["sensory_noise"].index)) > 0
            else None
        )

        # initialize the out dictionary
        out = [None] * len(self.ocp.nlp)
        for p, nlp in enumerate(self.ocp.nlp):
            out[p] = {}
            for key in self.ocp.nlp[0].states.keys():
                out[p][key] = [None] * nlp.n_states_nodes
                for i_node in range(nlp.ns):
                    out[p][key][i_node] = np.zeros((len(nlp.states[key].index), n_sub_nodes, size))
                out[p][key][nlp.ns] = np.zeros((len(nlp.states[key].index), 1, size))

        cov_matrix = StochasticBioModel.reshape_to_matrix(u[0][0][cov_index, 0], self.ocp.nlp[0].model.matrix_shape_cov)
        first_x = np.random.multivariate_normal(x[0][0][:, 0], cov_matrix, size=size).T
        for p, nlp in enumerate(self.ocp.nlp):
            d = []
            for n_idx in range(nlp.ns + 1):
                d_tp = get_numerical_timeseries(self.ocp, p, n_idx, 0)
                if d_tp.shape == (0, 0):
                    d += [np.array([])]
                else:
                    d += [np.array(d_tp)]

            motor_noise = np.zeros((len(params[motor_noise_index]), nlp.ns, size))
            for i in range(len(params[motor_noise_index])):
                motor_noise[i, :] = np.random.normal(0, params[motor_noise_index[i]], size=(nlp.ns, size))
            sensory_noise = (
                np.zeros((len(sensory_noise_index), nlp.ns, size)) if sensory_noise_index is not None else None
            )
            if sensory_noise_index is not None:
                for i in range(len(params[sensory_noise_index])):
                    sensory_noise[i, :] = np.random.normal(0, params[sensory_noise_index[i]], size=(nlp.ns, size))

            without_noise_idx = [
                i for i in range(len(params)) if i not in motor_noise_index and i not in sensory_noise_index
            ]
            parameters_cx = nlp.parameters.cx[without_noise_idx]
            parameters = params[without_noise_idx]
            for i_random in range(size):
                params_this_time = []
                list_of_dynamics = []
                for node in range(nlp.ns):
                    params_this_time += [nlp.parameters.cx]
                    params_this_time[node][motor_noise_index, :] = motor_noise[:, node, i_random]
                    if sensory_noise_index is not None:
                        params_this_time[node][sensory_noise_index, :] = sensory_noise[:, node, i_random]

                    if len(nlp.extra_dynamics_func) > 1:
                        raise NotImplementedError("Noisy integration is not available for multiple extra dynamics.")
                    cas_func = Function(
                        "noised_extra_dynamics",
                        [
                            nlp.time_cx,
                            nlp.states.cx,
                            nlp.controls.cx,
                            parameters_cx,
                            nlp.algebraic_states.cx,
                            nlp.numerical_timeseries.cx,
                        ],
                        [
                            nlp.extra_dynamics_func[0](
                                nlp.time_cx,
                                nlp.states.cx,
                                nlp.controls.cx,
                                params_this_time[node],
                                nlp.algebraic_states.cx,
                                nlp.numerical_timeseries.cx,
                            )
                        ],
                    )
                    list_of_dynamics += [cas_func]

                integrated_sol = solve_ivp_interface(
                    list_of_dynamics=list_of_dynamics,
                    shooting_type=Shooting.SINGLE,
                    nlp=nlp,
                    t=t_spans[p],
                    x=[np.reshape(first_x[:, i_random], (-1, 1))],
                    u=u[p],  # No need to add noise on the controls, the extra_dynamics should do it for us
                    a=a[p],
                    p=parameters,
                    d=d,
                    method=integrator,
                )
                for i_node in range(nlp.ns + 1):
                    for key in nlp.states.keys():
                        states_integrated = (
                            integrated_sol[i_node][nlp.states[key].index, :]
                            if n_sub_nodes > 1
                            else integrated_sol[i_node][nlp.states[key].index, 0].reshape(-1, 1)
                        )
                        out[p][key][i_node][:, :, i_random] = states_integrated
                first_x[:, i_random] = np.reshape(integrated_sol[-1], (-1,))
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
        decision_algebraic_states,
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
        decision_algebraic_states
            The algebraic_states merged with SolutionMerge.KEYS

        Returns
        -------
        The states to integrate
        """
        from ...interfaces.interface_utils import get_numerical_timeseries

        # In the case of multiple shootings, we don't need to do anything special
        if shooting_type == Shooting.MULTIPLE:
            return decision_states[phase_idx]

        # At first phase, return the normal decision states.
        if phase_idx == 0:
            return [decision_states[phase_idx][0]]

        penalty = self.ocp.phase_transitions[phase_idx - 1]

        t0 = PenaltyHelpers.t0(penalty, 0, lambda p, n: self._stepwise_times[p][n][0])
        dt = PenaltyHelpers.phases_dt(penalty, self.ocp, lambda p: np.array([self.phases_dt[idx] for idx in p]))
        # Compute the error between the last state of the previous phase and the first state of the next phase
        # based on the phase transition objective or constraint function. That is why we need to concatenate
        # twice the last state
        x = PenaltyHelpers.states(penalty, 0, lambda p, n, sn: integrated_states[-1])

        u = PenaltyHelpers.controls(
            penalty,
            0,
            lambda p, n, sn: decision_controls[p][n][:, sn] if n < len(decision_controls[p]) else np.ndarray((0, 1)),
        )
        a = PenaltyHelpers.states(
            penalty,
            0,
            lambda p, n, sn: (
                decision_algebraic_states[p][n][:, sn] if n < len(decision_algebraic_states[p]) else np.ndarray((0, 1))
            ),
        )
        d_tp = PenaltyHelpers.numerical_timeseries(
            penalty,
            0,
            lambda p, n, sn: get_numerical_timeseries(self.ocp, p, n, sn),
        )
        d = np.array([]) if d_tp.shape == (0, 0) else np.array(d_tp)

        dx = penalty.function[-1](t0, dt, x, u, params, a, d)
        if dx.shape[0] != decision_states[phase_idx][0].shape[0]:
            raise RuntimeError(
                f"Phase transition must have the same number of states ({dx.shape[0]}) "
                f"when integrating with Shooting.SINGLE. If it is not possible, "
                f"please integrate with Shooting.SINGLE_DISCONTINUOUS_PHASE."
            )

        return [(integrated_states[-1] if shooting_type == Shooting.SINGLE else decision_states[phase_idx][0]) + dx]

    def _integrate_stepwise(self) -> None:
        """
        This method integrate to stepwise level the states. That is the states that are used in the dynamics and
        continuity constraints.

        Returns
        -------
        dict
            The integrated data structure similar in structure to the original _decision_states
        """
        from ...interfaces.interface_utils import get_numerical_timeseries

        params = self._parameters.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)[0][0]
        t_spans = self.t_span(time_alignment=TimeAlignment.CONTROLS)
        if len(self.ocp.nlp) == 1:
            t_spans = [t_spans]
        x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        a = self._decision_algebraic_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)

        unscaled: list = [None] * len(self.ocp.nlp)
        for p, nlp in enumerate(self.ocp.nlp):
            d = []
            for n_idx in range(nlp.ns + 1):
                d_tp = get_numerical_timeseries(self.ocp, p, n_idx, 0)
                if d_tp.shape == (0, 0):
                    d += [np.array([])]
                else:
                    d += [np.array(d_tp)]

            integrated_sol = solve_ivp_interface(
                list_of_dynamics=[nlp.dynamics_func] * nlp.ns,
                shooting_type=Shooting.MULTIPLE,
                nlp=nlp,
                t=t_spans[p],
                x=x[p],
                u=u[p],
                a=a[p],
                p=params,
                d=d,
                method=SolutionIntegrator.OCP,
            )

            unscaled[p] = {}
            for key in nlp.states.keys():
                unscaled[p][key] = [None] * nlp.n_states_nodes
                for ns, sol_ns in enumerate(integrated_sol):
                    unscaled[p][key][ns] = sol_ns[nlp.states[key].index, :]

        self._stepwise_states = SolutionData.from_unscaled(self.ocp, unscaled, "x")

    def _return_time_vector(self, to_merge: SolutionMerge | list[SolutionMerge], duplicated_times: bool):
        """
        Returns the time vector at each node that matches stepwise_states or stepwise_controls
        Parameters
        ----------
        to_merge: SolutionMerge | list[SolutionMerge]
            The merge type to perform. If None, then no merge is performed.
        duplicated_times: bool
            If the times should be duplicated for each node.
            If False, then the returned time vector will not have any duplicated times.
        Returns
        -------
        The time vector at each node that matches stepwise_states or stepwise_controls
        """
        if to_merge is None:
            to_merge = []
        if isinstance(to_merge, SolutionMerge):
            to_merge = [to_merge]
        if SolutionMerge.NODES and SolutionMerge.PHASES in to_merge:
            time_vector = np.concatenate(self.stepwise_time(to_merge=to_merge, duplicated_times=duplicated_times))
        elif SolutionMerge.NODES in to_merge:
            time_vector = self.stepwise_time(to_merge=to_merge, duplicated_times=duplicated_times)
            for i in range(len(self.ocp.nlp)):
                time_vector[i] = np.concatenate(time_vector[i])
        else:
            time_vector = self.stepwise_time(to_merge=to_merge, duplicated_times=duplicated_times)
        return time_vector

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
            t_all = [self.stepwise_time(to_merge=[SolutionMerge.ALL])]
            states = [self._stepwise_states.to_dict(scaled=scaled, to_merge=SolutionMerge.ALL)]
            n_frames = [n_frames]

        elif not isinstance(n_frames, (list, tuple)) or len(n_frames) != len(self._stepwise_states.unscaled):
            raise ValueError(
                "n_frames should either be an int to merge_phases phases "
                "or a list of int of the number of phases dimension"
            )

        else:
            t_all = self.stepwise_time(to_merge=[SolutionMerge.NODES])
            if len(self.ocp.nlp) == 1:
                t_all = [t_all]
            states = self._stepwise_states.to_dict(scaled=scaled, to_merge=[SolutionMerge.KEYS, SolutionMerge.NODES])

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
        save_name: str = None,
    ) -> list[plt.figure]:
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
        save_name: str
            If a name is provided, the figures will be saved with this name
        """

        plot_ocp = self.ocp.prepare_plots(automatically_organize, show_bounds, shooting_type, integrator)
        plot_ocp.update_data(*plot_ocp.parse_data(**{"x": self.vector}))
        if save_name:
            if save_name.endswith(".png"):
                save_name = save_name[:-4]
            for i_fig, name_fig in enumerate(plt.get_figlabels()):
                fig = plt.figure(i_fig + 1)
                fig.savefig(f"{save_name}_{name_fig}.png", format="png")
        if show_now:
            plt.show()

        # Returning the figures for the tests
        fig_list = [plt.figure(i_fig + 1) for i_fig in range(len(plt.get_figlabels()))]
        return fig_list

    def animate(
        self,
        n_frames: int = 0,
        shooting_type: Shooting = None,
        show_now: bool = True,
        show_tracked_markers: bool = False,
        viewer: str = "bioviz",
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
        viewer: str
            The viewer to use. Currently, bioviz or pyorerun
        kwargs: Any
            Any parameters to pass to bioviz

        Returns
        -------
            A list of bioviz structures (one for each phase). So one can call exec() by hand
        """

        from ...models.biorbd.viewer_utils import _check_models_comes_from_same_super_class

        if shooting_type:
            self.integrate(shooting_type=shooting_type)

        _check_models_comes_from_same_super_class(self.ocp.nlp)

        type(self.ocp.nlp[0].model).animate(
            ocp=self.ocp,
            solution=self,
            show_now=show_now,
            show_tracked_markers=show_tracked_markers,
            viewer=viewer,
            n_frames=n_frames,
            **kwargs,
        )

    @staticmethod
    def _dispatch_params(params):
        values = [params[key][0] for key in params.keys()]
        if values:
            return np.concatenate(values)
        else:
            return np.ndarray((0, 1))

    def _get_penalty_cost(self, nlp, penalty):
        from ...interfaces.interface_utils import get_numerical_timeseries

        if nlp is None:
            raise NotImplementedError("penalty cost over the full ocp is not implemented yet")

        val = []
        val_weighted = []

        phases_dt = PenaltyHelpers.phases_dt(penalty, self.ocp, lambda p: np.array([self.phases_dt[idx] for idx in p]))
        params = PenaltyHelpers.parameters(
            penalty, 0, lambda p_idx, n_idx, sn_idx: self._dispatch_params(self._parameters.scaled[0])
        )

        merged_x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        merged_u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        merged_a = self._decision_algebraic_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        for idx in range(len(penalty.node_idx)):
            t0 = PenaltyHelpers.t0(penalty, idx, lambda p, n: self._stepwise_times[p][n][0])
            x = PenaltyHelpers.states(
                penalty,
                idx,
                lambda p_idx, n_idx, sn_idx: self._get_x(self.ocp, penalty, p_idx, n_idx, sn_idx, merged_x),
            )
            u = PenaltyHelpers.controls(
                penalty,
                idx,
                lambda p_idx, n_idx, sn_idx: self._get_u(self.ocp, penalty, p_idx, n_idx, sn_idx, merged_u),
            )
            a = PenaltyHelpers.states(
                penalty,
                idx,
                lambda p_idx, n_idx, sn_idx: self._get_x(self.ocp, penalty, p_idx, n_idx, sn_idx, merged_a),
            )
            d_tp = PenaltyHelpers.numerical_timeseries(
                penalty,
                idx,
                lambda p_idx, n_idx, sn_idx: get_numerical_timeseries(self.ocp, p_idx, n_idx, sn_idx),
            )
            d = np.array([]) if d_tp.shape == (0, 0) else np.array(d_tp)

            weight = PenaltyHelpers.weight(penalty)
            target = PenaltyHelpers.target(penalty, idx)

            node_idx = penalty.node_idx[idx]
            val.append(penalty.function[node_idx](t0, phases_dt, x, u, params, a, d))
            val_weighted.append(penalty.weighted_function[node_idx](t0, phases_dt, x, u, params, a, d, weight, target))

        if self.ocp.n_threads > 1:
            val = [v[:, 0] for v in val]
            val_weighted = [v[:, 0] for v in val_weighted]

        val = np.nansum(val)
        val_weighted = np.nansum(val_weighted)

        return val, val_weighted

    @staticmethod
    def _get_x(ocp, penalty, phase_idx, node_idx, subnodes_idx, merged_x):
        values = merged_x[phase_idx]
        x = PenaltyHelpers.get_states(ocp, penalty, phase_idx, node_idx, subnodes_idx, values)
        return x

    @staticmethod
    def _get_u(ocp, penalty, phase_idx, node_idx, subnodes_idx, merged_u):
        values = merged_u[phase_idx]
        u = PenaltyHelpers.get_controls(ocp, penalty, phase_idx, node_idx, subnodes_idx, values)
        return u

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
                            "params": penalty.extra_parameters,
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
                    print(f"{penalty.type}: {val_weighted} (non weighted  {val})")

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
