from itertools import chain
from math import inf
from typing import Union, Callable
from time import perf_counter

import numpy as np
import biorbd_casadi as biorbd

from .optimal_control_program import OptimalControlProgram
from .solution import Solution
from ..dynamics.configure_problem import Dynamics, DynamicsList
from ..limits.constraints import ConstraintFcn
from ..limits.objective_functions import ObjectiveFcn
from ..limits.path_conditions import InitialGuess, Bounds
from ..misc.enums import SolverType, InterpolationType
from ..interfaces.solver_options import Solver


class RecedingHorizonOptimization(OptimalControlProgram):
    """
    The main class to define an MHE. This class prepares the full program and gives all
    the needed interface to modify and solve the program

    Methods
    -------
    solve(self, solver: Solver, show_online_optim: bool, solver_options: dict) -> Solution
        Call the solver to actually solve the ocp
    """

    def __init__(
        self,
        biorbd_model: Union[str, biorbd.Model, list, tuple],
        dynamics: Union[Dynamics, DynamicsList],
        window_len: Union[int, list, tuple],
        window_duration: Union[int, float, list, tuple],
        use_sx=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        window_size: Union[int, list[int]]
            The number of shooting point of the moving window
        """

        if isinstance(biorbd_model, (list, tuple)) and len(biorbd_model) > 1:
            raise ValueError("Receding horizon optimization must be defined using only one biorbd_model")

        super(RecedingHorizonOptimization, self).__init__(
            biorbd_model=biorbd_model,
            dynamics=dynamics,
            n_shooting=window_len,
            phase_time=window_duration,
            use_sx=use_sx,
            **kwargs,
        )
        self.total_optimization_run = 0

    def solve(
        self,
        update_function: Callable,
        solver: Solver.Generic = None,
        warm_start: Solution = None,
        solver_first_iter: Solver.Generic = None,
        export_options: dict = None,
        max_consecutive_failing: int = inf,
        update_function_extra_params: dict = None,
        get_all_iterations: bool = False,
        **advance_options,
    ) -> Union[Solution, tuple]:
        """
        Solve MHE program. The program runs until 'update_function' returns False. This function can be used to
        modify the objective set, for instance. The warm_start_function can be provided by the user. Otherwise, the
        initial guess is the solution where the first frame is dropped and the last frame is duplicated. Moreover,
        the bounds at first frame is set to the new first frame of the initial guess

        Parameters
        ----------
        update_function: Callable
            A function with the signature: update_function(mhe, current_time_index, previous_solution), where the
            mhe is the current program, current_time_index starts at 0 and increments after each solve and
            previous_solution is None the first call and then is the Solution structure for the last solving of the MHE.
            The function 'update_function' is called before each solve. If it returns true, the next frame is solve.
            Otherwise, it finishes the MHE and the solution is returned. The `update_function` callback can also
            be used to modify the program (usually the targets of some objective functions) and initial condition and
            bounds.
        solver: Solver
            The Solver to use (default being ACADOS)
        solver: Solver
            The Solver to use for the first iteration (must be the same as solver, but more options can be changed)
        warm_start: Solution
            A Solution to initiate the first iteration from
        export_options: dict
            Any options related to the saving of the data at each iteration
        max_consecutive_failing: int
            The number of consecutive failing before stopping the nmpc. Default is infinite
        update_function_extra_params: dict
            Any parameters to pass to the update function
        get_all_iterations: bool
            If an extra output value that includes all the individual solution should be returned
        advance_options: Any
            The extra options to pass to the advancing methods

        Returns
        -------
        The solution of the MHE
        """

        if len(self.nlp) != 1:
            raise NotImplementedError("MHE is only available for 1 phase program")

        sol = None
        states = []
        controls = []

        solver_all_iter = Solver.ACADOS() if solver is None else solver
        if solver_first_iter is None and solver is not None:
            # If not first iter was sent, the all iter becomes the first and is not updated afterward
            solver_first_iter = solver_all_iter
            solver_all_iter = None
        solver_current = solver_first_iter

        self._initialize_frame_to_export(export_options)

        total_time = 0
        real_time = perf_counter()
        all_solutions = []
        consecutive_failing = 0
        update_function_extra_params = {} if update_function_extra_params is None else update_function_extra_params

        self.total_optimization_run = 0
        while (
            update_function(self, self.total_optimization_run, sol, **update_function_extra_params)
            and consecutive_failing < max_consecutive_failing
        ):
            sol = super(RecedingHorizonOptimization, self).solve(
                solver=solver_current,
                warm_start=warm_start,
            )
            consecutive_failing = 0 if sol.status == 0 else consecutive_failing + 1

            # Set the option for the next iteration
            if self.total_optimization_run == 0:
                # Update the solver if first and the rest are different
                if solver_all_iter:
                    solver_current = solver_all_iter
                    if solver_current.type == SolverType.ACADOS and solver_current.only_first_options_has_changed:
                        raise RuntimeError(
                            f"Some options has been changed for the second iteration of acados.\n"
                            f"Only {solver_current.get_tolerance_keys()} can be modified."
                        )
                if solver_current.type == SolverType.IPOPT:
                    solver_current.show_online_optim = False
            warm_start = None

            total_time += sol.real_time_to_optimize
            if solver_current == Solver.ACADOS and self.total_optimization_run == 0:
                real_time = perf_counter()  # Reset timer to skip the compiling time (so skip the first call to solve)

            # Solve and save the current window
            _states, _controls = self.export_data(sol)
            states.append(_states)
            controls.append(_controls)
            if get_all_iterations:
                all_solutions.append(sol)

            # Update the initial frame bounds and initial guess
            self.advance_window(sol, **advance_options)

            self.total_optimization_run += 1

        real_time = perf_counter() - real_time

        # Prepare the modified ocp that fits the solution dimension
        sol = self._initialize_solution(states, controls)
        sol.solver_time_to_optimize = total_time
        sol.real_time_to_optimize = real_time
        return (sol, all_solutions) if get_all_iterations else sol

    def _initialize_frame_to_export(self, export_options):
        if export_options is None:
            export_options = {"frame_to_export": 0}
        else:
            if "frame_to_export" not in export_options:
                export_options["frame_to_export"] = 0

        if isinstance(export_options["frame_to_export"], int):
            export_options["frame_to_export"] = slice(
                export_options["frame_to_export"], export_options["frame_to_export"] + 1
            )

        self.frame_to_export = export_options["frame_to_export"]

    def _initialize_solution(self, states: list, controls: list):
        _states = InitialGuess(np.concatenate(states, axis=1), interpolation=InterpolationType.EACH_FRAME)
        _controls = InitialGuess(np.concatenate(controls, axis=1), interpolation=InterpolationType.EACH_FRAME)

        solution_ocp = OptimalControlProgram(
            biorbd_model=self.original_values["biorbd_model"][0],
            dynamics=self.original_values["dynamics"][0],
            n_shooting=self.total_optimization_run - 1,
            phase_time=self.total_optimization_run * self.nlp[0].dt,
            skip_continuity=True,
        )
        return Solution(solution_ocp, [_states, _controls])

    def advance_window(self, sol: Solution, steps: int = 0, **advance_options):
        state_bounds_have_changed = self.advance_window_bounds_states(sol, **advance_options)
        control_bounds_have_changed = self.advance_window_bounds_controls(sol, **advance_options)
        if self.ocp_solver.opts.type != SolverType.ACADOS:
            self.update_bounds(
                self.nlp[0].x_bounds if state_bounds_have_changed else None,
                self.nlp[0].u_bounds if control_bounds_have_changed else None,
            )

        init_states_have_changed = self.advance_window_initial_guess_states(sol, **advance_options)
        init_controls_have_changed = self.advance_window_initial_guess_controls(sol, **advance_options)

        if self.ocp_solver.opts.type != SolverType.ACADOS:
            self.update_initial_guess(
                self.nlp[0].x_init if init_states_have_changed else None,
                self.nlp[0].u_init if init_controls_have_changed else None,
            )

    def advance_window_bounds_states(self, sol, **advance_options):
        if self.nlp[0].x_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if self.nlp[0].x_bounds.type == InterpolationType.CONSTANT:
                x_min = np.repeat(self.nlp[0].x_bounds.min[:, 0:1], 3, axis=1)
                x_max = np.repeat(self.nlp[0].x_bounds.max[:, 0:1], 3, axis=1)
                self.nlp[0].x_bounds = Bounds(x_min, x_max)
            else:
                raise NotImplementedError(
                    "The MHE is not implemented yet for x_bounds not being "
                    "CONSTANT or CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT"
                )
            self.nlp[0].x_bounds.check_and_adjust_dimensions(self.nlp[0].states.shape, 3)
        self.nlp[0].x_bounds[:, 0] = sol.states["all"][:, 1]
        return True

    def advance_window_bounds_controls(self, sol, **advance_options):
        return False

    def advance_window_initial_guess_states(self, sol, **advance_options):
        if self.nlp[0].x_init.type != InterpolationType.EACH_FRAME:
            self.nlp[0].x_init = InitialGuess(
                np.ndarray(sol.states["all"].shape), interpolation=InterpolationType.EACH_FRAME
            )
            self.nlp[0].x_init.check_and_adjust_dimensions(self.nlp[0].states.shape, self.nlp[0].ns)
        self.nlp[0].x_init.init[:, :] = np.concatenate(
            (sol.states["all"][:, 1:], sol.states["all"][:, -1][:, np.newaxis]), axis=1
        )
        return True

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        if self.nlp[0].u_init.type != InterpolationType.EACH_FRAME:
            self.nlp[0].u_init = InitialGuess(
                np.ndarray(sol.controls["all"][:, :-1].shape), interpolation=InterpolationType.EACH_FRAME
            )
            self.nlp[0].u_init.check_and_adjust_dimensions(self.nlp[0].controls.shape, self.nlp[0].ns - 1)
        self.nlp[0].u_init.init[:, :] = np.concatenate(
            (sol.controls["all"][:, 1:-1], sol.controls["all"][:, -2][:, np.newaxis]), axis=1
        )
        return True

    def export_data(self, sol) -> tuple:
        return sol.states["all"][:, self.frame_to_export], sol.controls["all"][:, self.frame_to_export]

    def _define_time(self, phase_time: Union[int, float, list, tuple], objective_functions, constraints):
        """
        Declare the phase_time vector in v. If objective_functions or constraints defined a time optimization,
        a sanity check is perform and the values of initial guess and bounds for these particular phases

        Parameters
        ----------
        phase_time: Union[int, float, list, tuple]
            The time of all the phases
        objective_functions: ObjectiveList
            All the objective functions. It is used to scan if any time optimization was defined
        constraints: ConstraintList
            All the constraint functions. It is used to scan if any free time was defined
        """

        def check_for_time_optimization(penalty_functions):
            """
            Make sure one does not try to optimize time

            Parameters
            ----------
            penalty_functions: Union[ObjectiveList, ConstraintList]
                The list to parse to ensure no double free times are declared

            """

            for i, penalty_functions_phase in enumerate(penalty_functions):
                for pen_fun in penalty_functions_phase:
                    if not pen_fun:
                        continue
                    if (
                        pen_fun.type == ObjectiveFcn.Mayer.MINIMIZE_TIME
                        or pen_fun.type == ObjectiveFcn.Lagrange.MINIMIZE_TIME
                        or pen_fun.type == ConstraintFcn.TIME_CONSTRAINT
                    ):
                        raise ValueError("Time cannot be optimized in Receding Horizon Optimization")

        check_for_time_optimization(objective_functions)
        check_for_time_optimization(constraints)

        super(RecedingHorizonOptimization, self)._define_time(phase_time, objective_functions, constraints)


class CyclicRecedingHorizonOptimization(RecedingHorizonOptimization):
    def __init__(
        self,
        biorbd_model: Union[str, biorbd.Model, list, tuple],
        dynamics: Union[Dynamics, DynamicsList],
        cycle_len: Union[int, list, tuple],
        cycle_duration: Union[int, float, list, tuple],
        use_sx=True,
        **kwargs,
    ):
        super(CyclicRecedingHorizonOptimization, self).__init__(
            biorbd_model=biorbd_model,
            dynamics=dynamics,
            window_len=cycle_len,
            window_duration=cycle_duration,
            use_sx=use_sx,
            **kwargs,
        )
        self.time_idx_to_cycle = -1

    def solve(
        self,
        update_function: Callable,
        solver: Solver.Generic = None,
        cyclic_options: dict = None,
        solver_first_iter: Solver.Generic = None,
        **extra_options,
    ) -> Union[Solution, tuple]:

        if solver is None:
            solver = Solver.ACADOS()

        if not cyclic_options:
            cyclic_options = {}
        self._initialize_state_idx_to_cycle(cyclic_options)

        self._set_cyclic_bound()
        if solver.type == SolverType.IPOPT:
            self.update_bounds(self.nlp[0].x_bounds)

        export_options = {"frame_to_export": slice(0, self.time_idx_to_cycle)}
        return super(CyclicRecedingHorizonOptimization, self).solve(
            update_function=update_function,
            solver=solver,
            solver_first_iter=solver_first_iter,
            export_options=export_options,
            **extra_options,
        )

    def _initialize_solution(self, states: list, controls: list):
        _states = InitialGuess(np.concatenate(states, axis=1), interpolation=InterpolationType.EACH_FRAME)
        _controls = InitialGuess(np.concatenate(controls, axis=1), interpolation=InterpolationType.EACH_FRAME)

        solution_ocp = OptimalControlProgram(
            biorbd_model=self.original_values["biorbd_model"][0],
            dynamics=self.original_values["dynamics"][0],
            n_shooting=self.total_optimization_run * self.nlp[0].ns - 1,
            phase_time=self.total_optimization_run * self.nlp[0].ns * self.nlp[0].dt,
            skip_continuity=True,
        )
        return Solution(solution_ocp, [_states, _controls])

    def _initialize_state_idx_to_cycle(self, options):
        if "states" not in options:
            options["states"] = self.nlp[0].states.keys()

        states = self.nlp[0].states
        self.state_idx_to_cycle = list(chain.from_iterable([states[key].index for key in options["states"]]))

    def _set_cyclic_bound(self, sol: Solution = None):
        if self.nlp[0].x_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            raise ValueError(
                "Cyclic bounds for x_bounds should be of "
                "type InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT"
            )
        if self.nlp[0].u_bounds.type != InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            raise ValueError(
                "Cyclic bounds for u_bounds should be of "
                "type InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT"
            )

        s = self.state_idx_to_cycle
        range_of_motion = self.nlp[0].x_bounds.max[s, 1] - self.nlp[0].x_bounds.min[s, 1]
        if sol is None:
            self.nlp[0].x_bounds.min[s, 2] = self.nlp[0].x_bounds.min[s, 0] - range_of_motion * 0.01
            self.nlp[0].x_bounds.max[s, 2] = self.nlp[0].x_bounds.max[s, 0] + range_of_motion * 0.01
        else:
            t = self.time_idx_to_cycle
            self.nlp[0].x_bounds.min[s, 2] = sol.states["all"][s, t] - range_of_motion * 0.01
            self.nlp[0].x_bounds.max[s, 2] = sol.states["all"][s, t] + range_of_motion * 0.01

    @staticmethod
    def _append_current_solution(sol: Solution, states: list, controls: list):
        states.append(sol.states["all"][:, :-1])
        controls.append(sol.controls["all"][:, :-1])

    def advance_window(self, sol: Solution, steps: int = 0, **advance_options):
        super(CyclicRecedingHorizonOptimization, self).advance_window(sol, steps, **advance_options)
        if self.ocp_solver.opts.type == SolverType.IPOPT:
            self.ocp_solver.set_lagrange_multiplier(sol)

    def advance_window_bounds_states(self, sol, **advance_options):
        # Update the initial frame bounds
        self.nlp[0].x_bounds[:, 0] = sol.states["all"][:, self.time_idx_to_cycle]
        self._set_cyclic_bound(sol)
        return True

    def advance_window_initial_guess_states(self, sol, **advance_options):
        if self.nlp[0].x_init.type != InterpolationType.EACH_FRAME:
            self.nlp[0].x_init = InitialGuess(
                np.ndarray(sol.states["all"].shape), interpolation=InterpolationType.EACH_FRAME
            )
            self.nlp[0].x_init.check_and_adjust_dimensions(self.nlp[0].states.shape, self.nlp[0].ns)

        self.nlp[0].x_init.init[:, :] = sol.states["all"]
        return True

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        if self.nlp[0].u_init.type != InterpolationType.EACH_FRAME:
            self.nlp[0].u_init = InitialGuess(
                np.ndarray((sol.controls["all"].shape[0], self.nlp[0].ns)), interpolation=InterpolationType.EACH_FRAME
            )
            self.nlp[0].u_init.check_and_adjust_dimensions(self.nlp[0].controls.shape, self.nlp[0].ns - 1)
        self.nlp[0].u_init.init[:, :] = sol.controls["all"][:, :-1]


class MultiCyclicRecedingHorizonOptimization(CyclicRecedingHorizonOptimization):
    def __init__(
        self,
        biorbd_model: Union[str, biorbd.Model, list, tuple],
        dynamics: Union[Dynamics, DynamicsList],
        cycle_len: Union[int, list, tuple],
        cycle_duration: Union[int, float, list, tuple],
        n_cycles_simultaneous: int,
        n_cycles_to_advance: int = 1,
        use_sx=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        window_size: Union[int, list[int]]
            The number of shooting point of the moving window
        n_cycles_simultaneous: int
            The number of simultaneous cycles
        n_cycles_to_advance: int
            The number of cycles to skip while advancing
        """

        if isinstance(biorbd_model, (list, tuple)) and len(biorbd_model) > 1:
            raise ValueError("Receding horizon optimization must be defined using only one biorbd_model")

        self.cycle_len = cycle_len
        self.n_cycles = n_cycles_simultaneous
        self.n_cycles_to_advance = n_cycles_to_advance

        self.initial_guess_frames = []
        for _ in range(self.n_cycles):
            self.initial_guess_frames.extend(
                list(range(self.n_cycles_to_advance * self.cycle_len, (self.n_cycles_to_advance + 1) * self.cycle_len))
            )
        self.initial_guess_frames.append((self.n_cycles_to_advance + 1) * self.cycle_len)

        super(MultiCyclicRecedingHorizonOptimization, self).__init__(
            biorbd_model=biorbd_model,
            dynamics=dynamics,
            cycle_len=cycle_len * self.n_cycles,
            cycle_duration=cycle_duration * self.n_cycles,
            use_sx=use_sx,
            **kwargs,
        )
        self.time_idx_to_cycle = self.n_cycles_to_advance * self.cycle_len

    def advance_window_initial_guess_states(self, sol, **advance_options):
        if self.nlp[0].x_init.type != InterpolationType.EACH_FRAME:
            self.nlp[0].x_init = InitialGuess(
                np.ndarray(sol.states["all"].shape), interpolation=InterpolationType.EACH_FRAME
            )
            self.nlp[0].x_init.check_and_adjust_dimensions(self.nlp[0].states.shape, self.nlp[0].ns)
        self.nlp[0].x_init.init[:, :] = sol.states["all"][:, self.initial_guess_frames]

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        if self.nlp[0].u_init.type != InterpolationType.EACH_FRAME:
            self.nlp[0].u_init = InitialGuess(
                np.ndarray((sol.controls["all"].shape[0], self.nlp[0].ns)), interpolation=InterpolationType.EACH_FRAME
            )
            self.nlp[0].u_init.check_and_adjust_dimensions(self.nlp[0].controls.shape, self.nlp[0].ns - 1)
        self.nlp[0].u_init.init[:, :] = sol.controls["all"][:, self.initial_guess_frames[:-1]]

    def _initialize_solution(self, states: list, controls: list):
        _states = InitialGuess(np.concatenate(states, axis=1), interpolation=InterpolationType.EACH_FRAME)
        _controls = InitialGuess(np.concatenate(controls, axis=1), interpolation=InterpolationType.EACH_FRAME)

        solution_ocp = OptimalControlProgram(
            biorbd_model=self.original_values["biorbd_model"][0],
            dynamics=self.original_values["dynamics"][0],
            n_shooting=self.cycle_len * self.total_optimization_run - 1,
            phase_time=self.cycle_len * self.total_optimization_run * self.nlp[0].dt,
            skip_continuity=True,
        )
        return Solution(solution_ocp, [_states, _controls])


class NonlinearModelPredictiveControl(RecedingHorizonOptimization):
    """
    NMPC version of receding horizon optimization
    """

    pass


class CyclicNonlinearModelPredictiveControl(CyclicRecedingHorizonOptimization):
    """
    NMPC version of cyclic receding horizon optimization
    """

    pass


class MultiCyclicNonlinearModelPredictiveControl(MultiCyclicRecedingHorizonOptimization):
    """
    NMPC version of cyclic receding horizon optimization
    """

    pass


class MovingHorizonEstimator(RecedingHorizonOptimization):
    """
    MHE version of receding horizon optimization
    """

    pass


class CyclicMovingHorizonEstimator(CyclicRecedingHorizonOptimization):
    """
    MHE version of cyclic receding horizon optimization
    """

    pass
