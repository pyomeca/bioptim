from copy import deepcopy
from math import inf
from typing import Callable
from time import perf_counter

import numpy as np

from .optimal_control_program import OptimalControlProgram
from ..optimization.solution.solution import Solution
from ..dynamics.configure_problem import Dynamics, DynamicsList
from ..limits.constraints import ConstraintFcn, ConstraintList
from ..limits.objective_functions import ObjectiveFcn, ObjectiveList
from ..limits.path_conditions import InitialGuessList
from ..misc.enums import SolverType, InterpolationType, MultiCyclicCycleSolutions, ControlType
from ..interfaces import Solver
from ..interfaces.abstract_options import GenericSolver
from ..models.protocols.biomodel import BioModel
from ..optimization.solution.solution_data import SolutionMerge


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
        bio_model: list | tuple | BioModel,
        dynamics: Dynamics | DynamicsList,
        window_len: int | list | tuple,
        window_duration: int | float | list | tuple,
        use_sx=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        bio_model
            The model to perform the optimization on
        dynamics
            The dynamics equation to use
        window_len:
            The length of the sliding window. It is translated into n_shooting in each individual optimization program
        window_duration
            The time in second of the sliding window
        use_sx
            Same as OCP, but has True as default value
        """

        if isinstance(bio_model, (list, tuple)) and len(bio_model) > 1:
            raise ValueError("Receding horizon optimization must be defined using only one bio_model")

        super(RecedingHorizonOptimization, self).__init__(
            bio_model=bio_model,
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
        solver: GenericSolver = None,
        warm_start: Solution = None,
        solver_first_iter: GenericSolver = None,
        export_options: dict = None,
        max_consecutive_failing: int = inf,
        update_function_extra_params: dict = None,
        get_all_iterations: bool = False,
        **advance_options,
    ) -> Solution | tuple:
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
        split_solutions = []
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

            # Solve and save the current window of interest
            _states, _controls = self.export_data(sol)
            states.append(_states)
            controls.append(_controls)
            # Solve and save the full window of the OCP
            if get_all_iterations:
                all_solutions.append(sol)
            # Update the initial frame bounds and initial guess
            self.advance_window(sol, **advance_options)

            self.total_optimization_run += 1

        states.append({key: sol.decision_states()[key][-1] for key in sol.decision_states().keys()})
        real_time = perf_counter() - real_time

        # Prepare the modified ocp that fits the solution dimension
        dt = sol.t_spans[0][-1]
        final_sol = self._initialize_solution(float(dt), states, controls)
        final_sol.solver_time_to_optimize = total_time
        final_sol.real_time_to_optimize = real_time

        return (final_sol, all_solutions, split_solutions) if get_all_iterations else final_sol

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

    def _initialize_solution(self, dt: float, states: list, controls: list):
        x_init = InitialGuessList()
        for key in self.nlp[0].states.keys():
            x_init.add(
                key,
                np.concatenate([state[key] for state in states], axis=1),
                interpolation=InterpolationType.EACH_FRAME,
                phase=0,
            )

        u_init = InitialGuessList()
        for key in self.nlp[0].controls.keys():
            controls_tp = np.concatenate([control[key] for control in controls], axis=1)
            u_init.add(key, controls_tp, interpolation=InterpolationType.EACH_FRAME, phase=0)

        model_class = self.original_values["bio_model"][0][0]
        model_initializer = self.original_values["bio_model"][0][1]

        solution_ocp = OptimalControlProgram(
            bio_model=model_class(**model_initializer),
            dynamics=self.dynamics[0],
            n_shooting=self.total_optimization_run,
            phase_time=self.total_optimization_run * dt,
            x_init=x_init,
            u_init=u_init,
            use_sx=self.original_values["use_sx"],
        )
        s_init = InitialGuessList()
        p_init = InitialGuessList()
        return Solution.from_initial_guess(solution_ocp, [np.array([dt]), x_init, u_init, p_init, s_init])

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
        states = sol.decision_states(to_merge=SolutionMerge.NODES)

        for key in self.nlp[0].x_bounds.keys():
            if self.nlp[0].x_bounds[key].type == InterpolationType.CONSTANT:
                self.nlp[0].x_bounds.add(
                    key,
                    min_bound=np.repeat(self.nlp[0].x_bounds[key].min[:, 0:1], 3, axis=1),
                    max_bound=np.repeat(self.nlp[0].x_bounds[key].max[:, 0:1], 3, axis=1),
                    phase=0,
                )
                self.nlp[0].x_bounds[key].check_and_adjust_dimensions(len(self.nlp[0].states[key]), 3)
            elif not self.nlp[0].x_bounds[key].type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
                raise NotImplementedError(
                    "The MHE is not implemented yet for x_bounds not being "
                    "CONSTANT or CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT"
                )

            self.nlp[0].x_bounds[key][:, 0] = states[key][:, 1]
        return True

    def advance_window_bounds_controls(self, sol, **advance_options):
        return False

    def advance_window_initial_guess_states(self, sol, **advance_options):
        states = sol.decision_states(to_merge=SolutionMerge.NODES)

        for key in states.keys():
            if self.nlp[0].x_init[key].type != InterpolationType.EACH_FRAME:
                # Override the previous x_init
                self.nlp[0].x_init.add(
                    key, np.ndarray(states[key].shape), interpolation=InterpolationType.EACH_FRAME, phase=0
                )
                self.nlp[0].x_init[key].check_and_adjust_dimensions(len(self.nlp[0].states[key]), self.nlp[0].ns)

            self.nlp[0].x_init[key].init[:, :] = np.concatenate(
                (states[key][:, 1:], states[key][:, -1][:, np.newaxis]), axis=1
            )
        return True

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        for key in self.nlp[0].u_init.keys():
            self.nlp[0].controls.node_index = 0

            if self.nlp[0].u_init[key].type != InterpolationType.EACH_FRAME:
                # Override the previous u_init
                self.nlp[0].u_init.add(
                    key,
                    np.ndarray((sol.controls[key].shape[0], self.nlp[0].n_controls_nodes)),
                    interpolation=InterpolationType.EACH_FRAME,
                    phase=0,
                )
                self.nlp[0].u_init[key].check_and_adjust_dimensions(len(self.nlp[0].controls[key]), self.nlp[0].n_controls_nodes - 1)
            self.nlp[0].u_init[key].init[:, :] = np.concatenate((sol.controls[key][:, 1:], sol.controls[key][:, -1][:, np.newaxis]), axis=1)
        return True

    def export_data(self, sol) -> tuple:
        merged_states = sol.decision_states(to_merge=SolutionMerge.NODES)
        merged_controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

        states = {}
        controls = {}

        for key in sol.states:
            states[key] = merged_states[key][:, self.frame_to_export]
        for key in sol.controls:
            controls[key] = merged_controls[key][:, self.frame_to_export]
        return states, controls

    def _define_time(
        self, phase_time: int | float | list | tuple, objective_functions: ObjectiveList, constraints: ConstraintList
    ):
        """
        Declare the phase_time vector in v. If objective_functions or constraints defined a time optimization,
        a sanity check is perform and the values of initial guess and bounds for these particular phases

        Parameters
        ----------
        phase_time: int | float | list | tuple
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
            penalty_functions: ObjectiveList | ConstraintList
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
        bio_model: list | tuple | BioModel,
        dynamics: Dynamics | DynamicsList,
        cycle_len: int | list | tuple,
        cycle_duration: int | float | list | tuple,
        use_sx=True,
        **kwargs,
    ):
        super(CyclicRecedingHorizonOptimization, self).__init__(
            bio_model=bio_model,
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
        solver: GenericSolver = None,
        cyclic_options: dict = None,
        solver_first_iter: GenericSolver = None,
        **extra_options,
    ) -> Solution | tuple:
        if solver is None:
            solver = Solver.ACADOS()

        if not cyclic_options:
            cyclic_options = {}
        self._initialize_state_idx_to_cycle(cyclic_options)

        self._set_cyclic_bound()
        if solver.type == SolverType.IPOPT:
            self.update_bounds(self.nlp[0].x_bounds)

        export_options = {"frame_to_export": slice(0, self.time_idx_to_cycle if self.time_idx_to_cycle >= 0 else None)}
        return super(CyclicRecedingHorizonOptimization, self).solve(
            update_function=update_function,
            solver=solver,
            solver_first_iter=solver_first_iter,
            export_options=export_options,
            **extra_options,
        )

    def _initialize_solution(self, dt: float, states: list, controls: list):
        x_init = InitialGuessList()
        for key in self.nlp[0].states.keys():
            x_init.add(
                key,
                np.concatenate([state[key][:, :-1] for state in states] + [states[-1][key][:, -1:]], axis=1),
                interpolation=InterpolationType.EACH_FRAME,
                phase=0,
            )

        u_init = InitialGuessList()
        for key in self.nlp[0].controls.keys():
            controls_tp = np.concatenate([control[key] for control in controls], axis=1)
            u_init.add(key, controls_tp, interpolation=InterpolationType.EACH_FRAME, phase=0)

        model_class = self.original_values["bio_model"][0][0]
        model_initializer = self.original_values["bio_model"][0][1]

        solution_ocp = OptimalControlProgram(
            bio_model=model_class(**model_initializer),
            dynamics=self.dynamics[0],
            n_shooting=self.total_optimization_run * self.nlp[0].ns,
            phase_time=self.total_optimization_run * self.nlp[0].ns * dt,
            x_init=x_init,
            u_init=u_init,
            use_sx=self.original_values["use_sx"],
        )
        s_init = InitialGuessList()
        p_init = InitialGuessList()
        return Solution.from_initial_guess(solution_ocp, [np.array([dt]), x_init, u_init, p_init, s_init])

    def _initialize_state_idx_to_cycle(self, options):
        if "states" not in options:
            options["states"] = self.nlp[0].states.keys()

        states = self.nlp[0].states
        self.state_idx_to_cycle = {key: range(len(states[key])) for key in options["states"]}

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

        for key in self.nlp[0].x_bounds.keys():
            s = self.state_idx_to_cycle[key]
            range_of_motion = self.nlp[0].x_bounds[key].max[s, 1] - self.nlp[0].x_bounds[key].min[s, 1]
            if sol is None:
                self.nlp[0].x_bounds[key].min[s, 2] = self.nlp[0].x_bounds[key].min[s, 0] - range_of_motion * 0.01
                self.nlp[0].x_bounds[key].max[s, 2] = self.nlp[0].x_bounds[key].max[s, 0] + range_of_motion * 0.01
            else:
                t = self.time_idx_to_cycle
                states = sol.decision_states(to_merge=SolutionMerge.NODES)
                self.nlp[0].x_bounds[key].min[s, 2] = states[key][s, t] - range_of_motion * 0.01
                self.nlp[0].x_bounds[key].max[s, 2] = states[key][s, t] + range_of_motion * 0.01

    @staticmethod
    def _append_current_solution(sol: Solution, states: list, controls: list):
        states.append(sol.states["all"][:, :-1])
        controls.append(sol.controls["all"][:, :-1])

    def advance_window(self, sol: Solution, steps: int = 0, **advance_options):
        super(CyclicRecedingHorizonOptimization, self).advance_window(sol, steps, **advance_options)
        if self.ocp_solver.opts.type == SolverType.IPOPT:
            self.ocp_solver.set_lagrange_multiplier(sol)

    def advance_window_bounds_states(self, sol, **advance_options):
        states = sol.decision_states(to_merge=SolutionMerge.NODES)

        # Update the initial frame bounds
        for key in states.keys():
            self.nlp[0].x_bounds[key][:, 0] = states[key][:, self.time_idx_to_cycle]
        self._set_cyclic_bound(sol)
        return True

    def advance_window_initial_guess_states(self, sol, **advance_options):
        states = sol.decision_states(to_merge=SolutionMerge.NODES)

        for key in states.keys():
            if self.nlp[0].x_init[key].type != InterpolationType.EACH_FRAME:
                self.nlp[0].x_init.add(
                    key, np.ndarray(states[key].shape), interpolation=InterpolationType.EACH_FRAME, phase=0
                )
                self.nlp[0].x_init[key].check_and_adjust_dimensions(len(self.nlp[0].states[key]), self.nlp[0].ns)

            self.nlp[0].x_init[key].init[:, :] = states[key]
        return True

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        for key in sol.controls.keys():
            self.nlp[0].controls.node_index = 0

            if self.nlp[0].u_init[key].type != InterpolationType.EACH_FRAME:
                self.nlp[0].u_init.add(
                    key,
                    np.ndarray((sol.controls[key].shape[0], self.nlp[0].n_controls_nodes)),
                    interpolation=InterpolationType.EACH_FRAME,
                    phase=0,
                )
                self.nlp[0].u_init[key].check_and_adjust_dimensions(len(self.nlp[0].controls[key]), self.nlp[0].n_controls_nodes - 1)

            self.nlp[0].u_init[key].init[:, :] = sol.controls[key][:, :]
        return True


class MultiCyclicRecedingHorizonOptimization(CyclicRecedingHorizonOptimization):
    def __init__(
        self,
        bio_model: list | tuple | BioModel,
        dynamics: Dynamics | DynamicsList,
        cycle_len: int | list | tuple,
        cycle_duration: int | float | list | tuple,
        n_cycles_simultaneous: int,
        n_cycles_to_advance: int = 1,
        use_sx=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        window_size: int | list[int]
            The number of shooting point of the moving window
        n_cycles_simultaneous: int
            The number of simultaneous cycles
        n_cycles_to_advance: int
            The number of cycles to skip while advancing
        """

        if isinstance(bio_model, (list, tuple)) and len(bio_model) > 1:
            raise ValueError("Receding horizon optimization must be defined using only one bio_model")

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
            bio_model=bio_model,
            dynamics=dynamics,
            cycle_len=cycle_len * self.n_cycles,
            cycle_duration=cycle_duration * self.n_cycles,
            use_sx=use_sx,
            **kwargs,
        )
        self.time_idx_to_cycle = self.n_cycles_to_advance * self.cycle_len

    def advance_window_initial_guess_states(self, sol, **advance_options):
        states = sol.decision_states(to_merge=SolutionMerge.NODES)

        for key in states.keys():
            if self.nlp[0].x_init[key].type != InterpolationType.EACH_FRAME:
                self.nlp[0].x_init.add(
                    key,
                    np.ndarray((states[key].shape[0], self.nlp[0].ns + 1)),
                    interpolation=InterpolationType.EACH_FRAME,
                    phase=0,
                )
                self.nlp[0].x_init[key].check_and_adjust_dimensions(self.nlp[0].states[key].shape, self.nlp[0].ns)
            self.nlp[0].x_init[key].init[:, :] = states[key][:, self.initial_guess_frames]

    def advance_window_initial_guess_controls(self, sol, **advance_options):
        for key in sol.controls.keys():
            self.nlp[0].controls.node_index = 0
            
            if self.nlp[0].u_init[key].type != InterpolationType.EACH_FRAME:
                self.nlp[0].u_init.add(
                    key,
                    np.ndarray((sol.controls[key].shape[0], self.nlp[0].n_controls_nodes)),
                    interpolation=InterpolationType.EACH_FRAME,
                    phase=0,
                )
                self.nlp[0].u_init[key].check_and_adjust_dimensions(self.nlp[0].controls[key].shape, self.nlp[0].n_controls_nodes - 1)
            self.nlp[0].u_init[key].init[:, :] = sol.controls[key][:, self.initial_guess_frames]

    def solve(
        self,
        update_function=None,
        cycle_solutions: MultiCyclicCycleSolutions = MultiCyclicCycleSolutions.NONE,
        **extra_options,
    ) -> Solution | tuple:
        """


        Parameters
        ----------
        update_function: callable
            A function that will be called at each iteration of the optimization.
        cycle_solutions: MultiCyclicCycleSolutions
            The extra solutions to return, e.g. none, the solution of each cycle, all cycles of the terminal window.
        """
        get_all_iterations = extra_options["get_all_iterations"] if "get_all_iterations" in extra_options else False
        extra_options["get_all_iterations"] = True if cycle_solutions is not MultiCyclicCycleSolutions.NONE else False

        solution = super(MultiCyclicRecedingHorizonOptimization, self).solve(
            update_function=update_function, **extra_options
        )

        final_solution = [solution[0]]

        if get_all_iterations:
            final_solution.append(solution[1])

        cycle_solutions_output = []
        if cycle_solutions in (MultiCyclicCycleSolutions.FIRST_CYCLES, MultiCyclicCycleSolutions.ALL_CYCLES):
            for sol in solution[1]:
                _states, _controls = self.export_cycles(sol)
                dt = float(sol.t_spans[0][-1])
                cycle_solutions_output.append(self._initialize_one_cycle(dt, _states, _controls))

        if cycle_solutions == MultiCyclicCycleSolutions.ALL_CYCLES:
            for cycle_number in range(1, self.n_cycles):
                _states, _controls = self.export_cycles(solution[1][-1], cycle_number=cycle_number)
                dt = float(sol.t_spans[0][-1])
                cycle_solutions_output.append(self._initialize_one_cycle(dt, _states, _controls))

        if cycle_solutions in (MultiCyclicCycleSolutions.FIRST_CYCLES, MultiCyclicCycleSolutions.ALL_CYCLES):
            final_solution.append(cycle_solutions_output)

        return tuple(final_solution) if len(final_solution) > 1 else final_solution[0]

    def export_cycles(self, sol: Solution, cycle_number: int = 0) -> tuple[dict, dict]:
        """Exports the solution of the desired cycle from the full window solution"""
        window_slice = slice(cycle_number * self.cycle_len, (cycle_number + 1) * self.cycle_len + 1)
        states = {}
        controls = {}
        for key in sol.states.keys():
            states[key] = sol.states[key][:, window_slice]
        for key in sol.controls.keys():
            controls[key] = sol.controls[key][:, window_slice]
        return states, controls

    def _initialize_solution(self, dt: float, states: list, controls: list):
        x_init = InitialGuessList()
        for key in self.nlp[0].states.keys():
            x_init.add(
                key,
                np.concatenate([state[key][:, :-1] for state in states] + [states[-1][key][:, -1:]], axis=1),
                interpolation=InterpolationType.EACH_FRAME,
                phase=0,
            )

        u_init = InitialGuessList()
        for key in self.nlp[0].controls.keys():
            controls_tp = np.concatenate([control[key] for control in controls], axis=1)
            u_init.add(key, controls_tp, interpolation=InterpolationType.EACH_FRAME, phase=0)

        model_class = self.original_values["bio_model"][0][0]
        model_initializer = self.original_values["bio_model"][0][1]

        solution_ocp = OptimalControlProgram(
            bio_model=model_class(**model_initializer),
            dynamics=self.dynamics[0],
            n_shooting=self.total_optimization_run * self.cycle_len,
            phase_time=self.total_optimization_run * self.cycle_len * dt,
            x_init=x_init,
            u_init=u_init,
            use_sx=self.original_values["use_sx"],
        )
        s_init = InitialGuessList()
        p_init = InitialGuessList()
        return Solution.from_initial_guess(solution_ocp, [np.array([dt]), x_init, u_init, p_init, s_init])

    def _initialize_one_cycle(self, dt: float, states: np.ndarray, controls: np.ndarray):
        """return a solution for a single window kept of the MHE"""
        x_init = InitialGuessList()
        for key in self.nlp[0].states.keys():
            x_init.add(
                key,
                states[key],
                interpolation=InterpolationType.EACH_FRAME,
                phase=0,
            )

        u_init = InitialGuessList()
        u_init_for_solution = InitialGuessList()
        for key in self.nlp[0].controls.keys():
            controls_tp = controls[key]
            u_init_for_solution.add(key, controls_tp, interpolation=InterpolationType.EACH_FRAME, phase=0)
            if self.original_values["control_type"] == ControlType.CONSTANT:
                controls_tp = controls_tp[:, :-1]
            u_init.add(key, controls_tp, interpolation=InterpolationType.EACH_FRAME, phase=0)

        original_values = self.original_values

        model_class = original_values["bio_model"][0][0]
        model_initializer = original_values["bio_model"][0][1]

        solution_ocp = OptimalControlProgram(
            bio_model=model_class(**model_initializer),
            dynamics=original_values["dynamics"][0],
            objective_functions=deepcopy(original_values["objective_functions"]),
            ode_solver=self.nlp[0].ode_solver,
            n_shooting=self.cycle_len,
            phase_time=self.cycle_len * dt,
            x_init=x_init,
            u_init=u_init,
            use_sx=original_values["use_sx"],
        )
        s_init = InitialGuessList()
        p_init = InitialGuessList()
        return Solution.from_initial_guess(solution_ocp, [np.array([dt]), x_init, u_init_for_solution, p_init, s_init])


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
