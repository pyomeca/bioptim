from typing import Callable
import sys

import pickle

from .non_linear_program import NonLinearProgram as NLP
from ..dynamics.configure_problem import DynamicsList, Dynamics
from ..dynamics.ode_solver import OdeSolver
from ..models.protocols.stochastic_biomodel import StochasticBioModel
from ..limits.constraints import (
    ConstraintFcn,
    ConstraintList,
    Constraint,
    ParameterConstraintList,
)
from ..limits.phase_transition import PhaseTransitionList, PhaseTransitionFcn
from ..limits.multinode_constraint import MultinodeConstraintList, MultinodeConstraintFcn
from ..limits.multinode_objective import MultinodeObjectiveList
from ..limits.objective_functions import ObjectiveList, Objective, ParameterObjectiveList
from ..limits.path_conditions import BoundsList
from ..limits.path_conditions import InitialGuessList
from ..misc.enums import PhaseDynamics, InterpolationType
from ..misc.__version__ import __version__
from ..misc.enums import Node, ControlType
from ..misc.mapping import BiMappingList, Mapping, NodeMappingList, BiMapping
from ..misc.utils import check_version
from ..optimization.optimal_control_program import OptimalControlProgram
from ..optimization.parameters import ParameterList
from ..optimization.problem_type import SocpType
from ..optimization.solution.solution import Solution
from ..optimization.variable_scaling import VariableScalingList


class StochasticOptimalControlProgram(OptimalControlProgram):
    """
    The main class to define a stochastic ocp. This class prepares the full program and gives all
    the needed interface to modify and solve the program
    """

    def __init__(
        self,
        bio_model: list | tuple | StochasticBioModel,
        dynamics: Dynamics | DynamicsList,
        n_shooting: int | list | tuple,
        phase_time: int | float | list | tuple,
        x_bounds: BoundsList = None,
        u_bounds: BoundsList = None,
        a_bounds: BoundsList = None,
        x_init: InitialGuessList | None = None,
        u_init: InitialGuessList | None = None,
        a_init: InitialGuessList | None = None,
        objective_functions: Objective | ObjectiveList = None,
        constraints: Constraint | ConstraintList = None,
        parameters: ParameterList = None,
        parameter_bounds: BoundsList = None,
        parameter_init: InitialGuessList = None,
        parameter_objectives: ParameterObjectiveList = None,
        parameter_constraints: ParameterConstraintList = None,
        control_type: ControlType | list = ControlType.CONSTANT,
        variable_mappings: BiMappingList = None,
        time_phase_mapping: BiMapping = None,
        node_mappings: NodeMappingList = None,
        plot_mappings: Mapping = None,
        phase_transitions: PhaseTransitionList = None,
        multinode_constraints: MultinodeConstraintList = None,
        multinode_objectives: MultinodeObjectiveList = None,
        x_scaling: VariableScalingList = None,
        xdot_scaling: VariableScalingList = None,
        u_scaling: VariableScalingList = None,
        a_scaling: VariableScalingList = None,
        n_threads: int = 1,
        use_sx: bool = False,
        integrated_value_functions: dict[str, Callable] = None,
        problem_type=SocpType.TRAPEZOIDAL_IMPLICIT,
        **kwargs,
    ):
        _check_multi_threading_and_problem_type(problem_type, **kwargs)
        _check_has_no_ode_solver_defined(**kwargs)
        _check_has_no_phase_dynamics_shared_during_the_phase(problem_type, **kwargs)

        self.problem_type = problem_type
        self._a_init = a_init
        self._a_bounds = a_bounds
        self._a_scaling = a_scaling

        # Parameters
        if parameters is None:
            parameters = ParameterList()
        if parameter_bounds is None:
            parameter_bounds = BoundsList()
        if parameter_init is None:
            parameter_init = InitialGuessList()

        if "motor_noise" not in parameters.keys():
            parameters.add("motor_noise", None, size=bio_model.motor_noise_magnitude.shape[0])
            parameter_bounds.add(
                "motor_noise",
                min_bound=bio_model.motor_noise_magnitude,
                max_bound=bio_model.motor_noise_magnitude,
                interpolation=InterpolationType.CONSTANT,
            )
            parameter_init.add(
                "motor_noise", initial_guess=bio_model.motor_noise_magnitude, interpolation=InterpolationType.CONSTANT
            )

        if "sensory_noise" not in parameters.keys():
            parameters.add("sensory_noise", None, size=bio_model.sensory_noise_magnitude.shape[0])
            parameter_bounds.add(
                "sensory_noise",
                min_bound=bio_model.sensory_noise_magnitude,
                max_bound=bio_model.sensory_noise_magnitude,
                interpolation=InterpolationType.CONSTANT,
            )
            parameter_init.add(
                "sensory_noise",
                initial_guess=bio_model.sensory_noise_magnitude,
                interpolation=InterpolationType.CONSTANT,
            )

        super(StochasticOptimalControlProgram, self).__init__(
            bio_model=bio_model,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=phase_time,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
            parameters=parameters,
            parameter_bounds=parameter_bounds,
            parameter_init=parameter_init,
            parameter_objectives=parameter_objectives,
            parameter_constraints=parameter_constraints,
            ode_solver=None,
            control_type=control_type,
            variable_mappings=variable_mappings,
            time_phase_mapping=time_phase_mapping,
            node_mappings=node_mappings,
            plot_mappings=plot_mappings,
            phase_transitions=phase_transitions,
            multinode_constraints=multinode_constraints,
            multinode_objectives=multinode_objectives,
            x_scaling=x_scaling,
            xdot_scaling=xdot_scaling,
            u_scaling=u_scaling,
            n_threads=n_threads,
            use_sx=use_sx,
            integrated_value_functions=integrated_value_functions,
        )

    def _declare_multi_node_penalties(
        self,
        multinode_constraints: ConstraintList,
        multinode_objectives: ObjectiveList,
        constraints: ConstraintList,
        phase_transition: PhaseTransitionList,
    ):
        """
        This function declares the multi node penalties (constraints and objectives) to the penalty pool.

        Note
        ----
        This function overrides the method in OptimalControlProgram
        """
        multinode_constraints.add_or_replace_to_penalty_pool(self)
        multinode_objectives.add_or_replace_to_penalty_pool(self)

        # Add the internal multi-node constraints for the stochastic ocp
        if isinstance(self.problem_type, SocpType.TRAPEZOIDAL_EXPLICIT):
            self._prepare_stochastic_dynamics_explicit(
                constraints=constraints,
            )
        elif isinstance(self.problem_type, SocpType.TRAPEZOIDAL_IMPLICIT):
            self._prepare_stochastic_dynamics_implicit(
                constraints=constraints,
            )
        elif isinstance(self.problem_type, SocpType.COLLOCATION):
            self._prepare_stochastic_dynamics_collocation(
                constraints=constraints,
                phase_transition=phase_transition,
            )
        else:
            raise RuntimeError("Wrong choice of problem_type, you must choose one of the SocpType.")

    def _prepare_stochastic_dynamics_explicit(self, constraints):
        """
        Adds the internal constraint needed for the explicit formulation of the stochastic ocp.
        """

        constraints.add(ConstraintFcn.STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE, node=Node.ALL)

        penalty_m_dg_dz_list = MultinodeConstraintList()
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns):
                penalty_m_dg_dz_list.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_EXPLICIT,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                )
            if i_phase > 0:
                penalty_m_dg_dz_list.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_EXPLICIT,
                    nodes_phase=(i_phase - 1, i_phase),
                    nodes=(-1, 0),
                )
        penalty_m_dg_dz_list.add_or_replace_to_penalty_pool(self)

    def _prepare_stochastic_dynamics_implicit(self, constraints):
        """
        Adds the internal constraint needed for the implicit formulation of the stochastic ocp.
        """

        constraints.add(ConstraintFcn.STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE, node=Node.ALL)

        multi_node_penalties = MultinodeConstraintList()
        # Constraints for M
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns):
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_IMPLICIT,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                )
            if i_phase > 0 and i_phase < len(self.nlp) - 1:
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_IMPLICIT,
                    nodes_phase=(i_phase - 1, i_phase),
                    nodes=(-1, 0),
                )

        # Constraints for P
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_IMPLICIT,
                node=Node.ALL,
                phase=i_phase,
            )

        # Constraints for A
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_DF_DX_IMPLICIT,
                node=Node.ALL,
                phase=i_phase,
            )

        # Constraints for C
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns):
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_DF_DW_IMPLICIT,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                )
            if i_phase > 0 and i_phase < len(self.nlp) - 1:
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_DF_DW_IMPLICIT,
                    nodes_phase=(i_phase, i_phase + 1),
                    nodes=(-1, 0),
                )

        multi_node_penalties.add_or_replace_to_penalty_pool(self)

    def _prepare_stochastic_dynamics_collocation(self, constraints, phase_transition):
        """
        Adds the internal constraint needed for the implicit formulation of the stochastic ocp using collocation
        integration. This is the real implementation suggested in Gillis 2013.
        """

        if "ref" in self.nlp[0].algebraic_states:
            constraints.add(ConstraintFcn.STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE, node=Node.ALL)

        # Constraints for M
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_HELPER_MATRIX_COLLOCATION,
                node=Node.ALL_SHOOTING,
                phase=i_phase,
                expand=True,
            )

        # Constraints for P inner-phase
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION,
                node=Node.ALL_SHOOTING,
                phase=i_phase,
                expand=True,
            )

        # Constraints for P inter-phase
        for i_phase, nlp in enumerate(self.nlp):
            if len(self.nlp) > 1 and i_phase < len(self.nlp) - 1:
                phase_transition.add(PhaseTransitionFcn.COVARIANCE_CONTINUOUS, phase_pre_idx=i_phase)

    @staticmethod
    def load(file_path: str) -> list:
        """
        Reload a previous optimization (*.bo) saved using save

        Parameters
        ----------
        file_path: str
            The path to the *.bo file

        Returns
        -------
        The ocp and sol structure. If it was saved, the iterations are also loaded
        """

        with open(file_path, "rb") as file:
            try:
                data = pickle.load(file)
            except BaseException as error_message:
                raise ValueError(
                    f"The file '{file_path}' cannot be loaded, maybe the version of bioptim (version {__version__})\n"
                    f"is not the same as the one that created the file (version unknown). For more information\n"
                    "please refer to the original error message below\n\n"
                    f"{type(error_message).__name__}: {error_message}"
                )
            ocp = StochasticOptimalControlProgram.from_loaded_data(data["ocp_initializer"])
            for key in data["versions"].keys():
                key_module = "biorbd_casadi" if key == "biorbd" else key
                try:
                    check_version(sys.modules[key_module], data["versions"][key], ocp.version[key], exclude_max=False)
                except ImportError:
                    raise ImportError(
                        f"Version of {key} from file ({data['versions'][key]}) is not the same as the "
                        f"installed version ({ocp.version[key]})"
                    )
            sol = data["sol"]
            sol.ocp = Solution.SimplifiedOCP(ocp)
            out = [ocp, sol]
        return out

    def _set_default_ode_solver(self):
        """It overrides the method in OptimalControlProgram that set a RK4 by default"""
        if isinstance(self.problem_type, SocpType.TRAPEZOIDAL_IMPLICIT) or isinstance(
            self.problem_type, SocpType.TRAPEZOIDAL_EXPLICIT
        ):
            return OdeSolver.TRAPEZOIDAL()
        elif isinstance(self.problem_type, SocpType.COLLOCATION):
            return OdeSolver.COLLOCATION(
                method=self.problem_type.method,
                polynomial_degree=self.problem_type.polynomial_degree,
                duplicate_starting_point=True,
            )
        else:
            raise RuntimeError("Wrong choice of problem_type, you must choose one of the SocpType.")

    def _set_internal_algebraic_states(self):
        """
        Set the algebraic_states variables to their internal values

        Note
        ----
        This method overrides the method in OptimalControlProgram
        """
        return (
            self._a_init,
            self._a_bounds,
            self._a_scaling,
        )  # Nothing to do here as they are already set before calling super().__init__

    def _set_nlp_is_stochastic(self):
        """
        Set the is_stochastic variable to True for all the nlp

        Note
        ----
        This method overrides the method in OptimalControlProgram
        """
        NLP.add(self, "is_stochastic", True, True)


def _check_multi_threading_and_problem_type(problem_type, **kwargs):
    if not isinstance(problem_type, SocpType.COLLOCATION):
        if "n_thread" in kwargs:
            if kwargs["n_thread"] != 1:
                raise ValueError(
                    "Multi-threading is not possible yet while solving a trapezoidal stochastic ocp."
                    "n_thread is set to 1 by default."
                )


def _check_has_no_ode_solver_defined(**kwargs):
    if "ode_solver" in kwargs:
        raise ValueError(
            "The ode_solver cannot be defined for a stochastic ocp. "
            "The value is chosen based on the type of problem solved:"
            "\n- TRAPEZOIDAL_EXPLICIT: OdeSolver.TRAPEZOIDAL() "
            "\n- TRAPEZOIDAL_IMPLICIT: OdeSolver.TRAPEZOIDAL() "
            "\n- COLLOCATION: "
            "OdeSolver.COLLOCATION("
            "method=problem_type.method, "
            "polynomial_degree=problem_type.polynomial_degree, "
            "duplicate_starting_point=True"
            ")"
        )


def _check_has_no_phase_dynamics_shared_during_the_phase(problem_type, **kwargs):
    if not isinstance(problem_type, SocpType.COLLOCATION):
        if "phase_dynamics" in kwargs:
            if kwargs["phase_dynamics"] == PhaseDynamics.SHARED_DURING_THE_PHASE:
                raise ValueError(
                    "The dynamics cannot be SHARED_DURING_THE_PHASE with a trapezoidal stochastic ocp."
                    "phase_dynamics is set to PhaseDynamics.ONE_PER_NODE by default."
                )
