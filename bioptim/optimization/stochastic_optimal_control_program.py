from typing import Callable, Any

from .non_linear_program import NonLinearProgram as NLP
from ..dynamics.configure_problem import DynamicsList, Dynamics
from ..dynamics.ode_solver import OdeSolver
from ..dynamics.configure_problem import ConfigureProblem
from ..interfaces.stochastic_bio_model import StochasticBioModel
from ..limits.constraints import (
    ConstraintFcn,
    ConstraintList,
    Constraint,
    ParameterConstraintList,
)
from ..limits.phase_transition import PhaseTransitionList
from ..limits.multinode_constraint import MultinodeConstraintList, MultinodeConstraintFcn
from ..limits.multinode_objective import MultinodeObjectiveList
from ..limits.objective_functions import ObjectiveList, Objective, ParameterObjectiveList
from ..limits.path_conditions import BoundsList
from ..limits.path_conditions import InitialGuessList
from ..misc.enums import Node, ControlType
from ..misc.mapping import BiMappingList, Mapping, NodeMappingList, BiMapping
from ..optimization.parameters import ParameterList
from ..optimization.problem_type import SocpType
from ..optimization.optimal_control_program import OptimalControlProgram
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
        s_bounds: BoundsList = None,
        x_init: InitialGuessList | None = None,
        u_init: InitialGuessList | None = None,
        s_init: InitialGuessList | None = None,
        objective_functions: Objective | ObjectiveList = None,
        constraints: Constraint | ConstraintList = None,
        parameters: ParameterList = None,
        parameter_bounds: BoundsList = None,
        parameter_init: InitialGuessList = None,
        parameter_objectives: ParameterObjectiveList = None,
        parameter_constraints: ParameterConstraintList = None,
        external_forces: list[list[Any], ...] | tuple[list[Any], ...] = None,
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
        s_scaling: VariableScalingList = None,
        state_continuity_weight: float = None,
        n_threads: int = 1,
        use_sx: bool = False,
        skip_continuity: bool = False,
        assume_phase_dynamics: bool = False,
        integrated_value_functions: dict[str, Callable] = None,
        problem_type=SocpType.TRAPEZOIDAL_IMPLICIT,
        **kwargs,
    ):
        """ """

        if "n_thread" in kwargs:
            if kwargs["n_thread"] != 1:
                raise ValueError(
                    "Multi-threading is not possible yet while solving a stochastic ocp."
                    "n_thread is set to 1 by default."
                )

        if "ode_solver" in kwargs:
            raise ValueError(
                "The ode_solver cannot be defined for a stochastic ocp. The value is chosen based on the type of problem solved:"
                "\n- TRAPEZOIDAL_EXPLICIT: OdeSolver.TRAPEZOIDAL(), "
                "\n- TRAPEZOIDAL_IMPLICIT: OdeSolver.TRAPEZOIDAL(), "
                "\n- COLLOCATION: OdeSolver.COLLOCATION(method=problem_type.method, polynomial_degree=problem_type.polynomial_degree)"
            )

        self.n_threads = 1

        if "assume_phase_dynamics" in kwargs:
            if kwargs["assume_phase_dynamics"]:
                raise ValueError(
                    "The dynamics cannot be assumed to be the same between phases with a stochastic ocp."
                    "assume_phase_dynamics is set to False by default."
                )

        self.assume_phase_dynamics = False

        self.check_bioptim_version()

        bio_model = self.initialize_model(bio_model)

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_IMPLICIT) or isinstance(
            problem_type, SocpType.TRAPEZOIDAL_EXPLICIT):
            ode_solver = OdeSolver.TRAPEZOIDAL()
        elif isinstance(problem_type, SocpType.COLLOCATION):
            ode_solver = OdeSolver.COLLOCATION(method=problem_type.method, polynomial_degree=problem_type.polynomial_degree)

        self.set_original_values(
            bio_model,
            dynamics,
            n_shooting,
            phase_time,
            x_init,
            u_init,
            s_init,
            x_bounds,
            u_bounds,
            s_bounds,
            x_scaling,
            xdot_scaling,
            u_scaling,
            s_scaling,
            external_forces,
            ode_solver,
            control_type,
            variable_mappings,
            time_phase_mapping,
            node_mappings,
            plot_mappings,
            phase_transitions,
            multinode_constraints,
            multinode_objectives,
            parameter_bounds,
            parameter_init,
            parameter_constraints,
            parameter_objectives,
            state_continuity_weight,
            n_threads,
            use_sx,
            assume_phase_dynamics,
            integrated_value_functions,
        )

        (
            constraints,
            objective_functions,
            parameter_constraints,
            parameter_objectives,
            multinode_constraints,
            multinode_objectives,
            phase_transitions,
            x_bounds,
            u_bounds,
            parameter_bounds,
            s_bounds,
            x_init,
            u_init,
            parameter_init,
            s_init,
        ) = self.check_arguments_and_build_nlp(
            dynamics,
            n_threads,
            n_shooting,
            phase_time,
            x_bounds,
            u_bounds,
            s_bounds,
            x_init,
            u_init,
            s_init,
            x_scaling,
            xdot_scaling,
            u_scaling,
            s_scaling,
            objective_functions,
            constraints,
            parameters,
            phase_transitions,
            multinode_constraints,
            multinode_objectives,
            parameter_bounds,
            parameter_init,
            parameter_constraints,
            parameter_objectives,
            ode_solver,
            use_sx,
            assume_phase_dynamics,
            bio_model,
            external_forces,
            plot_mappings,
            time_phase_mapping,
            control_type,
            variable_mappings,
            integrated_value_functions,
        )
        self.problem_type = problem_type
        NLP.add(self, "is_stochastic", True, True)
        self.prepare_node_mapping(node_mappings)
        self.prepare_dynamics()
        self.prepare_bounds_and_init(
            x_bounds, u_bounds, parameter_bounds, s_bounds, x_init, u_init, parameter_init, s_init
        )

        self._declare_multi_node_penalties(multinode_constraints, multinode_objectives, constraints)

        self.finalize_penalties(
            skip_continuity,
            state_continuity_weight,
            constraints,
            parameter_constraints,
            objective_functions,
            parameter_objectives,
            phase_transitions,
        )

    def prepare_dynamics(self):
        # Prepare the dynamics
        for i in range(self.n_phases):
            self.nlp[i].initialize(self.cx)
            ConfigureProblem.initialize(self, self.nlp[i])
            self.nlp[i].ode_solver.prepare_dynamic_integrator(self, self.nlp[i])
            self.nlp[i].ode_solver.prepare_noised_dynamic_integrator(self, self.nlp[i])

    def _declare_multi_node_penalties(
        self, multinode_constraints: ConstraintList, multinode_objectives: ObjectiveList, constraints: ConstraintList
    ):
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
            )

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

    def _prepare_stochastic_dynamics_collocation(self, constraints):
        """
        Adds the internal constraint needed for the implicit formulation of the stochastic ocp using collocation
        integration. This is the real implementation suggested in Gillis 2013.
        """

        constraints.add(ConstraintFcn.STOCHASTIC_MEAN_SENSORY_INPUT_EQUALS_REFERENCE, node=Node.ALL)

        # Constraints for M
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_HELPER_MATRIX_COLLOCATION,
                node=Node.ALL_SHOOTING,
                phase=i_phase,
            )

        # Constraints for P
        multi_node_penalties = MultinodeConstraintList()
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns):
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                )
            if i_phase > 0 and i_phase < len(self.nlp) - 1:
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION,
                    nodes_phase=(i_phase - 1, i_phase),
                    nodes=(-1, 0),
                )

        multi_node_penalties.add_or_replace_to_penalty_pool(self)
