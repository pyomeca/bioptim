from typing import Callable, Any

from ..dynamics.configure_problem import DynamicsList, Dynamics
from ..dynamics.ode_solver import OdeSolver, OdeSolverBase
from ..interfaces.biomodel import BioModel
from ..limits.constraints import (
    ConstraintFcn,
    ConstraintList,
    Constraint,
    ParameterConstraintList,
    ParameterConstraint,
)
from ..limits.phase_transition import PhaseTransitionList
from ..limits.multinode_constraint import MultinodeConstraintList, MultinodeConstraintFcn
from ..limits.multinode_objective import MultinodeObjectiveList
from ..limits.objective_functions import ObjectiveList, Objective, ParameterObjectiveList, ParameterObjective
from ..limits.path_conditions import BoundsList, Bounds
from ..limits.path_conditions import InitialGuess, InitialGuessList
from ..misc.enums import Node, ControlType
from ..misc.mapping import BiMappingList, Mapping, NodeMappingList, BiMapping
from ..optimization.parameters import ParameterList, Parameter
from ..optimization.problem_type import SocpType
from ..optimization.optimal_control_program import OptimalControlProgram
from ..optimization.variable_scaling import VariableScalingList, VariableScaling


class StochasticOptimalControlProgram(OptimalControlProgram):
    """
    The main class to define a stochastic ocp. This class prepares the full program and gives all
    the needed interface to modify and solve the program
    """

    def __init__(
        self,
        bio_model: list | tuple | BioModel,
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
        ode_solver: list | OdeSolverBase | OdeSolver = None,
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
        state_continuity_weight: float = None,  # TODO: docstring
        n_threads: int = 1,
        use_sx: bool = False,
        skip_continuity: bool = False,
        assume_phase_dynamics: bool = False,
        integrated_value_functions: dict[str, Callable] = None,
        problem_type: SocpType.SOCP_EXPLICIT | SocpType.SOCP_IMPLICIT = SocpType.SOCP_EXPLICIT,
        **kwargs,
    ):
        """ """

        if "n_thread" in kwargs:
            if kwargs["n_thread"] != 1:
                raise ValueError(
                    "Multi-threading is not possible yet while solving a stochactic ocp."
                    "n_thread is set to 1 by default."
                )

        self.n_threads = 1

        if "assume_phase_dynamics" in kwargs:
            if kwargs["assume_phase_dynamics"]:
                raise ValueError(
                    "The dynamics cannot be assumed to be the same between phases with a stochactic ocp."
                    "assume_phase_dynamics is set to False by default."
                )

        self.assume_phase_dynamics = False

        self.check_bioptim_version()

        bio_model = self.initialize_model(bio_model)

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
            node_mappings,
            state_continuity_weight,
        )

        self.problem_type = problem_type
        self._declare_multi_node_penalties(multinode_constraints, multinode_objectives, constraints)

        self.finalize_penalties(
            skip_continuity,
            state_continuity_weight,
            constraints,
            parameter_constraints,
            objective_functions,
            parameter_objectives,
        )

    def _declare_multi_node_penalties(
        self, multinode_constraints: ConstraintList, multinode_objectives: ObjectiveList, constraints: ConstraintList
    ):
        multinode_constraints.add_or_replace_to_penalty_pool(self)
        multinode_objectives.add_or_replace_to_penalty_pool(self)

        # Add the internal multi-node constraints for the stochastic ocp
        if isinstance(self.problem_type, SocpType.SOCP_EXPLICIT):
            self._prepare_stochastic_dynamics_explicit(
                motor_noise_magnitude=self.problem_type.motor_noise_magnitude,
                sensory_noise_magnitude=self.problem_type.sensory_noise_magnitude,
            )
        elif isinstance(self.problem_type, SocpType.SOCP_IMPLICIT):
            self._prepare_stochastic_dynamics_implicit(
                motor_noise_magnitude=self.problem_type.motor_noise_magnitude,
                sensory_noise_magnitude=self.problem_type.sensory_noise_magnitude,
                constraints=constraints,
            )

    def _prepare_stochastic_dynamics_explicit(self, motor_noise_magnitude, sensory_noise_magnitude):
        """
        Adds the internal constraint needed for the explicit formulation of the stochastic ocp.
        """
        penalty_m_dg_dz_list = MultinodeConstraintList()
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns - 1):  # TODO: Charbie -> check if this is correct
                penalty_m_dg_dz_list.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_EXPLICIT,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                    dynamics=nlp.dynamics_type.dynamic_function,
                    motor_noise_magnitude=motor_noise_magnitude,
                    sensory_noise_magnitude=sensory_noise_magnitude,
                )
            if i_phase > 0:
                penalty_m_dg_dz_list.add(
                    MultinodeConstraintFcn.STOCHASTIC_HELPER_MATRIX_EXPLICIT,
                    nodes_phase=(i_phase - 1, i_phase),
                    nodes=(-1, 0),
                    dynamics=nlp.dynamics_type.dynamic_function,
                    motor_noise_magnitude=motor_noise_magnitude,
                    sensory_noise_magnitude=sensory_noise_magnitude,
                )
        penalty_m_dg_dz_list.add_or_replace_to_penalty_pool(self)

    def _prepare_stochastic_dynamics_implicit(self, motor_noise_magnitude, sensory_noise_magnitude, constraints):
        """
        Adds the internal constraint needed for the implicit formulation of the stochastic ocp.
        """
        # constraint A, C, P, M
        multi_node_penalties = MultinodeConstraintList()
        # Constraints for M
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns - 1):  # TODO: Charbie -> check if this is correct
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
                node=Node.ALL_SHOOTING,
                phase=i_phase,
                motor_noise_magnitude=motor_noise_magnitude,
                sensory_noise_magnitude=sensory_noise_magnitude,
            )
            if i_phase > 0 and i_phase < len(self.nlp) - 1:
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_IMPLICIT,
                    nodes_phase=(i_phase - 1, i_phase),
                    nodes=(-1, 0),
                    motor_noise_magnitude=motor_noise_magnitude,
                    sensory_noise_magnitude=sensory_noise_magnitude,
                )

        # Constraints for A
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_DG_DX_IMPLICIT,
                node=Node.ALL_SHOOTING,
                phase=i_phase,
                dynamics=nlp.dynamics_type.dynamic_function,
                motor_noise_magnitude=motor_noise_magnitude,
                sensory_noise_magnitude=sensory_noise_magnitude,
            )

        # Constraints for C
        for i_phase, nlp in enumerate(self.nlp):
            for i_node in range(nlp.ns - 1):  # TODO: Charbie -> check if this is correct
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_DG_DW_IMPLICIT,
                    nodes_phase=(i_phase, i_phase),
                    nodes=(i_node, i_node + 1),
                    dynamics=nlp.dynamics_type.dynamic_function,
                    motor_noise_magnitude=motor_noise_magnitude,
                    sensory_noise_magnitude=sensory_noise_magnitude,
                )
            if i_phase > 0 and i_phase < len(self.nlp) - 1:
                multi_node_penalties.add(
                    MultinodeConstraintFcn.STOCHASTIC_DG_DW_IMPLICIT,
                    nodes_phase=(i_phase, i_phase + 1),
                    nodes=(-1, 0),
                    dynamics=nlp.dynamics_type.dynamic_function,
                    motor_noise_magnitude=motor_noise_magnitude,
                    sensory_noise_magnitude=sensory_noise_magnitude,
                )

        multi_node_penalties.add_or_replace_to_penalty_pool(self)
