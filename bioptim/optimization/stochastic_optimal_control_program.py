from typing import Callable, Any
import sys

import pickle

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
from ..limits.phase_transition import PhaseTransitionList, PhaseTransition, PhaseTransitionFcn
from ..limits.multinode_constraint import MultinodeConstraintList, MultinodeConstraintFcn
from ..limits.multinode_objective import MultinodeObjectiveList
from ..limits.objective_functions import ObjectiveList, Objective, ParameterObjectiveList
from ..limits.path_conditions import BoundsList
from ..limits.path_conditions import InitialGuessList
from ..misc.__version__ import __version__
from ..misc.enums import Node, ControlType
from ..misc.mapping import BiMappingList, Mapping, NodeMappingList, BiMapping
from ..misc.utils import check_version
from ..optimization.optimal_control_program import OptimalControlProgram
from ..optimization.parameters import ParameterList
from ..optimization.problem_type import SocpType
from ..optimization.solution import Solution
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

        if not isinstance(problem_type, SocpType.COLLOCATION):
            if "n_thread" in kwargs:
                if kwargs["n_thread"] != 1:
                    raise ValueError(
                        "Multi-threading is not possible yet while solving a trapezoidal stochastic ocp."
                        "n_thread is set to 1 by default."
                    )
        self.n_threads = n_threads

        if "ode_solver" in kwargs:
            raise ValueError(
                "The ode_solver cannot be defined for a stochastic ocp. The value is chosen based on the type of problem solved:"
                "\n- TRAPEZOIDAL_EXPLICIT: OdeSolver.TRAPEZOIDAL(), "
                "\n- TRAPEZOIDAL_IMPLICIT: OdeSolver.TRAPEZOIDAL(), "
                "\n- COLLOCATION: OdeSolver.COLLOCATION(method=problem_type.method, polynomial_degree=problem_type.polynomial_degree)"
            )

        if not isinstance(problem_type, SocpType.COLLOCATION):
            if "assume_phase_dynamics" in kwargs:
                if kwargs["assume_phase_dynamics"]:
                    raise ValueError(
                        "The dynamics cannot be assumed to be the same between nodes with a trapezoidal stochastic ocp."
                        "assume_phase_dynamics is set to False by default."
                    )
        self.assume_phase_dynamics = assume_phase_dynamics

        self._check_bioptim_version()

        bio_model = self._initialize_model(bio_model)

        if isinstance(problem_type, SocpType.TRAPEZOIDAL_IMPLICIT) or isinstance(
            problem_type, SocpType.TRAPEZOIDAL_EXPLICIT
        ):
            ode_solver = OdeSolver.TRAPEZOIDAL()
        elif isinstance(problem_type, SocpType.COLLOCATION):
            ode_solver = OdeSolver.COLLOCATION(
                method=problem_type.method, polynomial_degree=problem_type.polynomial_degree
            )
        else:
            raise ValueError("Wrong choice of ode_solver")

        # Placed here because of MHE
        self._check_and_prepare_dynamics(dynamics)

        self._set_original_values(
            bio_model,
            n_shooting,
            phase_time,
            x_init,
            u_init,
            x_bounds,
            u_bounds,
            x_scaling,
            xdot_scaling,
            u_scaling,
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
        self._set_stochastic_variables_to_original_values(s_init, s_bounds, s_scaling)

        self._check_and_set_threads(n_threads)
        self._check_and_set_shooting_points(n_shooting)
        self._check_and_set_phase_time(phase_time)

        x_bounds, x_init, x_scaling = self._check_and_prepare_decision_variables("x", x_bounds, x_init, x_scaling)
        u_bounds, u_init, u_scaling = self._check_and_prepare_decision_variables("u", u_bounds, u_init, u_scaling)
        s_bounds, s_init, s_scaling = self._check_and_prepare_decision_variables("s", s_bounds, s_init, s_scaling)

        xdot_scaling = self._prepare_option_dict_for_phase("xdot_scaling", xdot_scaling, VariableScalingList)

        (
            constraints,
            objective_functions,
            parameter_constraints,
            parameter_objectives,
            multinode_constraints,
            multinode_objectives,
            phase_transitions,
            parameter_bounds,
            parameter_init,
        ) = self._check_arguments_and_build_nlp(
            dynamics,
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

        # Do not copy singleton since x_scaling was already dealt with before
        NLP.add(self, "x_scaling", x_scaling, True)
        NLP.add(self, "xdot_scaling", xdot_scaling, True)
        NLP.add(self, "u_scaling", u_scaling, True)
        NLP.add(self, "s_scaling", s_scaling, True)

        self.problem_type = problem_type
        NLP.add(self, "is_stochastic", True, True)

        self._prepare_node_mapping(node_mappings)
        self._prepare_dynamics()
        self._prepare_bounds_and_init(
            x_bounds, u_bounds, parameter_bounds, s_bounds, x_init, u_init, parameter_init, s_init
        )

        self._declare_multi_node_penalties(multinode_constraints, multinode_objectives, constraints)

        self._finalize_penalties(
            skip_continuity,
            state_continuity_weight,
            constraints,
            parameter_constraints,
            objective_functions,
            parameter_objectives,
            phase_transitions,
        )

    def _prepare_dynamics(self):
        # Prepare the dynamics
        for i in range(self.n_phases):
            self.nlp[i].initialize(self.cx)
            ConfigureProblem.initialize(self, self.nlp[i])
            self.nlp[i].ode_solver.prepare_dynamic_integrator(self, self.nlp[i])

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

        # Constraints for P inner-phase
        covariance_phase_transition = PhaseTransitionList()
        for i_phase, nlp in enumerate(self.nlp):
            constraints.add(
                ConstraintFcn.STOCHASTIC_COVARIANCE_MATRIX_CONTINUITY_COLLOCATION,
                node=Node.ALL_SHOOTING,
                phase=i_phase,
            )
            if i_phase > 0 and i_phase < len(self.nlp) - 1:
                covariance_phase_transition.add(PhaseTransitionFcn.COVARIANCE_CONTINUOUS, phase_pre_idx=i_phase)

        # Constraints for P inter-phase
        for pt in covariance_phase_transition:
            pt.name = f"COVARIANCE_PHASE_TRANSITION ({pt.type.name}) {pt.nodes_phase[0] % self.n_phases}->{pt.nodes_phase[1] % self.n_phases}"
            pt.list_index = -1
            pt.add_or_replace_to_penalty_pool(self, self.nlp[pt.nodes_phase[0]])

    def _set_stochastic_variables_to_original_values(
        self,
        s_init: InitialGuessList,
        s_bounds: BoundsList,
        s_scaling: VariableScalingList,
    ):
        self.original_values["s_init"] = s_init
        self.original_values["s_bounds"] = s_bounds
        self.original_values["s_scaling"] = s_scaling

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
