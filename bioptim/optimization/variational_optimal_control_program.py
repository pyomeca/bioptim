"""
Optimal control program with the variational integrator for the dynamics.
"""

import numpy as np
from casadi import Function, vertcat

from .optimal_control_program import OptimalControlProgram
from ..dynamics.configure_problem import ConfigureProblem, DynamicsOptionsList
from ..dynamics.dynamics_evaluation import DynamicsEvaluation
from ..dynamics.ode_solvers import OdeSolver
from ..limits.constraints import ParameterConstraintList
from ..limits.multinode_constraint import MultinodeConstraintList
from ..limits.objective_functions import ParameterObjectiveList
from ..limits.path_conditions import BoundsList, InitialGuessList
from ..limits.penalty_controller import PenaltyController
from ..misc.enums import ControlType, ContactType
from ..models.biorbd.variational_biorbd_model import VariationalBiorbdModel
from ..models.protocols.variational_biomodel import VariationalBioModel
from ..models.biorbd.model_dynamics import VariationalTorqueBiorbdModel
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.parameters import ParameterList
from ..optimization.variable_scaling import VariableScaling


class VariationalOptimalControlProgram(OptimalControlProgram):
    """
    q_init and q_bounds only the positions initial guess and bounds since there are no velocities in the variational
    integrator.
    """

    def __init__(
        self,
        bio_model: VariationalBioModel,
        n_shooting: int,
        final_time: float,
        q_init: InitialGuessList = None,
        q_bounds: BoundsList = None,
        qdot_init: InitialGuessList = None,
        qdot_bounds: BoundsList = None,
        parameters: ParameterList = None,
        parameter_bounds: BoundsList = None,
        parameter_init: InitialGuessList = None,
        parameter_objectives: ParameterObjectiveList = None,
        parameter_constraints: ParameterConstraintList = None,
        multinode_constraints: MultinodeConstraintList = None,
        use_sx: bool = False,
        **kwargs,
    ):
        if type(bio_model) != VariationalBiorbdModel and type(bio_model) != VariationalTorqueBiorbdModel:
            raise TypeError("bio_model must be of type VariationalBiorbdModel")

        if "phase_time" in kwargs:
            raise NotImplementedError(
                "Multiphase problems have not been implemented yet with VariationalOptimalControlProgram. Please use "
                "final_time argument instead of phase_time."
            )

        if not isinstance(final_time, (float, int)):
            if (
                isinstance(final_time, (tuple, list))
                and len(final_time) != 1
                or isinstance(final_time, np.ndarray)
                and final_time.size != 1
            ):
                raise ValueError(
                    "Multiphase problems have not been implemented yet with "
                    "VariationalOptimalControlProgram. Please use final_time argument and use one float to"
                    " define it."
                )

        if "x_init" in kwargs or "x_bounds" in kwargs:
            raise ValueError(
                "In VariationalOptimalControlProgram q_init and q_bounds must be used instead of x_init and x_bounds "
                "since there are no velocities."
            )

        self.bio_model = bio_model
        n_qdot = n_q = self.bio_model.nb_q

        # Dynamics
        dynamics = DynamicsOptionsList()
        expand = True
        dynamics.add(
            expand_dynamics=expand,
            skip_continuity=True,
            ode_solver=OdeSolver.VARIATIONAL(),  # This is a fake ode_solver to be able to use the variational integrator
        )

        if qdot_bounds is None or not isinstance(qdot_bounds, BoundsList):
            raise ValueError(
                "qdot_bounds must be a BoundsList, moreover they must contain 'qdot_start' and 'qdot_end' keys"
            )
        for key in qdot_bounds.keys():
            # Make sure only these keys are defined
            if key not in ("qdot_start", "qdot_end"):
                raise ValueError(
                    "qdot_bounds must be a BoundsList, moreover they must contain 'qdot_start' and 'qdot_end' keys"
                )

        if qdot_init is None:
            qdot_init = InitialGuessList()
        if not isinstance(qdot_init, InitialGuessList):
            raise ValueError(
                "qdot_init must be a InitialGuessList, moreover they can only contain 'qdot_start' and 'qdot_end' keys"
            )
        for key in qdot_init.keys():
            # Make sure only these keys are defined
            if key not in ("qdot_start", "qdot_end"):
                raise ValueError(
                    "qdot_init must be a InitialGuessList, moreover they can only contain 'qdot_start' and 'qdot_end' "
                    "keys"
                )
        # Make sure all are declared
        for key in ("qdot_start", "qdot_end"):
            if key not in qdot_init.keys():
                qdot_init.add(key, [0] * n_q)

        # Declare parameters for the initial and final velocities
        if parameters is None:
            parameters = ParameterList(use_sx=use_sx)
        if not isinstance(parameters, ParameterList):
            raise ValueError("parameters must be a ParameterList")

        if parameter_init is None:
            parameter_init = InitialGuessList()
        if not isinstance(parameter_init, InitialGuessList):
            raise ValueError("parameter_init must be a InitialGuessList")

        if parameter_bounds is None:
            parameter_bounds = BoundsList()
        if not isinstance(parameter_bounds, BoundsList):
            raise ValueError("parameter_bounds must be a BoundsList")

        if parameter_constraints is None:
            parameter_constraints = ParameterConstraintList()
        if not isinstance(parameter_constraints, ParameterConstraintList):
            raise ValueError("parameter_constraints must be a ParameterConstraintList")

        if parameter_objectives is None:
            parameter_objectives = ParameterObjectiveList()
        if not isinstance(parameter_objectives, ParameterObjectiveList):
            raise ValueError("parameter_objectives must be a ParameterObjectiveList")

        if "qdot_start" in parameters.keys() or "qdot_end" in parameters.keys():
            raise KeyError(
                "'qdot_start' and 'qdot_end' cannot be declared in parameters as they are reserved words in "
                "VariationalOptimalControlProgram. To define the initial and final velocities, please use "
                "`qdot_init` and `qdot_bounds` instead."
            )
        parameters.add(
            "qdot_start",  # The name of the parameter
            function=self.qdot_function,  # The function that modifies the biorbd model
            size=n_qdot,  # The number of elements this particular parameter vector has
            scaling=VariableScaling("qdot_start", np.ones((n_qdot, 1))),
        )
        parameters.add(
            "qdot_end",  # The name of the parameter
            function=self.qdot_function,  # The function that modifies the biorbd model
            size=n_qdot,  # The number of elements this particular parameter vector has
            scaling=VariableScaling("qdot_end", np.ones((n_qdot, 1))),
        )

        for key in qdot_bounds.keys():
            if key in parameter_bounds.keys():
                raise KeyError(
                    f"{key} cannot be declared in parameters_bounds as it is a reserved word in "
                    f"VariationalOptimalControlProgram"
                )
            parameter_bounds.add(key, qdot_bounds[key], phase=0)

        for init in qdot_init.keys():
            if key in parameter_init.keys():
                raise KeyError(
                    f"{key} cannot be declared in parameters_init as it is a reserved word in "
                    f"VariationalOptimalControlProgram"
                )
            parameter_init.add(init, qdot_init[key], phase=0)

        if multinode_constraints is None:
            multinode_constraints = MultinodeConstraintList()
        if not isinstance(multinode_constraints, MultinodeConstraintList):
            raise ValueError("multinode_constraints must be a MultinodeConstraintList")
        self.variational_continuity(multinode_constraints, n_shooting, n_q)

        super().__init__(
            self.bio_model,
            n_shooting=n_shooting,
            phase_time=final_time,
            dynamics=dynamics,
            x_init=q_init,
            x_bounds=q_bounds,
            parameters=parameters,
            parameter_init=parameter_init,
            parameter_bounds=parameter_bounds,
            multinode_constraints=multinode_constraints,
            control_type=ControlType.LINEAR_CONTINUOUS,
            use_sx=use_sx,
            **kwargs,
        )

    @staticmethod
    def qdot_function(model, value):
        """
        It is currently mandatory to provide a function to the method add of ParameterList.
        Parameters
        ----------
        model
        value
        """
        pass

    def variational_integrator_three_nodes(
        self,
        controllers: list[PenaltyController, PenaltyController, PenaltyController],
    ):
        """
        The discrete Euler Lagrange equations for the main integration.

        Parameters
        ----------
        controllers: list[PenaltyController, PenaltyController, PenaltyController]
            The controllers of the three nodes considered in the input list.

        Returns
        -------
        The symbolic expression of the discrete Euler Lagrange equations
        for the integration from node i-1, i,to i+1.

        """
        if self.bio_model.has_holonomic_constraints:
            return controllers[0].get_nlp.dynamics_defects_func(
                controllers[0].dt.cx,
                controllers[0].states["q"].cx,
                controllers[1].states["q"].cx,
                controllers[2].states["q"].cx,
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
                controllers[2].controls["tau"].cx,
                controllers[1].states["lambdas"].cx,
            )
        else:
            return controllers[0].get_nlp.dynamics_defects_func(
                controllers[0].dt.cx,
                controllers[0].states["q"].cx,
                controllers[1].states["q"].cx,
                controllers[2].states["q"].cx,
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
                controllers[2].controls["tau"].cx,
            )

    def variational_integrator_initial(
        self,
        controllers: list[PenaltyController, PenaltyController],
        n_qdot: int,
    ):
        """
        The initial continuity constraint for the integration.

        Parameters
        ----------
        controllers: list[PenaltyController, PenaltyController]
            The controllers of the two first nodes considered in the input list.
        n_qdot:
            The number of generalized velocities

        Returns
        -------
        The symbolic expression of the initial continuity constraint for the integration.

        """
        if self.bio_model.has_holonomic_constraints:
            return controllers[0].get_nlp.dynamics_defects_func_first_node(
                controllers[0].dt.cx,
                controllers[0].states["q"].cx,
                controllers[0].parameters.cx[:n_qdot],  # hardcoded
                controllers[1].states["q"].cx,
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
                controllers[0].states["lambdas"].cx,
            )
        else:
            return controllers[0].get_nlp.dynamics_defects_func_first_node(
                controllers[0].dt.cx,
                controllers[0].states["q"].cx,
                controllers[0].parameters.cx[:n_qdot],  # hardcoded
                controllers[1].states["q"].cx,
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
            )

    def variational_integrator_final(
        self,
        controllers: list[PenaltyController, PenaltyController],
        n_qdot: int,
    ):
        """
        The final continuity constraint for the integration. Warning: When the system has holonomic constraints, there
        are more variables than equations so the lambda and the velocity of the last node are "free" variables.

        Parameters
        ----------
        controllers: list[PenaltyController, PenaltyController]
            The controllers of the two first nodes considered in the input list.
        n_qdot:
            The number of generalized velocities

        Returns
        -------
        The symbolic expression of the final continuity constraint for the integration.

        """
        if self.bio_model.has_holonomic_constraints:
            return controllers[0].get_nlp.dynamics_defects_func_last_node(
                controllers[0].dt.cx,
                controllers[0].states["q"].cx,
                controllers[1].states["q"].cx,
                controllers[0].parameters.cx[n_qdot : 2 * n_qdot],  # hardcoded
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
                controllers[1].states["lambdas"].cx,
            )
        else:
            return controllers[0].get_nlp.dynamics_defects_func_last_node(
                controllers[0].dt.cx,
                controllers[0].states["q"].cx,
                controllers[1].states["q"].cx,
                controllers[0].parameters.cx[n_qdot : 2 * n_qdot],  # hardcoded
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
            )

    def variational_continuity(
        self, multinode_constraints: MultinodeConstraintList, n_shooting: int, n_qdot: int
    ) -> MultinodeConstraintList:
        """
        The continuity constraint for the integration.

        Parameters
        ----------
        multinode_constraints: MultinodeConstraintList
        n_shooting: int
        n_qdot: int

        Returns
        -------
        The list of continuity constraints for the integration.
        """
        for i in range(n_shooting - 1):
            multinode_constraints.add(
                self.variational_integrator_three_nodes,
                nodes_phase=(0, 0, 0),
                nodes=(i, i + 1, i + 2),
            )
        # add initial and final constraints
        multinode_constraints.add(
            self.variational_integrator_initial,
            nodes_phase=(0, 0),
            nodes=(0, 1),
            n_qdot=n_qdot,
        )

        multinode_constraints.add(
            self.variational_integrator_final,
            nodes_phase=(0, 0),
            nodes=(n_shooting - 1, n_shooting),
            n_qdot=n_qdot,
        )
        return multinode_constraints
