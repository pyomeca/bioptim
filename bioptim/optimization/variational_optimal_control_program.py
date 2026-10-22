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
from ..misc.enums import ControlType, ContactType, PlotType
from ..models.biorbd.variational_biorbd_model import VariationalBiorbdModel
from ..models.protocols.variational_biomodel import VariationalBioModel
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.parameters import ParameterList
from ..misc.parameters_types import (
    Bool,
    Int,
    Float,
    NpArrayDictOptional,
)


class VariationalOptimalControlProgram(OptimalControlProgram):
    """
    q_init and q_bounds only the positions initial guess and bounds since there are no velocities in the variational
    integrator.

    The controls are linear continuous. The quadrature rules of the discretization are chosen with the ode_solver, e.g.
    OdeSolver.VARIATIONAL(discrete_approximation=QuadratureRule.MIDPOINT).
    """

    def __init__(
        self,
        bio_model: VariationalBioModel,
        n_shooting: Int,
        final_time: Float,
        q_init: InitialGuessList | None = None,
        q_bounds: BoundsList = None,
        qdot_init: InitialGuessList = None,
        qdot_bounds: BoundsList = None,
        parameters: ParameterList = None,
        parameter_bounds: BoundsList = None,
        parameter_init: InitialGuessList = None,
        parameter_objectives: ParameterObjectiveList = None,
        parameter_constraints: ParameterConstraintList = None,
        multinode_constraints: MultinodeConstraintList = None,
        use_sx: Bool = False,
        ode_solver: OdeSolver.VARIATIONAL | None = None,
        control_type: ControlType = ControlType.LINEAR_CONTINUOUS,
        **kwargs,
    ):
        if not isinstance(bio_model, VariationalBiorbdModel):
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

        if ode_solver is None:
            ode_solver = OdeSolver.VARIATIONAL()
        if not isinstance(ode_solver, OdeSolver.VARIATIONAL):
            raise TypeError("ode_solver must be of type OdeSolver.VARIATIONAL")

        self.bio_model = bio_model
        n_q = self.bio_model.nb_q

        # Dynamics
        dynamics = DynamicsOptionsList()
        expand = True
        dynamics.add(
            expand_dynamics=expand,
            ode_solver=ode_solver,
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

        for key in qdot_bounds.keys():
            if key in parameter_bounds.keys():
                raise KeyError(
                    f"{key} cannot be declared in parameters_bounds as it is a reserved word in "
                    f"VariationalOptimalControlProgram"
                )
            parameter_bounds.add(key, qdot_bounds[key], phase=0)

        for key in qdot_init.keys():
            if key in parameter_init.keys():
                raise KeyError(
                    f"{key} cannot be declared in parameters_init as it is a reserved word in "
                    f"VariationalOptimalControlProgram"
                )
            parameter_init.add(key, qdot_init[key], phase=0)

        if multinode_constraints is None:
            multinode_constraints = MultinodeConstraintList()
        if not isinstance(multinode_constraints, MultinodeConstraintList):
            raise ValueError("multinode_constraints must be a MultinodeConstraintList")

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
            parameter_objectives=parameter_objectives,
            parameter_constraints=parameter_constraints,
            multinode_constraints=multinode_constraints,
            control_type=control_type,
            use_sx=use_sx,
            **kwargs,
        )
        self._plot_states_through_nodes()

    def _plot_states_through_nodes(self) -> None:
        """
        Draw the states as one line through the nodes. With the variational integrator, the states only exist at the
        nodes, so the default PlotType.INTEGRATED, which draws one segment per interval, would draw nothing.
        """
        for key in self.nlp[0].states.keys():
            state_plot = self.nlp[0].plot.get(f"{key}_states")
            if state_plot is not None:
                state_plot.type = PlotType.PLOT
