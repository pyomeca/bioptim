"""
Optimal control program with the variational integrator for the dynamics.
"""

import numpy as np
from casadi import Function, vertcat

from .optimal_control_program import OptimalControlProgram
from ..dynamics.configure_problem import ConfigureProblem, DynamicsList
from ..dynamics.dynamics_evaluation import DynamicsEvaluation
from ..limits.constraints import ParameterConstraintList
from ..limits.multinode_constraint import MultinodeConstraintList
from ..limits.objective_functions import ParameterObjectiveList
from ..limits.path_conditions import BoundsList, InitialGuessList
from ..limits.penalty_controller import PenaltyController
from ..misc.enums import ControlType, ContactType
from ..models.biorbd.variational_biorbd_model import VariationalBiorbdModel
from ..models.protocols.variational_biomodel import VariationalBioModel
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
        if type(bio_model) != VariationalBiorbdModel:
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

        if "ode_solver" in kwargs:
            raise ValueError(
                "ode_solver cannot be defined in VariationalOptimalControlProgram since the integration is"
                " done by the variational integrator."
            )

        if "x_init" in kwargs or "x_bounds" in kwargs:
            raise ValueError(
                "In VariationalOptimalControlProgram q_init and q_bounds must be used instead of x_init and x_bounds "
                "since there are no velocities."
            )

        self.bio_model = bio_model
        n_qdot = n_q = self.bio_model.nb_q

        # Dynamics
        dynamics = DynamicsList()
        expand = True
        dynamics.add(
            self.configure_torque_driven,
            expand_dynamics=expand,
            skip_continuity=True,
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
            dynamics,
            n_shooting=n_shooting,
            phase_time=final_time,
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

    def configure_dynamics_function(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        expand: bool = True,
    ):
        """
        Configure the dynamics of the system. This is where the variational integrator equations are defined.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        expand: bool
            If the dynamics should be expanded with casadi
        """

        dynamics_eval = DynamicsEvaluation(nlp.cx(0), nlp.cx(0))
        dynamics_dxdt = dynamics_eval.dxdt
        if isinstance(dynamics_dxdt, (list, tuple)):
            dynamics_dxdt = vertcat(*dynamics_dxdt)

        # Note: useless but needed to run bioptim as it need to test the size of xdot
        nlp.dynamics_func = Function(
            "ForwardDyn",
            [
                vertcat(nlp.time_cx, nlp.dt),
                nlp.states.scaled.cx,
                nlp.controls.scaled.cx,
                nlp.parameters.cx,
                nlp.algebraic_states.scaled.cx,
                nlp.numerical_timeseries.cx,
            ],
            [dynamics_dxdt],
            ["t_span", "x", "u", "p", "a", "d"],
            ["xdot"],
        )

        dt = nlp.cx.sym("time_step")
        q_prev = nlp.cx.sym("q_prev", nlp.model.nb_q, 1)
        q_cur = nlp.cx.sym("q_cur", nlp.model.nb_q, 1)
        q_next = nlp.cx.sym("q_next", nlp.model.nb_q, 1)
        control_prev = nlp.cx.sym("control_prev", nlp.model.nb_q, 1)
        control_cur = nlp.cx.sym("control_cur", nlp.model.nb_q, 1)
        control_next = nlp.cx.sym("control_next", nlp.model.nb_q, 1)
        q0 = nlp.cx.sym("q0", nlp.model.nb_q, 1)
        qdot0 = nlp.cx.sym("qdot_start", nlp.model.nb_q, 1)
        q1 = nlp.cx.sym("q1", nlp.model.nb_q, 1)
        control0 = nlp.cx.sym("control0", nlp.model.nb_q, 1)
        control1 = nlp.cx.sym("control1", nlp.model.nb_q, 1)
        q_ultimate = nlp.cx.sym("q_ultimate", nlp.model.nb_q, 1)
        qdot_ultimate = nlp.cx.sym("qdot_ultimate", nlp.model.nb_q, 1)
        q_penultimate = nlp.cx.sym("q_penultimate", nlp.model.nb_q, 1)
        controlN_minus_1 = nlp.cx.sym("controlN_minus_1", nlp.model.nb_q, 1)
        controlN = nlp.cx.sym("controlN", nlp.model.nb_q, 1)

        three_nodes_input = [dt, q_prev, q_cur, q_next, control_prev, control_cur, control_next]
        two_first_nodes_input = [dt, q0, qdot0, q1, control0, control1]
        two_last_nodes_input = [dt, q_penultimate, q_ultimate, qdot_ultimate, controlN_minus_1, controlN]

        if self.bio_model.has_holonomic_constraints:
            lambdas = nlp.cx.sym("lambda", self.bio_model.nb_holonomic_constraints, 1)
            three_nodes_input.append(lambdas)
            two_first_nodes_input.append(lambdas)
            two_last_nodes_input.append(lambdas)
        else:
            lambdas = None

        nlp.implicit_dynamics_func = Function(
            "ThreeNodesIntegration",
            three_nodes_input,
            [
                self.bio_model.discrete_euler_lagrange_equations(
                    dt,
                    q_prev,
                    q_cur,
                    q_next,
                    control_prev,
                    control_cur,
                    control_next,
                    lambdas,
                )
            ],
        )

        nlp.implicit_dynamics_func_first_node = Function(
            "TwoFirstNodesIntegration",
            two_first_nodes_input,
            [
                self.bio_model.compute_initial_states(
                    dt,
                    q0,
                    qdot0,
                    q1,
                    control0,
                    control1,
                    lambdas,
                )
            ],
        )

        nlp.implicit_dynamics_func_last_node = Function(
            "TwoLastNodesIntegration",
            two_last_nodes_input,
            [
                self.bio_model.compute_final_states(
                    dt,
                    q_penultimate,
                    q_ultimate,
                    qdot_ultimate,
                    controlN_minus_1,
                    controlN,
                    lambdas,
                )
            ],
        )

        if expand:
            nlp.dynamics_func = nlp.dynamics_func.expand()
            nlp.implicit_dynamics_func = nlp.implicit_dynamics_func.expand()
            nlp.implicit_dynamics_func_first_node = nlp.implicit_dynamics_func_first_node.expand()
            nlp.implicit_dynamics_func_last_node = nlp.implicit_dynamics_func_last_node.expand()

    def configure_torque_driven(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        numerical_data_timeseries=None,
        contact_type: list[ContactType] = [],
    ):
        """
        Configure the problem to be torque driven for the variational integrator.
        The states are the q (and the lambdas if the system has holonomic constraints).
        The controls are the tau.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp.
        nlp: NonLinearProgram
            A reference to the phase.
        """

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        if self.bio_model.has_holonomic_constraints:
            lambdas = []
            for i in range(self.bio_model.nb_holonomic_constraints):
                lambdas.append(f"lambda_{i}")
            ConfigureProblem.configure_new_variable(
                "lambdas",
                lambdas,
                ocp,
                nlp,
                as_states=True,
                as_controls=False,
                as_states_dot=False,
            )

        self.configure_dynamics_function(ocp, nlp)

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
            return controllers[0].get_nlp.implicit_dynamics_func(
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
            return controllers[0].get_nlp.implicit_dynamics_func(
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
            return controllers[0].get_nlp.implicit_dynamics_func_first_node(
                controllers[0].dt.cx,
                controllers[0].states["q"].cx,
                controllers[0].parameters.cx[:n_qdot],  # hardcoded
                controllers[1].states["q"].cx,
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
                controllers[0].states["lambdas"].cx,
            )
        else:
            return controllers[0].get_nlp.implicit_dynamics_func_first_node(
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
            return controllers[0].get_nlp.implicit_dynamics_func_last_node(
                controllers[0].dt.cx,
                controllers[0].states["q"].cx,
                controllers[1].states["q"].cx,
                controllers[0].parameters.cx[n_qdot : 2 * n_qdot],  # hardcoded
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
                controllers[1].states["lambdas"].cx,
            )
        else:
            return controllers[0].get_nlp.implicit_dynamics_func_last_node(
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
