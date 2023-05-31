"""
Optimal control program with the variational integrator for the dynamics.
"""
from bioptim import (
    Bounds,
    BoundsList,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
    DynamicsList,
    InitialGuess,
    InitialGuessList,
    MultinodeConstraintList,
    NoisedInitialGuess,
    NonLinearProgram,
    Objective,
    ObjectiveList,
    OptimalControlProgram,
    ParameterList,
    PenaltyController,
)
from casadi import MX, vertcat, Function

from biorbd_model_holonomic import BiorbdModelCustomHolonomic


class VariationalOptimalControlProgram(OptimalControlProgram):
    def __init__(
        self,
        bio_model: BiorbdModelCustomHolonomic,
        n_shooting: int,
        final_time: float,
        q_init: InitialGuess | InitialGuessList | NoisedInitialGuess = None,
        u_init: InitialGuess | InitialGuessList | NoisedInitialGuess = None,
        q_bounds: Bounds | BoundsList = None,
        u_bounds: Bounds | BoundsList = None,
        qdot0_init: InitialGuess | InitialGuessList | NoisedInitialGuess = None,
        qdot0_bounds: Bounds | BoundsList = None,
        qdotN_bounds: Bounds | BoundsList = None,
        qdotN_init: InitialGuess | InitialGuessList | NoisedInitialGuess = None,
        holonomic_constraints: Function = None,
        holonomic_constraints_jacobian: Function = None,
        objective_functions: Objective | ObjectiveList = None,
        use_sx: bool = False,
    ):
        self.bio_model = bio_model
        n_qdot = n_q = self.bio_model.nb_q

        self.holonomic_constraints = holonomic_constraints
        self.holonomic_constraints_jacobian = holonomic_constraints_jacobian
        if self.holonomic_constraints is None:
            self.use_constraints = False
        else:
            self.use_constraints = True

        # Dynamics
        dynamics = DynamicsList()
        expand = True
        dynamics.add(self.custom_configure, expand=expand)

        # Declare parameters for the initial and final velocities
        parameters = ParameterList()
        parameters.add(
            "qdot0",  # The name of the parameter
            function=self.qdot_function,  # The function that modifies the biorbd model
            initial_guess=qdot0_init,  # The initial guess
            bounds=qdot0_bounds,  # The bounds
            size=n_qdot,  # The number of elements this particular parameter vector has
        )
        parameters.add(
            "qdotN",  # The name of the parameter
            function=self.qdot_function,  # The function that modifies the biorbd model
            initial_guess=qdotN_init,  # The initial guess
            bounds=qdotN_bounds,  # The bounds
            size=n_qdot,  # The number of elements this particular parameter vector has
        )

        multinode_constraints = self.variational_continuity(n_shooting, n_q)

        super().__init__(
            self.bio_model,
            dynamics,
            n_shooting,
            final_time,
            x_init=q_init,
            u_init=u_init,
            x_bounds=q_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            use_sx=use_sx,
            assume_phase_dynamics=True,
            skip_continuity=True,
            parameters=parameters,
            multinode_constraints=multinode_constraints,
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

    def custom_dynamics_function(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        expand: bool = True,
    ):
        """
        Configure the dynamics of the system

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        expand: bool
            If the dynamics should be expanded with casadi
        """

        nlp.parameters = ocp.v.parameters_in_list
        DynamicsFunctions.apply_parameters(nlp.parameters.mx, nlp)

        dynamics_eval = DynamicsEvaluation(MX(0), MX(0))
        dynamics_dxdt = dynamics_eval.dxdt
        if isinstance(dynamics_dxdt, (list, tuple)):
            dynamics_dxdt = vertcat(*dynamics_dxdt)

        # Note: useless but needed to run bioptim as it need to test the size of xdot
        nlp.dynamics_func = Function(
            "ForwardDyn",
            [nlp.states.scaled.mx_reduced, nlp.controls.scaled.mx_reduced, nlp.parameters.mx],
            [dynamics_dxdt],
            ["x", "u", "p"],
            ["xdot"],
        )

        ts = MX.sym("ts")
        q_prev = MX.sym("q_prev", nlp.model.nb_q, 1)
        q_cur = MX.sym("q_cur", nlp.model.nb_q, 1)
        q_next = MX.sym("q_next", nlp.model.nb_q, 1)
        control_prev = MX.sym("control_prev", nlp.model.nb_q, 1)
        control_cur = MX.sym("control_cur", nlp.model.nb_q, 1)
        control_next = MX.sym("control_next", nlp.model.nb_q, 1)
        q0 = MX.sym("q0", nlp.model.nb_q, 1)
        q0_dot = MX.sym("q0_dot", nlp.model.nb_q, 1)
        q1 = MX.sym("q1", nlp.model.nb_q, 1)
        control0 = MX.sym("control0", nlp.model.nb_q, 1)
        control1 = MX.sym("control1", nlp.model.nb_q, 1)
        qN = MX.sym("qN", nlp.model.nb_q, 1)
        qN_dot = MX.sym("qN_dot", nlp.model.nb_q, 1)
        qN_minus_1 = MX.sym("qN_minus_1", nlp.model.nb_q, 1)
        controlN_minus_1 = MX.sym("controlN_minus_1", nlp.model.nb_q, 1)
        controlN = MX.sym("controlN", nlp.model.nb_q, 1)

        three_nodes_input = [ts, q_prev, q_cur, q_next, control_prev, control_cur, control_next]
        two_first_nodes_input = [ts, q0, q0_dot, q1, control0, control1]
        two_last_nodes_input = [ts, qN_minus_1, qN, qN_dot, controlN_minus_1, controlN]

        if self.use_constraints:
            lambdas = MX.sym("lambda", self.holonomic_constraints.nnz_out(), 1)
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
                    ts,
                    q_prev,
                    q_cur,
                    q_next,
                    control_prev,
                    control_cur,
                    control_next,
                    self.holonomic_constraints,
                    self.holonomic_constraints_jacobian,
                    lambdas,
                )
            ],
        )

        nlp.implicit_dynamics_func_first_node = Function(
            "TwoFirstNodesIntegration",
            two_first_nodes_input,
            [
                self.bio_model.compute_initial_states(
                    ts,
                    q0,
                    q0_dot,
                    q1,
                    control0,
                    control1,
                    self.holonomic_constraints,
                    self.holonomic_constraints_jacobian,
                    lambdas,
                )
            ],
        )

        nlp.implicit_dynamics_func_last_node = Function(
            "TwoLastNodesIntegration",
            two_last_nodes_input,
            [
                self.bio_model.compute_final_states(
                    ts,
                    qN_minus_1,
                    qN,
                    qN_dot,
                    controlN_minus_1,
                    controlN,
                    self.holonomic_constraints,
                    self.holonomic_constraints_jacobian,
                    lambdas,
                )
            ],
        )

        if expand:
            nlp.dynamics_func = nlp.dynamics_func.expand()
            nlp.implicit_dynamics_func = nlp.implicit_dynamics_func.expand()
            nlp.implicit_dynamics_func_first_node = nlp.implicit_dynamics_func_first_node.expand()
            nlp.implicit_dynamics_func_last_node = nlp.implicit_dynamics_func_last_node.expand()

    def custom_configure(self, ocp: OptimalControlProgram, nlp: NonLinearProgram, expand: bool = True):
        """
        If the problem is not constrained, use this custom configuration.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp.
        nlp: NonLinearProgram
            A reference to the phase.
        expand: bool
            If the dynamics should be expanded with casadi.
        """

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        if self.use_constraints:
            lambdas = []
            for i in range(self.holonomic_constraints.nnz_out()):
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

        self.custom_dynamics_function(ocp, nlp, expand)

    def variational_integrator_three_nodes(
        self,
        controllers: list[PenaltyController, PenaltyController, PenaltyController],
    ):
        """
        The discrete Euler Lagrange equations for the main integration.

        Parameters
        ----------
        controllers

        Returns
        -------

        """
        if self.use_constraints:
            return controllers[0].get_nlp.implicit_dynamics_func(
                controllers[0].get_nlp.dt,
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
                controllers[0].get_nlp.dt,
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
        controllers
        n_qdot

        Returns
        -------

        """
        if self.use_constraints:
            return controllers[0].get_nlp.implicit_dynamics_func_first_node(
                controllers[0].get_nlp.dt,
                controllers[0].states["q"].cx,
                controllers[0].parameters.cx[:n_qdot],
                controllers[1].states["q"].cx,
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
                controllers[0].states["lambdas"].cx,
            )
        else:
            return controllers[0].get_nlp.implicit_dynamics_func_first_node(
                controllers[0].get_nlp.dt,
                controllers[0].states["q"].cx,
                controllers[0].parameters.cx[:n_qdot],
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
        The final continuity constraint for the integration.

        Parameters
        ----------
        n_qdot
        controllers

        Returns
        -------

        """
        if self.use_constraints:
            return controllers[0].get_nlp.implicit_dynamics_func_last_node(
                controllers[0].get_nlp.dt,
                controllers[0].states["q"].cx,
                controllers[1].states["q"].cx,
                controllers[0].parameters.cx[n_qdot : 2 * n_qdot],
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
                controllers[1].states["lambdas"].cx,
            )
        else:
            return controllers[0].get_nlp.implicit_dynamics_func_last_node(
                controllers[0].get_nlp.dt,
                controllers[0].states["q"].cx,
                controllers[1].states["q"].cx,
                controllers[0].parameters.cx[n_qdot : 2 * n_qdot],
                controllers[0].controls["tau"].cx,
                controllers[1].controls["tau"].cx,
            )

    def variational_continuity(self, n_shooting, n_qdot) -> MultinodeConstraintList:
        """
        The continuity constraint for the integration.

        Parameters
        ----------
        n_shooting
        n_qdot

        Returns
        -------
        The list of continuity constraints for the integration.
        """
        multinode_constraints = MultinodeConstraintList()
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
