from casadi import vertcat, MX
import biorbd


class DynamicsFunctions:
    """
    Implementation of all the dynamic functions

    Methods
    -------
    custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Interface to custom dynamic function provided by the user
    forward_dynamics_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by joint torques, optional external forces can be declared.
    forward_dynamics_torque_driven_with_contact(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by joint torques with contact constraints.
    forces_from_forward_dynamics_with_contact_for_torque_driven_problem(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Contact forces of a forward dynamics driven by joint torques with contact constraints.
    forces_from_forward_dynamics_with_contact_for_torque_activation_driven_problem(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Contact forces of a forward dynamics driven by muscle activation with contact constraints.
    forward_dynamics_torque_activations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by joint torques activations.
    forward_dynamics_torque_activations_driven_with_contact(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by joint torques activations with contact constraints.
    forward_dynamics_muscle_activations_and_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by muscle activations and joint torques.
    forward_dynamics_muscle_activations_and_torque_driven_with_contact(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by muscles activations and joint torques with contact constraints.
    forces_from_forward_dynamics_muscle_activations_and_torque_driven_with_contact(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.
    forward_dynamics_muscle_activations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by muscle activations.
    forward_dynamics_muscle_excitations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by muscle excitations.
    forward_dynamics_muscle_excitations_and_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by muscle excitations and joint torques.
    forward_dynamics_muscle_excitations_and_torque_driven_with_contact(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Forward dynamics driven by muscle excitations and joint torques with contact constraints..
    forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Contact forces of a forward dynamics driven by muscle excitations and joint torques with contact constraints.
    dispatch_q_qdot_tau_data(states: MX.sym, controls: MX.sym, nlp: NonLinearProgram) -> tuple[MX.sym, MX.sym, MX.sym]
        Extracting q, qdot and tau from states and controls, assuming state, state and control, respectively.
    apply_parameters(parameters: MX.sym, nlp: NonLinearProgram)
        Apply the parameter variables to the model. This should be called before calling the dynamics
    """

    @staticmethod
    def custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram") -> MX:
        """
        Interface to custom dynamic function provided by the user.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """
        qdot, qddot = nlp.dynamics_type.dynamic_function(states, controls, parameters, nlp)
        return vertcat(qdot, qddot)

    @staticmethod
    def forward_dynamics_torque_driven(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by joint torques, optional external forces can be declared.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)

        if nlp.external_forces:
            dxdt = MX(nlp.nx, nlp.ns)
            for i, f_ext in enumerate(nlp.external_forces):
                qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()
                qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
                dxdt[:, i] = vertcat(qdot_reduced, qddot_reduced)
        else:
            qddot = nlp.model.ForwardDynamics(q, qdot, tau).to_mx()
            qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
            dxdt = vertcat(qdot_reduced, qddot_reduced)

        return dxdt

    @staticmethod
    def forward_dynamics_torque_driven_with_contact(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forces_from_forward_dynamics_with_contact_for_torque_driven_problem(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        cs = nlp.model.getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau, cs)

        return cs.getForce().to_mx()

    @staticmethod
    def forces_from_forward_dynamics_with_contact_for_torque_activation_driven_problem(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by muscle activation with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, torque_act = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        tau = nlp.model.torque(torque_act, q, qdot).to_mx()

        cs = nlp.model.getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau, cs)

        return cs.getForce().to_mx()

    @staticmethod
    def forward_dynamics_torque_activations_driven(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by joint torques activations.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, torque_act = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        tau = nlp.model.torque(torque_act, q, qdot).to_mx()
        qddot = nlp.model.ForwardDynamics(q, qdot, tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_torque_activations_driven_with_contact(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by joint torques activations with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, torque_act = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        tau = nlp.model.torque(torque_act, q, qdot).to_mx()
        qddot = nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_muscle_activations_and_torque_driven(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by muscle activations and joint torques.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp.shape["muscle"])
        muscles_activations = controls[nlp.shape["tau"] :]

        for k in range(nlp.shape["muscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamics(nlp.model, q, qdot, tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_muscle_activations_and_torque_driven_with_contact(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by muscles activations and joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp.shape["muscle"])
        muscles_activations = controls[nlp.shape["tau"] :]

        for k in range(nlp.shape["muscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forces_from_forward_dynamics_muscle_activations_and_torque_driven_with_contact(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp.shape["muscle"])
        muscles_activations = controls[nlp.shape["tau"] :]

        for k in range(nlp.shape["muscle"]):
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()

        tau = muscles_tau + residual_tau

        cs = nlp.model.getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau, cs)

        return cs.getForce().to_mx()

    @staticmethod
    def forward_dynamics_muscle_activations_driven(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by muscle activations.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)

        nq = nlp.mapping["q"].to_first.len
        q = nlp.mapping["q"].to_second.map(states[:nq])
        qdot = nlp.mapping["q_dot"].to_second.map(states[nq:])

        muscles_states = biorbd.VecBiorbdMuscleState(nlp.shape["muscle"])
        muscles_activations = controls

        for k in range(nlp.shape["muscle"]):
            muscles_states[k].setActivation(muscles_activations[k])

        muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, muscles_tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced)

    @staticmethod
    def forward_dynamics_muscle_excitations_driven(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by muscle excitations.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)

        nq = nlp.mapping["q"].to_first.len
        q = nlp.mapping["q"].to_second.map(states[:nq])
        qdot = nlp.mapping["q_dot"].to_second.map(states[nq:])

        muscles_states = biorbd.VecBiorbdMuscleState(nlp.shape["muscle"])
        muscles_excitation = controls
        muscles_activations = states[nlp.shape["q"] + nlp.shape["q_dot"] :]

        for k in range(nlp.shape["muscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp.model.activationDot(muscles_states).to_mx()

        muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, muscles_tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forward_dynamics_muscle_excitations_and_torque_driven(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by muscle excitations and joint torques.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp.shape["muscle"])
        muscles_excitation = controls[nlp.shape["tau"] :]
        muscles_activations = states[nlp.shape["q"] + nlp.shape["q_dot"] :]

        for k in range(nlp.shape["muscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp.model.activationDot(muscles_states).to_mx()

        muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forward_dynamics_muscle_excitations_and_torque_driven_with_contact(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Forward dynamics driven by muscle excitations and joint torques with contact constraints..

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp.shape["muscle"])
        muscles_excitation = controls[nlp.shape["tau"] :]
        muscles_activations = states[nlp.shape["q"] + nlp.shape["q_dot"] :]

        for k in range(nlp.shape["muscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])
        muscles_activations_dot = nlp.model.activationDot(muscles_states).to_mx()

        muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau).to_mx()

        q_dot = nlp.model.computeQdot(q, qdot).to_mx()
        qdot_reduced = nlp.mapping["q"].to_first.map(q_dot)
        qddot_reduced = nlp.mapping["q_dot"].to_first.map(qddot)
        return vertcat(qdot_reduced, qddot_reduced, muscles_activations_dot)

    @staticmethod
    def forces_from_forward_dynamics_muscle_excitations_and_torque_driven_with_contact(
        states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: "NonLinearProgram"
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by muscle excitations and joint torques with contact constraints.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot, residual_tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

        muscles_states = biorbd.VecBiorbdMuscleState(nlp.shape["muscle"])
        muscles_excitation = controls[nlp.shape["tau"] :]
        muscles_activations = states[nlp.shape["q"] + nlp.shape["q_dot"] :]

        for k in range(nlp.shape["muscle"]):
            muscles_states[k].setExcitation(muscles_excitation[k])
            muscles_states[k].setActivation(muscles_activations[k])

        muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()
        tau = muscles_tau + residual_tau
        cs = nlp.model.getConstraints()
        biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau, cs)
        return cs.getForce().to_mx()

    @staticmethod
    def dispatch_q_qdot_tau_data(
        states: MX.sym, controls: MX.sym, nlp: "NonLinearProgram"
    ) -> tuple:
        """
        Extracting q, qdot and tau from states and controls, assuming state, state and control, respectively.

        Parameters
        ----------
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        nlp: NonLinearProgram
            The definition of the system

        Returns
        ----------
        MX.sym
            q, the generalized coordinates
        MX.sym
            qdot, the generalized velocities
        MX.sym
            tau, the generalized torques
        """

        nq = nlp.mapping["q"].to_first.len
        q = nlp.mapping["q"].to_second.map(states[:nq])
        qdot = nlp.mapping["q_dot"].to_second.map(states[nq:])
        tau = nlp.mapping["tau"].to_second.map(controls[: nlp.shape["tau"]])

        return q, qdot, tau

    @staticmethod
    def apply_parameters(parameters: MX.sym, nlp: "NonLinearProgram"):
        """
        Apply the parameter variables to the model. This should be called before calling the dynamics

        Parameters
        ----------
        parameters: MX.sym
            The state of the system
        nlp: NonLinearProgram
            The definition of the system
        """

        for key in nlp.parameters_to_optimize:
            param = nlp.parameters_to_optimize[key]

            # Call the pre dynamics function
            if param.function:
                param.function(nlp.model, parameters, **param.params)
