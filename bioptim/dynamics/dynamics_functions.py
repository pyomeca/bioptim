from typing import Union

from casadi import horzcat, vertcat, MX, SX

from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.optimization_variable import OptimizationVariable


class DynamicsFunctions:
    """
    Implementation of all the dynamic functions

    Methods
    -------
    custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp: NonLinearProgram) -> MX
        Interface to custom dynamic function provided by the user
    torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool)
        Forward dynamics driven by joint torques, optional external forces can be declared.
    torque_activations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact) -> MX:
        Forward dynamics driven by joint torques activations.
    torque_derivative_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        Forward dynamics driven by joint torques, optional external forces can be declared.
    forces_from_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        Contact forces of a forward dynamics driven by joint torques with contact constraints.
    muscles_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        Forward dynamics driven by muscle.
    forces_from_muscle_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.
    get(var: OptimizationVariable, cx: Union[MX, SX]):
        Main accessor to a variable in states or controls (cx)
    apply_parameters(parameters: MX.sym, nlp: NonLinearProgram)
        Apply the parameter variables to the model. This should be called before calling the dynamics
    compute_qdot(nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX]):
        Easy accessor to derivative of q
    forward_dynamics(nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX], tau: Union[MX, SX], with_contact: bool):
        Easy accessor to derivative of qdot
    compute_muscle_dot(nlp: NonLinearProgram, muscle_excitations: Union[MX, SX]):
        Easy accessor to derivative of muscle activations
    compute_tau_from_muscle(nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX], muscle_activations: Union[MX, SX]):
        Easy accessor to tau computed from muscles
    contact_forces(nlp: NonLinearProgram, q, qdot, tau):
        Easy accessor for the contact forces in contact dynamics
    """

    @staticmethod
    def custom(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
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
    def torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
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
        with_contact: bool
            If the dynamic with contact should be used

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)

        dq = horzcat(*[dq for _ in range(ddq.shape[1])])
        return vertcat(dq, ddq)

    @staticmethod
    def torque_activations_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact) -> MX:
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
        with_contact: bool
            If the dynamic with contact should be used

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau_activations = DynamicsFunctions.get(nlp.controls["tau"], controls)

        tau = nlp.model.torque(tau_activations, q, qdot).to_mx()
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)

        dq = horzcat(*[dq for _ in range(ddq.shape[1])])

        return vertcat(dq, ddq)

    @staticmethod
    def torque_derivative_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
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
        with_contact: bool
            If the dynamic with contact should be used

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.states["tau"], states)
        taudot = DynamicsFunctions.get(nlp.controls["taudot"], controls)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)
        dtau = nlp.controls["taudot"].mapping.to_first.map(taudot)

        dq = horzcat(*[dq for _ in range(ddq.shape[1])])
        dtau = horzcat(*[dtau for _ in range(ddq.shape[1])])

        return vertcat(dq, ddq, dtau)

    @staticmethod
    def forces_from_torque_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
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

        q_nlp, q_var = (nlp.states["q"], states) if "q" in nlp.states else (nlp.controls["q"], controls)
        qdot_nlp, qdot_var = (nlp.states["qdot"], states) if "qdot" in nlp.states else (nlp.controls["qdot"], controls)
        tau_nlp, tau_var = (nlp.states["tau"], states) if "tau" in nlp.states else (nlp.controls["tau"], controls)
        q = DynamicsFunctions.get(q_nlp, q_var)
        qdot = DynamicsFunctions.get(qdot_nlp, qdot_var)
        tau = DynamicsFunctions.get(tau_nlp, tau_var)

        return DynamicsFunctions.contact_forces(nlp, q, qdot, tau)

    @staticmethod
    def muscles_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool) -> MX:
        """
        Forward dynamics driven by muscle.

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
        with_contact: bool
            If the dynamic with contact should be used

        Returns
        ----------
        MX.sym
            The derivative of the states
        """

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if "tau" in nlp.controls else None

        mus_act_nlp, mus_act = (nlp.states, states) if "muscles" in nlp.states else (nlp.controls, controls)
        mus_activations = DynamicsFunctions.get(mus_act_nlp["muscles"], mus_act)
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact)

        dq = horzcat(*[dq for _ in range(ddq.shape[1])])
        dxdt = vertcat(dq, ddq)

        has_excitation = True if "muscles" in nlp.states else False
        if has_excitation:
            mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
            dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations)
            dmus = horzcat(*[dmus for _ in range(ddq.shape[1])])
            dxdt = vertcat(dxdt, dmus)

        return dxdt

    @staticmethod
    def forces_from_muscle_driven(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
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
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        residual_tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if "tau" in nlp.controls else None

        mus_act_nlp, mus_act = (nlp.states, states) if "muscles" in nlp.states else (nlp.controls, controls)
        mus_activations = DynamicsFunctions.get(mus_act_nlp["muscles"], mus_act)
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        return DynamicsFunctions.contact_forces(nlp, q, qdot, tau)

    @staticmethod
    def get(var: OptimizationVariable, cx: Union[MX, SX]):
        """
        Main accessor to a variable in states or controls (cx)

        Parameters
        ----------
        var: OptimizationVariable
            The variable from nlp.states["name"] or nlp.controls["name"]
        cx: Union[MX, SX]
            The actual SX or MX variables

        Returns
        -------
        The sliced values
        """

        return var.mapping.to_second.map(cx[var.index, :])

    @staticmethod
    def apply_parameters(parameters: MX.sym, nlp):
        """
        Apply the parameter variables to the model. This should be called before calling the dynamics

        Parameters
        ----------
        parameters: MX.sym
            The state of the system
        nlp: NonLinearProgram
            The definition of the system
        """

        offset = 0
        for param in nlp.parameters:
            # Call the pre dynamics function
            if param.function:
                param.function(nlp.model, parameters[offset : offset + param.size], **param.params)
                offset += param.size

    @staticmethod
    def compute_qdot(nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX]):
        """
        Easy accessor to derivative of q

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"

        Returns
        -------
        The derivative of q
        """

        q_nlp = nlp.states["q"] if "q" in nlp.states else nlp.controls["q"]
        return q_nlp.mapping.to_first.map(nlp.model.computeQdot(q, qdot).to_mx())

    @staticmethod
    def forward_dynamics(
        nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX], tau: Union[MX, SX], with_contact: bool
    ):
        """
        Easy accessor to derivative of qdot

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"
        tau: Union[MX, SX]
            The value of tau from "get"
        with_contact: bool
            If the dynamics with contact should be used

        Returns
        -------
        The derivative of qdot
        """
        qdot_var = nlp.states["qdot"] if "qdot" in nlp.states else nlp.controls["qdot"]

        if nlp.external_forces:
            dxdt = MX(len(qdot_var.mapping.to_first), nlp.ns)
            for i, f_ext in enumerate(nlp.external_forces):
                if with_contact:
                    qddot = nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau, f_ext).to_mx()
                else:
                    qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()
                dxdt[:, i] = qdot_var.mapping.to_first.map(qddot)
            return dxdt
        else:
            if with_contact:
                qddot = nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()
            else:
                qddot = nlp.model.ForwardDynamics(q, qdot, tau).to_mx()
            return qdot_var.mapping.to_first.map(qddot)

    @staticmethod
    def compute_muscle_dot(nlp: NonLinearProgram, muscle_excitations: Union[MX, SX]):
        """
        Easy accessor to derivative of muscle activations

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        muscle_excitations: Union[MX, SX]
            The value of muscle_excitations from "get"

        Returns
        -------
        The derivative of muscle activations
        """

        muscles_states = nlp.model.stateSet()
        for k in range(len(nlp.controls["muscles"])):
            muscles_states[k].setExcitation(muscle_excitations[k])
        return nlp.model.activationDot(muscles_states).to_mx()

    @staticmethod
    def compute_tau_from_muscle(
        nlp: NonLinearProgram, q: Union[MX, SX], qdot: Union[MX, SX], muscle_activations: Union[MX, SX]
    ):
        """
        Easy accessor to tau computed from muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"
        muscle_activations: Union[MX, SX]
            The value of muscle_activations from "get"

        Returns
        -------
        The generalized forces computed from the muscles
        """

        muscles_states = nlp.model.stateSet()
        for k in range(len(nlp.controls["muscles"])):
            muscles_states[k].setActivation(muscle_activations[k])
        return nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()

    @staticmethod
    def contact_forces(nlp: NonLinearProgram, q, qdot, tau):
        """
        Easy accessor for the contact forces in contact dynamics

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: Union[MX, SX]
            The value of q from "get"
        qdot: Union[MX, SX]
            The value of qdot from "get"
        tau: Union[MX, SX]
            The value of tau from "get"

        Returns
        -------
        The contact forces
        """

        cs = nlp.model.getConstraints()
        if nlp.external_forces:
            all_cs = MX()
            for i, f_ext in enumerate(nlp.external_forces):
                nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau, cs, f_ext).to_mx()
                raise NotImplementedError("Forward dynamics with contact is not implemented yet")
                # all_cs[:, i] = vertcat(cs.getForce().to_mx())  # TODO
            return all_cs
        else:
            nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau, cs).to_mx()
            return cs.getForce().to_mx()
