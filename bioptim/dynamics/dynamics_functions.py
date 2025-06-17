from casadi import horzcat, vertcat, MX, SX, DM

from .dynamics_evaluation import DynamicsEvaluation
from .ode_solvers import OdeSolver
from .fatigue.fatigue_dynamics import FatigueList
from ..limits.holonomic_constraints import HolonomicConstraintsFcn
from ..misc.enums import DefectType, ContactType
from ..misc.mapping import BiMapping
from ..optimization.optimization_variable import OptimizationVariable
from ..misc.parameters_types import Bool, AnyListOptional, CX, CXOptional, Str, Tuple


class DynamicsFunctions:
    """
    Implementation of all the dynamic functions

    Methods
    -------
    get:
        Main accessor to a variable in states or controls (cx)
    reshape_qdot:
        Easy accessor to derivative of q
    forward_dynamics:
        Easy accessor to derivative of qdot
    compute_muscle_dot:
        Easy accessor to derivative of muscle activations
    compute_tau_from_muscle:
        Easy accessor to tau computed from muscles
    """

    @staticmethod
    def get_fatigable_tau(nlp, states: CX, controls: CX, fatigue: FatigueList) -> CX:
        """
        Apply the forward dynamics including (or not) the torque fatigue

        Parameters
        ----------
        nlp: NonLinearProgram
            The current phase
        states: MX | SX
            The states variable that may contains the tau and the tau fatigue variables
        controls: MX | SX
            The controls variable that may contains the tau
        fatigue: FatigueList
            The dynamics for the torque fatigue

        Returns
        -------
        The generalized accelerations
        """
        tau_var, tau_cx = (nlp.controls, controls) if "tau" in nlp.controls else (nlp.states, states)
        tau = nlp.get_var("tau", states, controls)
        if fatigue is not None and "tau" in fatigue:
            tau_fatigue = fatigue["tau"]
            tau_suffix = fatigue["tau"].suffix

            # Only homogeneous state_only is implemented yet
            n_state_only = sum([t.models.state_only for t in tau_fatigue])
            if 0 < n_state_only < len(fatigue["tau"]):
                raise NotImplementedError("fatigue list without homogeneous state_only flag is not supported yet")
            apply_to_joint_dynamics = sum([t.models.apply_to_joint_dynamics for t in tau_fatigue])
            if 0 < n_state_only < len(fatigue["tau"]):
                raise NotImplementedError(
                    "fatigue list without homogeneous apply_to_joint_dynamics flag is not supported yet"
                )
            if apply_to_joint_dynamics != 0:
                raise NotImplementedError("apply_to_joint_dynamics is not implemented for joint torque")

            if not tau_fatigue[0].models.split_controls and "tau" in nlp.controls:
                pass
            elif tau_fatigue[0].models.state_only:
                tau = sum([DynamicsFunctions.get(tau_var[f"tau_{suffix}"], tau_cx) for suffix in tau_suffix])
            else:
                tau = nlp.cx()
                for i, t in enumerate(tau_fatigue):
                    tau_tp = nlp.cx(1, 1)
                    for suffix in tau_suffix:
                        model = t.models.models[suffix]
                        tau_tp += (
                            DynamicsFunctions.get(nlp.states[f"tau_{suffix}_{model.dynamics_suffix()}"], states)[i]
                            * model.scaling
                        )
                    tau = vertcat(tau, tau_tp)
        return tau

    @staticmethod
    def get_fatigue_states(
        states,
        nlp,
        fatigue,
        mus_activations,
    ):

        fatigue_states = None
        if fatigue is not None and "muscles" in fatigue:
            mus_fatigue = fatigue["muscles"]
            fatigue_name = mus_fatigue.suffix[0]

            # Sanity check
            n_state_only = sum([m.models.state_only for m in mus_fatigue])
            if 0 < n_state_only < len(fatigue["muscles"]):
                raise NotImplementedError(
                    f"{fatigue_name} list without homogeneous state_only flag is not supported yet"
                )
            apply_to_joint_dynamics = sum([m.models.apply_to_joint_dynamics for m in mus_fatigue])
            if 0 < apply_to_joint_dynamics < len(fatigue["muscles"]):
                raise NotImplementedError(
                    f"{fatigue_name} list without homogeneous apply_to_joint_dynamics flag is not supported yet"
                )

            dyn_suffix = mus_fatigue[0].models.models[fatigue_name].dynamics_suffix()
            fatigue_suffix = mus_fatigue[0].models.models[fatigue_name].fatigue_suffix()
            for m in mus_fatigue:
                for key in m.models.models:
                    if (
                        m.models.models[key].dynamics_suffix() != dyn_suffix
                        or m.models.models[key].fatigue_suffix() != fatigue_suffix
                    ):
                        raise ValueError(f"{fatigue_name} must be of all same types")

            if n_state_only == 0:
                mus_activations = DynamicsFunctions.get(nlp.states[f"muscles_{dyn_suffix}"], states)

            if apply_to_joint_dynamics > 0:
                fatigue_states = DynamicsFunctions.get(nlp.states[f"muscles_{fatigue_suffix}"], states)

        return fatigue_states, mus_activations

    @staticmethod
    def get(var: OptimizationVariable, cx: CX):
        """
        Main accessor to a variable in states or controls (cx)

        Parameters
        ----------
        var: OptimizationVariable
            The variable from nlp.states["name"] or nlp.controls["name"]
        cx: MX | SX
            The actual SX or MX variables

        Returns
        -------
        The sliced values
        """

        return var.mapping.to_second.map(cx[var.index, :])

    @staticmethod
    def compute_qdot(nlp, q: CX, qdot: CX):
        """
        Easy accessor to derivative of q

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"

        Returns
        -------
        The derivative of q
        """

        if "q" in nlp.states:
            mapping = nlp.states["q"].mapping
        elif "q_roots" and "q_joints" in nlp.states:
            mapping = BiMapping(
                to_first=list(nlp.states["q_roots"].mapping.to_first.map_idx)
                + [i + nlp.model.nb_root for i in nlp.states["q_joints"].mapping.to_first.map_idx],
                to_second=list(nlp.states["q_roots"].mapping.to_second.map_idx)
                + [i + nlp.model.nb_root for i in nlp.states["q_joints"].mapping.to_second.map_idx],
            )
        elif "q" in nlp.controls:
            mapping = nlp.controls["q"].mapping
        else:
            raise RuntimeError("Your q key combination was not found in states or controls")
        return mapping.to_first.map(nlp.model.reshape_qdot()(q, qdot, nlp.parameters.cx))

    @staticmethod
    def compute_qddot(nlp, q: CX, qdot: CX, tau: CX, external_forces: CX):
        """
        Easy accessor to derivative of qdot

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        tau: MX | SX
            The value of tau from "get"
        external_forces: MX | SX
            The value of external forces to apply to the model

        Returns
        -------
        The derivative of qdot
        """
        forward_dynamics_contact_types = ContactType.get_equivalent_explicit_contacts(nlp.model.contact_types)
        ddq_fd = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, forward_dynamics_contact_types, external_forces)
        return ddq_fd

    @staticmethod
    def collect_tau(
        nlp,
        q: CX,
        qdot: CX,
        parameters: CX,
        states: OptimizationVariable,
        controls: OptimizationVariable,
        fatigue: FatigueList | None = None,
    ):
        """
        Collect the additional joint torques to add to the torques from controls.

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        parameters: MX | SX
            The parameters of the system
        fatigue: FatigueList | None
            The fatigue elements to consider in the dynamics. If None, no fatigue will be considered.
        """
        tau = nlp.cx.zeros(nlp.model.nb_tau, 1)
        if nlp.model.nb_passive_joint_torques > 0:
            tau += nlp.model.passive_joint_torque()(q, qdot, parameters)
        if nlp.model.nb_ligaments > 0:
            tau += nlp.model.ligament_joint_torque()(q, qdot, parameters)
        if nlp.model.friction_coefficients is not None:
            tau -= nlp.model.friction_coefficients @ qdot
        return tau

    @staticmethod
    def get_contact_defects(nlp, q: CX, qdot: CX, slope_qdot: CX):
        """
        Get the defects associated with implicit contacts.

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        slope_qdot: MX | SX
            The slope of qdot from "get" states_dot
        """

        contact_defects = nlp.cx()
        # We append the defects with the algebraic states implicit constraints
        if ContactType.RIGID_IMPLICIT in nlp.model.contact_types:
            _, _, acceleration_constraint_func = HolonomicConstraintsFcn.rigid_contacts(nlp.model)
            contact_acceleration_defect = acceleration_constraint_func(q, qdot, slope_qdot, nlp.parameters.cx)
            contact_defects = vertcat(contact_defects, contact_acceleration_defect)

        if ContactType.SOFT_IMPLICIT in nlp.model.contact_types:
            soft_contact_defect = (
                nlp.model.soft_contact_forces().expand()(q, qdot, nlp.parameters.cx)
                - nlp.algebraic_states["soft_contact_forces"].cx
            )
            contact_defects = vertcat(contact_defects, soft_contact_defect)
        return contact_defects

    @staticmethod
    def get_fatigue_defects(
        key: Str, dxdt_defects: CX, slopes: CX, nlp, states: CX, controls: CX, fatigue: FatigueList
    ) -> Tuple[CX, CX]:
        """
        Get the dxdt and slopes associated with fatigue elements.
        These are added to compute the defects in the case where there is fatigue.

        Parameters
        ----------
        key: str
            The name of the fatigue element to consider
        dxdt_defects: MX | SX
            The states derivative before fatigue is applied
        slopes : MX | SX
            The slopes of the states before fatigue is applied
        nlp: NonLinearProgram
            The phase of the program
        states: MX | SX
            The states of the system
        controls: MX | SX
            The controls of the system
        fatigue: FatigueList
            The fatigue elements to consider in the dynamics. If None, no fatigue will be considered.
        """
        if fatigue is not None and key in fatigue:
            dxdt_defects = fatigue[key].dynamics(dxdt_defects, nlp, states, controls)
            state_keys = nlp.states.keys()
            if state_keys[0] != "q" or state_keys[1] != "qdot":
                raise NotImplementedError("The accession of fatigue states is not implemented generically yet.")

            slopes_fatigue = nlp.cx()
            fatigue_indices = []
            for key in state_keys[2:]:
                if not key.startswith("tau_"):
                    raise NotImplementedError("The accession of states is not implemented generically yet.")
                slopes_fatigue = vertcat(slopes_fatigue, nlp.states_dot[key].cx)
                fatigue_indices += list(nlp.states[key].index)

            slopes[fatigue_indices, 0] = slopes_fatigue

        return dxdt_defects, slopes

    @staticmethod
    def get_external_forces_from_contacts(nlp, q, qdot, contact_types, external_forces: MX | SX):
        """
        Get the external forces associated with the contacts defined in the model.

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        contact_types: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        external_forces: MX | SX
            The external forces to consider in the dynamics. If None, it will be initialized as an empty vector.
        """

        external_forces = nlp.cx() if external_forces is None else external_forces
        if ContactType.RIGID_IMPLICIT in contact_types:
            if external_forces.shape[0] != 0:
                raise NotImplementedError("ContactType.RIGID_IMPLICIT cannot be used with external forces yet")
            if "rigid_contact_forces" in nlp.states:
                contact_forces = nlp.states["rigid_contact_forces"].cx
            elif "rigid_contact_forces" in nlp.algebraic_states:
                contact_forces = nlp.algebraic_states["rigid_contact_forces"].cx
            else:
                raise RuntimeError("The key 'rigid_contact_forces' was not found in states or algebraic_states")
            external_forces = vertcat(
                external_forces,
                nlp.model.map_rigid_contact_forces_to_global_forces(contact_forces, q, nlp.parameters.cx),
            )

        if ContactType.SOFT_EXPLICIT in contact_types:
            contact_forces = nlp.model.soft_contact_forces().expand()(q, qdot, nlp.parameters.cx)
            external_forces = vertcat(
                external_forces, nlp.model.map_soft_contact_forces_to_global_forces(contact_forces)
            )

        if ContactType.SOFT_IMPLICIT in contact_types:
            contact_forces = nlp.algebraic_states["soft_contact_forces"].cx
            external_forces = vertcat(
                external_forces, nlp.model.map_soft_contact_forces_to_global_forces(contact_forces)
            )

        external_forces = [] if external_forces.shape == (0, 1) else external_forces

        return external_forces

    @staticmethod
    def forward_dynamics(
        nlp,
        q: CX,
        qdot: CX,
        tau: CX,
        contact_types: list[ContactType] | tuple[ContactType],
        external_forces: AnyListOptional = None,
    ):
        """
        Easy accessor to derivative of qdot

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        tau: MX | SX
            The value of tau from "get"
        contact_types: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        external_forces: MX | SX
            The external forces
        Returns
        -------
        The derivative of qdot
        """
        # Get the mapping of the output
        if "qdot" in nlp.states:
            qdot_var_mapping = nlp.states["qdot"].mapping.to_first
        elif "qdot_roots" and "qdot_joints" in nlp.states:
            qdot_var_mapping = BiMapping(
                to_first=list(nlp.states["qdot_roots"].mapping.to_first.map_idx)
                + [i + nlp.model.nb_root for i in nlp.states["qdot_joints"].mapping.to_first.map_idx],
                to_second=list(nlp.states["qdot_roots"].mapping.to_second.map_idx)
                + [i + nlp.model.nb_root for i in nlp.states["qdot_joints"].mapping.to_second.map_idx],
            ).to_first
        elif "qdot" in nlp.controls:
            qdot_var_mapping = nlp.controls["qdot"].mapping.to_first
        else:
            qdot_var_mapping = BiMapping([i for i in range(qdot.shape[0])], [i for i in range(qdot.shape[0])]).to_first

        external_forces = DynamicsFunctions.get_external_forces_from_contacts(
            nlp, q, qdot, contact_types, external_forces
        )
        with_contact = ContactType.RIGID_EXPLICIT in contact_types

        qddot = nlp.model.forward_dynamics(with_contact=with_contact)(
            q,
            qdot,
            tau,
            external_forces,
            nlp.parameters.cx,
        )
        return qdot_var_mapping.map(qddot)

    @staticmethod
    def inverse_dynamics(
        nlp,
        q: CX,
        qdot: CX,
        qddot: CX,
        contact_types: list[ContactType] | tuple[ContactType],
        external_forces: MX = None,
    ):
        """
        Easy accessor to torques from inverse dynamics

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        qddot: MX | SX
            The value of qddot from "get"
        contact_types: list[ContactType] | tuple[ContactType]
            The type of contacts to consider in the dynamics
        external_forces: MX
            The external forces

        Returns
        -------
        Torques in tau
        """
        # TODO: Charbie -> Check if the mapping can be applied or not.
        # Old comment: we ignore on purpose the mapping to keep zeros in the defects of the dynamic.

        # Get the mapping of the output
        if "tau" in nlp.states:
            tau_var_mapping = nlp.states["tau"].mapping.to_first
        elif "tau" in nlp.controls:
            tau_var_mapping = nlp.controls["tau"].mapping.to_first
        elif "tau_joints" in nlp.controls:
            if nlp.variable_mappings["tau_joints"].actually_does_a_mapping():
                raise NotImplementedError(
                    "Free floating base dynamics was used with a mapping. This is not implemented yet."
                )
            to_first = [None] * nlp.model.nb_root + list(range(nlp.model.nb_q - nlp.model.nb_root))
            to_second = list(range(nlp.model.nb_root, nlp.model.nb_q))
            tau_var_mapping = BiMapping(to_first, to_second).to_first
        else:
            raise RuntimeError("The key 'tau' was not found in states or controls")

        if ContactType.RIGID_EXPLICIT in contact_types:
            raise NotImplementedError("Inverse dynamics, cannot be used with ContactType.RIGID_EXPLICIT yet")

        external_forces = DynamicsFunctions.get_external_forces_from_contacts(
            nlp, q, qdot, contact_types, external_forces
        )

        tau = nlp.model.inverse_dynamics(with_contact=False)(q, qdot, qddot, external_forces, nlp.parameters.cx)

        return tau_var_mapping.map(tau)

    @staticmethod
    def compute_muscle_dot(nlp, muscle_excitations: CX, muscle_activations: CX):
        """
        Easy accessor to derivative of muscle activations

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        muscle_excitations: MX | SX
            The value of muscle_excitations from "get"
        muscle_activations: MX | SX
            The value of muscle_activations from "get"

        Returns
        -------
        The derivative of muscle activations
        """

        return nlp.model.muscle_activation_dot()(muscle_excitations, muscle_activations, nlp.parameters.cx)

    @staticmethod
    def compute_tau_from_muscle(
        nlp,
        q: CX,
        qdot: CX,
        muscle_activations: CX,
        fatigue_states: CXOptional = None,
    ):
        """
        Easy accessor to tau computed from muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            The phase of the program
        q: MX | SX
            The value of q from "get"
        qdot: MX | SX
            The value of qdot from "get"
        muscle_activations: MX | SX
            The value of muscle_activations from "get"
        fatigue_states: MX | SX
            The states of fatigue

        Returns
        -------
        The generalized forces computed from the muscles
        """

        activations = type(q)()
        for k in range(len(nlp.controls["muscles"])):
            if fatigue_states is not None:
                activations = vertcat(activations, muscle_activations[k] * (1 - fatigue_states[k]))
            else:
                activations = vertcat(activations, muscle_activations[k])
        return nlp.model.muscle_joint_torque()(activations, q, qdot, nlp.parameters.cx)

    @staticmethod
    def no_states_mapping(nlp):
        for key in nlp.states.keys():
            if nlp.variable_mappings[key].actually_does_a_mapping():
                raise NotImplementedError(
                    f"COLLOCATION transcription is not compatible with mapping for states. "
                    "Please note that concept of states mapping in already sketchy on it's own, but is particularly not appropriate for COLLOCATION transcriptions."
                )
