from typing import Callable, Any, Union

from casadi import MX, vertcat, Function
import numpy as np
from biorbd_casadi import biorbd

from .dynamics_functions import DynamicsFunctions
from .fatigue.fatigue_dynamics import FatigueList, MultiFatigueInterface
from .ode_solver import OdeSolver
from ..gui.plot import CustomPlot
from ..limits.path_conditions import Bounds
from ..misc.enums import (
    PlotType,
    ControlType,
    VariableType,
    Node,
    ConstraintType,
    RigidBodyDynamics,
    SoftContactDynamics,
)
from ..misc.fcn_enum import FcnEnum
from ..misc.mapping import BiMapping, Mapping
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric
from ..limits.constraints import ImplicitConstraintFcn


class ConfigureProblem:
    """
    Dynamics configuration for the most common ocp

    Methods
    -------
    initialize(ocp, nlp)
        Call the dynamics a first time
    custom(ocp, nlp, **extra_params)
        Call the user-defined dynamics configuration function
    torque_driven(ocp, nlp, with_contact=False)
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)
    torque_derivative_driven(ocp, nlp, with_contact=False)
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)
    torque_activations_driven(ocp, nlp, with_contact=False)
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau activations).
        The tau activations are bounded between -1 and 1 and actual tau is computed from torque-position-velocity
        relationship
    muscle_driven(
        ocp, nlp, with_excitations: bool = False, with_torque: bool = False, with_contact: bool = False
    )
        Configure the dynamics for a muscle driven program.
        If with_excitations is set to True, then the muscle muscle activations are computed from the muscle dynamics.
        The tau from muscle is computed using the muscle activations.
        If with_torque is set to True, then tau are used as supplementary force in the
        case muscles are too weak.
    configure_dynamics_function(ocp, nlp, dyn_func, **extra_params)
        Configure the dynamics of the system
    configure_contact_function(ocp, nlp, dyn_func: Callable, **extra_params)
        Configure the contact points
    configure_soft_contact_function
        Configure the soft contact function
    configure_new_variable(
        name: str, name_elements: list, nlp, as_states: bool, as_controls: bool, combine_state_control_plot: bool = False
    )
        Add a new variable to the states/controls pool
    configure_q(nlp, as_states: bool, as_controls: bool)
        Configure the generalized coordinates
    configure_qdot(nlp, as_states: bool, as_controls: bool)
        Configure the generalized velocities
    configure_qddot(nlp, as_states: bool, as_controls: bool)
        Configure the generalized accelerations
    configure_qdddot(nlp, as_states: bool, as_controls: bool)
        Configure the generalized jerks
    configure_tau(nlp, as_states: bool, as_controls: bool)
        Configure the generalized forces
    configure_taudot(nlp, as_states: bool, as_controls: bool)
        Configure the generalized forces derivative
    configure_muscles(nlp, as_states: bool, as_controls: bool)
        Configure the muscles
    _adjust_mapping(key_to_adjust: str, reference_keys: list, nlp)
        Automatic mapping duplicator basing the values on the another mapping
    """

    @staticmethod
    def _get_kinematics_based_names(nlp, type: str) -> list[str]:
        """
        To modify the names of the variables added to the plots if there is quaternions

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        type: str
            A string that refers to the decision variable such as (q, qdot, qddot, tau, etc...)

        Returns
        ----------
        new_name: list[str]
            The list of str to display on figures
        """

        if nlp.phase_mapping:
            idx = nlp.phase_mapping.map_idx
        else:
            idx = range(nlp.model.nbQ())
        if nlp.model.nbQuat() == 0:
            new_name = [nlp.model.nameDof()[i].to_string() for i in idx]
        else:
            new_name = []
            for i in nlp.phase_mapping.map_idx:
                if (
                    nlp.model.nameDof()[i].to_string()[-4:-1] == "Rot"
                    or nlp.model.nameDof()[i].to_string()[-6:-1] == "Trans"
                ):
                    new_name += [nlp.model.nameDof()[i].to_string()]
                else:
                    if nlp.model.nameDof()[i].to_string()[-5:] != "QuatW":
                        if type == "qdot":
                            new_name += [
                                nlp.model.nameDof()[i].to_string()[:-5]
                                + "omega"
                                + nlp.model.nameDof()[i].to_string()[-1]
                            ]
                        elif type == "qddot":
                            new_name += [
                                nlp.model.nameDof()[i].to_string()[:-5]
                                + "omegadot"
                                + nlp.model.nameDof()[i].to_string()[-1]
                            ]
                        elif type == "qdddot":
                            new_name += [
                                nlp.model.nameDof()[i].to_string()[:-5]
                                + "omegaddot"
                                + nlp.model.nameDof()[i].to_string()[-1]
                            ]
                        elif type == "tau" or type == "taudot":
                            new_name += [nlp.model.nameDof()[i].to_string()]

        return new_name

    @staticmethod
    def initialize(ocp, nlp):
        """
        Call the dynamics a first time

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.type(ocp, nlp, **nlp.dynamics_type.params)

    @staticmethod
    def custom(ocp, nlp, **extra_params):
        """
        Call the user-defined dynamics configuration function

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        nlp.dynamics_type.configure(ocp, nlp, **extra_params)

    @staticmethod
    def torque_driven(
        ocp,
        nlp,
        with_contact: bool = False,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        soft_contacts_dynamics: SoftContactDynamics = SoftContactDynamics.ODE,
        fatigue: FatigueList = None,
    ):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        soft_contacts_dynamics: SoftContactDynamics
            which soft contact dynamic should be used
        fatigue: FatigueList
            A list of fatigue elements
        """

        if nlp.model.nbSoftContacts() != 0:
            if (
                soft_contacts_dynamics != SoftContactDynamics.CONSTRAINT
                and soft_contacts_dynamics != SoftContactDynamics.ODE
            ):
                raise ValueError(
                    "soft_contacts_dynamics can be used only with SoftContactDynamics.ODE or SoftContactDynamics.CONSTRAINT"
                )

            if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
                if soft_contacts_dynamics == SoftContactDynamics.ODE:
                    raise ValueError(
                        "Soft contacts dynamics should not be used with SoftContactDynamics.ODE "
                        "when rigidbody dynamics is not RigidBodyDynamics.ODE . "
                        "Please set soft_contacts_dynamics=SoftContactDynamics.CONSTRAINT"
                    )

        # Declared rigidbody states and controls
        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
        ConfigureProblem.configure_tau(ocp, nlp, False, True, fatigue)

        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
        ):
            ConfigureProblem.configure_qddot(ocp, nlp, False, True, True)
        elif (
            rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
            or rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
        ):
            ConfigureProblem.configure_qddot(ocp, nlp, True, False, True)
            ConfigureProblem.configure_qdddot(ocp, nlp, False, True)
        else:
            ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)

        # Algebraic constraints of rigidbody dynamics if needed
        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
        ):
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.TAU_EQUALS_INVERSE_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
                with_contact=with_contact,
            )
            if with_contact:
                # qddot is continuous with RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
                # so the consistency constraint of the marker acceleration can only be set to zero
                # at the first shooting node
                node = Node.ALL_SHOOTING if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS else Node.ALL
                ConfigureProblem.configure_contact_forces(ocp, nlp, False, True)
                for ii in range(nlp.model.nbContacts()):
                    ocp.implicit_constraints.add(
                        ImplicitConstraintFcn.CONTACT_ACCELERATION_EQUALS_ZERO,
                        with_contact=with_contact,
                        contact_index=ii,
                        node=node,
                        constraint_type=ConstraintType.IMPLICIT,
                        phase=nlp.phase_idx,
                    )
        if (
            rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS
            or rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS_JERK
        ):
            # contacts forces are directly handled with this constraint
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.QDDOT_EQUALS_FORWARD_DYNAMICS,
                node=Node.ALL_SHOOTING,
                constraint_type=ConstraintType.IMPLICIT,
                with_contact=with_contact,
                phase=nlp.phase_idx,
            )

        # Declared soft contacts controls
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ConfigureProblem.configure_soft_contact_forces(ocp, nlp, False, True)

        # Configure the actual ODE of the dynamics
        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_driven,
                with_contact=with_contact,
                fatigue=fatigue,
                rigidbody_dynamics=rigidbody_dynamics,
            )

        # Configure the contact forces
        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)
        # Configure the soft contact forces
        ConfigureProblem.configure_soft_contact_function(ocp, nlp)
        # Algebraic constraints of soft contact forces if needed
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.SOFT_CONTACTS_EQUALS_SOFT_CONTACTS_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
            )

    @staticmethod
    def torque_derivative_driven(
        ocp,
        nlp,
        with_contact=False,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
        soft_contacts_dynamics: SoftContactDynamics = SoftContactDynamics.ODE,
    ):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used
        soft_contacts_dynamics: SoftContactDynamics
            which soft contact dynamic should be used
        """
        if rigidbody_dynamics not in (RigidBodyDynamics.DAE_INVERSE_DYNAMICS, RigidBodyDynamics.ODE):
            raise NotImplementedError("TORQUE_DERIVATIVE_DRIVEN cannot be used with this enum RigidBodyDynamics yet")

        if nlp.model.nbSoftContacts() != 0:
            if (
                soft_contacts_dynamics != SoftContactDynamics.CONSTRAINT
                and soft_contacts_dynamics != SoftContactDynamics.ODE
            ):
                raise ValueError(
                    "soft_contacts_dynamics can be used only with RigidBodyDynamics.ODE or SoftContactDynamics.CONSTRAINT"
                )

            if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
                if soft_contacts_dynamics == SoftContactDynamics.ODE:
                    raise ValueError(
                        "Soft contacts dynamics should not be used with RigidBodyDynamics.ODE "
                        "when rigidbody dynamics is not RigidBodyDynamics.ODE . "
                        "Please set soft_contacts_dynamics=SoftContactDynamics.CONSTRAINT"
                    )

        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False)
        ConfigureProblem.configure_tau(ocp, nlp, True, False)
        ConfigureProblem.configure_taudot(ocp, nlp, False, True)

        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            ConfigureProblem.configure_qddot(ocp, nlp, True, False)
            ConfigureProblem.configure_qdddot(ocp, nlp, False, True)
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.TAU_EQUALS_INVERSE_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
            )
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ConfigureProblem.configure_soft_contact_forces(ocp, nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.torque_derivative_driven,
                with_contact=with_contact,
                rigidbody_dynamics=rigidbody_dynamics,
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

        ConfigureProblem.configure_soft_contact_function(ocp, nlp)
        if soft_contacts_dynamics == SoftContactDynamics.CONSTRAINT:
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.SOFT_CONTACTS_EQUALS_SOFT_CONTACTS_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
            )

    @staticmethod
    def torque_activations_driven(ocp, nlp, with_contact=False):
        """
        Configure the dynamics for a torque driven program (states are q and qdot, controls are tau activations).
        The tau activations are bounded between -1 and 1 and actual tau is computed from torque-position-velocity
        relationship

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_contact: bool
            If the dynamic with contact should be used
        """

        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False)
        ConfigureProblem.configure_tau(ocp, nlp, False, True)

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp, nlp, DynamicsFunctions.torque_activations_driven, with_contact=with_contact
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(
                ocp, nlp, DynamicsFunctions.forces_from_torque_activation_driven
            )
        ConfigureProblem.configure_soft_contact_function(ocp, nlp)

    @staticmethod
    def joints_acceleration_driven(ocp, nlp, rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE):
        """
        Configure the dynamics for a joints acceleration driven program
        (states are q and qdot, controls are qddot_joints)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used

        """
        if rigidbody_dynamics != RigidBodyDynamics.ODE:
            raise NotImplementedError("Implicit dynamics not implemented yet.")

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        # Configure qddot joints
        nb_root = nlp.model.nbRoot()
        if not nb_root > 0:
            raise RuntimeError("Model must have at least one DoF on root.")

        name_qddot_joints = [str(i + nb_root) for i in range(nlp.model.nbQddot() - nb_root)]
        ConfigureProblem.configure_new_variable(
            "qddot_joints", name_qddot_joints, ocp, nlp, as_states=False, as_controls=True
        )
        ConfigureProblem.configure_dynamics_function(
            ocp, nlp, DynamicsFunctions.joints_acceleration_driven, expand=False
        )

    @staticmethod
    def muscle_driven(
        ocp,
        nlp,
        with_excitations: bool = False,
        fatigue: FatigueList = None,
        with_torque: bool = False,
        with_contact: bool = False,
        rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
    ):
        """
        Configure the dynamics for a muscle driven program.
        If with_excitations is set to True, then the muscle muscle activations are computed from the muscle dynamics.
        The tau from muscle is computed using the muscle activations.
        If with_torque is set to True, then tau are used as supplementary force in the
        case muscles are too weak.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_excitations: bool
            If the dynamic should include the muscle dynamics
        fatigue: FatigueList
            The list of fatigue parameters
        with_torque: bool
            If the dynamic should be added with residual torques
        with_contact: bool
            If the dynamic with contact should be used
        rigidbody_dynamics: RigidBodyDynamics
            which rigidbody dynamics should be used

        """

        if fatigue is not None and "tau" in fatigue and not with_torque:
            raise RuntimeError("Residual torques need to be used to apply fatigue on torques")

        if rigidbody_dynamics not in (RigidBodyDynamics.DAE_INVERSE_DYNAMICS, RigidBodyDynamics.ODE):
            raise NotImplementedError("MUSCLE_DRIVEN cannot be used with this enum RigidBodyDynamics yet")

        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False)
        if with_torque:
            ConfigureProblem.configure_tau(ocp, nlp, False, True, fatigue=fatigue)
        ConfigureProblem.configure_muscles(ocp, nlp, with_excitations, True, fatigue=fatigue)

        if rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
            ConfigureProblem.configure_qddot(ocp, nlp, False, True)
            ocp.implicit_constraints.add(
                ImplicitConstraintFcn.TAU_FROM_MUSCLE_EQUAL_INVERSE_DYNAMICS,
                node=Node.ALL_SHOOTING,
                penalty_type=ConstraintType.IMPLICIT,
                phase=nlp.phase_idx,
            )

        if nlp.dynamics_type.dynamic_function:
            ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        else:
            ConfigureProblem.configure_dynamics_function(
                ocp,
                nlp,
                DynamicsFunctions.muscles_driven,
                with_contact=with_contact,
                fatigue=fatigue,
                with_torque=with_torque,
                rigidbody_dynamics=rigidbody_dynamics,
            )

        if with_contact:
            ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_muscle_driven)
        ConfigureProblem.configure_soft_contact_function(ocp, nlp)

    @staticmethod
    def configure_dynamics_function(ocp, nlp, dyn_func, expand: bool = True, **extra_params):
        """
        Configure the dynamics of the system

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[states, controls, param]
            The function to get the derivative of the states
        expand: bool
            If the dynamics should be expanded with casadi
        """

        nlp.parameters = ocp.v.parameters_in_list

        dynamics_eval = dyn_func(nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx, nlp, **extra_params)
        dynamics_dxdt = dynamics_eval.dxdt
        if isinstance(dynamics_dxdt, (list, tuple)):
            dynamics_dxdt = vertcat(*dynamics_dxdt)

        nlp.dynamics_func = Function(
            "ForwardDyn",
            [nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx],
            [dynamics_dxdt],
            ["x", "u", "p"],
            ["xdot"],
        )
        if expand:
            nlp.dynamics_func = nlp.dynamics_func.expand()

        if dynamics_eval.defects is not None:
            nlp.implicit_dynamics_func = Function(
                "DynamicsDefects",
                [nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx, nlp.states_dot.mx_reduced],
                [dynamics_eval.defects],
                ["x", "u", "p", "xdot"],
                ["defects"],
            ).expand()

    @staticmethod
    def configure_contact_function(ocp, nlp, dyn_func: Callable, **extra_params):
        """
        Configure the contact points

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        dyn_func: Callable[states, controls, param]
            The function to get the values of contact forces from the dynamics
        """

        nlp.contact_forces_func = Function(
            "contact_forces_func",
            [nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx],
            [dyn_func(nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx, nlp, **extra_params)],
            ["x", "u", "p"],
            ["contact_forces"],
        ).expand()

        all_contact_names = []
        for elt in ocp.nlp:
            all_contact_names.extend(
                [name.to_string() for name in elt.model.contactNames() if name.to_string() not in all_contact_names]
            )

        if "contact_forces" in nlp.plot_mapping:
            phase_mappings = nlp.plot_mapping["contact_forces"]
        else:
            contact_names_in_phase = [name.to_string() for name in nlp.model.contactNames()]
            phase_mappings = Mapping([i for i, c in enumerate(all_contact_names) if c in contact_names_in_phase])

        nlp.plot["contact_forces"] = CustomPlot(
            lambda t, x, u, p: nlp.contact_forces_func(x, u, p),
            plot_type=PlotType.INTEGRATED,
            axes_idx=phase_mappings,
            legend=all_contact_names,
        )

    @staticmethod
    def configure_soft_contact_function(ocp, nlp):
        """
        Configure the soft contact sphere

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        global_soft_contact_force_func = MX.zeros(nlp.model.nbSoftContacts() * 6, 1)
        n = nlp.model.nbQ()
        component_list = ["Mx", "My", "Mz", "Fx", "Fy", "Fz"]

        for i_sc in range(nlp.model.nbSoftContacts()):
            soft_contact = nlp.model.softContact(i_sc)

            global_soft_contact_force_func[i_sc * 6 : (i_sc + 1) * 6, :] = (
                biorbd.SoftContactSphere(soft_contact)
                .computeForceAtOrigin(nlp.model, nlp.states.mx_reduced[:n], nlp.states.mx_reduced[n:])
                .to_mx()
            )
        nlp.soft_contact_forces_func = Function(
            "soft_contact_forces_func",
            [nlp.states.mx_reduced, nlp.controls.mx_reduced, nlp.parameters.mx],
            [global_soft_contact_force_func],
            ["x", "u", "p"],
            ["soft_contact_forces"],
        ).expand()

        for i_sc in range(nlp.model.nbSoftContacts()):
            all_soft_contact_names = []
            all_soft_contact_names.extend(
                [
                    f"{nlp.model.softContactName(i_sc).to_string()}_{name}"
                    for name in component_list
                    if nlp.model.softContactName(i_sc).to_string() not in all_soft_contact_names
                ]
            )

            if "soft_contact_forces" in nlp.plot_mapping:
                phase_mappings = nlp.plot_mapping["soft_contact_forces"]
            else:
                soft_contact_names_in_phase = [
                    f"{nlp.model.softContactName(i_sc).to_string()}_{name}"
                    for name in component_list
                    if nlp.model.softContactName(i_sc).to_string() not in all_soft_contact_names
                ]
                phase_mappings = Mapping(
                    [i for i, c in enumerate(all_soft_contact_names) if c in soft_contact_names_in_phase]
                )
            nlp.plot[f"soft_contact_forces_{nlp.model.softContactName(i_sc).to_string()}"] = CustomPlot(
                lambda t, x, u, p: nlp.soft_contact_forces_func(x, u, p)[(i_sc * 6) : ((i_sc + 1) * 6), :],
                plot_type=PlotType.INTEGRATED,
                axes_idx=phase_mappings,
                legend=all_soft_contact_names,
            )

    @staticmethod
    def _manage_fatigue_to_new_variable(
        name: str,
        name_elements: list,
        ocp,
        nlp,
        as_states: bool,
        as_controls: bool,
        fatigue: FatigueList = None,
    ):
        if fatigue is None or name not in fatigue:
            return False

        if not as_controls:
            raise NotImplementedError("Fatigue not applied on controls is not implemented yet")

        fatigue_var = fatigue[name]
        meta_suffixes = fatigue_var.suffix

        # Only homogeneous fatigue model are implement
        fatigue_suffix = fatigue_var[0].models.models[meta_suffixes[0]].suffix(VariableType.STATES)
        multi_interface = isinstance(fatigue_var[0].models, MultiFatigueInterface)
        split_controls = fatigue_var[0].models.split_controls
        for dof in fatigue_var:
            for key in dof.models.models:
                if dof.models.models[key].suffix(VariableType.STATES) != fatigue_suffix:
                    raise ValueError(f"Fatigue for {name} must be of all same types")
                if isinstance(dof.models, MultiFatigueInterface) != multi_interface:
                    raise ValueError("multi_interface must be the same for all the elements")
                if dof.models.split_controls != split_controls:
                    raise ValueError("split_controls must be the same for all the elements")

        # Prepare the plot that will combine everything
        n_elements = len(name_elements)

        legend = [f"{name}_{i}" for i in name_elements]
        fatigue_plot_name = f"fatigue_{name}"
        nlp.plot[fatigue_plot_name] = CustomPlot(
            lambda t, x, u, p: x[:n_elements, :] * np.nan,
            plot_type=PlotType.INTEGRATED,
            legend=legend,
            bounds=Bounds(-1, 1),
        )
        control_plot_name = f"{name}_controls" if not multi_interface and split_controls else f"{name}"
        nlp.plot[control_plot_name] = CustomPlot(
            lambda t, x, u, p: u[:n_elements, :] * np.nan, plot_type=PlotType.STEP, legend=legend
        )

        var_names_with_suffix = []
        color = fatigue_var[0].models.color()
        fatigue_color = [fatigue_var[0].models.models[m].color() for m in fatigue_var[0].models.models]
        plot_factor = fatigue_var[0].models.plot_factor()
        for i, meta_suffix in enumerate(meta_suffixes):
            var_names_with_suffix.append(f"{name}_{meta_suffix}" if not multi_interface else f"{name}")

            try:
                ConfigureProblem._adjust_mapping(var_names_with_suffix[-1], [name], nlp)
            except RuntimeError as message:
                if message.args[0] != "Could not adjust mapping with the reference_keys provided":
                    raise RuntimeError(message)

            if split_controls:
                ConfigureProblem.configure_new_variable(
                    var_names_with_suffix[-1], name_elements, ocp, nlp, as_states, as_controls, skip_plot=True
                )
                nlp.plot[f"{var_names_with_suffix[-1]}_controls"] = CustomPlot(
                    lambda t, x, u, p, key: u[nlp.controls[key].index, :],
                    plot_type=PlotType.STEP,
                    combine_to=control_plot_name,
                    key=var_names_with_suffix[-1],
                    color=color[i],
                )
            elif i == 0:
                ConfigureProblem.configure_new_variable(
                    f"{name}", name_elements, ocp, nlp, as_states, as_controls, skip_plot=True
                )
                nlp.plot[f"{name}_controls"] = CustomPlot(
                    lambda t, x, u, p, key: u[nlp.controls[key].index, :],
                    plot_type=PlotType.STEP,
                    combine_to=control_plot_name,
                    key=f"{name}",
                    color=color[i],
                )

            for p, params in enumerate(fatigue_suffix):
                name_tp = f"{var_names_with_suffix[-1]}_{params}"
                ConfigureProblem._adjust_mapping(name_tp, [var_names_with_suffix[-1]], ocp, nlp)
                ConfigureProblem.configure_new_variable(name_tp, name_elements, ocp, nlp, True, False, skip_plot=True)
                nlp.plot[name_tp] = CustomPlot(
                    lambda t, x, u, p, key, mod: mod * x[nlp.states[key].index, :],
                    plot_type=PlotType.INTEGRATED,
                    combine_to=fatigue_plot_name,
                    key=name_tp,
                    color=fatigue_color[i][p],
                    mod=plot_factor[i],
                )

        # Create a fake accessor for the name of the controls so it can be directly called in nlp.controls
        if split_controls:
            ConfigureProblem.append_faked_optim_var(name, nlp.controls, var_names_with_suffix)
        else:
            for meta_suffix in var_names_with_suffix:
                ConfigureProblem.append_faked_optim_var(meta_suffix, nlp.controls, [name])

        return True

    @staticmethod
    def configure_new_variable(
        name: str,
        name_elements: list,
        ocp,
        nlp,
        as_states: bool,
        as_controls: bool,
        as_states_dot: bool = False,
        fatigue: FatigueList = None,
        combine_name: str = None,
        combine_state_control_plot: bool = False,
        skip_plot: bool = False,
        axes_idx: Mapping = None,
    ):
        """
        Add a new variable to the states/controls pool

        Parameters
        ----------
        name: str
            The name of the new variable to add
        name_elements: list[str]
            The name of each element of the vector
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the new variable should be added to the state variable set
        as_states_dot: bool
            If the new variable should be added to the state_dot variable set
        as_controls: bool
            If the new variable should be added to the control variable set
        fatigue: FatigueList
            The list of fatigable item
        combine_name: str
            The name of a previously added plot to combine to
        combine_state_control_plot: bool
            If states and controls plot should be combined. Only effective if as_states and as_controls are both True
        skip_plot: bool
            If no plot should be automatically added
        """

        if combine_state_control_plot and combine_name is not None:
            raise ValueError("combine_name and combine_state_control_plot cannot be defined simultaneously")

        def define_cx(n_col: int) -> list:
            _cx = [nlp.cx() for _ in range(n_col)]
            for idx in nlp.variable_mappings[name].to_first.map_idx:
                if idx is None:
                    continue
                for j in range(len(_cx)):
                    sign = "-" if np.sign(idx) < 0 else ""
                    _cx[j] = vertcat(
                        _cx[j], nlp.cx.sym(f"{sign}{name}_{name_elements[abs(idx)]}_{nlp.phase_idx}_{j}", 1, 1)
                    )
            return _cx

        if ConfigureProblem._manage_fatigue_to_new_variable(
            name, name_elements, ocp, nlp, as_states, as_controls, fatigue
        ):
            # If the element is fatigable, this function calls back configure_new_variable to fill everything.
            # Therefore, we can exist now
            return

        if name not in nlp.variable_mappings:
            nlp.variable_mappings[name] = BiMapping(range(len(name_elements)), range(len(name_elements)))

        copy_states = False
        if nlp.use_states_from_phase_idx is not None:
            if nlp.use_states_from_phase_idx < nlp.phase_idx:
                if name in ocp.nlp[nlp.use_states_from_phase_idx].states:
                    copy_states = True

        copy_controls = False
        if nlp.use_controls_from_phase_idx is not None:
            if nlp.use_controls_from_phase_idx < nlp.phase_idx:
                if name in ocp.nlp[nlp.use_controls_from_phase_idx].controls:
                    copy_controls = True

        copy_states_dot = False
        if nlp.use_states_dot_from_phase_idx is not None:
            if nlp.use_states_dot_from_phase_idx < nlp.phase_idx:
                if name in ocp.nlp[nlp.use_states_dot_from_phase_idx].states_dot:
                    copy_states_dot = True

        mx_states = [] if not copy_states else [ocp.nlp[nlp.use_states_from_phase_idx].states[name].mx]
        mx_controls = [] if not copy_controls else [ocp.nlp[nlp.use_controls_from_phase_idx].controls[name].mx]
        mx_states_dot = [] if not copy_states_dot else [ocp.nlp[nlp.use_states_dot_from_phase_idx].states_dot[name].mx]

        # if mapping on variables, what do we do with mapping on the nodes ?

        for i in nlp.variable_mappings[name].to_second.map_idx:
            var_name = f"{'-' if np.sign(i) < 0 else ''}{name}_{name_elements[abs(i)]}_MX" if i is not None else "zero"

            if not copy_states:
                mx_states.append(MX.sym(var_name, 1, 1))

            if not copy_controls:
                mx_controls.append(MX.sym(var_name, 1, 1))

            if not copy_states_dot:
                mx_states_dot.append(MX.sym(var_name, 1, 1))

        mx_states = vertcat(*mx_states)
        mx_controls = vertcat(*mx_controls)
        mx_states_dot = vertcat(*mx_states_dot)

        if not axes_idx:
            axes_idx = Mapping(range(len(name_elements)))

        legend = [
            f"{name}_{name_el}"
            for idx, name_el in enumerate(name_elements)
            if idx is not None and idx in axes_idx.map_idx
        ]

        if as_states:
            n_cx = nlp.ode_solver.polynomial_degree + 2 if isinstance(nlp.ode_solver, OdeSolver.COLLOCATION) else 2
            cx = (
                ocp.nlp[nlp.use_states_from_phase_idx].states[name].original_cx
                if copy_states
                else define_cx(n_col=n_cx)
            )

            nlp.states.append(name, cx, mx_states, nlp.variable_mappings[name])
            if not skip_plot:
                nlp.plot[f"{name}_states"] = CustomPlot(
                    lambda t, x, u, p: x[nlp.states[name].index, :],
                    plot_type=PlotType.INTEGRATED,
                    axes_idx=axes_idx,
                    legend=legend,
                    combine_to=combine_name,
                )

        if as_controls:
            cx = (
                ocp.nlp[nlp.use_controls_from_phase_idx].controls[name].original_cx
                if copy_controls
                else define_cx(n_col=2)
            )

            nlp.controls.append(name, cx, mx_controls, nlp.variable_mappings[name])

            plot_type = PlotType.PLOT if nlp.control_type == ControlType.LINEAR_CONTINUOUS else PlotType.STEP
            if not skip_plot:
                nlp.plot[f"{name}_controls"] = CustomPlot(
                    lambda t, x, u, p: u[nlp.controls[name].index, :],
                    plot_type=plot_type,
                    axes_idx=axes_idx,
                    legend=legend,
                    combine_to=f"{name}_states" if as_states and combine_state_control_plot else combine_name,
                )

        if as_states_dot:
            n_cx = nlp.ode_solver.polynomial_degree + 1 if isinstance(nlp.ode_solver, OdeSolver.COLLOCATION) else 2
            cx = (
                ocp.nlp[nlp.use_states_dot_from_phase_idx].states_dot[name].original_cx
                if copy_states_dot
                else define_cx(n_col=n_cx)
            )

            nlp.states_dot.append(name, cx, mx_states_dot, nlp.variable_mappings[name])

    @staticmethod
    def configure_q(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure the generalized coordinates

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized coordinates should be a state
        as_controls: bool
            If the generalized coordinates should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """

        name = "q"
        name_q = [name.to_string() for name in nlp.model.nameDof()]
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name,
            name_q,
            ocp,
            nlp,
            as_states,
            as_controls,
            as_states_dot,
            axes_idx=axes_idx,
        )

    @staticmethod
    def configure_qdot(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure the generalized velocities

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        as_states_dot: bool
            If the generalized velocities should be a state_dot
        """

        name = "qdot"
        name_qdot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        ConfigureProblem._adjust_mapping(name, ["q", "qdot", "taudot"], nlp)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_qdot, ocp, nlp, as_states, as_controls, as_states_dot, axes_idx=axes_idx
        )

    @staticmethod
    def configure_qddot(ocp, nlp, as_states: bool, as_controls: bool, as_states_dot: bool = False):
        """
        Configure the generalized accelerations

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        as_states_dot: bool
            If the generalized accelerations should be a state_dot
        """

        name = "qddot"
        name_qddot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        ConfigureProblem._adjust_mapping(name, ["q", "qdot"], nlp)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_qddot, ocp, nlp, as_states, as_controls, as_states_dot, axes_idx=axes_idx
        )

    @staticmethod
    def configure_qdddot(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized accelerations

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized velocities should be a state
        as_controls: bool
            If the generalized velocities should be a control
        """

        name = "qdddot"
        name_qdddot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        ConfigureProblem._adjust_mapping(name, ["q", "qdot", "qddot"], nlp)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, name_qdddot, ocp, nlp, as_states, as_controls, axes_idx=axes_idx)

    @staticmethod
    def configure_tau(ocp, nlp, as_states: bool, as_controls: bool, fatigue: FatigueList = None):
        """
        Configure the generalized forces

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized forces should be a state
        as_controls: bool
            If the generalized forces should be a control
        fatigue: FatigueList
            If the dynamics with fatigue should be declared
        """

        name = "tau"
        name_tau = ConfigureProblem._get_kinematics_based_names(nlp, name)
        ConfigureProblem._adjust_mapping(name, ["qdot", "taudot"], nlp)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(
            name, name_tau, ocp, nlp, as_states, as_controls, fatigue=fatigue, axes_idx=axes_idx
        )

    @staticmethod
    def append_faked_optim_var(name, optim_var, keys: list):
        """
        Add a fake optim var by combining vars in keys

        Parameters
        ----------
        optim_var: OptimizationVariableList
            states or controls
        keys: list
            The list of keys to combine
        """

        index = []
        mx = MX()
        to_second = []
        to_first = []
        for key in keys:
            index.extend(list(optim_var[key].index))
            mx = vertcat(mx, optim_var[key].mx)
            to_second.extend(list(np.array(optim_var[key].mapping.to_second.map_idx) + len(to_second)))
            to_first.extend(list(np.array(optim_var[key].mapping.to_first.map_idx) + len(to_first)))

        optim_var.append_fake(name, index, mx, BiMapping(to_second, to_first))

    @staticmethod
    def configure_taudot(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        """

        name = "taudot"
        name_taudot = ConfigureProblem._get_kinematics_based_names(nlp, name)
        ConfigureProblem._adjust_mapping(name, ["qdot", "tau"], nlp)
        axes_idx = ConfigureProblem._apply_phase_mapping(ocp, nlp, name)
        ConfigureProblem.configure_new_variable(name, name_taudot, ocp, nlp, as_states, as_controls, axes_idx=axes_idx)

    @staticmethod
    def configure_contact_forces(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        """

        name_contact_forces = [name.to_string() for name in nlp.model.contactNames()]
        ConfigureProblem.configure_new_variable("fext", name_contact_forces, ocp, nlp, as_states, as_controls)

    @staticmethod
    def configure_soft_contact_forces(ocp, nlp, as_states: bool, as_controls: bool):
        """
        Configure the generalized forces derivative

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the generalized force derivatives should be a state
        as_controls: bool
            If the generalized force derivatives should be a control
        """
        name_soft_contact_forces = []
        component_list = ["fx", "fy", "fz"]  # TODO: find a better place to hold this or define it in biorbd ?
        for ii in range(nlp.model.nbSoftContacts()):
            name_soft_contact_forces.extend(
                [
                    f"{nlp.model.softContactName(ii).to_string()}_{name}"
                    for name in component_list
                    if nlp.model.softContactName(ii).to_string() not in name_soft_contact_forces
                ]
            )

        ConfigureProblem.configure_new_variable("fext", name_soft_contact_forces, ocp, nlp, as_states, as_controls)

    @staticmethod
    def configure_muscles(ocp, nlp, as_states: bool, as_controls: bool, fatigue: FatigueList = None):
        """
        Configure the muscles

        Parameters
        ----------
        nlp: NonLinearProgram
            A reference to the phase
        as_states: bool
            If the muscles should be a state
        as_controls: bool
            If the muscles should be a control
        fatigue: FatigueList
            The list of fatigue parameters
        """

        muscle_names = [names.to_string() for names in nlp.model.muscleNames()]
        ConfigureProblem.configure_new_variable(
            "muscles",
            muscle_names,
            ocp,
            nlp,
            as_states,
            as_controls,
            combine_state_control_plot=True,
            fatigue=fatigue,
        )

    @staticmethod
    def _adjust_mapping(key_to_adjust: str, reference_keys: list, nlp):
        """
        Automatic mapping duplicator basing the values on the another mapping

        Parameters
        ----------
        key_to_adjust: str
            The name of the variable to create if not already defined
        reference_keys: list[str]
            The reference keys, as soon one is found, it is used and the function returns
        nlp: NonLinearProgram
            A reference to the phase
        """

        if key_to_adjust not in nlp.variable_mappings:
            for n in reference_keys:
                if n in nlp.variable_mappings:
                    if n == "q":
                        q_map = list(nlp.variable_mappings[n].to_first.map_idx)
                        target = list(range(nlp.model.nbQ()))
                        if nlp.model.nbQuat() > 0:
                            if q_map != target:
                                raise RuntimeError(
                                    "It is not possible to define a q mapping without a qdot or tau mapping"
                                    "while the model has quaternions"
                                )
                            target = list(range(nlp.model.nbQdot()))
                        nlp.variable_mappings[key_to_adjust] = BiMapping(target, target)
                    else:
                        nlp.variable_mappings[key_to_adjust] = nlp.variable_mappings[n]
                    return
            raise RuntimeError("Could not adjust mapping with the reference_keys provided")

    @staticmethod
    def _apply_phase_mapping(ocp, nlp, name):
        if nlp.phase_mapping:
            if name in nlp.variable_mappings.keys():
                double_mapping = nlp.variable_mappings[name].to_first.map(nlp.phase_mapping.map_idx).T.tolist()[0]
                double_mapping = [int(double_mapping[i]) for i in range(len(double_mapping))]
            else:
                double_mapping = nlp.phase_mapping.map_idx
            axes_idx = Mapping(double_mapping)
        else:
            axes_idx = None
        return axes_idx


class DynamicsFcn(FcnEnum):
    """
    Selection of valid dynamics functions
    """

    TORQUE_DRIVEN = (ConfigureProblem.torque_driven,)
    TORQUE_DERIVATIVE_DRIVEN = (ConfigureProblem.torque_derivative_driven,)
    TORQUE_ACTIVATIONS_DRIVEN = (ConfigureProblem.torque_activations_driven,)
    JOINTS_ACCELERATION_DRIVEN = (ConfigureProblem.joints_acceleration_driven,)
    MUSCLE_DRIVEN = (ConfigureProblem.muscle_driven,)
    CUSTOM = (ConfigureProblem.custom,)


class Dynamics(OptionGeneric):
    """
    A placeholder for the chosen dynamics by the user

    Attributes
    ----------
    dynamic_function: Callable
        The custom dynamic function provided by the user
    configure: Callable
        The configuration function provided by the user that declares the NLP (states and controls),
        usually only necessary when defining custom functions
    expand: bool
        If the continuity constraint should be expand. This can be extensive on RAM

    """

    def __init__(
        self,
        dynamics_type: Union[Callable, DynamicsFcn],
        expand: bool = False,
        **params: Any,
    ):
        """
        Parameters
        ----------
        dynamics_type: Union[Callable, DynamicsFcn]
            The chosen dynamic functions
        params: Any
            Any parameters to pass to the dynamic and configure functions
        expand: bool
            If the continuity constraint should be expand. This can be extensive on RAM
        """

        configure = None
        if not isinstance(dynamics_type, DynamicsFcn):
            configure = dynamics_type
            dynamics_type = DynamicsFcn.CUSTOM
        else:
            if "configure" in params:
                configure = params["configure"]
                del params["configure"]

        dynamic_function = None
        if "dynamic_function" in params:
            dynamic_function = params["dynamic_function"]
            del params["dynamic_function"]

        super(Dynamics, self).__init__(type=dynamics_type, **params)
        self.dynamic_function = dynamic_function
        self.configure = configure
        self.expand = expand


class DynamicsList(UniquePerPhaseOptionList):
    """
    A list of Dynamics if more than one is required, typically when more than one phases are declared

    Methods
    -------
    add(dynamics: DynamicsFcn, **extra_parameters)
        Add a new Dynamics to the list
    print(self)
        Print the DynamicsList to the console
    """

    def add(self, dynamics_type: Union[Callable, Dynamics, DynamicsFcn], **extra_parameters: Any):
        """
        Add a new Dynamics to the list

        Parameters
        ----------
        dynamics_type: Union[Callable, Dynamics, DynamicsFcn]
            The chosen dynamic functions
        extra_parameters: dict
            Any parameters to pass to Dynamics
        """

        if isinstance(dynamics_type, Dynamics):
            self.copy(dynamics_type)

        else:
            super(DynamicsList, self)._add(dynamics_type=dynamics_type, option_type=Dynamics, **extra_parameters)

    def print(self):
        """
        Print the DynamicsList to the console
        """
        raise NotImplementedError("Printing of DynamicsList is not ready yet")
