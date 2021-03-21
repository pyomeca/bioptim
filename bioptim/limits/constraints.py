from typing import Callable, Union, Any
from math import inf
from enum import Enum

import numpy as np
from casadi import sum1, horzcat, if_else, vertcat, lt, MX, SX, Function
import biorbd

from .path_conditions import Bounds
from .penalty import PenaltyType, PenaltyFunctionAbstract, PenaltyOption, PenaltyNodes
from ..dynamics.ode_solver import OdeSolver
from ..misc.enums import Node, InterpolationType, ControlType
from ..misc.options import OptionList, OptionGeneric


class Constraint(PenaltyOption):
    """
    A placeholder for a constraint

    Attributes
    ----------
    min_bound: np.ndarray
        The vector of minimum bound of the constraint. Default is 0
    max_bound: np.ndarray
        The vector of maximal bound of the constraint. Default is 0
    """

    def __init__(
        self,
        constraint: Any,
        min_bound: Union[np.ndarray, float] = None,
        max_bound: Union[np.ndarray, float] = None,
        phase: int = 0,
        **params: Any,
    ):
        """
        Parameters
        ----------
        constraint: ConstraintFcn
            The chosen constraint
        min_bound: np.ndarray
            The vector of minimum bound of the constraint. Default is 0
        max_bound: np.ndarray
            The vector of maximal bound of the constraint. Default is 0
        phase: int
            The index of the phase to apply the constraint
        params:
            Generic parameters for options
        """
        custom_function = None
        if not isinstance(constraint, ConstraintFcn):
            custom_function = constraint
            constraint = ConstraintFcn.CUSTOM

        super(Constraint, self).__init__(penalty=constraint, phase=phase, custom_function=custom_function, **params)
        self.min_bound = min_bound
        self.max_bound = max_bound


class ConstraintList(OptionList):
    """
    A list of Constraint if more than one is required

    Methods
    -------
    add(self, constraint: Union[Callable, "ConstraintFcn"], **extra_arguments)
        Add a new Constraint to the list
    print(self)
        Print the ConstraintList to the console
    """

    def add(self, constraint: Union[Callable, Constraint, Any], **extra_arguments: Any):
        """
        Add a new constraint to the list

        Parameters
        ----------
        constraint: Union[Callable, Constraint, ConstraintFcn]
            The chosen constraint
        extra_arguments: dict
            Any parameters to pass to Constraint
        """

        if isinstance(constraint, Constraint):
            self.copy(constraint)

        else:
            super(ConstraintList, self)._add(option_type=Constraint, constraint=constraint, **extra_arguments)

    def print(self):
        """
        Print the ConstraintList to the console
        """
        # TODO: Print all elements in the console
        raise NotImplementedError("Printing of ConstraintList is not ready yet")


class ConstraintFunction(PenaltyFunctionAbstract):
    """
    Internal (re)implementation of the penalty functions

    Methods
    -------
    inter_phase_continuity(ocp: OptimalControlProgram, pt: "PhaseTransition")
        Add phase transition constraints between two phases.
    add_to_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, val: Union[MX, SX], penalty: Constraint)
        Add the constraint to the constraint pool
    clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: Constraint)
        Resets a penalty. A negative penalty index creates a new empty penalty.
    _parameter_modifier(constraint: Constraint)
        Apply some default parameters
    _span_checker(constraint, nlp)
        Check for any non sense in the requested times for the constraint. Raises an error if so
    penalty_nature() -> str
        Get the nature of the penalty
    """

    class Functions:
        """
        Implementation of all the constraint functions

        Methods
        -------
        time_constraint(constraint: Constraint, pn: PenaltyNodes)
            The time constraint is taken care elsewhere, but must be declared here. This function therefore does nothing
        torque_max_from_actuators(constraint: Constraint, pn: PenaltyNodes, min_torque=None)
            Non linear maximal values of joint torques computed from the torque-position-velocity relationship
        non_slipping(constraint: Constraint, pn: PenaltyNodes,
                tangential_component_idx: int, normal_component_idx: int, static_friction_coefficient: float)
            Add a constraint of static friction at contact points allowing for small tangential forces. This constraint
            assumes that the normal forces is positive
        contact_force(constraint: Constraint, pn: PenaltyNodes, contact_force_idx: int)
            Add a constraint of contact forces given by any forward dynamics with contact
        """

        @staticmethod
        def contact_force(
            constraint: Constraint,
            pn: PenaltyNodes,
            contact_force_idx: int,
        ):
            """
            Add a constraint of contact forces given by any forward dynamics with contact

            Parameters
            ----------
            constraint: Constraint
                The actual constraint to declare
            pn: PenaltyNodes
                The penalty node elements
            contact_force_idx: int
                The index of the contact force to add to the constraint set
            """

            for i in range(len(pn.u)):
                ConstraintFunction.add_to_penalty(
                    pn.ocp,
                    pn.nlp,
                    pn.nlp.contact_forces_func(pn.x[i], pn.u[i], pn.p)[contact_force_idx, 0],
                    constraint,
                )

        @staticmethod
        def non_slipping(
            constraint: Constraint,
            pn: PenaltyNodes,
            tangential_component_idx: int,
            normal_component_idx: int,
            static_friction_coefficient: float,
        ):
            """
            Add a constraint of static friction at contact points constraining for small tangential forces.
            This constraint assumes that the normal forces is positive

            Parameters
            ----------
            constraint: Constraint
                The actual constraint to declare
            pn: PenaltyNodes
                The penalty node elements
            tangential_component_idx: int
                Index of the tangential component of the contact force
            normal_component_idx: int
                Index of the normal component of the contact force
            static_friction_coefficient: float
                Static friction coefficient
            """

            if not isinstance(tangential_component_idx, int):
                raise RuntimeError("tangential_component_idx must be a unique integer")

            if isinstance(normal_component_idx, int):
                normal_component_idx = [normal_component_idx]

            mu = static_friction_coefficient
            constraint.min_bound = 0
            constraint.max_bound = inf
            for i in range(len(pn.u)):
                contact = pn.nlp.contact_forces_func(pn.x[i], pn.u[i], pn.p)
                normal_contact_force = sum1(contact[normal_component_idx, 0])
                tangential_contact_force = contact[tangential_component_idx, 0]

                # Since it is non-slipping normal forces are supposed to be greater than zero
                ConstraintFunction.add_to_penalty(
                    pn.ocp,
                    pn.nlp,
                    mu * normal_contact_force - tangential_contact_force,
                    constraint,
                )
                ConstraintFunction.add_to_penalty(
                    pn.ocp,
                    pn.nlp,
                    mu * normal_contact_force + tangential_contact_force,
                    constraint,
                )

        @staticmethod
        def torque_max_from_actuators(
            constraint: Constraint,
            pn: PenaltyNodes,
            min_torque=None,
        ):
            """
            Non linear maximal values of joint torques computed from the torque-position-velocity relationship

            Parameters
            ----------
            constraint: Constraint
                The actual constraint to declare
            pn: PenaltyNodes
                The penalty node elements
            min_torque: float
                Minimum joint torques. This prevent from having too small torques, but introduces an if statement
            """

            # TODO: Add index to select the u (control_idx)
            nlp = pn.nlp
            nq = nlp.mapping["q"].to_first.len
            q = [nlp.mapping["q"].to_second.map(mx[:nq]) for mx in pn.x]
            qdot = [nlp.mapping["qdot"].to_second.map(mx[nq:]) for mx in pn.x]

            if min_torque and min_torque < 0:
                raise ValueError("min_torque cannot be negative in tau_max_from_actuators")
            func = biorbd.to_casadi_func("torqueMax", nlp.model.torqueMax, nlp.q, nlp.qdot)
            constraint.min_bound = np.repeat([0, -np.inf], nlp.nu)
            constraint.max_bound = np.repeat([np.inf, 0], nlp.nu)
            for i in range(len(pn.u)):
                bound = func(q[i], qdot[i])
                if min_torque:
                    min_bound = nlp.mapping["tau"].to_first.map(
                        if_else(lt(bound[:, 1], min_torque), min_torque, bound[:, 1])
                    )
                    max_bound = nlp.mapping["tau"].to_first.map(
                        if_else(lt(bound[:, 0], min_torque), min_torque, bound[:, 0])
                    )
                else:
                    min_bound = nlp.mapping["tau"].to_first.map(bound[:, 1])
                    max_bound = nlp.mapping["tau"].to_first.map(bound[:, 0])

                ConstraintFunction.add_to_penalty(
                    pn.ocp, nlp, vertcat(*[pn.u[i] + min_bound, pn.u[i] - max_bound]), constraint
                )

        @staticmethod
        def time_constraint(
            constraint: Constraint,
            pn: PenaltyNodes,
            **unused_param,
        ):
            """
            The time constraint is taken care elsewhere, but must be declared here. This function therefore does nothing

            Parameters
            ----------
            constraint: Constraint
                The actual constraint to declare
            pn: PenaltyNodes
                The penalty node elements
            **unused_param: dict
                Since the function does nothing, we can safely ignore any argument
            """

            pass

    @staticmethod
    def add_or_replace(ocp, nlp, penalty: PenaltyOption):
        """
        Doing some configuration before calling the super.add_or_replace function that prepares the adding of the
        constraint to the constraint pool

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        penalty: PenaltyOption
            The actual constraint to declare
        """

        if penalty.type == ConstraintFcn.TIME_CONSTRAINT:
            penalty.node = Node.END
        PenaltyFunctionAbstract.add_or_replace(ocp, nlp, penalty)

    @staticmethod
    def inner_phase_continuity(ocp):
        """
        Add continuity constraints between each nodes of a phase.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """
        # Dynamics must be sound within phases
        for i, nlp in enumerate(ocp.nlp):
            penalty = Constraint([])
            penalty.name = f"CONTINUITY {i}"
            penalty.list_index = -1
            ConstraintFunction.clear_penalty(ocp, None, penalty)
            # Loop over shooting nodes or use parallelization
            if ocp.n_threads > 1:
                end_nodes = nlp.par_dynamics(horzcat(*nlp.X[:-1]), horzcat(*nlp.U), nlp.p)[0]
                val = horzcat(*nlp.X[1:]) - end_nodes
                ConstraintFunction.add_to_penalty(ocp, None, val.reshape((nlp.nx * nlp.ns, 1)), penalty)
            else:
                for k in range(nlp.ns):
                    # Create an evaluation node
                    if (
                        isinstance(nlp.ode_solver, OdeSolver.RK4)
                        or isinstance(nlp.ode_solver, OdeSolver.RK8)
                        or isinstance(nlp.ode_solver, OdeSolver.IRK)
                    ):
                        if nlp.control_type == ControlType.CONSTANT:
                            u = nlp.U[k]
                        elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                            u = horzcat(nlp.U[k], nlp.U[k + 1])
                        else:
                            raise NotImplementedError(f"Dynamics with {nlp.control_type} is not implemented yet")
                        end_node = nlp.dynamics[k](x0=nlp.X[k], p=u, params=nlp.p)["xf"]
                    else:
                        end_node = nlp.dynamics[k](x0=nlp.X[k], p=nlp.U[k])["xf"]

                    # Save continuity constraints
                    val = end_node - nlp.X[k + 1]
                    ConstraintFunction.add_to_penalty(ocp, None, val, penalty)

    @staticmethod
    def inter_phase_continuity(ocp, pt):
        """
        Add phase transition constraints between two phases.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        pt: PhaseTransition
            The phase transition to add
        """

        # Dynamics must be respected between phases
        penalty = OptionGeneric()
        penalty.name = f"PHASE_TRANSITION {pt.phase_pre_idx}->{pt.phase_pre_idx + 1}"
        penalty.min_bound = 0
        penalty.max_bound = 0
        penalty.list_index = -1
        penalty.sliced_target = None
        pt.base.clear_penalty(ocp, None, penalty)
        val = pt.type.value[0](ocp, pt)
        casadi_name = f"PHASE_TRANSITION_{pt.phase_pre_idx}_{pt.phase_pre_idx + 1}"
        pre_nlp, post_nlp = ocp.nlp[pt.phase_pre_idx], ocp.nlp[(pt.phase_pre_idx + 1) % ocp.n_phases]
        pt.casadi_function = Function(
            casadi_name, [pre_nlp.X[-1], pre_nlp.U[-1], post_nlp.X[0], post_nlp.U[0], ocp.v.parameters.cx], [val]
        ).expand()
        pt.base.add_to_penalty(ocp, None, val, penalty)

    @staticmethod
    def add_to_penalty(ocp, nlp, val: Union[MX, SX, float, int], penalty: Constraint):
        """
        Add the constraint to the constraint pool

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        val: Union[MX, SX, float, int]
            The actual constraint to add
        penalty: Constraint
            The actual constraint to declare
        """

        g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)
        penalty.min_bound = 0 if penalty.min_bound is None else penalty.min_bound
        penalty.max_bound = 0 if penalty.max_bound is None else penalty.max_bound
        for i in range(val.rows()):
            min_bound = (
                penalty.min_bound[i]
                if hasattr(penalty.min_bound, "__getitem__") and penalty.min_bound.shape[0] > 1
                else penalty.min_bound
            )
            max_bound = (
                penalty.max_bound[i]
                if hasattr(penalty.max_bound, "__getitem__") and penalty.max_bound.shape[0] > 1
                else penalty.max_bound
            )
            g_bounds.concatenate(Bounds(min_bound, max_bound, interpolation=InterpolationType.CONSTANT))

        g = {
            "constraint": penalty,
            "val": val,
            "bounds": g_bounds,
            "target": penalty.sliced_target,
        }
        if nlp:
            nlp.g[penalty.list_index].append(g)
        else:
            ocp.g[penalty.list_index].append(g)

    @staticmethod
    def clear_penalty(ocp, nlp, penalty: Constraint):
        """
        Resets a constraint. A negative penalty index creates a new empty constraint.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        penalty: Constraint
            The actual constraint to declare
        """

        if nlp:
            g_to_add_to = nlp.g
        else:
            g_to_add_to = ocp.g

        if penalty.list_index < 0:
            for i, j in enumerate(g_to_add_to):
                if not j:
                    penalty.list_index = i
                    return
            else:
                g_to_add_to.append([])
                penalty.list_index = len(g_to_add_to) - 1
        else:
            while penalty.list_index >= len(g_to_add_to):
                g_to_add_to.append([])
            g_to_add_to[penalty.list_index] = []

    @staticmethod
    def _parameter_modifier(constraint: Constraint):
        """
        Apply some default parameters

        Parameters
        ----------
        constraint: Constraint
            The actual constraint to declare
        """

        # Everything that should change the entry parameters depending on the penalty can be added here
        super(ConstraintFunction, ConstraintFunction)._parameter_modifier(constraint)

    @staticmethod
    def _span_checker(constraint: Constraint, pn: PenaltyNodes):
        """
        Check for any non sense in the requested times for the constraint. Raises an error if so

        Parameters
        ----------
        constraint: Constraint
            The actual constraint to declare
        pn: PenaltyNodes
            The penalty node elements
        """

        # Everything that is suspicious in terms of the span of the penalty function can be checked here
        super(ConstraintFunction, ConstraintFunction)._span_checker(constraint, pn)
        func = constraint.type.value[0]
        node = constraint.node
        if func == ConstraintFcn.CONTACT_FORCE.value[0] or func == ConstraintFcn.NON_SLIPPING.value[0]:
            if node == Node.END or node == pn.nlp.ns:
                raise RuntimeError("No control u at last node")

    @staticmethod
    def penalty_nature() -> str:
        """
        Get the nature of the penalty

        Returns
        -------
        The nature of the penalty
        """

        return "constraints"


class ConstraintFcn(Enum):
    """
    Selection of valid constraint functions

    Methods
    -------
    def get_type() -> Callable
        Returns the type of the penalty
    """

    TRACK_STATE = (PenaltyType.TRACK_STATE,)
    TRACK_MARKERS = (PenaltyType.TRACK_MARKERS,)
    TRACK_MARKERS_VELOCITY = (PenaltyType.TRACK_MARKERS_VELOCITY,)
    SUPERIMPOSE_MARKERS = (PenaltyType.SUPERIMPOSE_MARKERS,)
    PROPORTIONAL_STATE = (PenaltyType.PROPORTIONAL_STATE,)
    PROPORTIONAL_CONTROL = (PenaltyType.PROPORTIONAL_CONTROL,)
    TRACK_TORQUE = (PenaltyType.TRACK_TORQUE,)
    TRACK_MUSCLES_CONTROL = (PenaltyType.TRACK_MUSCLES_CONTROL,)
    TRACK_ALL_CONTROLS = (PenaltyType.TRACK_ALL_CONTROLS,)
    TRACK_CONTACT_FORCES = (PenaltyType.TRACK_CONTACT_FORCES,)
    TRACK_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.TRACK_SEGMENT_WITH_CUSTOM_RT,)
    TRACK_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.TRACK_MARKER_WITH_SEGMENT_AXIS,)
    TRACK_COM_POSITION = (PenaltyType.MINIMIZE_COM_POSITION,)
    TRACK_COM_VELOCITY = (PenaltyType.MINIMIZE_COM_VELOCITY,)
    CUSTOM = (PenaltyType.CUSTOM,)
    CONTACT_FORCE = (ConstraintFunction.Functions.contact_force,)
    NON_SLIPPING = (ConstraintFunction.Functions.non_slipping,)
    TORQUE_MAX_FROM_ACTUATORS = (ConstraintFunction.Functions.torque_max_from_actuators,)
    TIME_CONSTRAINT = (ConstraintFunction.Functions.time_constraint,)

    @staticmethod
    def get_type():
        """
        Returns the type of the penalty
        """

        return ConstraintFunction


class ContinuityFunctions:
    """
    Interface between continuity and constraint
    """

    @staticmethod
    def continuity(ocp):
        """
        The declaration of inner- and inter-phase continuity constraints

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        """

        ConstraintFunction.inner_phase_continuity(ocp)

        # Dynamics must be respected between phases
        for pt in ocp.phase_transitions:
            pt.base.inter_phase_continuity(ocp, pt)
