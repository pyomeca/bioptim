from math import inf
from enum import Enum

from casadi import sum1, horzcat

from .path_conditions import Bounds
from .penalty import PenaltyType, PenaltyFunctionAbstract, PenaltyOption
from ..misc.enums import Node, InterpolationType, OdeSolver, ControlType
from ..misc.options_lists import OptionList, OptionGeneric


class ConstraintOption(PenaltyOption):
    def __init__(self, constraint, min_bound=None, max_bound=None, phase=0, **params):
        custom_function = None
        if not isinstance(constraint, Constraint):
            custom_function = constraint
            constraint = Constraint.CUSTOM

        super(ConstraintOption, self).__init__(
            penalty=constraint, phase=phase, custom_function=custom_function, **params
        )
        self.min_bound = min_bound
        self.max_bound = max_bound


class ConstraintList(OptionList):
    def add(self, constraint, **extra_arguments):
        if isinstance(constraint, ConstraintOption):
            self.copy(constraint)

        else:
            super(ConstraintList, self)._add(option_type=ConstraintOption, constraint=constraint, **extra_arguments)


class ConstraintFunction(PenaltyFunctionAbstract):
    """
    Different conditions between biorbd geometric structures.
    """

    class Functions:
        """
        Biomechanical constraints
        """

        @staticmethod
        def contact_force(constraint, ocp, nlp, t, x, u, p, contact_force_idx):
            """
            To be completed when this function will be fully developed, in particular the fact that policy is either a
            tuple/list or a tuple of tuples/list of lists,
            with in the 1st index the number of the contact force and in the 2nd index the associated bound.
            """
            # To be modified later so that it can handle something other than lower bounds for greater than
            for i in range(len(u)):
                ConstraintFunction.add_to_penalty(
                    ocp,
                    nlp,
                    nlp.contact_forces_func(x[i], u[i], p)[contact_force_idx, 0],
                    constraint,
                )

        @staticmethod
        def non_slipping(
            constraint,
            ocp,
            nlp,
            t,
            x,
            u,
            p,
            tangential_component_idx,
            normal_component_idx,
            static_friction_coefficient,
        ):
            """
            Constraint preventing the contact point from slipping tangentially to the contact surface
            with a chosen static friction coefficient.
            One constraint per tangential direction.
            Normal forces are considered to be greater than zero.
            :param tangential_component_idx: index of the tangential portion of the contact force (integer)
            :param normal_component_idx: index of the normal portion of the contact force (integer)
            :param static_friction_coefficient: static friction coefficient (float)
            """
            if not isinstance(tangential_component_idx, int):
                raise RuntimeError("tangential_component_idx must be a unique integer")

            if isinstance(normal_component_idx, int):
                normal_component_idx = [normal_component_idx]

            mu = static_friction_coefficient
            constraint.min_bound = 0
            constraint.max_bound = inf
            for i in range(len(u)):
                contact = nlp.contact_forces_func(x[i], u[i], p)
                normal_contact_force = sum1(contact[normal_component_idx, 0])
                tangential_contact_force = contact[tangential_component_idx, 0]

                # Since it is non-slipping normal forces are supposed to be greater than zero
                ConstraintFunction.add_to_penalty(
                    ocp,
                    nlp,
                    mu * normal_contact_force - tangential_contact_force,
                    constraint,
                )
                ConstraintFunction.add_to_penalty(
                    ocp,
                    nlp,
                    mu * normal_contact_force + tangential_contact_force,
                    constraint,
                )

        @staticmethod
        def time_constraint(constraint_type, ocp, nlp, t, x, u, p, **unused_params):
            pass

    @staticmethod
    def add_or_replace(ocp, nlp, penalty):
        if penalty.type == Constraint.TIME_CONSTRAINT:
            penalty.node = Node.END
        PenaltyFunctionAbstract.add_or_replace(ocp, nlp, penalty)

    @staticmethod
    def inner_phase_continuity(ocp):
        """
        Adds continuity constraints between each nodes and its neighbours. It is possible to add a continuity
        constraint between first and last nodes to have a loop (nlp.is_cyclic_constraint).
        :param ocp: An OptimalControlProgram class.
        """
        # Dynamics must be sound within phases
        penalty = ConstraintOption([])
        for i, nlp in enumerate(ocp.nlp):
            penalty.list_index = -1
            ConstraintFunction.clear_penalty(ocp, None, penalty)
            # Loop over shooting nodes or use parallelization
            if ocp.nb_threads > 1:
                end_nodes = nlp.par_dynamics(horzcat(*nlp.X[:-1]), horzcat(*nlp.U), nlp.p)[0]
                vals = horzcat(*nlp.X[1:]) - end_nodes
                ConstraintFunction.add_to_penalty(ocp, None, vals.reshape((nlp.nx * nlp.ns, 1)), penalty)
            else:
                for k in range(nlp.ns):
                    # Create an evaluation node
                    if nlp.ode_solver == OdeSolver.RK or nlp.ode_solver == OdeSolver.IRK:
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
        # Dynamics must be respected between phases
        penalty = OptionGeneric()
        penalty.min_bound = 0
        penalty.max_bound = 0
        penalty.list_index = -1
        pt.base.clear_penalty(ocp, None, penalty)
        val = pt.type.value[0](ocp, pt)
        pt.base.add_to_penalty(ocp, None, val, penalty)

    @staticmethod
    def add_to_penalty(ocp, nlp, val, penalty):
        """
        Sets minimal and maximal bounds of the parameter g to be constrained.
        :param g: Parameter to be constrained. (?)
        :param penalty: Index of the parameter g in the penalty array nlp.g. (integer)
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

        if nlp:
            nlp.g[penalty.list_index].append(val)
            nlp.g_bounds[penalty.list_index].append(g_bounds)
        else:
            ocp.g[penalty.list_index].append(val)
            ocp.g_bounds[penalty.list_index].append(g_bounds)

    @staticmethod
    def clear_penalty(ocp, nlp, penalty):
        """
        Resets specified penalty.
        Negative penalty index leads to enlargement of the array by one empty space.
        :param penalty: Index of the penalty to be reset. (integer)
        :return: penalty: Index of the penalty reset. (integer)
        """
        if nlp:
            g_to_add_to = nlp.g
            g_bounds_to_add_to = nlp.g_bounds
        else:
            g_to_add_to = ocp.g
            g_bounds_to_add_to = ocp.g_bounds

        if penalty.list_index < 0:
            for i, j in enumerate(g_to_add_to):
                if not j:
                    penalty.list_index = i
                    return
            else:
                g_to_add_to.append([])
                g_bounds_to_add_to.append([])
                penalty.list_index = len(g_to_add_to) - 1
        else:
            while penalty.list_index >= len(g_to_add_to):
                g_to_add_to.append([])
                g_bounds_to_add_to.append([])
            g_to_add_to[penalty.list_index] = []
            g_bounds_to_add_to[penalty.list_index] = []

    @staticmethod
    def _parameter_modifier(constraint_function, parameters):
        """Modification of parameters"""
        # Everything that should change the entry parameters depending on the penalty can be added here
        super(ConstraintFunction, ConstraintFunction)._parameter_modifier(constraint_function, parameters)

    @staticmethod
    def _span_checker(constraint_function, node, nlp):
        """Raises errors on the span of penalty functions"""
        # Everything that is suspicious in terms of the span of the penalty function can be checked here
        super(ConstraintFunction, ConstraintFunction)._span_checker(constraint_function, node, nlp)
        if (
            constraint_function == Constraint.CONTACT_FORCE.value[0]
            or constraint_function == Constraint.NON_SLIPPING.value[0]
        ):
            if node == Node.END or node == nlp.ns:
                raise RuntimeError("No control u at last node")


class Constraint(Enum):
    """
    Different conditions between biorbd geometric structures.
    """

    TRACK_STATE = (PenaltyType.TRACK_STATE,)
    TRACK_MARKERS = (PenaltyType.TRACK_MARKERS,)
    TRACK_MARKERS_VELOCITY = (PenaltyType.TRACK_MARKERS_VELOCITY,)
    ALIGN_MARKERS = (PenaltyType.ALIGN_MARKERS,)
    PROPORTIONAL_STATE = (PenaltyType.PROPORTIONAL_STATE,)
    PROPORTIONAL_CONTROL = (PenaltyType.PROPORTIONAL_CONTROL,)
    TRACK_TORQUE = (PenaltyType.TRACK_TORQUE,)
    TRACK_MUSCLES_CONTROL = (PenaltyType.TRACK_MUSCLES_CONTROL,)
    TRACK_ALL_CONTROLS = (PenaltyType.TRACK_ALL_CONTROLS,)
    TRACK_CONTACT_FORCES = (PenaltyType.TRACK_CONTACT_FORCES,)
    ALIGN_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT,)
    ALIGN_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS,)
    CUSTOM = (PenaltyType.CUSTOM,)
    CONTACT_FORCE = (ConstraintFunction.Functions.contact_force,)
    NON_SLIPPING = (ConstraintFunction.Functions.non_slipping,)
    TIME_CONSTRAINT = (ConstraintFunction.Functions.time_constraint,)

    @staticmethod
    def get_type():
        """Returns the type of the constraint function"""
        return ConstraintFunction
