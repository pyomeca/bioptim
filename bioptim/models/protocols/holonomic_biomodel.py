from typing import Protocol, Callable
from functools import wraps

from biorbd_casadi import GeneralizedCoordinates
from casadi import MX, DM, Function
from .biomodel import BioModel
from ..holonomic_constraints import HolonomicConstraintsList


class HolonomicBioModel(BioModel, Protocol):

    def _cache_function(method):
        """Decorator to cache CasADi functions automatically"""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Create a unique key based on the method name and arguments
            key = (method.__name__, args, frozenset(kwargs.items()))
            if key in self._cached_functions:
                return self._cached_functions[key]

            # Call the original function to create the CasADi function
            casadi_fun = method(self, *args, **kwargs)

            # Store in the cache
            self._cached_functions[key] = casadi_fun
            return casadi_fun

        return wrapper

    def set_holonomic_configuration(
        self,
        constraints_list: HolonomicConstraintsList,
        dependent_joint_index: list = None,
        independent_joint_index: list = None,
    ):
        """
        Set the holonomic constraints of the model and if necessary the partitioned dynamics.
        The joint indexes are not mandatory because a HolonomicBiorbdModel can be used without the partitioned dynamics,
        for instance in VariationalOptimalControlProgram.

        Parameters
        ----------
        dependent_joint_index: list
            The list of the index of the dependent joints
        independent_joint_index: list
            The list of the index of the independent joints
        constraints_list: HolonomicConstraintsList
            The list of the holonomic constraints
        """

    @property
    def nb_independent_joints(self) -> int:
        """
        Get the number of independent joints

        Returns
        -------
        int
            The number of independent joints
        """
        return -1

    @property
    def nb_dependent_joints(self) -> int:
        """
        Get the number of dependent joints

        Returns
        -------
        int
            The number of dependent joints
        """
        return -1

    @property
    def dependent_joint_index(self) -> list:
        """
        Get the index of the dependent joints

        Returns
        -------
        list
            The index of the dependent joints
        """
        return []

    @property
    def independent_joint_index(self) -> list:
        """
        Get the index of the independent joints

        Returns
        -------
        list
            The index of the independent joints
        """
        return []

    def _add_holonomic_constraint(
        self,
        constraints: Function | Callable[[GeneralizedCoordinates], MX],
        constraints_jacobian: Function | Callable[[GeneralizedCoordinates], MX],
        constraints_double_derivative: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        """
        Add a holonomic constraint to the model

        Parameters
        ----------
        constraints: Function | Callable[[GeneralizedCoordinates], MX]
            The holonomic constraint
        constraints_jacobian: Function | Callable[[GeneralizedCoordinates], MX]
            The jacobian of the holonomic constraint
        constraints_double_derivative: Function | Callable[[GeneralizedCoordinates], MX]
            The double derivative of the holonomic constraint
        """

    @property
    def has_holonomic_constraints(self):
        """
        Check if the model has holonomic constraints

        Returns
        -------
        bool
            If the model has holonomic constraints
        """
        return False

    @property
    def nb_holonomic_constraints(self) -> int:
        """
        Get the number of holonomic constraints

        Returns
        -------
        int
            The number of holonomic constraints
        """
        return -1

    def holonomic_constraints(self, q: MX) -> MX:
        """
        Get the holonomic constraints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The holonomic constraints
        """

    def holonomic_constraints_jacobian(self, q: MX) -> MX:
        """
        Get the jacobian of the holonomic constraints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The holonomic constraints jacobian
        """

    def holonomic_constraints_derivative(self, q: MX, qdot: MX) -> MX:
        """
        Get the derivative of the holonomic constraints

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities

        Returns
        -------
        MX
            The holonomic constraints derivative
        """

    def holonomic_constraints_double_derivative(self, q: MX, qdot: MX, qddot: MX) -> MX:
        """
        Get the double derivative of the holonomic constraints

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        qddot: MX
            The generalized accelerations

        Returns
        -------
        MX
            The holonomic constraints double derivative
        """

    def partitioned_mass_matrix(self, q: MX) -> MX:
        """
        This function returns the partitioned mass matrix, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned mass matrix, reordered in function independent and dependent joints
        """

    def partitioned_non_linear_effect(self, q: MX, qdot: MX) -> MX:
        """
        This function returns the partitioned non-linear effect, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        f_ext: MX
            The external forces
        f_contacts: MX
            The contact forces
        """

    def partitioned_q(self, q: MX) -> MX:
        """
        This function returns the partitioned q, reordered in function independent and dependent joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned q, reorder in function independent and dependent joints
        """

    def partitioned_qdot(self, qdot: MX) -> MX:
        """
        This function returns the partitioned qdot, reordered in function independent and dependent joints

        Parameters
        ----------
        qdot: MX
            The generalized velocities

        Returns
        -------
        MX
            The partitioned qdot, reordered in function independent and dependent joints
        """

    def partitioned_tau(self, tau: MX) -> MX:
        """
        This function returns the partitioned tau, reordered in function independent and dependent joints

        Parameters
        ----------
        tau: MX
            The generalized torques

        Returns
        -------
        MX
            The partitioned tau, reordered in function independent and dependent joints
        """

    def partitioned_constrained_jacobian(self, q: MX) -> MX:
        """
        This function returns the partitioned constrained jacobian, reordered in function independent and dependent
        joints

        Parameters
        ----------
        q: MX
            The generalized coordinates

        Returns
        -------
        MX
            The partitioned constrained jacobian, reordered in function independent and dependent joints
        """

    def coupling_matrix(self, q: MX) -> MX:
        """
        Compute the coupling matrix, denoted Bvu in the paper :

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """

    def biais_vector(self, q: MX, qdot: MX) -> MX:
        """
        Compute the biais vector, denoted b in the paper :

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.

        The right term of the equation (15) in the paper.
        """

    def state_from_partition(self, state_u: MX, state_v: MX) -> MX:
        """
        Compute the generalized coordinates from the independent and dependent joint coordinates
        qddot_u, qddot_v -> qddot
        qdot_u, qdot_v -> qdot
        q_u, q_v -> q

        Parameters
        ----------
        state_u: MX
            The independent joint coordinates
        state_v: MX
            The dependent joint coordinates

        Returns
        -------
        MX
            The generalized coordinates

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        """

    def compute_q_v(self) -> Function:
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints
        """

    def compute_q(self) -> Function:
        """
        Compute the generalized coordinates from the independent joint coordinates
        """

    def compute_qdot_v(self) -> Function:
        """
        Compute the dependent joint velocities from the independent joint velocities and the positions.
        """

    def compute_qdot(self, q: MX, qdot_u: MX) -> MX:
        """
        Compute the velocities from the independent joint velocities and the positions.

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot_u: MX
            The independent joint velocities

        Returns
        -------
        MX
            The dependent joint velocities
        """

    def compute_qddot_v(self) -> Function:
        """
        Compute the dependent joint accelerations from the independent joint accelerations and the velocities and
        positions.

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """

    def compute_qddot(self) -> Function:
        """
        Compute the accelerations from the independent joint accelerations and the velocities and positions.

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """

    def compute_the_lagrangian_multipliers(self) -> Function:
        """
        Compute the Lagrangian multiplier, denoted lambda in the paper:

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """

    def check_state_u_size(self, state_u: MX):
        """
        Check if the size of the independent state vector matches the number of independent joints

        Parameters
        ----------
        state_u: MX
            The independent state vector to check
        """

    def check_state_v_size(self, state_v: MX):
        """
        Check if the size of the dependent state vector matches the number of dependent joints

        Parameters
        ----------
        state_v: MX
            The dependent state vector to check
        """

    def holonomic_forward_dynamics(self) -> Function:
        """
        Compute the forward dynamics while respecting holonomic constraints.
        This combines the regular forward dynamics with constraint forces.

        Returns
        -------
        Function
            The holonomic forward dynamics function
        """

    def holonomic_inverse_dynamics(self) -> Function:
        """
        Compute the inverse dynamics while respecting holonomic constraints.
        This combines the regular inverse dynamics with constraint forces.

        Returns
        -------
        Function
            The holonomic inverse dynamics function
        """

    def constraint_forces(self) -> Function:
        """
        Compute the forces required to maintain the holonomic constraints.

        Returns
        -------
        Function
            The constraint forces function
        """
