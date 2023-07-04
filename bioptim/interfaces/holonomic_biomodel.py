from typing import Protocol, Callable

from biorbd_casadi import GeneralizedCoordinates
from casadi import MX, DM, Function
from ..interfaces.biomodel import BioModel


class HolonomicBioModel(BioModel, Protocol):
    def set_dependencies(self, dependent_joint_index: list, independent_joint_index: list):
        """
        Set the dependencies between the joints of the model

        Parameters
        ----------
        dependent_joint_index: list
            The list of the index of the dependent joints
        independent_joint_index: list
            The list of the index of the independent joints
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

    def add_holonomic_constraint(
        self,
        constraint: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX],
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX],
    ):
        """
        Add a holonomic constraint to the model

        Parameters
        ----------
        constraint: Function | Callable[[GeneralizedCoordinates], MX]
            The holonomic constraint
        constraint_jacobian: Function | Callable[[GeneralizedCoordinates], MX]
            The jacobian of the holonomic constraint
        constraint_double_derivative: Function | Callable[[GeneralizedCoordinates], MX]
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

    def partitioned_non_linear_effect(self, q: MX, qdot: MX, f_ext=None, f_contacts=None) -> MX:
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

    def partitioned_forward_dynamics(
        self, q_u, qdot_u, tau, external_forces=None, f_contacts=None, q_v_init=None
    ) -> MX:
        """
        This is the forward dynamics of the model, but only for the independent joints

        Parameters
        ----------
        q_u: MX
            The independent generalized coordinates
        qdot_u: MX
            The independent generalized velocities
        tau: MX
            The generalized torques
        external_forces: MX
            The external forces
        f_contacts: MX
            The contact forces

        Returns
        -------
        MX
            The generalized accelerations

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
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

    def compute_q_v(self, u: MX) -> MX:
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! Symbolic version of the function

        Parameters
        ----------
        u: MX
            The generalized coordinates

        Returns
        -------
        MX
            The dependent joint
        """

    def compute_q(self, q_u, q_v_init: MX = None) -> MX:
        """
        Compute the generalized coordinates from the independent joint coordinates

        Parameters
        ----------
        q_u: MX
            The independent joint coordinates
        q_v_init: MX
            The initial guess for the dependent joint coordinates

        Returns
        -------
        MX
            The generalized coordinates
        """

    def compute_qdot_v(self, q: MX, qdot_u: MX) -> MX:
        """
        Compute the dependent joint velocities from the independent joint velocities and the positions.

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

    def compute_qddot_v(self, q: MX, qdot: MX, qddot_u: MX) -> MX:
        """
        Compute the dependent joint accelerations from the independent joint accelerations and the velocities and
        positions.

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot:
            The generalized velocities
        qddot_u:
            The independent joint accelerations

        Returns
        -------
        MX
            The dependent joint accelerations

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """

    def compute_qddot(self, q: MX, qdot: MX, qddot_u: MX) -> MX:
        """
        Compute the accelerations from the independent joint accelerations and the velocities and positions.

        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot:
            The generalized velocities
        qddot_u:
            The independent joint accelerations

        Returns
        -------
        MX
            The generalized accelerations

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """

    def compute_q_v_numeric(self, u: DM, v_init=None) -> DM:
        """
        Compute the dependent joint from the independent joint,
        This is done by solving the system of equations given by the holonomic constraints
        At the end of this step, we get admissible generalized coordinates w.r.t. the holonomic constraints

        !! Numeric version of the function

        Parameters
        ----------
        u: DM
            The generalized coordinates
        v_init: DM
            The initial guess for the dependent joint

        Returns
        -------
        DM
            The numerical values of the dependent joint for a given independent joint state_u
        """

    def compute_the_Lagrangian_multiplier(
        self, q: MX, qdot: MX, qddot: MX, tau: MX, external_forces: MX = None, f_contacts: MX = None
    ) -> MX:
        """
        Compute the Lagrangian multiplier, denoted lambda in the paper:
        Parameters
        ----------
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        qddot: MX
            The generalized accelerations
        tau: MX
            The generalized torques
        external_forces: MX
            The external forces
        f_contacts: MX
            The contact forces

        Returns
        -------
        MX
            The Lagrangian multipliers

        Sources
        -------
        Docquier, N., Poncelet, A., and Fisette, P.:
        ROBOTRAN: a powerful symbolic gnerator of multibody models, Mech. Sci., 4, 199–219,
        https://doi.org/10.5194/ms-4-199-2013, 2013.
        Equation (17) in the paper.
        """
