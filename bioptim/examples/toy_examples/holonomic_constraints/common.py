import numpy as np
import warnings
from casadi import DM

from bioptim import HolonomicTorqueBiorbdModel, SolutionMerge


def compute_all_q(sol, bio_model: HolonomicTorqueBiorbdModel):
    """
    Compute all the states from the solution of the optimal control program.

    .. deprecated:: 
        This function is deprecated. Use `bio_model.compute_q_from_u_iterative(states["q_u"])` instead.

    Parameters
    ----------
    bio_model : HolonomicTorqueBiorbdModel
        The biorbd model with holonomic constraints.
    sol : Solution
        The solution of the optimal control program.

    Returns
    -------
    np.ndarray
        Full coordinate trajectory, shape (nb_q × n_nodes).

    Examples
    --------
    New recommended approach:
    
    >>> states = sol.decision_states(to_merge=SolutionMerge.NODES)
    >>> q_full = bio_model.compute_q_from_u_iterative(states["q_u"])
    """
    warnings.warn(
        "compute_all_q() is deprecated and will be removed in a future version. "
        "Use bio_model.compute_q_from_u_iterative(states['q_u']) instead.",
        DeprecationWarning,
        stacklevel=2
    )

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    return bio_model.compute_q_from_u_iterative(states["q_u"])


def compute_all_states(sol, bio_model: HolonomicTorqueBiorbdModel):
    """
    Compute all the states from the solution of the optimal control program.

    .. deprecated::
        This function is deprecated. Use `bio_model.compute_all_states_from_u_iterative()` instead.

    Parameters
    ----------
    bio_model : HolonomicTorqueBiorbdModel
        The biorbd model with holonomic constraints.
    sol : Solution
        The solution of the optimal control program.

    Returns
    -------
    q : np.ndarray
        Full coordinate trajectory, shape (nb_q × n_nodes).
    qdot : np.ndarray
        Full velocity trajectory, shape (nb_q × n_nodes).
    qddot : np.ndarray
        Full acceleration trajectory, shape (nb_q × n_nodes).
    lambdas : np.ndarray
        Lagrange multiplier trajectory, shape (n_v × n_nodes).

    Examples
    --------
    New recommended approach:
    
    >>> states = sol.decision_states(to_merge=SolutionMerge.NODES)
    >>> controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    >>> 
    >>> # Prepare tau array
    >>> n_nodes = states["q_u"].shape[1]
    >>> tau = np.zeros((bio_model.nb_tau, n_nodes))
    >>> tau[:, :-1] = controls["tau"]
    >>> 
    >>> # Compute all states
    >>> q, qdot, qddot, lambdas = bio_model.compute_all_states_from_u_iterative(
    ...     states["q_u"],
    ...     states["qdot_u"],
    ...     tau
    ... )
    """
    warnings.warn(
        "compute_all_states() is deprecated and will be removed in a future version. "
        "Use bio_model.compute_all_states_from_u_iterative() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)

    # Prepare tau array with proper dimensions
    n_nodes = states["q_u"].shape[1]
    tau = np.zeros((bio_model.nb_tau, n_nodes))
    
    # Copy control torques (handle both independent and dependent joints)
    n_controls = controls["tau"].shape[1]
    for independent_joint_index in bio_model.independent_joint_index:
        tau[independent_joint_index, :n_controls] = controls["tau"][independent_joint_index, :]
    for dependent_joint_index in bio_model.dependent_joint_index:
        tau[dependent_joint_index, :n_controls] = controls["tau"][dependent_joint_index, :]

    return bio_model.compute_all_states_from_u_iterative(
        states["q_u"],
        states["qdot_u"],
        tau
    )
