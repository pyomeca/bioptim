import numpy as np

from ..misc.enums import InterpolationType
from ..optimization.optimization_variable import OptimizationVariableContainer

DEFAULT_INITIAL_GUESS = 0
DEFAULT_MIN_BOUND = -np.inf
DEFAULT_MAX_BOUND = np.inf


def _compute_values_for_all_nodes(
    nlp: "NonLinearProgram",
    default_value: np.ndarray | int,
    variable_container: OptimizationVariableContainer,
    defined_values: dict,  # "min" or "max" only, not both
    scaling: "VariableScalingList",
    n_nodes: int,
    repeat: int,
) -> np.ndarray:
    """
    Compute bounds for all nodes in the discretized problem.

    This function iterates through all nodes and their intervals to compute
    either minimum or maximum bounds for the state variables.

    Parameters
    ----------
    nlp: NonLinearProgram
        The non-linear program containing the optimization problem
    default_value: np.ndarray
        The default value to use if the variable is not defined
        (either DEFAULT_MIN_BOUND, DEFAULT_MAX_BOUND, or DEFAULT_INITIAL_GUESS)
    variable_container: OptimizationVariableContainer
        The container for the optimization variables for states, controls, or algebraic variables that we refer to
    defined_values : dict or InitialGuessList
        The defined values for the variable, which can be min bounds, max bounds, or initial guesses.
    variable_scaling: "VariableScalingList"
        The scaling factors for the variables, which are used to scale the evaluated values.
    n_nodes : int
        The number of nodes for the given variable considered.
    repeat : int
        Number of steps in each interval (for direct collocation, this is the number of nodes per interval)
        Only relevant for states or algebraic variables NOT for controls.

    Returns
    -------
    np.ndarray
        The concatenated bounds for all nodes
    """
    all_bounds = []
    for node in range(n_nodes):
        nlp.set_node_index(node)
        is_final_node = node == nlp.ns
        sub_node_bounds = []
        for sub_node in range(1 if is_final_node else repeat):
            collapsed = _compute_value_for_node(
                node,
                sub_node,
                default_value,
                repeat,
                variable_container,
                defined_values,
                scaling,
            )
            sub_node_bounds += [np.reshape(collapsed.T, (-1, 1))]
        sub_node_bounds = np.concatenate(sub_node_bounds, axis=0)
        all_bounds += [sub_node_bounds]
    # return np.concatenate(all_bounds, axis=0)
    return all_bounds


def _compute_value_for_node(
    node: int,
    sub_node: int,
    default_value: np.ndarray,
    repeat: int,
    variable_container: OptimizationVariableContainer,
    defined_values: dict,
    variable_scaling: "VariableScalingList",
) -> np.ndarray:
    """
    Compute the value for a specific node and sub node.

    Parameters
    ----------
    node: int
        The current node index of the interval
    sub_node: int
        The current interval node index (0 for the first node, 1 for the second node, etc.) within the interval ,
    default_value: np.ndarray
        The default value to use if the variable is not defined
        (either DEFAULT_MIN_BOUND, DEFAULT_MAX_BOUND, or DEFAULT_INITIAL_GUESS)
    repeat: int
        Number of steps for direct collocation, this is the number of nodes per interval
    variable_container: OptimizationVariableContainer
        The container for the optimization variables for states, controls, or algebraic variables that we refer to
    defined_values: dict
        The defined values for the variable, which can be min bounds, max bounds, or initial guesses.
        They are defined through InterpolationType and will be used to evaluate the value at the given node.
    variable_scaling: "VariableScalingList"
        The scaling factors for the variables, which are used to scale the evaluated values.

    Returns
    -------

    """
    collapsed_values = np.ndarray((variable_container.shape, 1))

    real_keys = [key for key in defined_values.keys() if key is not "None"]
    for key in real_keys:

        if defined_values[key].type == InterpolationType.ALL_POINTS:
            point = node * repeat + sub_node
        else:
            point = _get_interpolation_point(node, sub_node)

        value = (
            defined_values[key].evaluate_at(shooting_point=point, repeat=repeat)[:, np.newaxis]
            / variable_scaling[key].scaling
        )

        collapsed_values[variable_container[key].index, :] = value

    keys_not_defined = set(variable_container.keys()) - set(defined_values.keys())
    for key in keys_not_defined:
        collapsed_values[variable_container[key].index, :] = default_value

    return collapsed_values


def _get_interpolation_point(node: int, sub_node: int) -> int:
    """
    This function determines the interpolation point to use
        for InterpolationType except interpolationType.ALL_POINTS

    NOTE: This logic allows CONSTANT_WITH_FIRST_AND_LAST to work with OdeSolver.COLLOCATION,
    OdeSolver.COLLOCATION would also work for InterpolationType.CONSTANT, and ALL_POINTS, but not for the others.

    In the case of direct collocation, we ENFORCE the nodes within the first interval to take the value of
     the node of index 1 instead of 0.

    n = node, i = interval, p = returned point

    Standard Case:                  Collocation Case:
    (Direct multiple shooting)      (Direct Collocation)
    n0 n1 n2 n3 ... nN              n_{0,0} n_{0,1} n_{0,2} ... n_{0,N}, n_{1,0} ...
    |  |  |  |      |               /       |       |             |       |
    p0 p1 p2 p3 ... pN             0        1       1             1       1
                                      (enforced) (enforced)   (enforced)

    Parameters
    ----------
    node: int
        The current node index
    sub_node: int
        The current interval node index (0 for the first node, 1 for the second node, etc.),
            in the case of direct collocation more decision variable exist within an interval.
    Returns
    -------
    int
        The new point/node to use for the given node and sub_node

    """
    is_first_node = node == 0

    # always true for direct multiple shooting, but not for direct collocation
    is_first_subnode = sub_node == 0

    if is_first_node:
        return 0 if is_first_subnode else 1  # This the enforced hack
    else:
        return node


def dimension_check(
    variable_container: OptimizationVariableContainer, defined_values: dict, nlp_n_shooting: int, repeat: int
):
    for key in defined_values.real_keys():
        repeat_for_key = repeat if defined_values[key].type == InterpolationType.ALL_POINTS else 1
        n_shooting = nlp_n_shooting * repeat_for_key
        defined_values[key].check_and_adjust_dimensions(variable_container[key].cx.shape[0], n_shooting)
