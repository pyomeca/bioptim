"""
This script implements a custom dynamics to work with bioptim. Bioptim has a deep connection with biorbd,
but it is possible to use bioptim without biorbd.
This is an example of how to use bioptim with a custom dynamics.
"""

from casadi import MX, vertcat

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    ContactType,
)


def custom_dynamics(
    time: MX,
    states: MX,
    controls: MX,
    parameters: MX,
    algebraic_states: MX,
    numerical_timeseries: MX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    Parameters
    ----------
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase
    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    return DynamicsEvaluation(
        dxdt=vertcat(states[1], nlp.model.forward_dynamics(with_contact=False)(states[0], states[1], controls[0], [])),
        defects=None,
    )


def custom_configure_my_dynamics(ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None, contact_type:list[ContactType]=[]):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.
    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)
